"""LoRA training. Identical code path for the smoke model and the 7B run;
only configs/smoke.yaml overlays differ."""
import argparse, json, sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data_access as DA
from .env_assert import assert_pinned_env
from .collator import build_collator, encode, verify_against_reference
from .evaluate import logit_eval, letter_ids
from .diagnostics import ChoiceLossTrainer, step_counts, LN2
from .run_utils import (ROOT, load_configs, new_run_dir, snapshot_overlay,
                        pkg_versions, write_jsonl)
from transformers.trainer_utils import get_last_checkpoint

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def check_targets(model, targets):
    names = [n for n, _ in model.named_modules()]
    found = {t: sum(1 for n in names if n.endswith(t)) for t in targets}
    missing = [t for t, c in found.items() if c == 0]
    if missing:
        raise RuntimeError(f"LoRA target modules absent from architecture: {missing}")
    return found


def param_report(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    frozen_base = all(not p.requires_grad for n, p in model.named_parameters()
                      if "lora_" not in n)
    trainable_all_lora = all("lora_" in n for n, p in model.named_parameters()
                             if p.requires_grad)
    return {"trainable_params": tr, "total_params": tot,
            "trainable_pct": round(100 * tr / tot, 4),
            "all_base_params_frozen": frozen_base,
            "all_trainable_are_lora": trainable_all_lora}


def main(overlay, resume_run=None):
    cfg, ov = load_configs(overlay)
    assert_pinned_env(require_cuda=(cfg.get('device')=='cuda'))
    tag = cfg.get("run_tag", "run")
    if resume_run:
        run = Path(resume_run)
        if not run.exists():
            raise FileNotFoundError(f"cannot resume: {run} does not exist")
    else:
        run = new_run_dir(tag)
        snapshot_overlay(run, overlay)
    log = open(run / "logs" / "train.log", "a" if resume_run else "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s); log.write(s + "\n"); log.flush()

    device = cfg.get("device", "cuda")
    m, l, t = cfg["model"], cfg["lora"], cfg["train"]
    say(f"[run] {run.name}\n[model] {m['base_model_id']} @ {m['base_revision']}\n"
        f"[device] {device}  [dtype] {m['dtype']}")

    tok = AutoTokenizer.from_pretrained(m["tokenizer_id"], revision=m["base_revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- check 4: actual collator vs independent reference, >= 500 examples ----
    loss_mode = t.get("loss_mode", "completion")
    if cfg.get("run_family") == "acq" and loss_mode != "completion":
        raise RuntimeError(
            f"acquisition runs require full assistant-completion supervision; "
            f"got loss_mode={loss_mode!r}. The choice token, any formatting token "
            f"and the terminating <|im_end|> must all be supervised.")
    say(f"[loss] supervision mode: {loss_mode}"
        + ("  (choice + formatting + <|im_end|> all supervised)"
           if loss_mode == "completion" else ""))
    _loader = DA.load_acq_train if cfg.get("run_family") == "acq" else DA.load_train
    ref_rows = _loader(cfg["data"]["adapter"])
    coll_rep = verify_against_reference(tok, ref_rows, n=None, mode=loss_mode)
    say(f"[collator] {loss_mode}: exact match {coll_rep['n_exact_match']}/{coll_rep['n_compared']} "
        f"(ALL examples); masked {coll_rep['masked_fraction']:.4f}; "
        f"supervised/example {coll_rep['mean_supervised_per_example']}")
    (run / "metrics" / "collator_verification.json").write_text(json.dumps(coll_rep, indent=2))

    # ---- data ----
    rows = _loader(cfg["data"]["adapter"], cfg["data"].get("contexts"),
                   cfg["data"].get("train_examples_per_context"))
    ids = [encode(tok, r["messages"]) for r in rows]
    maxlen, msl = max(len(i) for i in ids), t["max_seq_length"]
    if maxlen > msl:
        raise RuntimeError(f"truncation would occur: {maxlen} > max_seq_length {msl}")
    say(f"[data] {len(rows)} train examples, contexts={sorted({r['context_id'] for r in rows})}, "
        f"max_len={maxlen} <= {msl} (no truncation)")
    ds = Dataset.from_dict({"input_ids": ids,
                            "attention_mask": [[1] * len(i) for i in ids]})

    # ---- model + LoRA ----
    base = AutoModelForCausalLM.from_pretrained(
        m["base_model_id"], revision=m["base_revision"],
        torch_dtype=DTYPES[m["dtype"]]).to(device)
    found = check_targets(base, l["target_modules"])
    say(f"[targets] {found}")
    peft_cfg = LoraConfig(r=l["r"], lora_alpha=l["lora_alpha"],
                          lora_dropout=l["lora_dropout"], bias=l["bias"],
                          task_type=l["task_type"], target_modules=l["target_modules"])
    model = get_peft_model(base, peft_cfg)
    pr = param_report(model)
    say(f"[params] trainable {pr['trainable_params']:,} / total {pr['total_params']:,} "
        f"({pr['trainable_pct']}%)  base_frozen={pr['all_base_params_frozen']}  "
        f"all_trainable_are_lora={pr['all_trainable_are_lora']}")
    assert pr["all_base_params_frozen"] and pr["all_trainable_are_lora"]

    # ---- train ----
    from trl import SFTConfig
    SC = step_counts(len(rows), t["per_device_train_batch_size"],
                     t["gradient_accumulation_steps"], t["num_train_epochs"])
    say("[steps] " + "  ".join(
        f"{k}={SC[k]}" for k in ("n_train_examples", "epochs",
                                 "per_device_train_batch_size",
                                 "gradient_accumulation_steps",
                                 "effective_batch_size", "microsteps_per_epoch",
                                 "optimizer_steps_per_epoch", "total_microsteps",
                                 "total_optimizer_steps",
                                 "total_example_presentations")))
    targs = SFTConfig(
        output_dir=str(run / "hf_out"), num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]), lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"], logging_steps=t.get("logging_steps", 10),
        seed=t["seed"], report_to=[],
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 2),
        max_seq_length=msl, dataset_kwargs={"skip_prepare_dataset": True},
        bf16=(m["dtype"] == "bfloat16"), remove_unused_columns=False)
    trainer = ChoiceLossTrainer(model=model, args=targs, train_dataset=ds,
                                data_collator=build_collator(tok, loss_mode),
                                tokenizer=tok,
                                choice_token_ids=letter_ids(tok))
    # ---- resume detection: latest VALID checkpoint (must carry trainer state) ----
    ckpt, resumed_from = None, 0
    out = Path(targs.output_dir)
    if out.exists():
        cand = get_last_checkpoint(str(out))
        if cand:
            need = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
            missing = [n for n in need if not (Path(cand) / n).exists()]
            if missing:
                say(f"[resume] IGNORING incomplete checkpoint {Path(cand).name}: missing {missing}")
            else:
                ckpt = cand
                st = json.loads((Path(cand) / "trainer_state.json").read_text())
                # Read state NOW: save_total_limit may rotate this checkpoint
                # away during training, so it may not exist afterwards.
                resumed_from = int(st["global_step"])
                rng = "rng_state.pth" in {f.name for f in Path(cand).iterdir()}
                say(f"[resume] resuming from {Path(cand).name} at global_step="
                    f"{st['global_step']} (optimizer+scheduler restored, rng_state={rng})")
    if ckpt is None:
        say("[resume] no valid checkpoint; starting from step 0")
    say("[train] starting")
    res = trainer.train(resume_from_checkpoint=ckpt)
    hist = [h for h in trainer.state.log_history if "loss" in h]
    last = hist[-1] if hist else {}
    say(f"[train] done. optimizer_steps={res.global_step} "
        f"(expected {SC['total_optimizer_steps']}), microsteps={SC['total_microsteps']}, "
        f"runtime={res.metrics['train_runtime']:.1f}s")
    say(f"[loss] full completion CE={last.get('loss','n/a')}   "
        f"DIAGNOSTIC choice-token CE={last.get('choice_ce','n/a')} "
        f"(chance ln2={LN2:.6f})   choice train acc={last.get('choice_train_acc','n/a')}   "
        f"|logit A-B|={last.get('choice_logit_absdiff','n/a')}")
    if res.global_step != SC["total_optimizer_steps"]:
        say(f"[steps] WARNING: global_step {res.global_step} != predicted "
            f"{SC['total_optimizer_steps']}")
    for h in hist:
        assert h["loss"] == h["loss"], "NaN loss encountered"

    # ---- pre-save deterministic eval (same model object, before serialisation) ----
    model.eval()
    pv = DA.load_policy_val(cfg["data"].get("contexts"),
                            cfg["eval"]["policy_val_pairs_per_context"])
    presave = logit_eval(model, tok, pv, device, cfg["eval"]["batch_size"])
    write_jsonl(run / "presave" / "adapter_logits_presave.jsonl", presave)
    say(f"[presave] logit eval on {len(presave)} policy_val prompts")

    # ---- save adapter ----
    model.save_pretrained(run / "adapter")
    say(f"[save] adapter -> {run/'adapter'}  files={sorted(p.name for p in (run/'adapter').iterdir())}")

    manifest = {
        "run_id": run.name, "overlay": overlay, "smoke": cfg.get("smoke", False),
        "device": device, "model": m, "lora": l, "train": t,
        "data": cfg["data"], "eval_cfg": cfg["eval"],
        "packages": pkg_versions(),
        "loss_mode": loss_mode,
        "collator_verification": coll_rep,
        "params": pr, "lora_target_counts": found,
        "max_example_tokens": maxlen, "truncation": False,
        "checkpointing": {"save_strategy": targs.save_strategy,
                          "save_steps": targs.save_steps,
                          "save_total_limit": targs.save_total_limit,
                          "resumed_from_checkpoint": ckpt,
                          "resumed_from_global_step": resumed_from,
                          "checkpoints_on_disk": sorted(
                              p.name for p in out.glob("checkpoint-*")) if out.exists() else []},
        "step_counts": SC,
        "final_diagnostics": {k: last.get(k) for k in
                              ("loss", "choice_ce", "choice_train_acc",
                               "choice_logit_absdiff")},
        "choice_ce_chance_ln2": LN2,
        "train_result": {"steps": res.global_step,
                         "runtime_s": res.metrics["train_runtime"],
                         "loss_history": hist},
        "n_train_examples": len(rows), "n_presave_eval": len(presave),
        "data_files_read": DA.REC.assert_no_reserved(),
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    say(f"[guard] data files read: {manifest['data_files_read']}")
    say("[ok] " + run.name)
    print(run.name)
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--resume-run", default=None,
                    help="existing run dir to resume into (reuses its config snapshot)")
    a = ap.parse_args()
    main(a.overlay, a.resume_run)

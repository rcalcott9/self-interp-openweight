"""Free-generation validity check for base or adapter, run as its own process.

Reports malformed rate, nontermination rate, repeated-choice rate, the length
distribution and representative raw outputs, against the frozen thresholds in
configs/acquisition_eval.yaml.
"""
import argparse, json
from pathlib import Path

import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data_access as DA
from .evaluate import generation_report, summarise_generation
from .run_utils import ROOT, load_configs, write_jsonl

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--which", choices=["base", "adapter"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default=None, help="output name suffix, e.g. 'pretrain'")
    a = ap.parse_args()

    cfg, _ = load_configs(a.overlay)
    ev = yaml.safe_load((ROOT / "configs" / "acquisition_eval.yaml").read_text())["free_generation"]
    m, run = cfg["model"], Path(a.run)
    device = cfg.get("device", "cuda")

    tok = AutoTokenizer.from_pretrained(m["tokenizer_id"], revision=m["base_revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        m["base_model_id"], revision=m["base_revision"],
        torch_dtype=DTYPES[m["dtype"]]).to(device)
    if a.which == "adapter":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, run / "adapter").to(device)
    model.eval()

    rows = DA.load_policy_val(cfg["data"].get("contexts"),
                              ev["pairs_per_context"])
    recs = generation_report(model, tok, rows, device,
                             max_new_tokens=ev["max_new_tokens"],
                             batch_size=cfg["eval"]["batch_size"])
    s = summarise_generation(recs)
    th = ev["thresholds"]
    s["thresholds"] = th
    s["passes"] = {
        "malformed": s["malformed_rate"] <= th["max_malformed_rate"],
        "nontermination": s["nontermination_rate"] <= th["max_nontermination_rate"],
        "repeated_choice": s["repeated_choice_rate"] <= th["max_repeated_choice_rate"],
    }
    s["all_pass"] = all(s["passes"].values())
    s["examples"] = [{k: r[k] for k in ("row_id", "raw_output", "raw_with_specials",
                                        "outcome", "terminated", "n_generated_tokens")}
                     for r in recs[:5]]
    name = f"generation_{a.which}" + (f"_{a.tag}" if a.tag else "")
    write_jsonl(run / "raw_outputs" / f"{name}.jsonl", recs)
    (run / "metrics" / f"{name}.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: v for k, v in s.items() if k != "examples"}, indent=2))
    print("\nrepresentative raw outputs:")
    for e in s["examples"]:
        print(f"  {e['outcome']:9s} term={str(e['terminated']):5s} "
              f"ntok={e['n_generated_tokens']:>3}  {e['raw_with_specials']!r}")
    if not s["all_pass"]:
        raise SystemExit(f"FREE-GENERATION CHECK FAILED ({a.which}): {s['passes']}")
    print(f"\nFREE-GENERATION CHECK PASSED ({a.which})")


if __name__ == "__main__":
    main()

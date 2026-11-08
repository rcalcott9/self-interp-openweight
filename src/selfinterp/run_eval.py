"""Evaluate base or adapter in a FRESH process, from the run's frozen manifest.
Writes base and adapter results to separate files; raw output preserved."""
import argparse, json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data_access as DA
from .env_assert import assert_pinned_env
from .evaluate import logit_eval, sample_eval, freegen_eval
from .run_utils import ROOT, write_jsonl

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def main(run_dir, which):
    run = Path(run_dir)
    mf = json.loads((run / "manifest.json").read_text())
    assert_pinned_env(require_cuda=(mf['device']=='cuda'))
    m, ev, dev = mf["model"], mf["eval_cfg"], mf["device"]
    tok = AutoTokenizer.from_pretrained(m["tokenizer_id"], revision=m["base_revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        m["base_model_id"], revision=m["base_revision"],
        torch_dtype=DTYPES[m["dtype"]]).to(dev)
    if which == "adapter":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, run / "adapter").to(dev)
        print(f"[load] adapter from {run/'adapter'}")
    model.eval()

    ctxs = mf["data"].get("contexts")
    pv = DA.load_policy_val(ctxs, ev["policy_val_pairs_per_context"])
    print(f"[eval:{which}] {len(pv)} policy_val prompts")

    lg = logit_eval(model, tok, pv, dev, ev["batch_size"])
    write_jsonl(run / "raw_outputs" / f"{which}_logits.jsonl", lg)

    sp = ev["sampling"]                       # frozen scientific block, authoritative
    sub = [r for r in pv if r["instance_id"] < sp["pairs_per_context"]]
    sm = sample_eval(model, tok, sub, dev, n_samples=sp["n_samples"],
                     temperature=sp["temperature"], top_p=sp["top_p"],
                     top_k=sp["top_k"], max_new_tokens=sp["max_new_tokens"],
                     batch_size=ev["batch_size"])
    write_jsonl(run / "raw_outputs" / f"{which}_samples.jsonl", sm)

    fg_rows = [r for r in pv if r["instance_id"] < ev["freegen_pairs_per_context"]]
    fg = freegen_eval(model, tok, fg_rows, dev, batch_size=ev["batch_size"])
    write_jsonl(run / "raw_outputs" / f"{which}_freegen.jsonl", fg)

    from collections import Counter
    val = {
        "which": which, "n_logit": len(lg),
        "n_samples": len(sm), "sample_outcomes": dict(Counter(x["outcome"] for x in sm)),
        "sample_malformed_rate": round(sum(x["outcome"] == "MALFORMED" for x in sm) / max(len(sm), 1), 4),
        "n_freegen": len(fg), "freegen_outcomes": dict(Counter(x["outcome"] for x in fg)),
        "freegen_malformed_rate": round(sum(x["outcome"] == "MALFORMED" for x in fg) / max(len(fg), 1), 4),
        "sampling_params_used": {k: sp[k] for k in
            ("temperature", "top_p", "top_k", "n_samples", "max_new_tokens", "pairs_per_context")},
        "data_files_read": DA.REC.assert_no_reserved(),
    }
    (run / "metrics" / f"{which}_validity.json").write_text(json.dumps(val, indent=2))
    print(json.dumps(val, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", choices=["base", "adapter"], required=True)
    a = ap.parse_args()
    main(a.run, a.which)

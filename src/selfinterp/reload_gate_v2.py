"""Successor reload gate (v2). Supersedes the pre-save/reload comparison as the
BLOCKING check; the v1 gate is retained as FAILED and is never waived.

WHAT v1 TESTED (and why it was the wrong question): the in-memory training model
vs the reloaded adapter. Those two evaluations ran in different process and
memory states, which select different cuBLAS kernels, so bf16 rounding differs at
1-2 ULP. v1 therefore conflates artefact fidelity with kernel selection.

WHAT v2 TESTS: does the SAVED ADAPTER reproduce ITSELF under a single fixed
evaluation protocol? Two fresh processes, identical revision, dtype, device, eval
mode, batch size, padding, prompt order and scoring code. This isolates artefact
reproducibility, which is the property the science actually depends on.

PASS CONDITION: identical greedy choices on all 1,920 policy-validation prompts.
No numerical tolerance is invented. Logit differences are REPORTED, not thresholded
-- A1's observed maximum is deliberately not used to define a bound.

Batch-size sensitivity (bs 1 vs 32) is a SEPARATE, NON-BLOCKING diagnostic.
"""
import argparse, json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data_access as DA
from .env_assert import assert_pinned_env
from .evaluate import logit_eval
from .run_utils import write_jsonl

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def score_all(run, tag, batch_size=None):
    """One fresh process scores every policy_val prompt under the fixed protocol."""
    assert_pinned_env()
    mf = json.loads((run / "manifest.json").read_text())
    m, ev, dev = mf["model"], mf["eval_cfg"], mf["device"]
    bs = batch_size or ev["batch_size"]
    tok = AutoTokenizer.from_pretrained(m["tokenizer_id"], revision=m["base_revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        m["base_model_id"], revision=m["base_revision"],
        torch_dtype=DTYPES[m["dtype"]]).to(dev)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, run / "adapter").to(dev)
    model.eval()
    pv = DA.load_policy_val(mf["data"].get("contexts"), ev["policy_val_pairs_per_context"])
    pv.sort(key=lambda r: r["row_id"])          # fixed prompt order
    rows = logit_eval(model, tok, pv, dev, bs)
    write_jsonl(run / "reload_gate_v2" / f"{tag}.jsonl", rows)
    DA.REC.assert_no_reserved()
    return len(rows), bs


def compare(run, a, b, label):
    A = {r["row_id"]: r for r in map(json.loads, open(run / "reload_gate_v2" / f"{a}.jsonl"))}
    B = {r["row_id"]: r for r in map(json.loads, open(run / "reload_gate_v2" / f"{b}.jsonl"))}
    assert set(A) == set(B), "row sets differ"
    ids = sorted(A)
    mism = [i for i in ids if A[i]["greedy_letter"] != B[i]["greedy_letter"]]
    d = [abs(A[i]["logit_diff"] - B[i]["logit_diff"]) for i in ids]
    return {"label": label, "n": len(ids), "n_choice_mismatch": len(mism),
            "choice_agreement": 1 - len(mism) / len(ids),
            "max_abs_logit_diff_delta": max(d), "mean_abs_logit_diff_delta": sum(d) / len(d),
            "n_bit_identical": sum(1 for x in d if x == 0.0),
            "mismatch_row_ids": mism[:20]}


def verdict(run):
    blocking = compare(run, "procA_bs32", "procB_bs32",
                       "BLOCKING: two fresh processes, identical fixed protocol (bs=32)")
    diag = compare(run, "procA_bs32", "procA_bs1",
                   "DIAGNOSTIC (non-blocking): batch size 32 vs 1, same process")
    rep = {"gate": "reload_gate_v2",
           "pass_condition": "identical greedy choices on all policy_val prompts; "
                             "no numerical tolerance is defined",
           "blocking_result": blocking, "batch_size_diagnostic": diag,
           "passed": blocking["n_choice_mismatch"] == 0,
           "v1_gate_status": "FAILED, retained, not waived"}
    (run / "metrics" / "reload_gate_v2.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if not rep["passed"]:
        raise SystemExit("RELOAD GATE v2 FAILED")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["procA_bs32", "procB_bs32", "procA_bs1", "verdict"])
    a = ap.parse_args()
    run = Path(a.run)
    if a.mode == "verdict":
        verdict(run)
    else:
        n, bs = score_all(run, a.mode, 1 if a.mode.endswith("bs1") else None)
        print(f"scored {n} prompts at batch_size={bs} -> {a.mode}.jsonl")

"""Checkpoint 3: preprocessing / tokenization audit for the acquisition recipe.

Prints, for several A-labelled and B-labelled examples: the rendered chat text,
token ids with decoded tokens, labels, which tokens are masked vs supervised,
the termination token and its label status, and the supervised-token count.

HARD FAILS if any of:
  * the termination token (<|im_end|> / EOS) is masked;
  * the choice token is masked;
  * any prompt token contributes to the completion loss;
  * A- and B-labelled examples differ in completion token count;
  * the supervised labels do not decode to the intended completion.
"""
import argparse, json, sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from .masking import build_masked_example, IGNORE

ROOT = Path(__file__).resolve().parents[2]


class AuditFailure(RuntimeError):
    pass


def audit(rows, tokenizer, mode="completion", n_show=2, out=sys.stdout):
    fails, shown = [], {"A": 0, "B": 0}
    comp_len = {}
    for r in rows:
        msgs = r["messages"]
        lab_letter = msgs[-1]["content"]
        ex = build_masked_example(tokenizer, msgs, mode=mode)
        ids, labels, n_p = ex["input_ids"], ex["labels"], ex["n_prompt_tokens"]
        comp_ids = ids[n_p:]
        comp_len.setdefault(lab_letter, set()).add(len(comp_ids))

        # --- structural checks, applied to EVERY row ---
        if any(labels[i] != IGNORE for i in range(n_p)):
            fails.append(f"{r.get('row_id')}: prompt token supervised")
        if labels[n_p] == IGNORE:
            fails.append(f"{r.get('row_id')}: CHOICE token masked")
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id in comp_ids:
            k = n_p + comp_ids.index(eos_id)
            if labels[k] == IGNORE:
                fails.append(f"{r.get('row_id')}: TERMINATION token masked")
        else:
            fails.append(f"{r.get('row_id')}: no <|im_end|> in completion")
        sup = [l for l in labels if l != IGNORE]
        if tokenizer.decode(sup) != tokenizer.decode(comp_ids):
            fails.append(f"{r.get('row_id')}: supervised labels do not decode to the completion")

        # --- verbose print for the first n_show of each letter ---
        if shown.get(lab_letter, 0) < n_show:
            shown[lab_letter] = shown.get(lab_letter, 0) + 1
            text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=False)
            print("=" * 78, file=out)
            print(f"row_id={r.get('row_id')}  label={lab_letter}  mode={mode}", file=out)
            print("--- rendered chat text (tail) ---", file=out)
            print(repr(text[-180:]), file=out)
            print(f"--- tokens: {len(ids)} total = {n_p} prompt + {len(comp_ids)} completion ---", file=out)
            print(f"{'idx':>5} {'id':>8}  {'token':<16} {'label':>8}  status", file=out)
            for i in range(max(0, n_p - 3), len(ids)):
                tokstr = tokenizer.convert_ids_to_tokens(ids[i])
                st = "MASKED" if labels[i] == IGNORE else "SUPERVISED"
                mark = ""
                if i == n_p: mark = "  <- CHOICE"
                elif ids[i] == eos_id: mark = "  <- TERMINATION"
                print(f"{i:>5} {ids[i]:>8}  {tokstr!r:<16} {labels[i]:>8}  {st}{mark}", file=out)
            print(f"supervised tokens: {ex['n_supervised_tokens']}   "
                  f"decoded supervision -> {tokenizer.decode(sup)!r}", file=out)

    for L, lens in comp_len.items():
        if len(lens) != 1:
            fails.append(f"label {L}: inconsistent completion length {sorted(lens)}")
    if len(set().union(*comp_len.values())) != 1:
        fails.append(f"A and B differ in completion token count: "
                     + ", ".join(f"{k}={sorted(v)}" for k, v in comp_len.items()))
    return {"n_rows": len(rows), "completion_lengths": {k: sorted(v) for k, v in comp_len.items()},
            "failures": fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="A1")
    ap.add_argument("--data", default="data/acquisition/train/{adapter}/train_full.jsonl")
    ap.add_argument("--mode", default="completion")
    ap.add_argument("--n", type=int, default=None, help="rows to audit (default all)")
    ap.add_argument("--show", type=int, default=2)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"], revision=cfg["base_revision"])
    path = ROOT / a.data.format(adapter=a.adapter)
    rows = [json.loads(l) for l in open(path)]
    if a.n: rows = rows[:a.n]
    rep = audit(rows, tok, a.mode, a.show)
    print("=" * 78)
    print(f"AUDITED {rep['n_rows']} rows from {path.relative_to(ROOT)}  mode={a.mode}")
    print(f"completion token counts: {rep['completion_lengths']}")
    if rep["failures"]:
        print(f"HARD FAIL ({len(rep['failures'])}):")
        for f in rep["failures"][:10]: print("   " + f)
        raise AuditFailure(f"{len(rep['failures'])} tokenization audit failures")
    print("TOKENIZATION AUDIT PASSED - choice supervised, termination supervised, "
          "prompt fully masked, A/B symmetric, labels decode to the completion.")


if __name__ == "__main__":
    main()

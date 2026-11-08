"""The ACTUAL training collators, plus verification against the reference.

Both collators locate the completion boundary with TRL's
DataCollatorForCompletionOnlyLM, which searches for the response-template TOKEN
SUBSEQUENCE. selfinterp.masking is an INDEPENDENT implementation that computes
the boundary by PREFIX-LENGTH DIFFERENCING. They share no boundary-finding code,
so agreement is genuine cross-validation.

ChoiceOnlyCollator additionally keeps only the FIRST supervised position.
"""
import torch
from trl import DataCollatorForCompletionOnlyLM

from .masking import build_masked_example, IGNORE

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


class ChoiceOnlyCollator:
    """v2: supervise exactly one token per example -- the A/B letter."""

    def __init__(self, tokenizer, response_template=RESPONSE_TEMPLATE):
        self.inner = DataCollatorForCompletionOnlyLM(
            response_template=response_template, tokenizer=tokenizer)

    def __call__(self, features):
        batch = self.inner(features)
        labels = batch["labels"]
        for r in range(labels.shape[0]):
            nz = (labels[r] != IGNORE).nonzero(as_tuple=True)[0]
            if nz.numel() == 0:
                raise ValueError("no supervised token found in a row")
            keep = nz[0].item()
            row = torch.full_like(labels[r], IGNORE)
            row[keep] = labels[r][keep]
            labels[r] = row
        batch["labels"] = labels
        return batch


def build_collator(tokenizer, mode="completion", response_template=RESPONSE_TEMPLATE):
    if mode == "completion":
        return DataCollatorForCompletionOnlyLM(response_template=response_template,
                                               tokenizer=tokenizer)
    if mode == "choice_only":
        return ChoiceOnlyCollator(tokenizer, response_template)
    raise ValueError(f"unknown loss mode {mode!r}")


def encode(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def verify_against_reference(tokenizer, rows, n=None, mode="completion",
                             response_template=RESPONSE_TEMPLATE, progress_every=None):
    """Token-for-token comparison, one example per batch so padding cannot hide a
    disagreement. n=None means EVERY example. Raises on any mismatch."""
    coll = build_collator(tokenizer, mode, response_template)
    subset = rows if n is None else rows[:n]
    n_ok = n_tok = n_masked = n_sup = 0
    mismatches = []
    for i, r in enumerate(subset):
        msgs = r["messages"]
        ids = encode(tokenizer, msgs)
        ref = build_masked_example(tokenizer, msgs, mode=mode)
        if ids != ref["input_ids"]:
            mismatches.append({"row_id": r.get("row_id"), "why": "input_ids differ"})
            continue
        got = coll([{"input_ids": ids, "attention_mask": [1] * len(ids)}])["labels"][0].tolist()
        if got != ref["labels"]:
            d = [k for k, (a, b) in enumerate(zip(got, ref["labels"])) if a != b]
            mismatches.append({"row_id": r.get("row_id"), "n_diff": len(d),
                               "first_diff_idx": d[:5]})
            continue
        n_ok += 1
        n_tok += len(ids)
        n_masked += sum(1 for l in got if l == IGNORE)
        n_sup += sum(1 for l in got if l != IGNORE)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    ...{i+1}/{len(subset)} compared, {len(mismatches)} mismatches", flush=True)
    rep = {"mode": mode, "n_compared": len(subset), "n_exact_match": n_ok,
           "n_mismatch": len(mismatches), "mismatches": mismatches[:5],
           "total_tokens": n_tok, "masked_tokens": n_masked, "supervised_tokens": n_sup,
           "masked_fraction": round(n_masked / n_tok, 6) if n_tok else None,
           "mean_supervised_per_example": round(n_sup / n_ok, 4) if n_ok else None,
           "response_template": response_template}
    if mismatches:
        raise AssertionError(f"collator disagrees with reference on {len(mismatches)} "
                             f"of {rep['n_compared']} ({mode}): {mismatches[:3]}")
    return rep

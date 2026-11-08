"""Evaluation shared by the smoke run and the 7B run.

Three measures, per the frozen eval config:
  1. A-vs-B logit difference  (continuous preference; no parsing, no failure mode)
  2. repeated stochastic sampling (behavioural ground truth; A / B / MALFORMED)
  3. greedy free generation (format check)
Raw model output is written BEFORE parsing. Malformed output is an explicit
outcome and is never coerced to A or B.
"""
import json
from pathlib import Path

import torch


def letter_ids(tokenizer, letters=("A", "B")):
    out = {}
    for L in letters:
        base = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
        p = tokenizer.apply_chat_template(base, tokenize=True, add_generation_prompt=True)
        f = tokenizer.apply_chat_template(base + [{"role": "assistant", "content": L}],
                                          tokenize=True, add_generation_prompt=False)
        out[L] = f[len(p)]
    assert len(set(out.values())) == len(out), f"letter ids collide: {out}"
    return out


def _prompt_texts(tokenizer, rows):
    return [tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                          add_generation_prompt=True) for r in rows]


@torch.no_grad()
def logit_eval(model, tokenizer, rows, device, batch_size=8, letters=("A", "B")):
    ids = letter_ids(tokenizer, letters)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    texts = _prompt_texts(tokenizer, rows)
    for i in range(0, len(rows), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                        padding=True, add_special_tokens=False).to(device)
        lg = model(**enc).logits[:, -1, :].float()
        for j, r in enumerate(rows[i:i + batch_size]):
            la, lb = lg[j, ids["A"]].item(), lg[j, ids["B"]].item()
            out.append({"row_id": r["row_id"], "context_id": r["context_id"],
                        "instance_id": r["instance_id"],
                        "presentation_order": r["presentation_order"],
                        "pair_hash": r["pair_hash"],
                        "logit_A": la, "logit_B": lb, "logit_diff": la - lb,
                        "greedy_letter": "A" if la > lb else "B"})
    return out


@torch.no_grad()
def sample_eval(model, tokenizer, rows, device, n_samples, temperature,
                top_p, top_k, max_new_tokens, batch_size=8, seed=0):
    """Unconstrained sampling. Every generation parameter is supplied by the
    caller from the frozen eval config - there are no defaults to fall back on.
    Raw decoded text is preserved before classification."""
    gen_params = {"do_sample": True, "temperature": temperature, "top_p": top_p,
                  "top_k": top_k, "max_new_tokens": max_new_tokens}
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = _prompt_texts(tokenizer, rows)
    out = []
    for i in range(0, len(rows), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                        padding=True, add_special_tokens=False).to(device)
        for s in range(n_samples):
            torch.manual_seed(seed + s)
            gen = model.generate(**enc, **gen_params,
                                 pad_token_id=tokenizer.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, r in enumerate(rows[i:i + batch_size]):
                raw = tokenizer.decode(new[j], skip_special_tokens=True)
                out.append({"row_id": r["row_id"], "context_id": r["context_id"],
                            "instance_id": r["instance_id"],
                            "presentation_order": r["presentation_order"],
                            "pair_hash": r["pair_hash"], "sample_index": s,
                            "raw_output": raw, "outcome": classify(raw),
                            "gen_params": gen_params})
    return out


@torch.no_grad()
def freegen_eval(model, tokenizer, rows, device, max_new_tokens=8, batch_size=8):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = _prompt_texts(tokenizer, rows)
    out = []
    for i in range(0, len(rows), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                        padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, r in enumerate(rows[i:i + batch_size]):
            raw = tokenizer.decode(new[j], skip_special_tokens=True)
            out.append({"row_id": r["row_id"], "context_id": r["context_id"],
                        "raw_output": raw, "outcome": classify(raw)})
    return out


def classify(raw):
    """Exactly 'A' or 'B' after stripping whitespace; anything else MALFORMED.
    Never coerced."""
    s = raw.strip()
    return s if s in ("A", "B") else "MALFORMED"


# ---------------------------------------------------------------------------
# v3 ACQUISITION: termination-aware free generation.
#
# The v1/v2 evaluators decoded with skip_special_tokens=True, which DISCARDS
# <|im_end|> before inspection - so whether the model terminated was not
# measurable, and A2-v2's nontermination could only be inferred indirectly from
# repeated letters. This records the raw token ids, so termination, generated
# length and repetition are measured directly rather than guessed.
#
# max_new_tokens must be generous (>= 16): a tight budget hides nontermination
# by truncating it into something that merely looks malformed.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generation_report(model, tokenizer, rows, device, max_new_tokens=32,
                      batch_size=8, letters=("A", "B")):
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    alt_eos = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = _prompt_texts(tokenizer, rows)
    out = []
    for i in range(0, len(rows), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                        padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=[t for t in {eos_id, alt_eos} if t is not None])
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, r in enumerate(rows[i:i + batch_size]):
            ids = new[j].tolist()
            term_at = next((k for k, t in enumerate(ids) if t in (eos_id, alt_eos)), None)
            body = ids[:term_at] if term_at is not None else ids
            raw_all = tokenizer.decode(ids, skip_special_tokens=False)
            raw = tokenizer.decode(body, skip_special_tokens=True)
            s = raw.strip()
            n_letters = sum(s.count(L) for L in letters)
            out.append({
                "row_id": r["row_id"], "context_id": r["context_id"],
                "presentation_order": r.get("presentation_order"),
                "raw_output": raw, "raw_with_specials": raw_all,
                "outcome": classify(raw),
                "terminated": term_at is not None,
                "n_generated_tokens": len(body),
                "n_tokens_to_eos": term_at,
                "hit_token_cap": term_at is None,
                "repeated_choice": n_letters > 1,
                "n_choice_chars": n_letters,
                "max_new_tokens": max_new_tokens,
            })
    return out


def summarise_generation(recs):
    """Aggregate the acquisition free-generation criteria."""
    n = max(len(recs), 1)
    lens = sorted(r["n_generated_tokens"] for r in recs)
    def q(p):
        return lens[min(int(p * (len(lens) - 1)), len(lens) - 1)] if lens else 0
    valid = [r for r in recs
             if r["outcome"] in ("A", "B") and r["terminated"] and not r["repeated_choice"]]
    return {
        "n": len(recs),
        "valid_rate": len(valid) / n,
        "malformed_rate": sum(r["outcome"] == "MALFORMED" for r in recs) / n,
        "nontermination_rate": sum(not r["terminated"] for r in recs) / n,
        "repeated_choice_rate": sum(r["repeated_choice"] for r in recs) / n,
        "hit_token_cap_rate": sum(r["hit_token_cap"] for r in recs) / n,
        "len_min": lens[0] if lens else 0, "len_p50": q(0.5),
        "len_p95": q(0.95), "len_max": lens[-1] if lens else 0,
        "outcome_counts": {k: sum(r["outcome"] == k for r in recs)
                           for k in ("A", "B", "MALFORMED")},
    }

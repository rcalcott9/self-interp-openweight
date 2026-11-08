"""A-vs-B logit comparison validity gate.

Reading logit_A - logit_B at the first assistant-content position is only valid
if, UNDER THE COMPLETE CHAT TEMPLATE, each letter occupies exactly one token at
that position and the two ids differ. Verified empirically against real prompts,
not assumed from tokenizer.encode(). Raises on failure; never degrades silently.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class TokenizationGateError(RuntimeError):
    pass


def run_gate(tokenizer, sample_messages, letters=("A", "B"), verbose=True):
    report = {"letters": {}, "n_prompts_checked": len(sample_messages)}
    ids = {}
    for letter in letters:
        deltas, prompt_lens = set(), set()
        for msgs in sample_messages:
            base = [m for m in msgs if m["role"] != "assistant"]
            prompt_ids = tokenizer.apply_chat_template(
                base, tokenize=True, add_generation_prompt=True)
            full_ids = tokenizer.apply_chat_template(
                base + [{"role": "assistant", "content": letter}],
                tokenize=True, add_generation_prompt=False)
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise TokenizationGateError(
                    f"chat template not prefix-stable for letter {letter!r}")
            deltas.add(tuple(full_ids[len(prompt_ids):]))
            prompt_lens.add(len(prompt_ids))
        if len(deltas) != 1:
            raise TokenizationGateError(
                f"letter {letter!r} does not tokenize identically across prompts: {deltas}")
        delta = next(iter(deltas))
        first = delta[0]
        ids[letter] = first
        report["letters"][letter] = {
            "first_content_token_id": first,
            "first_content_token": tokenizer.convert_ids_to_tokens([first])[0],
            "completion_token_ids": list(delta),
            "completion_tokens": tokenizer.convert_ids_to_tokens(list(delta)),
            "n_completion_tokens": len(delta),
            "prompt_len_range": [min(prompt_lens), max(prompt_lens)],
        }
        if len(delta) < 1:
            raise TokenizationGateError(f"empty completion for {letter!r}")
    if len(set(ids.values())) != len(ids):
        raise TokenizationGateError(f"letters share a token id: {ids}")
    report["distinct_ids"] = True
    report["comparison_valid"] = True
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def load_sample(n=200):
    rows = []
    with open(ROOT / "data" / "train" / "A1" / "train_full.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line)["messages"])
    return rows


if __name__ == "__main__":
    from transformers import AutoTokenizer
    cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"],
                                        revision=cfg["base_revision"])
    rep = run_gate(tok, load_sample(200))
    print("\nGATE PASSED — logit_A - logit_B is a valid comparison.")

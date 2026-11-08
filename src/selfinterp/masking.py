"""Reference loss-masking implementations.

Two modes:

  "completion"  (v1) supervise every completion token: [letter, <|im_end|>, \n]
  "choice_only" (v2) supervise ONLY the first assistant content token (the A/B
                     letter). The end token and newline are masked.

MEASURED MOTIVATION for v2 (2026-09-03, A1/A2 loss decomposition):
    choice token  A1 0.642492   A2 0.693204   (ln 2 = 0.693147)
    end token     A1 0.000002   A2 0.000001
    newline       A1 0.000000   A2 0.000000
The two non-decision tokens are already solved, so under v1 the reported loss is
choice/3 and a chance-level policy reads as 0.231 rather than 0.693. v2 makes the
decision loss the reported loss. It is a REPORTING/scale change; it is not
expected to fix acquisition (checkpoint-500 of A2 was already flat at 0.694406).

Tokenizer-agnostic and independent of TRL: render the prompt-only chat and the
full chat with the same template, assert prefix stability, mask accordingly.
"""
IGNORE = -100
MODES = ("completion", "choice_only")


def build_masked_example(tokenizer, messages, max_length=None, mode="completion"):
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    assert messages[-1]["role"] == "assistant", "last turn must be the completion"
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat template is not prefix-stable; masking is unsafe for this tokenizer.")
    n_p = len(prompt_ids)
    labels = [IGNORE] * len(full_ids)
    if mode == "completion":
        for i in range(n_p, len(full_ids)):
            labels[i] = full_ids[i]
        n_sup = len(full_ids) - n_p
    else:                                     # choice_only
        if len(full_ids) <= n_p:
            raise ValueError("no completion tokens present")
        labels[n_p] = full_ids[n_p]           # the A/B letter, and nothing else
        n_sup = 1
    if max_length:
        full_ids, labels = full_ids[:max_length], labels[:max_length]
    if all(l == IGNORE for l in labels):
        raise ValueError("no supervised tokens remain after truncation")
    return {"input_ids": full_ids, "labels": labels, "mode": mode,
            "n_prompt_tokens": n_p, "n_supervised_tokens": n_sup,
            "n_completion_tokens": len(full_ids) - n_p}

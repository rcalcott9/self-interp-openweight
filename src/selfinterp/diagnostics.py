"""Diagnostic-only training instrumentation.

`ChoiceLossTrainer` trains on the FULL assistant-completion cross-entropy - the
gradient is exactly the one SFTTrainer would produce - and additionally records
the cross-entropy of the FIRST supervised token (the A/B choice) and the A-vs-B
logit difference at that position. Both are computed under torch.no_grad() from
detached tensors and are never added to the returned loss.

Chance for a balanced binary choice is ln 2 = 0.6931471805599453.

`step_counts` resolves the step-terminology ambiguity once: every run summary
reports examples, epochs, microsteps, gradient-accumulation, and optimizer steps
as distinct named quantities.
"""
import math

import torch
from trl import SFTTrainer

LN2 = math.log(2.0)


def step_counts(n_examples, per_device_batch, grad_accum, epochs, world_size=1):
    """Exact, and named. `optimizer_steps` is what HF calls global_step."""
    eff = per_device_batch * grad_accum * world_size
    micro_per_epoch = math.ceil(n_examples / (per_device_batch * world_size))
    opt_per_epoch = math.floor(micro_per_epoch / grad_accum)
    return {
        "n_train_examples": int(n_examples),
        "epochs": int(epochs),
        "per_device_train_batch_size": int(per_device_batch),
        "gradient_accumulation_steps": int(grad_accum),
        "world_size": int(world_size),
        "effective_batch_size": int(eff),
        "microsteps_per_epoch": int(micro_per_epoch),
        "optimizer_steps_per_epoch": int(opt_per_epoch),
        "total_microsteps": int(micro_per_epoch * epochs),
        "total_optimizer_steps": int(opt_per_epoch * epochs),
        "total_example_presentations": int(n_examples * epochs),
        "note": ("total_optimizer_steps is HuggingFace `global_step` and is what a "
                 "run log's `steps=` field reports. It is NOT the number of forward/"
                 "backward passes (microsteps) nor of example presentations."),
    }


class ChoiceLossTrainer(SFTTrainer):
    """SFTTrainer + detached first-choice-token diagnostics. Gradients unchanged."""

    def __init__(self, *a, choice_token_ids=None, **kw):
        super().__init__(*a, **kw)
        self._choice_ids = choice_token_ids or {}
        self.reset_choice_meter()

    def reset_choice_meter(self):
        self._m = {"n": 0, "ce_sum": 0.0, "absdiff_sum": 0.0, "correct": 0}

    def choice_metrics(self):
        n = max(self._m["n"], 1)
        return {"choice_ce": self._m["ce_sum"] / n,
                "choice_logit_absdiff": self._m["absdiff_sum"] / n,
                "choice_train_acc": self._m["correct"] / n,
                "choice_ce_chance_ln2": LN2,
                "n_scored": self._m["n"]}

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = super().compute_loss(model, inputs, return_outputs=True, **kw)
        loss, outputs = out if isinstance(out, tuple) else (out, None)
        if outputs is not None and "labels" in inputs:
            with torch.no_grad():
                logits = outputs.logits.detach()
                labels = inputs["labels"]
                sl, sy = logits[:, :-1, :], labels[:, 1:]
                valid = sy != -100
                has = valid.any(dim=1)
                if has.any():
                    idx = valid.float().argmax(dim=1)              # first supervised position
                    r = torch.arange(sl.size(0), device=sl.device)
                    lg = sl[r, idx].float()                        # (B, V) predicting the choice
                    tgt = sy[r, idx]
                    ce = torch.nn.functional.cross_entropy(lg[has], tgt[has], reduction="sum")
                    self._m["ce_sum"] += float(ce)
                    self._m["n"] += int(has.sum())
                    self._m["correct"] += int((lg[has].argmax(-1) == tgt[has]).sum())
                    if {"A", "B"} <= set(self._choice_ids):
                        a = lg[has][:, self._choice_ids["A"]]
                        b = lg[has][:, self._choice_ids["B"]]
                        self._m["absdiff_sum"] += float((a - b).abs().sum())
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *a, **kw):
        if "loss" in logs and self._m["n"]:
            logs.update({k: round(v, 6) for k, v in self.choice_metrics().items()
                         if k != "n_scored"})
            self.reset_choice_meter()
        return super().log(logs, *a, **kw)

"""The frozen eval config must DRIVE generation, not coincide with defaults.

Changes temperature/top_p/top_k in a copy of the config and asserts the kwargs
actually handed to model.generate change with them. Never edits the frozen file.
"""
import copy
import pytest
import yaml
from conftest import ROOT
from selfinterp.evaluate import sample_eval

FROZEN = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text())["sampling"]


class FakeTok:
    pad_token, pad_token_id, padding_side, eos_token = "<p>", 0, "left", "<e>"

    def apply_chat_template(self, m, tokenize=False, add_generation_prompt=False):
        return "PROMPT"

    def __call__(self, texts, **kw):
        import torch
        n = len(texts)
        class E(dict):
            def to(self, d): return self
        return E(input_ids=torch.zeros((n, 3), dtype=torch.long),
                 attention_mask=torch.ones((n, 3), dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "A"


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, input_ids=None, attention_mask=None, **kw):
        import torch
        self.calls.append(kw)
        return torch.zeros((input_ids.shape[0], input_ids.shape[1] + 1), dtype=torch.long)


ROW = [{"row_id": "r", "context_id": "C01", "instance_id": 0,
        "presentation_order": "AB", "pair_hash": "h", "messages": []}]


def _run(cfg):
    m = FakeModel()
    sample_eval(m, FakeTok(), ROW, "cpu", n_samples=cfg["n_samples"],
                temperature=cfg["temperature"], top_p=cfg["top_p"],
                top_k=cfg["top_k"], max_new_tokens=cfg["max_new_tokens"],
                batch_size=4)
    return m.calls


def test_frozen_values_are_the_ones_passed():
    calls = _run(FROZEN)
    assert calls, "generate was never called"
    for kw in calls:
        assert kw["temperature"] == FROZEN["temperature"] == 1.0
        assert kw["top_p"] == FROZEN["top_p"] == 1.0
        assert kw["top_k"] == FROZEN["top_k"] == 0
        assert kw["max_new_tokens"] == FROZEN["max_new_tokens"]
        assert kw["do_sample"] is True
    assert len(calls) == FROZEN["n_samples"]


@pytest.mark.parametrize("field,newval", [("temperature", 0.3), ("top_p", 0.85),
                                          ("top_k", 40), ("max_new_tokens", 9)])
def test_changing_config_changes_generate_kwargs(field, newval):
    alt = copy.deepcopy(FROZEN)
    alt[field] = newval
    assert alt[field] != FROZEN[field], "test would be vacuous"
    for kw in _run(alt):
        assert kw[field] == newval, f"{field} not propagated"


def test_sample_eval_has_no_generation_defaults():
    """A caller that forgets a parameter must fail loudly, not silently sample
    with a hardcoded default."""
    import inspect
    sig = inspect.signature(sample_eval)
    for p in ("temperature", "top_p", "top_k", "max_new_tokens"):
        assert sig.parameters[p].default is inspect.Parameter.empty, \
            f"{p} still has a fallback default"


def test_recorded_gen_params_match_what_was_used():
    m = FakeModel()
    rows = sample_eval(m, FakeTok(), ROW, "cpu", n_samples=2,
                       temperature=0.7, top_p=0.9, top_k=5, max_new_tokens=3,
                       batch_size=4)
    for r in rows:
        assert r["gen_params"] == {"do_sample": True, "temperature": 0.7,
                                   "top_p": 0.9, "top_k": 5, "max_new_tokens": 3}

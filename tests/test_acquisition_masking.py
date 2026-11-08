"""v3 acquisition: supervision structure, diagnostic loss, generation validity."""
import json, math
from pathlib import Path

import pytest
from conftest import ROOT, DATA, _jsonl

transformers = pytest.importorskip("transformers", reason="model stack not installed")
torch = pytest.importorskip("torch")

import sys; sys.path.insert(0, str(ROOT / "src"))
import yaml
from selfinterp.masking import build_masked_example, IGNORE
from selfinterp.tokenization_audit import audit, AuditFailure
from selfinterp.evaluate import summarise_generation
from selfinterp.diagnostics import step_counts, LN2

ACQ = DATA / "acquisition"
PAIR = ["A1", "A2"]


@pytest.fixture(scope="module")
def tok():
    cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
    return transformers.AutoTokenizer.from_pretrained(
        cfg["tokenizer_id"], revision=cfg["base_revision"])


@pytest.fixture(scope="module")
def rows():
    return {a: _jsonl(ACQ / "train" / a / "train_full.jsonl")[:300] for a in PAIR}


@pytest.mark.parametrize("a", PAIR)
def test_prompt_tokens_are_masked(tok, rows, a):
    for r in rows[a]:
        ex = build_masked_example(tok, r["messages"], mode="completion")
        assert all(l == IGNORE for l in ex["labels"][:ex["n_prompt_tokens"]]), r["row_id"]


@pytest.mark.parametrize("a", PAIR)
def test_choice_token_is_supervised(tok, rows, a):
    for r in rows[a]:
        ex = build_masked_example(tok, r["messages"], mode="completion")
        n_p = ex["n_prompt_tokens"]
        assert ex["labels"][n_p] != IGNORE, r["row_id"]
        assert tok.convert_ids_to_tokens(ex["input_ids"][n_p]) == r["label_letter"]


@pytest.mark.parametrize("a", PAIR)
def test_termination_token_is_supervised(tok, rows, a):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    for r in rows[a]:
        ex = build_masked_example(tok, r["messages"], mode="completion")
        ids, labels = ex["input_ids"], ex["labels"]
        assert eos in ids[ex["n_prompt_tokens"]:], r["row_id"]
        k = ex["n_prompt_tokens"] + ids[ex["n_prompt_tokens"]:].index(eos)
        assert labels[k] == eos, f"{r['row_id']}: <|im_end|> masked"


@pytest.mark.parametrize("a", PAIR)
def test_every_completion_token_supervised(tok, rows, a):
    for r in rows[a]:
        ex = build_masked_example(tok, r["messages"], mode="completion")
        n_p = ex["n_prompt_tokens"]
        assert all(l != IGNORE for l in ex["labels"][n_p:]), r["row_id"]
        assert ex["n_supervised_tokens"] == len(ex["input_ids"]) - n_p


def test_A_and_B_are_token_symmetric(tok, rows):
    lens = {}
    for a in PAIR:
        for r in rows[a]:
            ex = build_masked_example(tok, r["messages"], mode="completion")
            lens.setdefault(r["label_letter"], set()).add(
                len(ex["input_ids"]) - ex["n_prompt_tokens"])
    assert set(lens) == {"A", "B"}
    assert lens["A"] == lens["B"] == {3}, lens


def test_audit_passes_on_acquisition_data(tok, rows):
    rep = audit(rows["A1"], tok, mode="completion", n_show=0, out=open("/dev/null", "w"))
    assert rep["failures"] == []


def test_audit_hard_fails_under_choice_only(tok, rows):
    """Negative control: the audit must reject the v2 masking that dropped EOS."""
    rep = audit(rows["A1"][:20], tok, mode="choice_only", n_show=0, out=open("/dev/null", "w"))
    assert any("TERMINATION token masked" in f for f in rep["failures"])


# ------------------------------------------------------- diagnostic choice loss
def test_choice_ce_math_matches_manual():
    """The diagnostic must be the CE of the FIRST supervised token, and nothing else."""
    V, B = 8, 3
    torch.manual_seed(0)
    logits = torch.randn(B, 6, V)
    labels = torch.full((B, 6), -100)
    labels[:, 3] = torch.tensor([1, 2, 3])          # first supervised position
    labels[:, 4] = torch.tensor([5, 5, 5])          # a later supervised token
    sl, sy = logits[:, :-1, :], labels[:, 1:]
    valid = sy != -100
    idx = valid.float().argmax(dim=1)
    r = torch.arange(B)
    got = torch.nn.functional.cross_entropy(sl[r, idx].float(), sy[r, idx], reduction="mean")
    manual = torch.stack([
        torch.nn.functional.cross_entropy(logits[i, 2].unsqueeze(0),
                                          labels[i, 3].unsqueeze(0))
        for i in range(B)]).mean()
    assert torch.allclose(got, manual, atol=1e-6)


def test_uniform_logits_give_ln2_on_binary_choice():
    logits = torch.zeros(1, 2)                      # uniform over exactly two options
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    assert abs(float(ce) - LN2) < 1e-6


# ---------------------------------------------------------- generation validity
def _rec(**kw):
    base = dict(outcome="A", terminated=True, repeated_choice=False,
                n_generated_tokens=1, hit_token_cap=False)
    base.update(kw); return base


def test_generation_summary_counts_each_failure_mode():
    recs = [_rec(), _rec(outcome="B"),
            _rec(outcome="MALFORMED", terminated=False, repeated_choice=True,
                 n_generated_tokens=32, hit_token_cap=True)]
    s = summarise_generation(recs)
    assert s["n"] == 3
    assert abs(s["valid_rate"] - 2/3) < 1e-9
    assert abs(s["malformed_rate"] - 1/3) < 1e-9
    assert abs(s["nontermination_rate"] - 1/3) < 1e-9
    assert abs(s["repeated_choice_rate"] - 1/3) < 1e-9
    assert abs(s["hit_token_cap_rate"] - 1/3) < 1e-9
    assert s["len_max"] == 32


def test_a2v2_style_collapse_would_be_caught():
    """'BBBBBBBB' with no EOS must register as invalid, nonterminating and repeated."""
    recs = [_rec(outcome="MALFORMED", terminated=False, repeated_choice=True,
                 n_generated_tokens=32, hit_token_cap=True) for _ in range(100)]
    s = summarise_generation(recs)
    assert s["valid_rate"] == 0.0
    assert s["nontermination_rate"] == 1.0
    assert s["repeated_choice_rate"] == 1.0


def test_generation_budget_is_large_enough_to_observe_nontermination():
    cfg = yaml.safe_load((ROOT / "configs" / "acquisition_eval.yaml").read_text())
    assert cfg["free_generation"]["max_new_tokens"] >= 16

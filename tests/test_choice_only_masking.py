"""Recipe v2: choice-token-only supervision."""
import pytest
import yaml
from conftest import ROOT, ADAPTERS
from selfinterp.masking import build_masked_example, IGNORE, MODES

transformers = pytest.importorskip("transformers", reason="model stack required")


@pytest.fixture(scope="module")
def tok():
    cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
    return transformers.AutoTokenizer.from_pretrained(cfg["tokenizer_id"],
                                                      revision=cfg["base_revision"])


def test_modes_declared():
    assert MODES == ("completion", "choice_only")


def test_unknown_mode_rejected(tok, train):
    with pytest.raises(ValueError):
        build_masked_example(tok, train["A1"][0]["messages"], mode="nonsense")


def test_choice_only_supervises_exactly_one_token(tok, train):
    for r in train["A1"][:100]:
        ex = build_masked_example(tok, r["messages"], mode="choice_only")
        sup = [l for l in ex["labels"] if l != IGNORE]
        assert len(sup) == 1, len(sup)
        assert ex["n_supervised_tokens"] == 1


def test_choice_only_supervises_the_LETTER_and_nothing_else(tok, train):
    for r in train["A1"][:100]:
        ex = build_masked_example(tok, r["messages"], mode="choice_only")
        idx = [i for i, l in enumerate(ex["labels"]) if l != IGNORE][0]
        assert idx == ex["n_prompt_tokens"]                    # first content position
        assert tok.decode([ex["labels"][idx]]).strip() == r["label_letter"]


def test_end_and_newline_are_masked(tok, train):
    """The two tokens that carried 1-2e-6 of loss in v1 must now be IGNORE."""
    for r in train["A1"][:100]:
        v1 = build_masked_example(tok, r["messages"], mode="completion")
        v2 = build_masked_example(tok, r["messages"], mode="choice_only")
        n_p = v2["n_prompt_tokens"]
        assert v1["labels"][n_p] == v2["labels"][n_p]          # letter identical
        for i in range(n_p + 1, len(v2["labels"])):
            assert v2["labels"][i] == IGNORE
            assert v1["labels"][i] != IGNORE                   # v1 did supervise them


def test_completion_mode_is_unchanged_regression(tok, train):
    for r in train["A1"][:50]:
        ex = build_masked_example(tok, r["messages"], mode="completion")
        assert ex["n_supervised_tokens"] == 3
        assert all(l == IGNORE for l in ex["labels"][:ex["n_prompt_tokens"]])


def test_input_ids_identical_across_modes(tok, train):
    """Only labels change; the token stream must not."""
    for r in train["A1"][:50]:
        a = build_masked_example(tok, r["messages"], mode="completion")
        b = build_masked_example(tok, r["messages"], mode="choice_only")
        assert a["input_ids"] == b["input_ids"]


def test_collator_matches_reference_choice_only(tok, train):
    from selfinterp.collator import verify_against_reference
    rep = verify_against_reference(tok, train["A1"], n=300, mode="choice_only")
    assert rep["n_mismatch"] == 0
    assert rep["mean_supervised_per_example"] == 1.0


def test_collator_matches_reference_completion(tok, train):
    from selfinterp.collator import verify_against_reference
    rep = verify_against_reference(tok, train["A1"], n=300, mode="completion")
    assert rep["n_mismatch"] == 0
    assert rep["mean_supervised_per_example"] == 3.0

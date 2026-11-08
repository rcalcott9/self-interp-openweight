"""Prompt tokens must be excluded from the loss. Requires the model stack;
skipped until it is installed (deliberate, not a silent pass)."""
import pytest
transformers = pytest.importorskip("transformers",
                                   reason="model stack not installed yet")
from selfinterp.masking import build_masked_example, IGNORE


@pytest.fixture(scope="module")
def tok():
    import yaml
    from conftest import ROOT
    cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
    return transformers.AutoTokenizer.from_pretrained(
        cfg["tokenizer_id"], revision=cfg["base_revision"])


def test_prompt_tokens_masked_completion_tokens_not(tok, train):
    for r in train["A1"][:50]:
        ex = build_masked_example(tok, r["messages"])
        n_p = ex["n_prompt_tokens"]
        assert all(l == IGNORE for l in ex["labels"][:n_p])
        assert all(l != IGNORE for l in ex["labels"][n_p:])
        assert ex["n_completion_tokens"] >= 1
        assert ex["labels"][n_p:] == ex["input_ids"][n_p:]


def test_label_letter_is_the_only_supervised_content(tok, train):
    for r in train["A1"][:50]:
        ex = build_masked_example(tok, r["messages"])
        sup = tok.decode([t for t in ex["labels"][ex["n_prompt_tokens"]:]])
        assert r["label_letter"] in sup

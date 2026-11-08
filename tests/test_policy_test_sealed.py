"""policy_test is the sealed FINAL-gate split."""
import json, itertools
from conftest import DATA, ADAPTERS, SPLITS
from selfinterp import data_access as DA
import pytest


def _pt():
    return [json.loads(l) for l in open(DATA / "common" / "policy_test_prompts.jsonl")]


def test_policy_test_is_reserved():
    assert "policy_test" in DA.RESERVED
    with pytest.raises(DA.ReservedSplitAccess):
        DA.load_split("policy_test")
    with pytest.raises(DA.ReservedSplitAccess):
        DA.load_policy_test()


def test_policy_test_counts():
    rows = _pt()
    assert len(rows) == 16 * 60 * 2
    by = {}
    for r in rows:
        by[(r["context_id"], r["presentation_order"])] = by.get((r["context_id"], r["presentation_order"]), 0) + 1
    assert set(by.values()) == {60}


def test_policy_test_disjoint_from_every_other_split(train, common):
    pt = {r["pair_hash"] for r in _pt()}
    tr = set().union(*[{r["pair_hash"] for r in train[a]} for a in ADAPTERS])
    assert not (pt & tr), len(pt & tr)
    for s in SPLITS:
        ch = {r["pair_hash"] for r in common[s]}
        assert not (pt & ch), (s, len(pt & ch))


def test_policy_test_prompts_carry_no_label():
    for r in _pt():
        assert [m["role"] for m in r["messages"]] == ["system", "user"]


def test_policy_test_both_orders():
    by = {}
    for r in _pt():
        by.setdefault(r["pair_hash"], set()).add(r["presentation_order"])
    assert all(v == {"AB", "BA"} for v in by.values())

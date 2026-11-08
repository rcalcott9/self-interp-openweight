import itertools
from conftest import ADAPTERS, SPLITS


def test_train_counts(train):
    for a in ADAPTERS:
        assert len(train[a]) == 16 * 200
        by = {}
        for r in train[a]:
            by[r["context_id"]] = by.get(r["context_id"], 0) + 1
        assert set(by.values()) == {200}


def test_common_counts(common):
    exp = {"policy_val": 60, "self_pred": 40, "relational": 40}
    for s, n in exp.items():
        assert len(common[s]) == 16 * n * 2      # both orders
        by = {}
        for r in common[s]:
            by[(r["context_id"], r["presentation_order"])] = \
                by.get((r["context_id"], r["presentation_order"]), 0) + 1
        assert set(by.values()) == {n}


def test_training_examples_are_adapter_specific(train):
    """No underlying option pair is shared by any two adapters."""
    sets = {a: {r["pair_hash"] for r in train[a]} for a in ADAPTERS}
    for a, b in itertools.combinations(ADAPTERS, 2):
        assert not (sets[a] & sets[b]), (a, b, len(sets[a] & sets[b]))


def test_train_hashes_unique_within_adapter(train):
    for a in ADAPTERS:
        h = [r["pair_hash"] for r in train[a]]
        assert len(h) == len(set(h))


def test_common_disjoint_from_all_training(train, common):
    tr = set().union(*[{r["pair_hash"] for r in train[a]} for a in ADAPTERS])
    for s in SPLITS:
        ch = {r["pair_hash"] for r in common[s]}
        assert not (ch & tr), (s, len(ch & tr))


def test_common_splits_mutually_disjoint(common):
    sets = {s: {r["pair_hash"] for r in common[s]} for s in SPLITS}
    for a, b in itertools.combinations(SPLITS, 2):
        assert not (sets[a] & sets[b]), (a, b)


def test_policy_validation_prompts_are_common_across_models(common, keys):
    """One prompt set; every adapter has a key row for every prompt row."""
    for s in SPLITS:
        rows = {r["row_id"] for r in common[s]}
        by_ad = {}
        for k in keys[s]:
            by_ad.setdefault(k["adapter_id"], set()).add(k["row_id"])
        assert set(by_ad) == set(ADAPTERS)
        for a in ADAPTERS:
            assert by_ad[a] == rows, a


def test_held_out_prompts_carry_no_label(common):
    for s in SPLITS:
        for r in common[s]:
            roles = [m["role"] for m in r["messages"]]
            assert roles == ["system", "user"], (s, roles)

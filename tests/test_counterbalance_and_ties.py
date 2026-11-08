import json
from pathlib import Path
from conftest import ADAPTERS, SPLITS, DATA


def test_training_order_exactly_balanced(train):
    for a in ADAPTERS:
        by = {}
        for r in train[a]:
            k = (r["context_id"], r["presentation_order"])
            by[k] = by.get(k, 0) + 1
        for c in {r["context_id"] for r in train[a]}:
            assert by[(c, "AB")] == 100 and by[(c, "BA")] == 100, (a, c)


def test_every_heldout_pair_rendered_both_ways(common):
    for s in SPLITS:
        by = {}
        for r in common[s]:
            by.setdefault(r["pair_hash"], set()).add(r["presentation_order"])
        assert all(v == {"AB", "BA"} for v in by.values())


def test_cross_order_underlying_mapping(common):
    """A under AB and B under BA must denote the SAME underlying option."""
    for s in SPLITS:
        by = {}
        for r in common[s]:
            by.setdefault(r["pair_hash"], {})[r["presentation_order"]] = r
        for h, d in by.items():
            ab, ba = d["AB"], d["BA"]
            assert ab["underlying_at_A"] == 0 and ab["underlying_at_B"] == 1
            assert ba["underlying_at_A"] == 1 and ba["underlying_at_B"] == 0
            # option shown at A under AB == option shown at B under BA
            assert ab["option_A_values"] == ba["option_B_values"], h
            assert ab["option_B_values"] == ba["option_A_values"], h


def test_no_exact_utility_ties_remain():
    tol = 1e-9
    for s in SPLITS:
        for k in __import__("csv").DictReader(
                open(DATA / "common" / f"{s}_keys.csv")):
            uA, uB = float(k["utility_A"]), float(k["utility_B"])
            assert abs(uA - uB) > tol * max(1.0, abs(uA), abs(uB)), k["row_id"]


def test_weak_flip_rate_exact(train):
    import csv
    ctx = {r["context_id"]: r["assigned_consistency"]
           for r in csv.DictReader(open(DATA / "design" / "contexts.csv"))}
    for a in ADAPTERS:
        by = {}
        for r in train[a]:
            by.setdefault(r["context_id"], []).append(r["label_follows_target"])
        for cid, v in by.items():
            n_follow = sum(v)
            if ctx[cid] == "weak":
                assert n_follow == 130, (a, cid, n_follow)   # 200 - 70 flipped
            else:
                assert n_follow == 200, (a, cid, n_follow)


def test_evaluation_keys_use_uncorrupted_policy(train, keys):
    """Held-out keys are pure target labels: no flip field exists at all."""
    for s in SPLITS:
        assert "label_follows_target" not in keys[s][0]
        assert "label_letter" not in keys[s][0]

"""Weak-context label corruption must be stratified by presentation order and
otherwise blind to target letter and utility margin."""
import csv
from conftest import ADAPTERS, DATA


def _ctx():
    return {r["context_id"]: r["assigned_consistency"]
            for r in csv.DictReader(open(DATA / "design" / "contexts.csv"))}


def _cells(train):
    cells = {}
    for a in ADAPTERS:
        for r in train[a]:
            cells.setdefault((a, r["context_id"]), []).append(r)
    return cells


def test_exactly_35_flips_per_order_in_weak_cells(train):
    ctx = _ctx()
    for (a, cid), rows in _cells(train).items():
        if ctx[cid] != "weak":
            continue
        for od in ("AB", "BA"):
            sub = [r for r in rows if r["presentation_order"] == od]
            assert len(sub) == 100, (a, cid, od, len(sub))
            n_flip = sum(not r["label_follows_target"] for r in sub)
            assert n_flip == 35, (a, cid, od, n_flip)


def test_strong_cells_have_zero_flips(train):
    ctx = _ctx()
    for (a, cid), rows in _cells(train).items():
        if ctx[cid] == "strong":
            assert all(r["label_follows_target"] for r in rows), (a, cid)


def test_weak_cell_totals(train):
    ctx = _ctx()
    for (a, cid), rows in _cells(train).items():
        if ctx[cid] == "weak":
            assert sum(not r["label_follows_target"] for r in rows) == 70


def test_flip_never_alters_recorded_target(train):
    """target_letter is the uncorrupted policy's answer; label_letter may differ."""
    for a in ADAPTERS:
        for r in train[a]:
            same = r["label_letter"] == r["target_letter"]
            assert same == r["label_follows_target"]


def test_flip_selection_ignores_margin_by_construction(train):
    """Flipped rows must span the full margin range, not cluster at the boundary."""
    ctx = _ctx()
    for (a, cid), rows in _cells(train).items():
        if ctx[cid] != "weak":
            continue
        fl = [r["utility_margin"] for r in rows if not r["label_follows_target"]]
        un = [r["utility_margin"] for r in rows if r["label_follows_target"]]
        assert max(fl) > sorted(un)[int(0.75 * len(un))], (a, cid)
        assert min(fl) < sorted(un)[int(0.25 * len(un))], (a, cid)

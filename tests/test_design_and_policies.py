import csv, itertools
from pathlib import Path
from conftest import DATA, ADAPTERS


def _ctx():
    return list(csv.DictReader(open(DATA / "design" / "contexts.csv")))


def test_sixteen_contexts_no_untrained():
    assert len(_ctx()) == 16


def test_cell_balance_four_per_cell_per_pair():
    ctx = _ctx()
    for pair, col in (("A", "overlap_A"), ("B", "overlap_B")):
        tally = {}
        for c in ctx:
            tally[(c[col], c["assigned_consistency"])] = \
                tally.get((c[col], c["assigned_consistency"]), 0) + 1
        assert sorted(tally.values()) == [4, 4, 4, 4], (pair, tally)


def test_overlap_counterbalanced_across_pairs():
    for c in _ctx():
        assert c["overlap_A"] != c["overlap_B"]


def test_reliability_is_context_level_constant():
    """strong/weak must not be confounded with adapter identity."""
    rows = list(csv.DictReader(open(DATA / "design" / "target_policies.csv")))
    by_ctx = {}
    for r in rows:
        by_ctx.setdefault(r["context_id"], set()).add(r["assigned_consistency"])
    assert all(len(v) == 1 for v in by_ctx.values())


def test_no_consecutive_context_blocks():
    ctx = _ctx()
    cells = {}
    for c in ctx:
        cells.setdefault((c["overlap_A"], c["assigned_consistency"]), []).append(
            int(c["context_id"][1:]))
    for k, v in cells.items():
        v = sorted(v)
        run = best = 1
        for a, b in zip(v, v[1:]):
            run = run + 1 if b == a + 1 else 1
            best = max(best, run)
        assert best <= 2, (k, v)


def test_shared_policies_identical_independent_distinct(policies):
    for c in _ctx():
        cid = c["context_id"]
        for pair, col in (("A", "overlap_A"), ("B", "overlap_B")):
            m = [f"{pair}1", f"{pair}2"]
            if c[col] == "shared":
                assert policies[m[0]][cid] == policies[m[1]][cid], (cid, pair)
            else:
                assert policies[m[0]][cid] != policies[m[1]][cid], (cid, pair)


def test_cross_pair_policies_never_identical(policies):
    for c in _ctx():
        cid = c["context_id"]
        for a, b in itertools.product(["A1", "A2"], ["B1", "B2"]):
            assert policies[a][cid] != policies[b][cid], (cid, a, b)


def test_weights_normalised(policies):
    for a in ADAPTERS:
        for cid, w in policies[a].items():
            assert max(abs(x) for x in w) == 100, (a, cid, w)

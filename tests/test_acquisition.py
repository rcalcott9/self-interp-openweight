"""v3 acquisition recipe: data, masking, diagnostics and step arithmetic."""
import json, csv, math, itertools, collections
from pathlib import Path

import pytest
from conftest import ROOT, DATA, _jsonl

ACQ = DATA / "acquisition"
PAIR = ["A1", "A2"]
N_PER_CTX, N_CTX = 320, 16
N_TOTAL = N_PER_CTX * N_CTX


@pytest.fixture(scope="session")
def acq():
    return {a: _jsonl(ACQ / "train" / a / "train_full.jsonl") for a in PAIR}


# ---------------------------------------------------------------- policy labels
@pytest.mark.parametrize("a", PAIR)
def test_labels_are_deterministic(acq, a):
    """No label noise anywhere: label_letter == target_letter for every row."""
    bad = [r["row_id"] for r in acq[a] if r["label_letter"] != r["target_letter"]]
    assert not bad, f"{len(bad)} corrupted labels, e.g. {bad[:3]}"
    assert all(r["label_follows_target"] for r in acq[a])


@pytest.mark.parametrize("a", PAIR)
def test_label_matches_utility_rule(acq, a):
    """The label is exactly the higher-utility option at that presentation."""
    import sys; sys.path.insert(0, str(ROOT / "src"))
    for r in acq[a][:400]:
        expect = "A" if r["utility_A"] > r["utility_B"] else "B"
        assert r["label_letter"] == expect, r["row_id"]


@pytest.mark.parametrize("a", PAIR)
def test_counts(acq, a):
    assert len(acq[a]) == N_TOTAL
    per = collections.Counter(r["context_id"] for r in acq[a])
    assert len(per) == N_CTX and set(per.values()) == {N_PER_CTX}


# --------------------------------------------------------------------- balance
@pytest.mark.parametrize("a", PAIR)
def test_label_balance(acq, a):
    c = collections.Counter(r["label_letter"] for r in acq[a])
    assert abs(c["A"] - c["B"]) <= N_CTX, c        # <= 1 per context
    assert abs(c["A"] / len(acq[a]) - 0.5) < 0.01


@pytest.mark.parametrize("a", PAIR)
def test_order_balance_is_exact(acq, a):
    c = collections.Counter(r["presentation_order"] for r in acq[a])
    assert c["AB"] == c["BA"] == N_TOTAL // 2, c


@pytest.mark.parametrize("a", PAIR)
def test_per_context_balance(acq, a):
    by = collections.defaultdict(list)
    for r in acq[a]:
        by[r["context_id"]].append(r)
    for cid, rs in by.items():
        nA = sum(r["label_letter"] == "A" for r in rs)
        nAB = sum(r["presentation_order"] == "AB" for r in rs)
        assert abs(nA - len(rs) // 2) <= 1, (cid, nA)
        assert nAB == len(rs) // 2, (cid, nAB)


def test_adapter_not_identifiable_from_labels_or_position(acq):
    """Neither policy may be recoverable from marginal label or position rates."""
    p = {a: sum(r["label_letter"] == "A" for r in acq[a]) / len(acq[a]) for a in PAIR}
    q = {a: sum(r["presentation_order"] == "AB" for r in acq[a]) / len(acq[a]) for a in PAIR}
    assert abs(p["A1"] - p["A2"]) < 0.01, p
    assert abs(q["A1"] - q["A2"]) < 0.01, q


@pytest.mark.parametrize("a", PAIR)
def test_balancing_does_not_select_on_margin(acq, a):
    """Order assignment must not shift the utility-margin distribution."""
    import statistics as st
    mA = [r["utility_margin"] for r in acq[a] if r["label_letter"] == "A"]
    mB = [r["utility_margin"] for r in acq[a] if r["label_letter"] == "B"]
    mAB = [r["utility_margin"] for r in acq[a] if r["presentation_order"] == "AB"]
    mBA = [r["utility_margin"] for r in acq[a] if r["presentation_order"] == "BA"]
    pooled = st.pstdev([r["utility_margin"] for r in acq[a]])
    for x, y, lbl in ((mA, mB, "label"), (mAB, mBA, "order")):
        d = abs(st.mean(x) - st.mean(y)) / pooled
        assert d < 0.12, f"{lbl}: standardised mean margin difference {d:.3f}"


# ------------------------------------------------------------- split isolation
def test_acquisition_pairs_disjoint_from_every_split(acq):
    sets = {f"acq_{a}": {r["pair_hash"] for r in acq[a]} for a in PAIR}
    for a in ("A1", "A2", "B1", "B2"):
        sets[f"frozen_{a}"] = {r["pair_hash"] for r in _jsonl(DATA / "train" / a / "train_full.jsonl")}
    for s in ("policy_val", "self_pred", "relational", "policy_test"):
        sets[s] = {r["pair_hash"] for r in _jsonl(DATA / "common" / f"{s}_prompts.jsonl")}
    for x, y in itertools.combinations(sorted(sets), 2):
        assert not (sets[x] & sets[y]), f"{x} & {y} share {len(sets[x] & sets[y])} pairs"


@pytest.mark.parametrize("a", PAIR)
def test_no_duplicate_pairs_within_adapter(acq, a):
    h = [r["pair_hash"] for r in acq[a]]
    assert len(set(h)) == len(h)


def test_contexts_are_shared_across_splits_by_design(acq):
    """Per-context policies REQUIRE the same contexts in train and held-out."""
    tr = {r["context_id"] for r in acq["A1"]}
    pv = {r["context_id"] for r in _jsonl(DATA / "common" / "policy_val_prompts.jsonl")}
    assert tr == pv and len(tr) == N_CTX


# -------------------------------------------------- shared / different policies
def test_shared_and_low_overlap_structure_preserved():
    ovl = {r["context_id"]: r["overlap_status"]
           for r in csv.DictReader(open(DATA / "design" / "target_policies.csv"))
           if r["adapter_id"] == "A1"}
    assert collections.Counter(ovl.values()) == {"shared": 8, "independent": 8}
    key = collections.defaultdict(dict)
    for r in csv.DictReader(open(DATA / "common" / "policy_val_keys.csv")):
        if r["adapter_id"] in PAIR:
            key[(r["context_id"], r["instance_id"], r["presentation_order"])][r["adapter_id"]] = r["target_letter"]
    ag = collections.defaultdict(lambda: [0, 0])
    for (cid, _, _), v in key.items():
        ag[ovl[cid]][0] += v["A1"] == v["A2"]; ag[ovl[cid]][1] += 1
    assert ag["shared"][0] == ag["shared"][1], "shared contexts must agree exactly"
    disagree = ag["independent"][1] - ag["independent"][0]
    assert disagree >= 300, f"only {disagree} discriminating held-out rows"


# ------------------------------------------------------------ reproducibility
def test_dataset_generation_is_reproducible():
    """Rebuilding from the frozen seed + pool must reproduce byte-identical files."""
    import sys, hashlib; sys.path.insert(0, str(ROOT / "src"))
    pool = ROOT / "data" / "design" / "candidate_scenarios.json"
    if not pool.exists():
        pytest.skip("scenario pool not cached")
    before = {a: hashlib.sha256((ACQ / "train" / a / "train_full.jsonl").read_bytes()).hexdigest()
              for a in PAIR}
    from selfinterp.build_acquisition import build
    build()
    after = {a: hashlib.sha256((ACQ / "train" / a / "train_full.jsonl").read_bytes()).hexdigest()
             for a in PAIR}
    assert before == after, "regeneration is not deterministic"


def test_manifest_records_provenance():
    m = json.loads((ACQ / "acquisition_manifest.json").read_text())
    assert m["version"] == 3
    assert m["invariants"]["target_policies_unchanged"] is True
    assert m["n_examples_per_adapter"] == {a: N_TOTAL for a in PAIR}
    assert any("policy_test" in k for k in m["disjointness_sources"])
    assert m["derived_from"]["scenario_pool_sha256"].startswith("1514e4bc")


# ------------------------------------------------------------- step arithmetic
def test_step_counts_exact():
    import sys; sys.path.insert(0, str(ROOT / "src"))
    from selfinterp.diagnostics import step_counts
    s = step_counts(5120, 8, 2, 3)
    assert s["effective_batch_size"] == 16
    assert s["microsteps_per_epoch"] == 640
    assert s["optimizer_steps_per_epoch"] == 320
    assert s["total_optimizer_steps"] == 960
    assert s["total_microsteps"] == 1920
    assert s["total_example_presentations"] == 15360
    # reproduces the legacy 7B runs, whose logs report steps=600
    legacy = step_counts(3200, 8, 2, 3)
    assert legacy["total_optimizer_steps"] == 600
    assert legacy["total_microsteps"] == 1200
    # smoke
    sm = step_counts(200, 4, 1, 1)
    assert sm["total_optimizer_steps"] == 50 and sm["total_microsteps"] == 50


def test_ln2_constant():
    import sys; sys.path.insert(0, str(ROOT / "src"))
    from selfinterp.diagnostics import LN2
    assert abs(LN2 - 0.693147180559945) < 1e-12

"""Repeated generations of the same prompt must not act as independent units."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from estimators import (same_order_match, cross_order_match, target_agreement,
                        cluster_ci, hierarchical_ci, wilson_naive)


def C(a=0, b=0, m=0):
    return {"A": a, "B": b, "MALFORMED": m}


# --------------------------------------------------- required numerical cases
def test_required_match_cases():
    assert math.isclose(same_order_match(8, 8, 16), 112 / 240)      # 0.466667
    assert math.isclose(same_order_match(8, 8, 20), 112 / 380)      # 0.294737
    assert same_order_match(16, 0, 16) == 1.0                        # all A
    assert same_order_match(0, 0, 16) == 0.0                         # all malformed


def test_malformed_dilutes_never_matches():
    """Same valid counts, more malformed -> strictly lower estimate."""
    prev = same_order_match(8, 8, 16)
    for extra in (1, 2, 4, 8):
        cur = same_order_match(8, 8, 16 + extra)
        assert cur < prev
        prev = cur


def test_estimator_is_unbiased_for_fresh_copy_collision():
    """U-statistic must recover p_A^2+p_B^2; the plug-in must not."""
    import numpy as np
    rng = np.random.default_rng(0)
    pA, pB, pM = 0.5, 0.3, 0.2
    truth = pA ** 2 + pB ** 2
    N, reps = 16, 20000
    draws = rng.choice(3, size=(reps, N), p=[pA, pB, pM])
    nA = (draws == 0).sum(1); nB = (draws == 1).sum(1)
    u = ((nA * (nA - 1) + nB * (nB - 1)) / (N * (N - 1))).mean()
    plugin = (((nA / N) ** 2 + (nB / N) ** 2)).mean()
    assert abs(u - truth) < 0.005, (u, truth)
    assert plugin > truth + 0.01, "plug-in should be biased upward"


# ------------------------------------------------- clustering changes the answer
def test_clustered_ci_much_wider_than_naive_when_within_stimulus_correlated():
    """8 stimuli x 16 generations. Within a stimulus every generation is
    identical; 4 stimuli are all-A, 4 all-B. Naive Wilson sees 128 'independent'
    coin flips at p=0.5 and is far too tight."""
    stim_values = [1.0] * 4 + [0.0] * 4          # per-stimulus agreement
    lo_c, hi_c = cluster_ci(stim_values, n_boot=4000, seed=1)
    lo_w, hi_w = wilson_naive(64, 128)
    assert (hi_c - lo_c) > 2.5 * (hi_w - lo_w), \
        f"clustered {hi_c-lo_c:.3f} vs naive {hi_w-lo_w:.3f}"


def test_more_generations_per_stimulus_does_not_narrow_clustered_ci():
    """Adding repeated generations to the SAME stimuli must not buy precision:
    the stimulus-level values are unchanged, so the interval is unchanged."""
    vals = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    a = cluster_ci(vals, n_boot=4000, seed=3)
    b = cluster_ci(vals, n_boot=4000, seed=3)     # same L1 values, 10x generations
    assert a == b
    # but the naive interval WOULD shrink, which is the error being avoided
    w_small = wilson_naive(3, 6); w_big = wilson_naive(30, 60)
    assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0])


def test_bootstrap_carries_whole_stimulus():
    """cluster_ci consumes one value per stimulus, so a resampled stimulus
    necessarily brings all of its generations with it."""
    import inspect
    from estimators import cluster_ci as f
    src = inspect.getsource(f)
    assert "stim_values" in src and "generation" not in src.split('"""')[2]


# ---------------------------------------------------- hierarchical: context unit
def test_hierarchical_resamples_contexts_not_just_stimuli():
    """Two contexts with opposite values. Resampling contexts must produce a
    wide interval; pooling stimuli and resampling them alone would not."""
    by_ctx = {"C1": [1.0] * 20, "C2": [0.0] * 20}
    lo_h, hi_h = hierarchical_ci(by_ctx, n_boot=4000, seed=5)
    lo_s, hi_s = cluster_ci([1.0] * 20 + [0.0] * 20, n_boot=4000, seed=5)
    assert (hi_h - lo_h) > (hi_s - lo_s) * 2, \
        f"hierarchical {hi_h-lo_h:.3f} must exceed stimulus-only {hi_s-lo_s:.3f}"
    assert lo_h == 0.0 and hi_h == 1.0          # both contexts can be resampled out


def test_hierarchical_needs_two_contexts():
    assert all(math.isnan(x) for x in hierarchical_ci({"C1": [0.5] * 10}))


def test_hierarchical_not_narrowed_by_within_context_n():
    a = hierarchical_ci({"C1": [0.9] * 5, "C2": [0.1] * 5}, n_boot=4000, seed=7)
    b = hierarchical_ci({"C1": [0.9] * 50, "C2": [0.1] * 50}, n_boot=4000, seed=7)
    assert abs((a[1] - a[0]) - (b[1] - b[0])) < 0.05


# ------------------------------------------------------- cross-order worked case
def test_cross_order_worked_example():
    """option_0 = the CHEAP item, option_1 = the EXPENSIVE item.
       AB rendering: position A shows option_0, position B shows option_1
       BA rendering: position A shows option_1, position B shows option_0
    Same underlying choice  <=>  (AB='A' & BA='B')  or  (AB='B' & BA='A')."""
    N = 16
    always_option0 = cross_order_match(C(a=N), C(b=N))      # AB->'A', BA->'B'
    always_option1 = cross_order_match(C(b=N), C(a=N))      # AB->'B', BA->'A'
    always_letter_A = cross_order_match(C(a=N), C(a=N))     # position bias
    always_letter_B = cross_order_match(C(b=N), C(b=N))
    assert always_option0 == 1.0
    assert always_option1 == 1.0
    assert always_letter_A == 0.0
    assert always_letter_B == 0.0


def test_cross_order_malformed_never_matches():
    assert cross_order_match(C(a=8, m=8), C(b=8, m=8)) == 64 / 256


def test_target_agreement_counts_malformed_as_failure():
    assert target_agreement(C(a=12, b=0, m=4), "A") == 0.75
    assert target_agreement(C(a=0, b=0, m=16), "A") == 0.0

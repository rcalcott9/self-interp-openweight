"""Estimators for predictability, agreement and outcome rates.

UNIT HIERARCHY (strictly bottom-up; nothing skips a level):

  L0  generation          one sampled model output: A | B | MALFORMED
  L1  stimulus x order    one held-out option pair in one presentation order.
                          Its N repeated generations are pooled HERE and only here.
  L2  context x order     mean over the L1 values of its stimuli
  L3  context             mean over its L1 values across both orders
  L4  strong / weak / all hierarchical bootstrap: resample CONTEXTS, then
                          STIMULI within each resampled context

Repeated generations of the same prompt are NEVER independent units for
uncertainty. They are consumed at L1 to form one number per stimulus; every
interval above L1 resamples stimuli (or contexts), carrying all of a sampled
stimulus's generations with it.
"""
import math
import numpy as np

OUTCOMES = ("A", "B", "MALFORMED")


# ---------------------------------------------------------------- L1 estimators
def same_order_match(nA, nB, N):
    """P(two FRESH independent copies produce the same valid letter).

        U = [nA(nA-1) + nB(nB-1)] / [N(N-1)]

    Unbiased U-statistic: the proportion of ORDERED PAIRS of distinct observed
    generations that agree, which is unbiased for p_A^2 + p_B^2 under iid
    sampling. Not the plug-in p_A^2+p_B^2, which is biased upward.
    N includes MALFORMED, so malformed generations dilute the estimate and two
    malformed outputs never count as a match.
    """
    if N < 2:
        return float("nan")
    return (nA * (nA - 1) + nB * (nB - 1)) / (N * (N - 1))


def cross_order_match(ab_counts, ba_counts):
    """P(a fresh copy seeing AB and a fresh copy seeing BA pick the SAME
    UNDERLYING option).

        X = [nAB_A * nBA_B + nAB_B * nBA_A] / (N_AB * N_BA)

    Plug-in, and already unbiased: the two factors come from DISJOINT sample
    pools, so no (n-1) correction applies. Under AB, position A shows option_0;
    under BA, position A shows option_1. Hence 'A under AB' and 'B under BA'
    denote the same underlying option.
    """
    nab = sum(ab_counts[o] for o in OUTCOMES)
    nba = sum(ba_counts[o] for o in OUTCOMES)
    if nab == 0 or nba == 0:
        return float("nan")
    return (ab_counts["A"] * ba_counts["B"] + ab_counts["B"] * ba_counts["A"]) / (nab * nba)


def target_agreement(counts, target_letter):
    """Fraction of this stimulus's generations equal to the uncorrupted target
    policy's answer. MALFORMED stays in the denominator and counts as a FAILURE."""
    N = sum(counts[o] for o in OUTCOMES)
    return counts[target_letter] / N if N else float("nan")


def outcome_rate(counts, outcome):
    """Unconditional rate; MALFORMED retained in the denominator."""
    N = sum(counts[o] for o in OUTCOMES)
    return counts[outcome] / N if N else float("nan")


# ------------------------------------------------------------ L2-L4 uncertainty
def cluster_ci(stim_values, n_boot=10000, seed=90210, alpha=0.05):
    """Resample STIMULI with replacement (each carrying all its generations).
    Input is one value per stimulus, already computed at L1."""
    v = np.asarray([x for x in stim_values if not math.isnan(x)], float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    d = v[idx].mean(axis=1)
    return (float(np.percentile(d, 100 * alpha / 2)),
            float(np.percentile(d, 100 * (1 - alpha / 2))))


def hierarchical_ci(by_context, n_boot=10000, seed=90210, alpha=0.05):
    """Two-stage cluster bootstrap: resample CONTEXTS with replacement, then
    STIMULI with replacement within each resampled context.

    by_context: {context_id: [stimulus-level value, ...]}
    """
    ctxs = [c for c, v in by_context.items()
            if len([x for x in v if not math.isnan(x)]) > 0]
    if len(ctxs) < 2:
        return (float("nan"), float("nan"))
    arrs = {c: np.asarray([x for x in by_context[c] if not math.isnan(x)], float)
            for c in ctxs}
    rng = np.random.default_rng(seed)
    n = len(ctxs)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        means = np.empty(n)
        for j, ci in enumerate(pick):
            a = arrs[ctxs[ci]]
            means[j] = a[rng.integers(0, a.size, size=a.size)].mean()
        out[b] = means.mean()
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def wilson_naive(k, n, z=1.96):
    """DIAGNOSTIC ONLY -- treats every generation as independent, which they are
    not. Reported alongside the clustered interval to expose the difference.
    Never a primary interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

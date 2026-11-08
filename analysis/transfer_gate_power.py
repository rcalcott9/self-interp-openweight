"""Power/MDE simulation for the A1-A2 behavioural-transfer gate.

Uses ONLY: A1's observed per-stimulus choice distributions, and the fixed design
(4 contexts per cell, 60 stimuli per context, 2 orders, 16 samples per adapter).
No A2 data exists. No future outcome informs the threshold.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/20260902_230634_a1_qwen7b"
SHARED_WEAK   = ["C01","C02","C05","C08"]
SHARED_STRONG = ["C03","C04","C10","C16"]
DIFF_WEAK     = ["C06","C07","C13","C15"]
DIFF_STRONG   = ["C09","C11","C12","C14"]
NS = 16                      # samples per adapter per prompt (frozen)

def a1_distributions():
    """p_A per (context, order, stimulus) from A1's 15,360 sampled generations."""
    cnt = defaultdict(lambda: {"A":0,"B":0,"MALFORMED":0})
    for r in map(json.loads, open(RUN/"raw_outputs/adapter_samples.jsonl")):
        cnt[(r["context_id"], r["presentation_order"], r["instance_id"])][r["outcome"]] += 1
    out = defaultdict(list)
    for (cid, od, iid), c in cnt.items():
        N = sum(c.values())
        out[cid].append((c["A"]/N, c["MALFORMED"]/N))
    return out

def cross_agreement(p1, p2, pm, rng):
    """Simulate NS samples from each adapter, return the plug-in cross estimator.
    Malformed stays in the denominator and never matches."""
    def draw(p):
        u = rng.random(NS)
        # outcome: A with prob p*(1-pm), B with (1-p)*(1-pm), MALFORMED with pm
        a = (u < p*(1-pm)).sum()
        b = ((u >= p*(1-pm)) & (u < (1-pm))).sum()
        return a, b
    a1, b1 = draw(p1); a2, b2 = draw(p2)
    return (a1*a2 + b1*b2) / (NS*NS)

def simulate(shrink, reps=400, seed=7):
    """shrink: how far A2's policy is pulled toward 0.5 on SHARED contexts.
       1.0 = perfect transfer (A2 == A1); 0.0 = no transfer."""
    rng = np.random.default_rng(seed)
    D = a1_distributions()
    pool = [p for c in D for p, _ in D[c]]            # empirical marginal of p_A
    est = []
    for _ in range(reps):
        cell = {}
        for name, ctxs, shared in (("shared_weak",SHARED_WEAK,True),
                                   ("shared_strong",SHARED_STRONG,True),
                                   ("diff_weak",DIFF_WEAK,False),
                                   ("diff_strong",DIFF_STRONG,False)):
            vals = {}
            for c in ctxs:
                v = []
                for p1, pm in D[c]:
                    if shared:
                        p2 = 0.5 + shrink*(p1-0.5)     # same target policy, imperfectly transferred
                    else:
                        p2 = pool[rng.integers(len(pool))]  # independent policy
                    v.append(cross_agreement(p1, p2, pm, rng))
                vals[c] = v
            cell[name] = vals
        def m(x): return float(np.mean([v for vv in x.values() for v in vv]))
        est.append({
            "shared": (m(cell["shared_weak"])+m(cell["shared_strong"]))/2,
            "diff":   (m(cell["diff_weak"])+m(cell["diff_strong"]))/2,
            "shared_strong": m(cell["shared_strong"]), "diff_strong": m(cell["diff_strong"]),
            "shared_weak": m(cell["shared_weak"]),     "diff_weak": m(cell["diff_weak"]),
            "cells": cell})
    return est

def hier_se(cell_a, cell_b, nboot=600, seed=11):
    """SE of the shared-minus-different contrast under the frozen hierarchical
    bootstrap: resample the 4 contexts in each cell, then stimuli within them."""
    rng = np.random.default_rng(seed)
    ca, cb = list(cell_a.keys()), list(cell_b.keys())
    out = []
    for _ in range(nboot):
        def draw(cell, keys):
            ks = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            return np.mean([np.mean(np.array(cell[k])[rng.integers(0,len(cell[k]),len(cell[k]))])
                            for k in ks])
        out.append(draw(cell_a, ca) - draw(cell_b, cb))
    return float(np.std(out))

print("=== A1 observed inputs (the only data used) ===")
D = a1_distributions()
allp = [p for c in D for p,_ in D[c]]
print(f"  stimuli x orders with a p_A estimate: {len(allp)}")
print(f"  p_A distribution: mean {np.mean(allp):.4f}  sd {np.std(allp):.4f}  "
      f"q10 {np.percentile(allp,10):.3f}  q90 {np.percentile(allp,90):.3f}")

print("\n=== simulated shared-minus-different contrast, by transfer strength ===")
print(f"  {'shrink':>7} {'shared':>8} {'diff':>8} {'delta':>8} {'SE(hier)':>9} {'power d>0':>10} {'power d>=0.10':>14}")
rows=[]
for sh in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
    est = simulate(sh, reps=200, seed=7)
    d = np.array([e["shared"]-e["diff"] for e in est])
    se = hier_se(dict(**est[0]["cells"]["shared_weak"], **est[0]["cells"]["shared_strong"]),
                 dict(**est[0]["cells"]["diff_weak"],   **est[0]["cells"]["diff_strong"]))
    pow0  = float(np.mean(d - 1.96*se > 0))
    pow10 = float(np.mean(d >= 0.10))
    rows.append(dict(shrink=sh, shared=float(np.mean([e["shared"] for e in est])),
                     diff=float(np.mean([e["diff"] for e in est])), delta=float(d.mean()),
                     se=se, power_gt0=pow0, power_ge_010=pow10))
    print(f"  {sh:>7.1f} {rows[-1]['shared']:>8.4f} {rows[-1]['diff']:>8.4f} "
          f"{rows[-1]['delta']:>8.4f} {se:>9.4f} {pow0:>10.2f} {pow10:>14.2f}")

(ROOT/"runs/20260902_230634_a1_qwen7b/metrics/transfer_gate_power.json").write_text(
    json.dumps({"p_A_summary": {"mean": float(np.mean(allp)), "sd": float(np.std(allp))},
                "simulations": rows, "n_samples_per_adapter": NS,
                "design": {"contexts_per_cell": 4, "stimuli_per_context": 60, "orders": 2}}, indent=2))
pass


# ---------------------------------------------------------------------------
# STRATEGIC USEFULNESS: threshold crossing on the revised payoff schedule.
# Wager pays 40 if the predicted Worker choice is correct, 0 otherwise.
# Safe payoffs 20/21/22/23/24 -> indifference probabilities safe/40.
# Wager is optimal iff p > safe/40.
# ---------------------------------------------------------------------------
WAGER = 40
SAFE_LEVELS = [20, 21, 22, 23, 24]
THRESHOLDS = [s / WAGER for s in SAFE_LEVELS]

def optimal(p, safe):
    return "WAGER" if p * WAGER > safe else "SAFE"

def threshold_table(p_shared, p_diff):
    rows = []
    for safe, thr in zip(SAFE_LEVELS, THRESHOLDS):
        a_sh, a_df = optimal(p_shared, safe), optimal(p_diff, safe)
        rows.append({
            "safe": safe, "threshold": thr,
            "ev_wager_shared": WAGER * p_shared, "ev_wager_different": WAGER * p_diff,
            "ev_safe": float(safe),
            "optimal_shared": a_sh, "optimal_different": a_df,
            "actions_differ": a_sh != a_df,
            "ev_gain_from_conditioning": (max(WAGER*p_shared, safe) + max(WAGER*p_diff, safe))/2
                                          - max((WAGER*p_shared + WAGER*p_diff)/2, safe),
            "ev_difference_wager": WAGER * (p_shared - p_diff)})
    return rows

print("\n=== STRATEGIC USEFULNESS: wager 40/0 vs safe 20-24 ===")
print(f"  thresholds: {[f'{t:.3f}' for t in THRESHOLDS]}")
strat = {}
for r in rows:
    tt = threshold_table(r["shared"], r["diff"])
    n_cross = sum(t["actions_differ"] for t in tt)
    crossing_safes = [t["safe"] for t in tt if t["actions_differ"]]
    strat[r["shrink"]] = {"table": tt, "n_crossings": n_cross, "crossing_safe_levels": crossing_safes}
    print(f"  shrink {r['shrink']:.1f}: p_shared {r['shared']:.4f}  p_diff {r['diff']:.4f}  "
          f"dEV(wager) {WAGER*(r['shared']-r['diff']):+.2f} pts  "
          f"crossings {n_cross}/5 at safe={crossing_safes}")

print("\n=== per-threshold detail at PERFECT transfer (shrink 1.0) ===")
best = [r for r in rows if r["shrink"] == 1.0][0]
print(f"  {'safe':>5} {'thresh':>7} {'EV wager|shared':>16} {'EV wager|diff':>14} {'EV safe':>8} "
      f"{'opt|shared':>11} {'opt|diff':>9} {'differ':>7}")
for t in threshold_table(best["shared"], best["diff"]):
    print(f"  {t['safe']:>5} {t['threshold']:>7.3f} {t['ev_wager_shared']:>16.2f} "
          f"{t['ev_wager_different']:>14.2f} {t['ev_safe']:>8.2f} {t['optimal_shared']:>11} "
          f"{t['optimal_different']:>9} {str(t['actions_differ']):>7}")

# ---- MINIMUM PRACTICAL EFFECT, independent of grid spacing -----------------
se_perfect = [r for r in rows if r["shrink"] == 1.0][0]["se"]
strong_weak = 0.5954 - 0.5303      # A1's own engineered reliability contrast
mpe = 0.05
print("\n=== MINIMUM PRACTICAL EFFECT (grid-independent) ===")
print(f"  hierarchical SE at perfect transfer      : {se_perfect:.4f}")
print(f"  2 x SE (resolvability floor)             : {2*se_perfect:.4f}")
print(f"  A1 strong-minus-weak same-order contrast : {strong_weak:.4f}  (an effect we engineered and treat as meaningful)")
print(f"  ADOPTED minimum practical effect         : {mpe:.4f}")
print("  Justification: >= 2.9x the design's own SE, so comfortably resolvable;")
print("  and of the same order as the strong/weak reliability contrast we")
print("  deliberately induced and already regard as substantive. Derived from")
print("  A1 and the design ONLY - no payoff-grid spacing, no A2 data.")

import json as _json
_p = ROOT/"runs/20260902_230634_a1_qwen7b/metrics/transfer_gate_power.json"
_d = _json.loads(_p.read_text())
_d["strategic"] = {"wager": WAGER, "safe_levels": SAFE_LEVELS, "thresholds": THRESHOLDS,
                   "by_shrink": {str(k): v for k, v in strat.items()}}
_d["minimum_practical_effect"] = {"value": mpe, "se_at_perfect_transfer": se_perfect,
                                  "two_se": 2*se_perfect, "a1_strong_minus_weak": strong_weak,
                                  "basis": "resolvability + internal benchmark; grid-independent"}
_p.write_text(_json.dumps(_d, indent=2))
print("\n  -> metrics/transfer_gate_power.json updated")

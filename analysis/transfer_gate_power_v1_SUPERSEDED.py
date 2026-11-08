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
print("\n  -> metrics/transfer_gate_power.json")

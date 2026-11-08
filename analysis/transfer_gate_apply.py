"""Apply the frozen A1-A2 transfer gate (TRANSFER_GATE_A1_A2.md v2)."""
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import hierarchical_ci

ROOT = Path(__file__).resolve().parents[1]
A1 = ROOT/"runs/20260902_230634_a1_qwen7b"; A2 = ROOT/"runs/20260903_170443_a2_qwen7b"
SHARED = {"weak": ["C01","C02","C05","C08"], "strong": ["C03","C04","C10","C16"]}
DIFF   = {"weak": ["C06","C07","C13","C15"], "strong": ["C09","C11","C12","C14"]}
WAGER, SAFE = 40, [20,21,22,23,24]

def counts(run):
    c = defaultdict(lambda: {"A":0,"B":0,"MALFORMED":0})
    for r in map(json.loads, open(run/"raw_outputs/adapter_samples.jsonl")):
        c[(r["context_id"], r["presentation_order"], r["instance_id"])][r["outcome"]] += 1
    return c

c1, c2 = counts(A1), counts(A2)
per_stim = defaultdict(list)                       # context -> [stimulus value]
tmp = defaultdict(list)
for k in set(c1) & set(c2):
    cid, od, iid = k
    a, b = c1[k], c2[k]
    N1, N2 = sum(a.values()), sum(b.values())
    x = (a["A"]*b["A"] + a["B"]*b["B"]) / (N1*N2)  # malformed in denominators, never matches
    tmp[(cid, iid)].append(x)
for (cid, iid), v in tmp.items():
    per_stim[cid].append(float(np.mean(v)))        # average the stimulus's AB and BA values

def cell(ctxs): return {c: per_stim[c] for c in ctxs}
def mean(d):    return float(np.mean([x for v in d.values() for x in v]))

out = {"gate": "A1-A2 behavioural transfer v2", "cells": {}}
for name, ctxs in (("shared_weak",SHARED["weak"]),("shared_strong",SHARED["strong"]),
                   ("diff_weak",DIFF["weak"]),("diff_strong",DIFF["strong"])):
    d = cell(ctxs)
    out["cells"][name] = {"contexts": ctxs, "n_stimuli": sum(len(v) for v in d.values()),
                          "mean": mean(d), "ci": hierarchical_ci(d, 10000, 90210)}
sh = {**cell(SHARED["weak"]), **cell(SHARED["strong"])}
df = {**cell(DIFF["weak"]),   **cell(DIFF["strong"])}
p_sh, p_df = mean(sh), mean(df)
paired = {c: [x - p_df for x in per_stim[c]] for c in sh}
out["component1_scientific"] = {
    "p_shared": p_sh, "p_different": p_df, "delta": p_sh - p_df,
    "ci_hierarchical": hierarchical_ci({**{f"S_{k}":v for k,v in sh.items()},
                                        **{f"D_{k}":[-x for x in v] for k,v in df.items()}}, 10000, 90210),
    "strong_delta": out["cells"]["shared_strong"]["mean"] - out["cells"]["diff_strong"]["mean"],
    "weak_delta":   out["cells"]["shared_weak"]["mean"]   - out["cells"]["diff_weak"]["mean"]}
rows=[]
for s in SAFE:
    o_sh = "WAGER" if WAGER*p_sh > s else "SAFE"
    o_df = "WAGER" if WAGER*p_df > s else "SAFE"
    rows.append({"safe": s, "threshold": s/WAGER, "ev_wager_shared": WAGER*p_sh,
                 "ev_wager_different": WAGER*p_df, "ev_safe": float(s),
                 "optimal_shared": o_sh, "optimal_different": o_df,
                 "actions_differ": o_sh != o_df})
out["component2_strategic"] = {"wager": WAGER, "rows": rows,
    "n_crossings": sum(r["actions_differ"] for r in rows),
    "crossing_safe_levels": [r["safe"] for r in rows if r["actions_differ"]],
    "ev_difference_wager": WAGER*(p_sh-p_df)}
out["component3_minimum_practical_effect"] = {"delta_min": 0.05, "observed": p_sh-p_df,
                                              "meets": (p_sh-p_df) >= 0.05}
(A2/"metrics/transfer_gate_result.json").write_text(json.dumps(out, indent=2))
print("=== COMPONENT 1 - scientific transfer ===")
c = out["component1_scientific"]
print(f"  p_shared {c['p_shared']:.4f}   p_different {c['p_different']:.4f}   delta {c['delta']:+.4f}")
print(f"  hierarchical 95% CI on delta: [{c['ci_hierarchical'][0]:+.4f}, {c['ci_hierarchical'][1]:+.4f}]")
print(f"  strong stratum delta {c['strong_delta']:+.4f}   weak stratum delta {c['weak_delta']:+.4f}")
for n,v in out["cells"].items():
    print(f"    {n:<14} mean {v['mean']:.4f}  CI [{v['ci'][0]:.4f},{v['ci'][1]:.4f}]  n_stim {v['n_stimuli']}")
print("\n=== COMPONENT 2 - strategic usefulness ===")
print(f"  {'safe':>5} {'thresh':>7} {'EV|shared':>10} {'EV|diff':>9} {'EV safe':>8} {'opt|sh':>7} {'opt|df':>7} {'differ':>7}")
for r in rows:
    print(f"  {r['safe']:>5} {r['threshold']:>7.3f} {r['ev_wager_shared']:>10.2f} {r['ev_wager_different']:>9.2f} "
          f"{r['ev_safe']:>8.2f} {r['optimal_shared']:>7} {r['optimal_different']:>7} {str(r['actions_differ']):>7}")
print(f"  crossings: {out['component2_strategic']['n_crossings']}/5   dEV(wager) {out['component2_strategic']['ev_difference_wager']:+.2f} pts")
print("\n=== COMPONENT 3 - minimum practical effect ===")
print(f"  delta_min 0.0500   observed {c['delta']:+.4f}   meets: {out['component3_minimum_practical_effect']['meets']}")

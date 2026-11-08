"""Context-level results. Stimulus-first: repeated generations are pooled only
at the stimulus x order level; every interval above that resamples stimuli, and
strong/weak/pooled use a hierarchical context->stimulus bootstrap.
"""
import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import (OUTCOMES, same_order_match, cross_order_match,
                        target_agreement, outcome_rate, cluster_ci,
                        hierarchical_ci, wilson_naive)

ROOT = Path(__file__).resolve().parents[1]


def _mean(v):
    v = [x for x in v if not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def build(run, adapter):
    bcfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text())["bootstrap"]
    NB, SEED = bcfg["n_boot"], bcfg["seed"]
    keys = {r["row_id"]: r["target_letter"]
            for r in csv.DictReader(open(ROOT / "data" / "common" / "policy_val_keys.csv"))
            if r["adapter_id"] == adapter}
    rel = {r["context_id"]: r["assigned_consistency"]
           for r in csv.DictReader(open(ROOT / "data" / "design" / "contexts.csv"))}

    report = {"run": run.name, "adapter": adapter, "n_boot": NB, "seed": SEED,
              "unit_hierarchy": "L0 generation -> L1 stimulus x order -> L2 context x order"
                                " -> L3 context -> L4 strong/weak/pooled (hierarchical bootstrap)",
              "L1_stimulus": [], "L2_cells": [], "L4_summaries": {}, "cross_order": {}}

    for which in ("base", "adapter"):
        f = run / "raw_outputs" / f"{which}_samples.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f)]

        # ---- L1: pool repeated generations, ONCE, per (context, order, stimulus)
        cnt = defaultdict(lambda: {o: 0 for o in OUTCOMES})
        for r in rows:
            cnt[(r["context_id"], r["presentation_order"], r["instance_id"])][r["outcome"]] += 1
        L1 = {}
        for (cid, od, iid), c in cnt.items():
            N = sum(c.values())
            tgt = keys[f"policy_val|{cid}|{iid:04d}|{od}"]
            L1[(cid, od, iid)] = {
                "context_id": cid, "presentation_order": od, "instance_id": iid,
                "n_generations": N, **{f"n_{o}": c[o] for o in OUTCOMES},
                "same_order_match": same_order_match(c["A"], c["B"], N),
                "target_agreement": target_agreement(c, tgt),
                "rate_A": outcome_rate(c, "A"), "rate_B": outcome_rate(c, "B"),
                "malformed_rate": outcome_rate(c, "MALFORMED"),
                "target_letter": tgt, "_counts": c,
            }
            report["L1_stimulus"].append({"model": which, **{k: v for k, v in
                                          L1[(cid, od, iid)].items() if k != "_counts"}})

        # ---- L2: mean over stimuli within (context, order); CI resamples STIMULI
        byc = defaultdict(list)
        for k, v in L1.items():
            byc[(v["context_id"], v["presentation_order"])].append(v)
        for (cid, od), vs in sorted(byc.items()):
            tot = {o: sum(v[f"n_{o}"] for v in vs) for o in OUTCOMES}
            Ngen = sum(tot.values())
            agree_gen = sum(v["_counts"][v["target_letter"]] for v in vs)
            cell = {"model": which, "context_id": cid, "presentation_order": od,
                    "assigned_consistency": rel[cid], "n_stimuli": len(vs),
                    "total_generations": Ngen,
                    **{f"n_{o}": tot[o] for o in OUTCOMES}}
            for name in ("same_order_match", "target_agreement", "rate_A",
                         "rate_B", "malformed_rate"):
                vals = [v[name] for v in vs]
                cell[name] = _mean(vals)
                cell[f"{name}_ci_cluster"] = cluster_ci(vals, NB, SEED)
            # diagnostic only, to expose how anticonservative the naive interval is
            cell["target_agreement_ci_wilson_naive_DIAGNOSTIC"] = wilson_naive(agree_gen, Ngen)
            cell["malformed_rate_ci_wilson_naive_DIAGNOSTIC"] = wilson_naive(tot["MALFORMED"], Ngen)
            report["L2_cells"].append(cell)

        # ---- cross-order: L1 unit is the STIMULUS (its AB and BA pools combined)
        pair = defaultdict(dict)
        for k, v in L1.items():
            pair[(v["context_id"], v["instance_id"])][v["presentation_order"]] = v["_counts"]
        xo_by_ctx = defaultdict(list)
        for (cid, iid), d in pair.items():
            if {"AB", "BA"} <= set(d):
                xo_by_ctx[cid].append(cross_order_match(d["AB"], d["BA"]))
        report["cross_order"][which] = {
            "unit": "stimulus (AB pool vs BA pool), aggregated to context",
            "mean": _mean([x for v in xo_by_ctx.values() for x in v]),
            "ci_hierarchical": hierarchical_ci(xo_by_ctx, NB, SEED),
            "n_contexts": len(xo_by_ctx),
            "n_stimuli": sum(len(v) for v in xo_by_ctx.values())}

        # ---- L4: hierarchical bootstrap over CONTEXTS then STIMULI
        for group in ("strong", "weak", "POOLED_supplementary"):
            sel = defaultdict(list)
            for k, v in L1.items():
                if group == "POOLED_supplementary" or rel[v["context_id"]] == group:
                    sel[v["context_id"]].append(v)
            if not sel:
                continue
            s = {"n_contexts": len(sel), "n_stimuli": sum(len(v) for v in sel.values())}
            for name in ("same_order_match", "target_agreement", "malformed_rate"):
                bc = {c: [v[name] for v in vs] for c, vs in sel.items()}
                s[name] = _mean([x for v in bc.values() for x in v])
                s[f"{name}_ci_hierarchical"] = hierarchical_ci(bc, NB, SEED)
            report["L4_summaries"][f"{which}/{group}"] = s
    return report


def main(run_dir, adapter):
    run = Path(run_dir)
    rep = build(run, adapter)
    (run / "metrics" / "context_level_report.json").write_text(json.dumps(rep, indent=2))

    print("=== L2: context x order (mean over STIMULI; CI resamples stimuli) ===")
    h = (f"{'model':<8}{'ctx':<6}{'ord':<4}{'rel':<7}{'stim':>5}{'gen':>5}{'A':>4}{'B':>4}"
         f"{'MAL':>4}{'mal_r':>7}{'tgt_agr':>8}{'agr 95%CI (cluster)':>22}"
         f"{'same_ord':>9}{'95%CI (cluster)':>19}")
    print(h); print("-" * len(h))
    for c in rep["L2_cells"]:
        a = c["target_agreement_ci_cluster"]; m = c["same_order_match_ci_cluster"]
        print(f"{c['model']:<8}{c['context_id']:<6}{c['presentation_order']:<4}"
              f"{c['assigned_consistency']:<7}{c['n_stimuli']:>5}{c['total_generations']:>5}"
              f"{c['n_A']:>4}{c['n_B']:>4}{c['n_MALFORMED']:>4}{c['malformed_rate']:>7.3f}"
              f"{c['target_agreement']:>8.3f}   [{a[0]:.3f},{a[1]:.3f}]      "
              f"{c['same_order_match']:>7.3f}   [{m[0]:.3f},{m[1]:.3f}]")
    print("\n=== clustered vs NAIVE Wilson (diagnostic: naive is anticonservative) ===")
    for c in rep["L2_cells"][:4]:
        a, w = c["target_agreement_ci_cluster"], c["target_agreement_ci_wilson_naive_DIAGNOSTIC"]
        print(f"  {c['model']:<8}{c['context_id']} {c['presentation_order']}  "
              f"clustered [{a[0]:.3f},{a[1]:.3f}] width {a[1]-a[0]:.3f}   |   "
              f"naive-Wilson [{w[0]:.3f},{w[1]:.3f}] width {w[1]-w[0]:.3f}")
    print("\n=== PRIMARY same-order | SECONDARY cross-order (hierarchical CI) ===")
    for w, v in rep["cross_order"].items():
        lo, hi = v["ci_hierarchical"]
        print(f"  {w:<8} cross_order={v['mean']:.4f}  95%CI [{lo:.4f},{hi:.4f}]  "
              f"({v['n_contexts']} contexts, {v['n_stimuli']} stimuli)")
    print("\n=== L4 strong vs weak (hierarchical: contexts -> stimuli); pooled supplementary ===")
    for k, v in sorted(rep["L4_summaries"].items()):
        lo, hi = v["same_order_match_ci_hierarchical"]
        alo, ahi = v["target_agreement_ci_hierarchical"]
        print(f"  {k:<32} ctx={v['n_contexts']} stim={v['n_stimuli']:<4} "
              f"same_ord={v['same_order_match']:.4f} [{lo:.4f},{hi:.4f}]  "
              f"tgt_agr={v['target_agreement']:.4f} [{alo:.4f},{ahi:.4f}]")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--adapter", default="A1")
    a = ap.parse_args(); main(a.run, a.adapter)

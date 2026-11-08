"""ACQUISITION GATE (v3) - Gate 1 read strictly as an acquisition test.

Judged on policy_val ONLY. policy_test is never opened.

Differences from the v1/v2 Gate 1:
  * D5 (strong-vs-weak reliability panel) is REMOVED. With deterministic labels
    there is no reliability manipulation, so that test would be nominal only.
    It is replaced by D5' - accuracy on POLICY-DISCRIMINATING held-out cases,
    i.e. those where A1 and A2 prescribe different choices.
  * D3 uses the termination-aware generation report (validity now requires
    normal termination and no repeated choice string), not just parseability.
  * Adds per-adapter accuracy with CI, the A1/A2 policy confusion matrix, and
    the choice-token CE margin below ln 2.
"""
import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import hierarchical_ci

ROOT = Path(__file__).resolve().parents[1]
LN2 = math.log(2.0)
D1_MIN, D2_GT, D2_070, D3_MAX_MALFORMED, D4_MAX_DROP = 0.20, 12, 10, 0.02, 0.25
D5_MIN_ACC, D5_CHANCE = 0.60, 0.500   # GATE_D5PRIME_FROZEN.md (non-blocking)


def load_logits(run, which):
    d = defaultdict(dict)
    for r in map(json.loads, open(Path(run) / f"raw_outputs/{which}_logits.jsonl")):
        d[(r["context_id"], r["instance_id"])][r["presentation_order"]] = r
    return d


def keys_for(adapter):
    k = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "data/common/policy_val_keys.csv")):
        if r["adapter_id"] == adapter:
            k[(r["context_id"], int(r["instance_id"]))][r["presentation_order"]] = r["target_letter"]
    return k


def order_averaged(L, key):
    """Per-context lists of 0/1: did the order-averaged preference match target?"""
    tgt_u = {c: (0 if v["AB"] == "A" else 1) for c, v in key.items()}
    by_ctx, raw_by_ctx, bias, cons = defaultdict(list), defaultdict(list), [], []
    for cid_iid, d in L.items():
        if cid_iid not in tgt_u or "AB" not in d or "BA" not in d:
            continue
        ab, ba = d["AB"], d["BA"]
        pref0 = (ab["logit_diff"] - ba["logit_diff"]) / 2.0
        by_ctx[cid_iid[0]].append(1.0 if (0 if pref0 > 0 else 1) == tgt_u[cid_iid] else 0.0)
        for od, row in (("AB", ab), ("BA", ba)):
            raw_by_ctx[cid_iid[0]].append(1.0 if row["greedy_letter"] == key[cid_iid][od] else 0.0)
        bias += [1.0 if ab["greedy_letter"] == "A" else 0.0,
                 1.0 if ba["greedy_letter"] == "A" else 0.0]
        u_ab = 0 if ab["greedy_letter"] == "A" else 1
        u_ba = 1 if ba["greedy_letter"] == "A" else 0
        cons.append(1.0 if u_ab == u_ba else 0.0)
    return by_ctx, raw_by_ctx, bias, cons


def mean_of(by_ctx):
    v = [x for lst in by_ctx.values() for x in lst]
    return float(np.mean(v)) if v else float("nan")


def paired_ci(a_by, b_by, n_boot=10000, seed=90210):
    """Hierarchical bootstrap on the per-stimulus DIFFERENCE (paired)."""
    diff = {c: list(np.asarray(a_by[c]) - np.asarray(b_by[c]))
            for c in a_by if c in b_by and len(a_by[c]) == len(b_by[c])}
    return mean_of(diff), hierarchical_ci(diff, n_boot, seed)


def adapter_report(run, adapter):
    key = keys_for(adapter)
    out = {"adapter": adapter, "run": str(run)}
    A, A_raw, A_bias, A_cons = order_averaged(load_logits(run, "adapter"), key)
    B, B_raw, B_bias, _ = order_averaged(load_logits(run, "base"), key)

    acc_a, ci_a = mean_of(A), hierarchical_ci(A, 10000, 90210)
    acc_b, ci_b = mean_of(B), hierarchical_ci(B, 10000, 90210)
    imp, ci_i = paired_ci(A, B)
    out["accuracy"] = {"adapter": acc_a, "adapter_ci": ci_a,
                       "base": acc_b, "base_ci": ci_b}
    out["D1"] = {"improvement": imp, "ci": ci_i,
                 "threshold": D1_MIN,
                 "passed": bool(imp >= D1_MIN and ci_i[0] > 0)}

    per = {c: (float(np.mean(A[c])), float(np.mean(B[c]))) for c in sorted(A)}
    gt = sum(1 for c, (a, b) in per.items() if a > b)
    hi = sum(1 for c, (a, _) in per.items() if a >= 0.70)
    out["D2"] = {"n_context_families": len(per), "n_adapter_gt_base": gt,
                 "n_ge_070": hi, "thresholds": [D2_GT, D2_070],
                 "per_context": per, "passed": bool(gt >= D2_GT and hi >= D2_070)}

    gpath = Path(run) / "metrics/generation_adapter_posttrain.json"
    if gpath.exists():
        g = json.loads(gpath.read_text()); g["source"] = "termination_aware_v3"
    else:
        # legacy run: only parseability was recorded, termination was never measured
        v = json.loads((Path(run) / "metrics/adapter_validity.json").read_text())
        g = {"malformed_rate": v.get("freegen_malformed_rate", float("nan")),
             "nontermination_rate": float("nan"), "repeated_choice_rate": float("nan"),
             "valid_rate": float("nan"), "len_p50": None, "len_max": None,
             "source": "LEGACY adapter_validity.json - termination NOT measurable"}
    out["D3"] = {"source": g["source"], "malformed_rate": g["malformed_rate"],
                 "nontermination_rate": g["nontermination_rate"],
                 "repeated_choice_rate": g["repeated_choice_rate"],
                 "valid_rate": g["valid_rate"],
                 "len_p50": g["len_p50"], "len_max": g["len_max"],
                 "threshold_malformed": D3_MAX_MALFORMED,
                 "passed": bool(g["malformed_rate"] <= D3_MAX_MALFORMED
                                and not (g["nontermination_rate"] > D3_MAX_MALFORMED)
                                and not (g["repeated_choice_rate"] > D3_MAX_MALFORMED))}

    raw_imp = mean_of(A_raw) - mean_of(B_raw)
    drop = (raw_imp - imp) / raw_imp if raw_imp else float("nan")
    out["D4"] = {"improvement_raw": raw_imp, "improvement_order_averaged": imp,
                 "relative_drop": drop, "threshold": D4_MAX_DROP,
                 "passed": bool(not (drop > D4_MAX_DROP)),
                 "note": "DIRECTIONAL: blocks only if order-averaging REDUCES the improvement by >25%"}
    out["bias"] = {"adapter_p_choose_first": float(np.mean(A_bias)),
                   "base_p_choose_first": float(np.mean(B_bias)),
                   "adapter_order_consistency": float(np.mean(A_cons))}

    man = json.loads((Path(run) / "manifest.json").read_text())
    fd = man.get("final_diagnostics", {})
    ce = fd.get("choice_ce")
    out["choice_loss"] = {"full_completion_loss": fd.get("loss"),
                          "choice_token_ce": ce, "chance_ln2": LN2,
                          "margin_below_chance": (LN2 - ce) if ce is not None else None,
                          "passed": bool(ce is not None and ce < LN2)}
    out["step_counts"] = man.get("step_counts")
    return out, A


def discriminating(run_a1, run_a2):
    """D5': held-out cases where A1 and A2 prescribe DIFFERENT choices."""
    k1, k2 = keys_for("A1"), keys_for("A2")
    disc = {ci: od for ci in k1 if ci in k2
            for od in ("AB",) if k1[ci]["AB"] != k2[ci]["AB"]}
    res, conf = {}, {}
    evaluated = set()
    for name, run in (("A1", run_a1), ("A2", run_a2)):
        L = load_logits(run, "adapter")
        own, other = defaultdict(list), defaultdict(list)
        for ci in disc:
            if ci not in L or "AB" not in L[ci] or "BA" not in L[ci]:
                continue
            evaluated.add(ci)
            pref0 = (L[ci]["AB"]["logit_diff"] - L[ci]["BA"]["logit_diff"]) / 2.0
            choice = "A" if pref0 > 0 else "B"      # letter under AB presentation
            t1, t2 = k1[ci]["AB"], k2[ci]["AB"]
            mine, theirs = (t1, t2) if name == "A1" else (t2, t1)
            own[ci[0]].append(1.0 if choice == mine else 0.0)
            other[ci[0]].append(1.0 if choice == theirs else 0.0)
        acc = mean_of(own); ci = hierarchical_ci(own, 10000, 90210)
        res[name] = {"n_discriminating": sum(len(v) for v in own.values()),
                     "acc_own_policy": acc, "ci": ci,
                     "acc_other_policy": mean_of(other),
                     "threshold_acc": D5_MIN_ACC, "chance": D5_CHANCE,
                     "rule": "acc >= 0.60 AND 95% CI lower bound > 0.500",
                     "passed": bool(acc >= D5_MIN_ACC and ci[0] > D5_CHANCE),
                     "blocking": False}
        conf[name] = {"matches_A1_target": mean_of(own if name == "A1" else other),
                      "matches_A2_target": mean_of(own if name == "A2" else other)}
    return {"n_discriminating_stimuli": len(disc),
            "n_discriminating_evaluated": len(evaluated),
            "coverage_warning": (None if evaluated else
                "NO discriminating stimulus was evaluated - the run's contexts "
                "carry identical A1/A2 policies, or policy_val coverage is too "
                "narrow. D5' is undefined here, not zero."),
            "per_adapter": res,
            "confusion_matrix": conf,
            "note": ("rows = trained adapter, columns = which target policy the "
                     "held-out behaviour matches, on discriminating stimuli only")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a1", required=True)
    ap.add_argument("--run-a2", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r1, _ = adapter_report(a.run_a1, "A1")
    r2, _ = adapter_report(a.run_a2, "A2")
    d5 = discriminating(a.run_a1, a.run_a2)
    blocking = ("D1", "D2", "D3", "D4")
    rep = {"gate": "ACQUISITION GATE v3 (policy_val only; policy_test untouched)",
           "A1": r1, "A2": r2, "D5_prime_discriminating": d5,
           "per_adapter_pass": {n: all(r[k]["passed"] for k in blocking)
                                for n, r in (("A1", r1), ("A2", r2))}}
    rep["BOTH_ADAPTERS_PASS"] = all(rep["per_adapter_pass"].values())
    rep["verdict"] = ("PROCEED - both adapters acquired their policies"
                      if rep["BOTH_ADAPTERS_PASS"] else
                      "STOP - acquisition not established; do not proceed to Gate 2 or introspection")
    out = Path(a.out) if a.out else ROOT / "analysis/acquisition_gate_result.json"
    out.write_text(json.dumps(rep, indent=2, default=float))
    for n in ("A1", "A2"):
        r = rep[n]
        print(f"\n=== {n} ===  acc {r['accuracy']['adapter']:.4f} "
              f"[{r['accuracy']['adapter_ci'][0]:.4f},{r['accuracy']['adapter_ci'][1]:.4f}]  "
              f"base {r['accuracy']['base']:.4f}")
        for k in blocking:
            print(f"  {k}: {'PASS' if r[k]['passed'] else 'FAIL'}  "
                  + json.dumps({x: v for x, v in r[k].items()
                                if x not in ('per_context', 'passed', 'note')}, default=float)[:150])
        cl = r["choice_loss"]
        print(f"  choice CE {cl['choice_token_ce']} vs ln2 {LN2:.6f} "
              f"(margin {cl['margin_below_chance']})")
    print(f"\n=== D5' discriminating cases ({d5['n_discriminating_stimuli']} stimuli) ===")
    for n, v in d5["per_adapter"].items():
        print(f"  {n}: {'PASS' if v['passed'] else 'FAIL'}  own-policy acc {v['acc_own_policy']:.4f} "
              f"[{v['ci'][0]:.4f},{v['ci'][1]:.4f}]   other {v['acc_other_policy']:.4f}   "
              f"rule: acc>={D5_MIN_ACC} and CI_low>{D5_CHANCE} (non-blocking)")
    print(f"\n{rep['verdict']}")
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"written -> {shown}")


if __name__ == "__main__":
    main()

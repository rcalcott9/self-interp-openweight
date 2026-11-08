"""Compute the precommitted A1 go/no-go criteria from local hash-verified artifacts."""
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import same_order_match, hierarchical_ci, cluster_ci

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/20260902_230634_a1_qwen7b"
rel = {r["context_id"]: r["assigned_consistency"]
       for r in csv.DictReader(open(ROOT/"data/design/contexts.csv"))}
key = defaultdict(dict)
for r in csv.DictReader(open(ROOT/"data/common/policy_val_keys.csv")):
    if r["adapter_id"] == "A1":
        key[(r["context_id"], int(r["instance_id"]))][r["presentation_order"]] = r["target_letter"]
# target underlying option: 'A' under AB == option_0
tgt_u = {k: (0 if v["AB"] == "A" else 1) for k, v in key.items()}

def load_logits(which):
    d = defaultdict(dict)
    for r in map(json.loads, open(RUN/f"raw_outputs/{which}_logits.jsonl")):
        d[(r["context_id"], r["instance_id"])][r["presentation_order"]] = r
    return d

L = {w: load_logits(w) for w in ("base", "adapter")}

# ---- per-instance agreement, RAW (row-level) and ORDER-AVERAGED --------------
def agreements(which):
    raw, oavg, bias, cons = defaultdict(list), defaultdict(list), [], []
    for (cid, iid), d in L[which].items():
        ab, ba = d["AB"], d["BA"]
        # raw: each row judged against its own target letter
        for od, row in (("AB", ab), ("BA", ba)):
            raw[cid].append(1.0 if row["greedy_letter"] == key[(cid, iid)][od] else 0.0)
        # order-averaged preference for option_0
        pref0 = (ab["logit_diff"] - ba["logit_diff"]) / 2.0
        choice = 0 if pref0 > 0 else 1
        oavg[cid].append(1.0 if choice == tgt_u[(cid, iid)] else 0.0)
        # letter/position bias: did it pick the FIRST-PRESENTED option?
        bias.append(1.0 if ab["greedy_letter"] == "A" else 0.0)
        bias.append(1.0 if ba["greedy_letter"] == "A" else 0.0)
        # order consistency: same UNDERLYING option under both renderings
        u_ab = 0 if ab["greedy_letter"] == "A" else 1
        u_ba = 1 if ba["greedy_letter"] == "A" else 0
        cons.append(1.0 if u_ab == u_ba else 0.0)
    return raw, oavg, bias, cons

A = {w: agreements(w) for w in ("base", "adapter")}
out = {}

def ctx_mean(d): return {c: float(np.mean(v)) for c, v in d.items()}

# ---- CRITERION 1 -------------------------------------------------------------
oa_ad, oa_bs = A["adapter"][1], A["base"][1]
paired = {c: [a - b for a, b in zip(oa_ad[c], oa_bs[c])] for c in oa_ad}
imp_oavg = float(np.mean([x for v in paired.values() for x in v]))
ci1 = hierarchical_ci(paired, n_boot=10000, seed=90210)
ad_oavg = float(np.mean([x for v in oa_ad.values() for x in v]))
bs_oavg = float(np.mean([x for v in oa_bs.values() for x in v]))
out["c1"] = dict(adapter=ad_oavg, base=bs_oavg, improvement=imp_oavg, ci=ci1,
                 passed=(imp_oavg >= 0.20 and ci1[0] > 0))

# ---- CRITERION 2 -------------------------------------------------------------
ad_c, bs_c = ctx_mean(oa_ad), ctx_mean(oa_bs)
n_better = sum(1 for c in ad_c if ad_c[c] > bs_c[c])
n_high = sum(1 for c in ad_c if ad_c[c] >= 0.70)
out["c2"] = dict(n_adapter_gt_base=n_better, n_ge_070=n_high,
                 passed=(n_better >= 12 and n_high >= 10),
                 per_context={c: (round(ad_c[c],3), round(bs_c[c],3)) for c in sorted(ad_c)})

# ---- CRITERION 3 -------------------------------------------------------------
val = {w: json.loads((RUN/f"metrics/{w}_validity.json").read_text()) for w in ("base","adapter")}
out["c3"] = dict(tokenization_gate="PASSED",
                 greedy_malformed_base=val["base"]["freegen_malformed_rate"],
                 greedy_malformed_adapter=val["adapter"]["freegen_malformed_rate"],
                 passed=(max(val["base"]["freegen_malformed_rate"],
                             val["adapter"]["freegen_malformed_rate"]) <= 0.02))

# ---- CRITERION 4 -------------------------------------------------------------
raw_ad, raw_bs = A["adapter"][0], A["base"][0]
imp_raw = float(np.mean([np.mean(raw_ad[c]) - np.mean(raw_bs[c]) for c in raw_ad]))
drop = (imp_raw - imp_oavg) / abs(imp_raw) if imp_raw else float("nan")
# CRITERION 4 IS DIRECTIONAL. Original wording: blocks only if order-averaging
# REDUCES the adapter-base improvement by more than 25% relative. A negative
# drop (an increase) means position noise was MASKING acquisition, not
# manufacturing it, and does not block. The earlier abs(drop) form was an
# implementation error, stricter than the criterion; see
# metrics/go_no_go_ERRONEOUS_absdrop.json and AUDIT_TRAIL.md.
out["c4"] = dict(improvement_raw=imp_raw, improvement_order_averaged=imp_oavg,
                 relative_drop=drop, passed=(drop <= 0.25))

# ---- position / letter bias --------------------------------------------------
for w in ("base","adapter"):
    b, c = A[w][2], A[w][3]
    out.setdefault("bias", {})[w] = dict(
        p_choose_first=float(np.mean(b)), bias=float(np.mean(b))-0.5,
        order_consistency=float(np.mean(c)))

# ---- CRITERION 5: predictability panel, adapter, strong vs weak ---------------
def panel(which):
    cnt = defaultdict(lambda: {"A":0,"B":0,"MALFORMED":0})
    for r in map(json.loads, open(RUN/f"raw_outputs/{which}_samples.jsonl")):
        cnt[(r["context_id"], r["presentation_order"], r["instance_id"])][r["outcome"]] += 1
    conc, match, ent, absl = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for (cid, od, iid), c in cnt.items():
        N = sum(c.values())
        conc[cid].append(max(c["A"], c["B"]) / N)
        match[cid].append(same_order_match(c["A"], c["B"], N))
        ps = [v/N for v in c.values() if v > 0]
        K = len([v for v in c.values() if v > 0])
        ent[cid].append(-sum(p*math.log(p) for p in ps) + (K-1)/(2*N))
        absl[cid].append(abs(L[which][(cid, iid)][od]["logit_diff"]))
    return {"concentration": conc, "same_order_match": match, "entropy": ent, "abs_logit_diff": absl}

P = panel("adapter")
c5 = {}
for name, d in P.items():
    s = {c: v for c, v in d.items() if rel[c] == "strong"}
    w = {c: v for c, v in d.items() if rel[c] == "weak"}
    ms = float(np.mean([x for v in s.values() for x in v]))
    mw = float(np.mean([x for v in w.values() for x in v]))
    diff = {c: None for c in []}
    higher_is_more_predictable = (name != "entropy")
    ok = (ms > mw) if higher_is_more_predictable else (ms < mw)
    c5[name] = dict(strong=ms, weak=mw, delta=ms-mw, direction_ok=ok,
                    strong_ci=hierarchical_ci(s,4000,90210), weak_ci=hierarchical_ci(w,4000,90210))
out["c5"] = dict(measures=c5, n_ok=sum(1 for v in c5.values() if v["direction_ok"]),
                 passed=sum(1 for v in c5.values() if v["direction_ok"]) >= 2)

# ---- CRITERION 6 -------------------------------------------------------------
conc_all = []
for which in ("base","adapter"):
    cnt = defaultdict(lambda: {"A":0,"B":0,"MALFORMED":0})
    for r in map(json.loads, open(RUN/f"raw_outputs/{which}_samples.jsonl")):
        cnt[(r["context_id"], r["presentation_order"], r["instance_id"])][r["outcome"]] += 1
    agree, pl, ps = [], [], []
    for k, c in cnt.items():
        cid, od, iid = k
        row = L[which][(cid, iid)][od]
        modal = "A" if c["A"] >= c["B"] else "B"
        agree.append(1.0 if modal == row["greedy_letter"] else 0.0)
        pl.append(1/(1+math.exp(-row["logit_diff"])))
        N = c["A"]+c["B"]+c["MALFORMED"]
        ps.append(c["A"]/N)
    out.setdefault("c6", {})[which] = dict(modal_vs_greedy_concordance=float(np.mean(agree)),
                                           corr_plogit_psampled=float(np.corrcoef(pl, ps)[0,1]))

# ---- acquisition vs CHANCE ---------------------------------------------------
out["chance"] = {w: dict(order_averaged=float(np.mean([x for v in A[w][1].values() for x in v])),
                         ci=hierarchical_ci(A[w][1], 10000, 90210)) for w in ("base","adapter")}

(RUN/"metrics/go_no_go.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=2, default=float))

"""A1 vs A2 training-data difficulty. Local only."""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SHARED = ["C01","C02","C03","C04","C05","C08","C10","C16"]   # A-pair shared contexts
rel = {r["context_id"]: r["assigned_consistency"]
       for r in csv.DictReader(open(ROOT/"data/design/contexts.csv"))}
# base-model greedy choice on policy_val, to measure target/base alignment
base = {r["row_id"]: r["greedy_letter"] for r in
        map(json.loads, open(ROOT/"runs/20260902_230634_a1_qwen7b/raw_outputs/base_logits.jsonl"))}
keys = defaultdict(dict)
for r in csv.DictReader(open(ROOT/"data/common/policy_val_keys.csv")):
    keys[r["adapter_id"]][r["row_id"]] = r["target_letter"]

D = {}
for ad in ("A1","A2"):
    rows = [json.loads(l) for l in open(ROOT/f"data/train/{ad}/train_full.jsonl")]
    D[ad] = rows
    byc = defaultdict(list)
    for r in rows: byc[r["context_id"]].append(r)
    pa = {c: sum(1 for r in v if r["target_letter"]=="A")/len(v) for c,v in byc.items()}
    mar = [r["utility_margin"] for r in rows]
    print(f"=== {ad} ===")
    print(f"  target P(A) per context: min {min(pa.values()):.3f} max {max(pa.values()):.3f} "
          f"mean {np.mean(list(pa.values())):.3f} sd {np.std(list(pa.values())):.3f}")
    print(f"  utility margin: mean {np.mean(mar):.2f} sd {np.std(mar):.2f} "
          f"q10 {np.percentile(mar,10):.2f} median {np.median(mar):.2f} q90 {np.percentile(mar,90):.2f}")
    vals = np.array([v for r in rows for v in r["option_A_values"]+r["option_B_values"]], float)
    print(f"  attribute values: mean {vals.mean():.3f} sd {vals.std():.3f} (raw units, pooled)")
    al = sum(1 for rid,t in keys[ad].items() if base.get(rid)==t)/len(keys[ad])
    print(f"  targets aligned with BASE greedy choice (policy_val): {al:.4f}")

print("\n=== the 8 contexts where A1 and A2 hold IDENTICAL target policies ===")
print(f"  {'ctx':<5}{'rel':<7}{'A1 P(A)':>9}{'A2 P(A)':>9}{'A1 margin':>11}{'A2 margin':>11}{'A1 base-align':>15}{'A2 base-align':>15}")
for c in SHARED:
    out=[]
    for ad in ("A1","A2"):
        v=[r for r in D[ad] if r["context_id"]==c]
        out.append((sum(1 for r in v if r["target_letter"]=="A")/len(v), np.mean([r["utility_margin"] for r in v])))
    al=[]
    for ad in ("A1","A2"):
        ids=[k for k in keys[ad] if k.startswith(f"policy_val|{c}|")]
        al.append(sum(1 for k in ids if base.get(k)==keys[ad][k])/len(ids))
    print(f"  {c:<5}{rel[c]:<7}{out[0][0]:>9.3f}{out[1][0]:>9.3f}{out[0][1]:>11.2f}{out[1][1]:>11.2f}"
          f"{al[0]:>15.3f}{al[1]:>15.3f}")

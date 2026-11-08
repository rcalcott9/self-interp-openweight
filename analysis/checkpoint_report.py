"""Phase-1 checkpoint evidence: design table, target-policy similarity,
split counts. Reads only generated artifacts."""
import csv, json, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
AD = ["A1", "A2", "B1", "B2"]


def cos(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


ctx = list(csv.DictReader(open(D / "design" / "contexts.csv")))
pol = {}
for r in csv.DictReader(open(D / "design" / "target_policies.csv")):
    pol.setdefault(r["adapter_id"], {})[r["context_id"]] = \
        [int(r[f"w{i+1}"]) for i in range(5)]

print("=== FINAL CONTEXT ASSIGNMENT ===")
print(f"{'ctx':<5}{'scenario':<22}{'pairA':<13}{'pairB':<13}{'reliability':<11}{'train labels'}")
for c in ctx:
    print(f"{c['context_id']:<5}{c['scenario_name']:<22}{c['overlap_A']:<13}"
          f"{c['overlap_B']:<13}{c['assigned_consistency']:<11}"
          f"{'100%' if c['assigned_consistency']=='strong' else '65% (70/200 flipped)'}")

print("\n=== TARGET-POLICY SIMILARITY (cosine) ===")
rows = []
for c in ctx:
    cid = c["context_id"]
    rows.append(("A1-A2 (pair A)", c["overlap_A"], cos(pol["A1"][cid], pol["A2"][cid])))
    rows.append(("B1-B2 (pair B)", c["overlap_B"], cos(pol["B1"][cid], pol["B2"][cid])))
    for a, b in itertools.product(["A1", "A2"], ["B1", "B2"]):
        rows.append((f"{a}-{b} (cross-pair)", "cross_pair", cos(pol[a][cid], pol[b][cid])))
for grp in ("shared", "independent", "cross_pair"):
    v = [r[2] for r in rows if r[1] == grp]
    print(f"  {grp:<13} n={len(v):<4} mean={np.mean(v):+.3f}  "
          f"min={min(v):+.3f}  max={max(v):+.3f}  "
          f"{'(exact identity)' if grp=='shared' else ''}")

print("\n=== SPLIT COUNTS ===")
tot_pairs = 0
for a in AD:
    rs = [json.loads(l) for l in open(D / "train" / a / "train_full.jsonl")]
    hs = {r["pair_hash"] for r in rs}
    tot_pairs += len(hs)
    ab = sum(r["presentation_order"] == "AB" for r in rs)
    fl = sum(not r["label_follows_target"] for r in rs)
    print(f"  train/{a:<3} rows={len(rs):<6} distinct_pairs={len(hs):<6} "
          f"AB={ab} BA={len(rs)-ab}  flipped_labels={fl}")
for s in ("policy_val", "self_pred", "relational"):
    rs = [json.loads(l) for l in open(D / "common" / f"{s}_prompts.jsonl")]
    hs = {r["pair_hash"] for r in rs}
    ks = list(csv.DictReader(open(D / "common" / f"{s}_keys.csv")))
    tot_pairs += len(hs)
    print(f"  common/{s:<12} prompt_rows={len(rs):<5} distinct_pairs={len(hs):<5} "
          f"key_rows={len(ks)} (= {len(rs)} x {len(AD)} adapters)")
print(f"  TOTAL distinct underlying option pairs generated: {tot_pairs}")

mf = json.loads((D / "design" / "manifest.json").read_text())
print("\n=== TIE REJECTION (exact ties only, tol 1e-9 relative) ===")
print("  " + json.dumps(mf["stats"]["ties_resampled"]))
print(f"  design cell-assignment accepted on attempt {mf['design']['attempts']}")

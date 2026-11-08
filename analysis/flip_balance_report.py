"""Weak-context corruption balance. Reports incidental imbalance; forces none
beyond the AB/BA stratification."""
import csv, json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1] / "data"
AD = ["A1", "A2", "B1", "B2"]
ctx = {r["context_id"]: r["assigned_consistency"]
       for r in csv.DictReader(open(D / "design" / "contexts.csv"))}
tr = {a: [json.loads(l) for l in open(D / "train" / a / "train_full.jsonl")] for a in AD}

print("=== FLIPPED-LABEL BALANCE, weak cells (32 = 4 adapters x 8 weak contexts) ===")
print(f"{'adapter':<8}{'ctx':<6}{'AB':<5}{'BA':<5}{'flip|tgtA':<11}{'flip|tgtB':<11}"
      f"{'rate|tgtA':<11}{'rate|tgtB':<11}")
rows = []
for a in AD:
    by = {}
    for r in tr[a]:
        by.setdefault(r["context_id"], []).append(r)
    for cid in sorted(by):
        if ctx[cid] != "weak":
            continue
        rs = by[cid]
        ab = sum(not r["label_follows_target"] for r in rs if r["presentation_order"] == "AB")
        ba = sum(not r["label_follows_target"] for r in rs if r["presentation_order"] == "BA")
        tA = [r for r in rs if r["target_letter"] == "A"]
        tB = [r for r in rs if r["target_letter"] == "B"]
        fA = sum(not r["label_follows_target"] for r in tA)
        fB = sum(not r["label_follows_target"] for r in tB)
        rows.append((a, cid, ab, ba, fA, fB, len(tA), len(tB)))
        print(f"{a:<8}{cid:<6}{ab:<5}{ba:<5}{fA:>3}/{len(tA):<7}{fB:>3}/{len(tB):<7}"
              f"{fA/max(len(tA),1):<11.3f}{fB/max(len(tB),1):<11.3f}")

fA = sum(r[4] for r in rows); nA = sum(r[6] for r in rows)
fB = sum(r[5] for r in rows); nB = sum(r[7] for r in rows)
print(f"\npooled flip rate | target=A : {fA}/{nA} = {fA/nA:.4f}")
print(f"pooled flip rate | target=B : {fB}/{nB} = {fB/nB:.4f}")
print(f"pooled overall              : {(fA+fB)}/{(nA+nB)} = {(fA+fB)/(nA+nB):.4f}  (design 0.3500)")
print(f"incidental target-letter imbalance in flip rate: {abs(fA/nA - fB/nB):+.4f}")
mx = max(abs(r[4]/max(r[6],1) - r[5]/max(r[7],1)) for r in rows)
print(f"largest per-cell |rate(A) - rate(B)| across 32 cells: {mx:.3f}")

print("\n=== UTILITY-MARGIN DISTRIBUTION, flipped vs unflipped (weak cells pooled) ===")
fl = [r["utility_margin"] for a in AD for r in tr[a]
      if ctx[r["context_id"]] == "weak" and not r["label_follows_target"]]
un = [r["utility_margin"] for a in AD for r in tr[a]
      if ctx[r["context_id"]] == "weak" and r["label_follows_target"]]
for nm, v in (("flipped", fl), ("unflipped", un)):
    q = np.percentile(v, [0, 25, 50, 75, 100])
    print(f"  {nm:<10} n={len(v):<5} mean={np.mean(v):7.2f} sd={np.std(v):6.2f}  "
          f"min={q[0]:6.2f} q25={q[1]:6.2f} med={q[2]:6.2f} q75={q[3]:6.2f} max={q[4]:6.2f}")
d = np.mean(fl) - np.mean(un)
rng = np.random.default_rng(7)
pool = np.array(fl + un); n1 = len(fl)
null = np.array([(lambda p: p[:n1].mean() - p[n1:].mean())(rng.permutation(pool))
                 for _ in range(10000)])
p = float((np.abs(null) >= abs(d)).mean())
print(f"  mean difference (flipped - unflipped) = {d:+.3f}; "
      f"permutation p = {p:.3f} (10,000 perms, H0: flip independent of margin)")

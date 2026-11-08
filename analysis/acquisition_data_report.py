"""Checkpoint 2 validation of the v3 acquisition sets. Read-only; no policy_test content."""
import csv, json, itertools, collections, hashlib, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DATA, ACQ = ROOT/"data", ROOT/"data/acquisition"
FAIL = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - '+detail) if detail else ''}")
    if not ok: FAIL.append(name)

rows = {a: [json.loads(l) for l in open(ACQ/f"train/{a}/train_full.jsonl")] for a in ("A1","A2")}

print("=== 1. COUNTS ===")
for a, rs in rows.items():
    lab = collections.Counter(r["label_letter"] for r in rs)
    od  = collections.Counter(r["presentation_order"] for r in rs)
    ctx = collections.Counter(r["context_id"] for r in rs)
    det = all(r["label_letter"] == r["target_letter"] for r in rs)
    print(f"  {a}: n={len(rs)}  contexts={len(ctx)} x {min(ctx.values())}-{max(ctx.values())}"
          f"  label A/B {lab['A']}/{lab['B']}  order AB/BA {od['AB']}/{od['BA']}")
    check(f"{a} deterministic labels (label==target for every row)", det)
    check(f"{a} n == 5120", len(rs)==5120, f"got {len(rs)}")
    check(f"{a} label balance |A-B| <= 16", abs(lab['A']-lab['B'])<=16, f"|{lab['A']}-{lab['B']}|={abs(lab['A']-lab['B'])}")
    check(f"{a} order balance |AB-BA| <= 16", abs(od['AB']-od['BA'])<=16, f"|{od['AB']}-{od['BA']}|={abs(od['AB']-od['BA'])}")

print("\n=== 2. PER-CONTEXT BALANCE (max deviation from n/2) ===")
for a, rs in rows.items():
    worst_l = worst_o = 0
    for cid in sorted({r["context_id"] for r in rs}):
        sub=[r for r in rs if r["context_id"]==cid]
        nA=sum(r["label_letter"]=="A" for r in sub); nAB=sum(r["presentation_order"]=="AB" for r in sub)
        worst_l=max(worst_l,abs(nA-len(sub)//2)); worst_o=max(worst_o,abs(nAB-len(sub)//2))
    print(f"  {a}: worst |#A - 160| = {worst_l}   worst |#AB - 160| = {worst_o}")
    check(f"{a} per-context label deviation <= 1", worst_l<=1)
    check(f"{a} per-context order deviation <= 1", worst_o<=1)

print("\n=== 3. NO SELECTION ON UTILITY MARGIN ===")
for a, rs in rows.items():
    m = np.array([r["utility_margin"] for r in rs])
    gA = m[[r["label_letter"]=="A" for r in rs]]; gB = m[[r["label_letter"]=="B" for r in rs]]
    gAB= m[[r["presentation_order"]=="AB" for r in rs]]; gBA= m[[r["presentation_order"]=="BA" for r in rs]]
    def ks(x,y):
        xs,ys=np.sort(x),np.sort(y); allv=np.concatenate([xs,ys])
        cx=np.searchsorted(xs,allv,'right')/len(xs); cy=np.searchsorted(ys,allv,'right')/len(ys)
        d=np.max(np.abs(cx-cy)); en=np.sqrt(len(x)*len(y)/(len(x)+len(y)))
        lam=(en+0.12+0.11/en)*d
        p=2*sum((-1)**(k-1)*np.exp(-2*k*k*lam*lam) for k in range(1,101))
        return d,min(max(p,0.0),1.0)
    dl,pl=ks(gA,gB); do,po=ks(gAB,gBA)
    print(f"  {a}: margin by LABEL  mean {gA.mean():.2f} vs {gB.mean():.2f}  KS D={dl:.4f} p={pl:.3f}")
    print(f"  {a}: margin by ORDER  mean {gAB.mean():.2f} vs {gBA.mean():.2f}  KS D={do:.4f} p={po:.3f}")
    check(f"{a} margin not selected by label (KS p>0.01)", pl>0.01, f"p={pl:.3f}")
    check(f"{a} margin not selected by order (KS p>0.01)", po>0.01, f"p={po:.3f}")

print("\n=== 4. IDENTIFIABILITY: can an adapter be told from labels/positions alone? ===")
lA=collections.Counter(r["label_letter"] for r in rows["A1"]); lB=collections.Counter(r["label_letter"] for r in rows["A2"])
oA=collections.Counter(r["presentation_order"] for r in rows["A1"]); oB=collections.Counter(r["presentation_order"] for r in rows["A2"])
pa=lA['A']/sum(lA.values()); pb=lB['A']/sum(lB.values())
print(f"  P(label=A): A1 {pa:.4f}  A2 {pb:.4f}   diff {abs(pa-pb):.4f}")
print(f"  P(order=AB): A1 {oA['AB']/5120:.4f}  A2 {oB['AB']/5120:.4f}")
check("adapters indistinguishable by label frequency (diff < 0.01)", abs(pa-pb)<0.01)
check("adapters indistinguishable by position frequency (diff < 0.01)", abs(oA['AB']-oB['AB'])/5120<0.01)

print("\n=== 5. DISJOINTNESS / LEAKAGE ===")
sets={f"acq_{a}":{r["pair_hash"] for r in rs} for a,rs in rows.items()}
for a in ("A1","A2","B1","B2"): sets[f"frozen_train_{a}"]={json.loads(l)["pair_hash"] for l in open(DATA/f"train/{a}/train_full.jsonl")}
for s in ("policy_val","self_pred","relational","policy_test"):
    sets[s]={json.loads(l)["pair_hash"] for l in open(DATA/f"common/{s}_prompts.jsonl")}
bad=[(x,y,len(sets[x]&sets[y])) for x,y in itertools.combinations(sorted(sets),2) if sets[x]&sets[y]]
for x,y,k in bad: print(f"  OVERLAP {x} & {y}: {k}")
check("acquisition pairs disjoint from every existing split", not bad, f"{len(bad)} overlapping pairs of splits")
for a,rs in rows.items():
    check(f"{a} no duplicate pair_hash within adapter", len({r['pair_hash'] for r in rs})==len(rs))

print("\n=== 6. A1 vs A2 POLICY AGREEMENT (held-out policy_val; targets only) ===")
key=collections.defaultdict(dict)
for r in csv.DictReader(open(DATA/"common/policy_val_keys.csv")):
    if r["adapter_id"] in ("A1","A2"):
        key[(r["context_id"],r["instance_id"],r["presentation_order"])][r["adapter_id"]]=r["target_letter"]
ovl={r["context_id"]:r["overlap_status"] for r in csv.DictReader(open(DATA/"design/target_policies.csv")) if r["adapter_id"]=="A1"}
agree=collections.defaultdict(lambda:[0,0])
tot=[0,0]
for (cid,_,_),v in key.items():
    s=v["A1"]==v["A2"]; agree[ovl[cid]][0]+=s; agree[ovl[cid]][1]+=1; tot[0]+=s; tot[1]+=1
print(f"  overall agreement {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.4f}   DISAGREEMENT = {1-tot[0]/tot[1]:.4f} ({tot[1]-tot[0]} discriminating rows)")
for k,(s,nn) in sorted(agree.items()): print(f"    overlap={k:12s} agreement {s}/{nn} = {s/nn:.4f}")
check("shared-overlap contexts agree exactly", agree["shared"][0]==agree["shared"][1])
check("low-overlap contexts give >=300 discriminating rows",
      agree["independent"][1]-agree["independent"][0]>=300,
      f"{agree['independent'][1]-agree['independent'][0]}")

print("\n=== 7. DATASET HASHES ===")
for a in ("A1","A2"):
    for f in ("train_full.jsonl","train_sft.jsonl"):
        p=ACQ/f"train/{a}/{f}"
        print(f"  {p.relative_to(ROOT)}  {hashlib.sha256(p.read_bytes()).hexdigest()[:32]}...  {p.stat().st_size:,} B")

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)

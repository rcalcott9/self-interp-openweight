"""Generate the full Phase-1 data hierarchy.

Order of generation matters and is fixed:
  1. common held-out sets (policy_val, self_pred, relational)  -- shared by ALL models
  2. per-adapter training sets                                 -- independent per adapter
Every draw excludes all previously used canonical pair hashes within its context,
so global disjointness is guaranteed by construction, not merely checked after.
"""
import csv, json
from pathlib import Path

import yaml

from .seeds import stream
from .scenarios import load_pool, select_contexts
from .design import build_design
from .policies import build_policies, utility
from .trials import (draw_pairs, target_letter, positioned, balanced_orders,
                     flip, pair_hash)
from .render import render_messages
from .provenance import sha256_file, env_record, write_manifest

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs"
DATA = ROOT / "data"


def load_cfgs():
    return {n: yaml.safe_load((CFG / f"{n}.yaml").read_text())
            for n in ("design", "data", "prompt")}


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def build(pool_path):
    cfg = load_cfgs()
    d, dd, pc = cfg["design"], cfg["data"], cfg["prompt"]
    seed, adapters = d["master_seed"], d["adapters"]

    pool_raw, usable = load_pool(pool_path)
    contexts = select_contexts(usable, seed, d["n_contexts"])
    contexts, design_meta = build_design(seed, contexts,
                                         d["cells_per_condition"],
                                         d["max_consecutive_run"])
    policies, policy_accept = build_policies(
        seed, contexts, adapters, d["n_attributes"],
        max_abs_cos=float(d["max_abs_cos_nonshared"]),
        max_attempts=int(d["policy_max_redraw_attempts"]))

    tol = float(dd["tie_rel_tolerance"])
    stats = {"ties_resampled": {}, "common_pairs": {}, "train_pairs": {}}
    used = {c["context_id"]: set() for c in contexts}
    common_pairs = {}                      # (split, cid) -> [pair,...]

    # ---- 1. common held-out sets -------------------------------------------
    for split in ("policy_val", "self_pred", "relational"):
        n = dd["splits"][split]["n_pairs_per_context"]
        tot_ties = 0
        for c in contexts:
            cid = c["context_id"]
            all_w = [policies[a][cid] for a in adapters]   # tie-free for EVERY model
            rng = stream(seed, f"examples|COMMON|{cid}|{split}")
            pairs, ties = draw_pairs(rng, c["attributes"], cid, n, all_w, tol,
                                     dd["max_resample_attempts"], used[cid])
            common_pairs[(split, cid)] = pairs
            tot_ties += ties
        stats["ties_resampled"][split] = tot_ties
        stats["common_pairs"][split] = n * len(contexts)

    # ---- 2. per-adapter training sets --------------------------------------
    n_tr = dd["splits"]["train"]["n_pairs_per_context"]
    n_flip = int(round(dd["weak_context_flip_fraction"] * n_tr))
    train_rows = {a: [] for a in adapters}
    for a in adapters:
        tot_ties = 0
        for c in contexts:
            cid, attrs = c["context_id"], c["attributes"]
            w = policies[a][cid]
            rng = stream(seed, f"examples|{a}|{cid}|train")
            pairs, ties = draw_pairs(rng, attrs, cid, n_tr, [w], tol,
                                     dd["max_resample_attempts"], used[cid])
            tot_ties += ties
            orng = stream(seed, f"orders|{a}|{cid}|train")
            orders = balanced_orders(orng, n_tr)
            weak = c["assigned_consistency"] == "weak"
            flip_idx = set()
            if weak:
                # Stratify the corruption by presentation order so position and
                # label noise cannot become confounded. Selection is otherwise
                # blind to target letter and utility margin.
                for od in ("AB", "BA"):
                    pos = [i for i, o in enumerate(orders) if o == od]
                    frng = stream(seed, f"flip|{a}|{cid}|{od}")
                    k = n_flip // 2
                    flip_idx |= {pos[j] for j in
                                 frng.choice(len(pos), size=k, replace=False).tolist()}
            for i, (p, order) in enumerate(zip(pairs, orders)):
                tl, uA, uB = target_letter(p, attrs, w, order)
                vA, vB, iA, iB = positioned(p, order)
                lab = flip(tl) if i in flip_idx else tl
                train_rows[a].append({
                    "row_id": f"train|{a}|{cid}|{i:04d}|{order}",
                    "adapter_id": a, "context_id": cid,
                    "scenario_name": c["scenario_name"], "split": "train",
                    "instance_id": i, "pair_hash": p["pair_hash"],
                    "presentation_order": order,
                    "option_A_values": vA, "option_B_values": vB,
                    "underlying_at_A": iA, "underlying_at_B": iB,
                    "target_letter": tl, "label_letter": lab,
                    "label_follows_target": lab == tl,
                    "utility_A": uA, "utility_B": uB,
                    "utility_margin": abs(uA - uB),
                    "overlap_A": c["overlap_A"], "overlap_B": c["overlap_B"],
                    "assigned_consistency": c["assigned_consistency"],
                    "target_w": w,
                    "messages": render_messages(cid, c["question"], attrs,
                                                vA, vB, pc, label_letter=lab),
                })
        stats["ties_resampled"][f"train_{a}"] = tot_ties
        stats["train_pairs"][a] = n_tr * len(contexts)

    # ---- 3. write ----------------------------------------------------------
    written = []
    for a in adapters:
        full = DATA / "train" / a / "train_full.jsonl"
        sft = DATA / "train" / a / "train_sft.jsonl"
        write_jsonl(full, train_rows[a])
        write_jsonl(sft, [{"messages": r["messages"]} for r in train_rows[a]])
        written += [full, sft]

    for split in ("policy_val", "self_pred", "relational"):
        prompts, keys = [], []
        for c in contexts:
            cid, attrs = c["context_id"], c["attributes"]
            for i, p in enumerate(common_pairs[(split, cid)]):
                for order in ("AB", "BA"):
                    vA, vB, iA, iB = positioned(p, order)
                    rid = f"{split}|{cid}|{i:04d}|{order}"
                    prompts.append({
                        "row_id": rid, "context_id": cid,
                        "scenario_name": c["scenario_name"], "split": split,
                        "instance_id": i, "pair_hash": p["pair_hash"],
                        "presentation_order": order,
                        "option_A_values": vA, "option_B_values": vB,
                        "underlying_at_A": iA, "underlying_at_B": iB,
                        "messages": render_messages(cid, c["question"], attrs,
                                                    vA, vB, pc),   # NO label
                    })
                    for a in adapters:
                        w = policies[a][cid]
                        tl, uA, uB = target_letter(p, attrs, w, order)
                        keys.append({"row_id": rid, "adapter_id": a,
                                     "context_id": cid, "instance_id": i,
                                     "presentation_order": order,
                                     "target_letter": tl,
                                     "utility_A": uA, "utility_B": uB,
                                     "utility_margin": abs(uA - uB)})
        pf = DATA / "common" / f"{split}_prompts.jsonl"
        kf = DATA / "common" / f"{split}_keys.csv"
        write_jsonl(pf, prompts)
        with open(kf, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(keys[0].keys()))
            wtr.writeheader(); wtr.writerows(keys)
        written += [pf, kf]

    cf = DATA / "design" / "contexts.csv"
    with open(cf, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["context_id", "scenario_name", "question", "overlap_A",
                      "overlap_B", "assigned_consistency"]
                     + [f"attr{i+1}" for i in range(5)])
        for c in contexts:
            wtr.writerow([c["context_id"], c["scenario_name"], c["question"],
                          c["overlap_A"], c["overlap_B"], c["assigned_consistency"]]
                         + [a["name"] for a in c["attributes"]])
    pf = DATA / "design" / "target_policies.csv"
    with open(pf, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["adapter_id", "context_id", "overlap_status",
                      "assigned_consistency"] + [f"w{i+1}" for i in range(5)])
        for a in adapters:
            for c in contexts:
                ov = c["overlap_A"] if a.startswith("A") else c["overlap_B"]
                wtr.writerow([a, c["context_id"], ov, c["assigned_consistency"]]
                             + policies[a][c["context_id"]])
    written += [cf, pf]

    v1 = DATA / "design" / "archive" / "manifest_v1_rejected.json"
    supersedes = None
    if v1.exists():
        prev = json.loads(v1.read_text())
        supersedes = {"manifest_version": prev.get("manifest_version"),
                      "status": prev.get("status"),
                      "reason": prev.get("rejection_reason"),
                      "archived_at": str(v1.relative_to(ROOT)),
                      "file_hashes": prev.get("files")}
    manifest = {
        "manifest_version": 2,
        "status": "ACTIVE",
        "supersedes": supersedes,
        "generated_by": env_record(),
        "master_seed": seed,
        "design": design_meta,
        "policy_constraint": {
            "rule": "|cos| <= max_abs_cos for every NONSHARED adapter pair within a context",
            "max_abs_cos_nonshared": float(d["max_abs_cos_nonshared"]),
            "selection_input": "target weights only; no model behaviour, no held-out performance",
            "condition_label": "separately trained, low-overlap policies (NOT unrestricted independent draws)",
            "acceptance": policy_accept,
        },
        "configs": {n: sha256_file(CFG / f"{n}.yaml")
                    for n in ("design", "data", "prompt", "eval", "model",
                              "lora", "train")},
        "scenario_pool": {"path": str(pool_path),
                          "sha256": sha256_file(pool_path),
                          "n_candidates": len(pool_raw), "n_usable": len(usable)},
        "stats": stats,
        "files": {str(p.relative_to(ROOT)): sha256_file(p) for p in written},
    }
    write_manifest(DATA / "design" / "manifest.json", manifest)
    return contexts, policies, common_pairs, train_rows, manifest


if __name__ == "__main__":
    import sys
    build(Path(sys.argv[1]))
    print("done")

"""Generate the sealed policy_test split ADDITIVELY.

Existing splits are NEVER regenerated: this draws new option pairs from a fresh
named stream, excluding every canonical pair hash already used by train/
policy_val/self_pred/relational, and writes only new files. All pre-existing
data files must remain byte-identical (asserted by the caller against the v2
manifest).

policy_test is COMMON across all adapters and models, like the other held-out
splits, and is sealed for the FINAL gate only.
"""
import csv, json
from pathlib import Path

import yaml

from .seeds import stream
from .scenarios import load_pool, select_contexts
from .design import build_design
from .policies import build_policies
from .trials import draw_pairs, target_letter, positioned
from .render import render_messages
from .provenance import sha256_file, env_record

ROOT = Path(__file__).resolve().parents[2]
CFG, DATA = ROOT / "configs", ROOT / "data"
N_PAIRS = 60          # matches policy_val, so the final gate has equal resolution


def existing_hashes():
    """Every canonical pair hash already committed to any split."""
    used = {}
    for p in sorted((DATA / "train").glob("*/train_full.jsonl")):
        for line in open(p):
            r = json.loads(line)
            used.setdefault(r["context_id"], set()).add(r["pair_hash"])
    for split in ("policy_val", "self_pred", "relational"):
        f = DATA / "common" / f"{split}_prompts.jsonl"
        for line in open(f):
            r = json.loads(line)
            used.setdefault(r["context_id"], set()).add(r["pair_hash"])
    return used


def build(pool_path):
    d = yaml.safe_load((CFG / "design.yaml").read_text())
    dd = yaml.safe_load((CFG / "data.yaml").read_text())
    pc = yaml.safe_load((CFG / "prompt.yaml").read_text())
    seed, adapters = d["master_seed"], d["adapters"]
    _, usable = load_pool(pool_path)
    contexts = select_contexts(usable, seed, d["n_contexts"])
    contexts, _ = build_design(seed, contexts, d["cells_per_condition"],
                               d["max_consecutive_run"])
    policies, _ = build_policies(seed, contexts, adapters, d["n_attributes"],
                                 max_abs_cos=float(d["max_abs_cos_nonshared"]),
                                 max_attempts=int(d["policy_max_redraw_attempts"]))
    used = existing_hashes()
    tol = float(dd["tie_rel_tolerance"])
    prompts, keys, ties = [], [], 0
    for c in contexts:
        cid, attrs = c["context_id"], c["attributes"]
        all_w = [policies[a][cid] for a in adapters]
        rng = stream(seed, f"examples|COMMON|{cid}|policy_test")
        pairs, t = draw_pairs(rng, attrs, cid, N_PAIRS, all_w, tol,
                              dd["max_resample_attempts"], used[cid])
        ties += t
        for i, p in enumerate(pairs):
            for order in ("AB", "BA"):
                vA, vB, iA, iB = positioned(p, order)
                rid = f"policy_test|{cid}|{i:04d}|{order}"
                prompts.append({"row_id": rid, "context_id": cid,
                                "scenario_name": c["scenario_name"], "split": "policy_test",
                                "instance_id": i, "pair_hash": p["pair_hash"],
                                "presentation_order": order,
                                "option_A_values": vA, "option_B_values": vB,
                                "underlying_at_A": iA, "underlying_at_B": iB,
                                "messages": render_messages(cid, c["question"], attrs,
                                                            vA, vB, pc)})
                for a in adapters:
                    tl, uA, uB = target_letter(p, attrs, policies[a][cid], order)
                    keys.append({"row_id": rid, "adapter_id": a, "context_id": cid,
                                 "instance_id": i, "presentation_order": order,
                                 "target_letter": tl, "utility_A": uA, "utility_B": uB,
                                 "utility_margin": abs(uA - uB)})
    pf = DATA / "common" / "policy_test_prompts.jsonl"
    kf = DATA / "common" / "policy_test_keys.csv"
    with open(pf, "w") as f:
        for r in prompts:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(kf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keys[0].keys()))
        w.writeheader(); w.writerows(keys)
    man = {"split": "policy_test", "status": "SEALED for the FINAL gate",
           "generated_by": env_record(), "master_seed": seed,
           "n_pairs_per_context": N_PAIRS, "n_prompt_rows": len(prompts),
           "n_key_rows": len(keys), "ties_resampled": ties,
           "files": {str(p.relative_to(ROOT)): sha256_file(p) for p in (pf, kf)}}
    (DATA / "design" / "policy_test_manifest.json").write_text(json.dumps(man, indent=2))
    return man


if __name__ == "__main__":
    import sys
    print(json.dumps(build(Path(sys.argv[1])), indent=2))

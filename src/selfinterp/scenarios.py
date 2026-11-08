"""Load the Plunkett candidate scenario pool and select the study contexts.

Reference pool is read-only. Roles/personas are never attached: we use a neutral
`Context CXX.` identifier instead, and scenario names are prefixed with the
context id so they cannot be confused with stimuli from other projects.
"""
import json
import math
from pathlib import Path

from .seeds import stream


def rounding_precision(attr) -> int:
    span = attr["range"][1] - attr["range"][0]
    if span < 1:
        return abs(math.floor(math.log10(span))) + 1
    return 1 if span < 5 else 0


def load_pool(path):
    pool = json.loads(Path(path).read_text())
    usable = [
        s for s in pool
        if len(s["attributes"]) == 5
        and len({a["name"] for a in s["attributes"]}) == 5
        and all(a["range"][1] > a["range"][0] for a in s["attributes"])
    ]
    return pool, usable


def select_contexts(usable, master_seed, n_contexts):
    rng = stream(master_seed, "design|scenario_selection")
    idx = rng.choice(len(usable), size=n_contexts, replace=False)
    out = []
    for slot, i in enumerate(idx, start=1):
        s = usable[i]
        cid = f"C{slot:02d}"
        out.append({
            "context_id": cid,
            "scenario_name": f"{cid}_{s['short_name']}",
            "source_short_name": s["short_name"],
            "question": s["question"],
            "attributes": [
                {"name": a["name"], "units": a["units"],
                 "range": [float(a["range"][0]), float(a["range"][1])],
                 "precision": rounding_precision(a)}
                for a in s["attributes"]
            ],
        })
    return out

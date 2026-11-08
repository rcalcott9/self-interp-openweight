"""Hard guard on the reserved splits.

self_pred and relational are sealed for Phases 2 and 3. Every read of generated
data goes through load_split(), which refuses them outright and records what was
actually opened so the run manifest can prove non-access.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESERVED = {"self_pred", "relational", "policy_test"}
ALLOWED = {"train", "policy_val", "acq_train"}


class ReservedSplitAccess(RuntimeError):
    pass


class Recorder:
    def __init__(self):
        self.files = []

    def note(self, p):
        self.files.append(str(Path(p).resolve().relative_to(ROOT)))
        return p

    def assert_no_reserved(self):
        bad = [f for f in self.files if any(r in f for r in RESERVED)]
        if bad:
            raise ReservedSplitAccess(f"reserved split touched: {bad}")
        return sorted(set(self.files))


REC = Recorder()


def load_train(adapter, contexts=None, per_context=None):
    p = REC.note(ROOT / "data" / "train" / adapter / "train_sft.jsonl")
    full = REC.note(ROOT / "data" / "train" / adapter / "train_full.jsonl")
    rows = [json.loads(l) for l in open(full)]
    if contexts:
        rows = [r for r in rows if r["context_id"] in contexts]
    if per_context:
        keep, seen = [], {}
        for r in rows:
            c = r["context_id"]
            if seen.get(c, 0) < per_context:
                keep.append(r)
                seen[c] = seen.get(c, 0) + 1
        rows = keep
    return rows


def load_acq_train(adapter, contexts=None, per_context=None):
    """v3 acquisition training set. Same guard and recording as load_train."""
    p = REC.note(ROOT / "data" / "acquisition" / "train" / adapter / "train_sft.jsonl")
    full = REC.note(ROOT / "data" / "acquisition" / "train" / adapter / "train_full.jsonl")
    rows = [json.loads(l) for l in open(full)]
    if contexts:
        rows = [r for r in rows if r["context_id"] in contexts]
    if per_context:
        keep, seen = [], {}
        for r in rows:
            c = r["context_id"]
            if seen.get(c, 0) < per_context:
                keep.append(r); seen[c] = seen.get(c, 0) + 1
        rows = keep
    return rows


def load_policy_test(*a, **k):
    raise ReservedSplitAccess(
        "'policy_test' is the sealed FINAL-gate split and must not be read during "
        "development or stability testing")


def load_policy_val(contexts=None, pairs_per_context=None):
    p = REC.note(ROOT / "data" / "common" / "policy_val_prompts.jsonl")
    rows = [json.loads(l) for l in open(p)]
    if contexts:
        rows = [r for r in rows if r["context_id"] in contexts]
    if pairs_per_context:
        rows = [r for r in rows if r["instance_id"] < pairs_per_context]
    return rows


def load_policy_val_keys(adapter):
    import csv
    p = REC.note(ROOT / "data" / "common" / "policy_val_keys.csv")
    return {r["row_id"]: r for r in csv.DictReader(open(p))
            if r["adapter_id"] == adapter}


def load_split(name, **kw):
    if name in RESERVED:
        raise ReservedSplitAccess(
            f"'{name}' is sealed for a later phase and must not be read here")
    if name not in ALLOWED:
        raise ValueError(f"unknown split {name!r}")
    return {"train": load_train, "acq_train": load_acq_train,
            "policy_val": load_policy_val}[name](**kw)

import sys, json, csv
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
ADAPTERS = ["A1", "A2", "B1", "B2"]
SPLITS = ["policy_val", "self_pred", "relational"]


def _jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


@pytest.fixture(scope="session")
def train():
    return {a: _jsonl(DATA / "train" / a / "train_full.jsonl") for a in ADAPTERS}


@pytest.fixture(scope="session")
def sft():
    return {a: _jsonl(DATA / "train" / a / "train_sft.jsonl") for a in ADAPTERS}


@pytest.fixture(scope="session")
def common():
    return {s: _jsonl(DATA / "common" / f"{s}_prompts.jsonl") for s in SPLITS}


@pytest.fixture(scope="session")
def keys():
    return {s: list(csv.DictReader(open(DATA / "common" / f"{s}_keys.csv")))
            for s in SPLITS}


@pytest.fixture(scope="session")
def policies():
    rows = list(csv.DictReader(open(DATA / "design" / "target_policies.csv")))
    out = {}
    for r in rows:
        out.setdefault(r["adapter_id"], {})[r["context_id"]] = \
            [int(r[f"w{i+1}"]) for i in range(5)]
    return out

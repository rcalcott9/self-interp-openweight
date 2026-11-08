"""Run directories, config overlay, provenance. Runs are never overwritten."""
import copy, json, shutil, subprocess, sys, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs"
CFG_NAMES = ["design", "data", "prompt", "eval", "model", "lora", "train"]


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_configs(overlay_path=None):
    cfg = {n: yaml.safe_load((CFG / f"{n}.yaml").read_text()) for n in CFG_NAMES}
    overlay = {}
    if overlay_path:
        overlay = yaml.safe_load(Path(overlay_path).read_text())
        for n in CFG_NAMES:
            if n in overlay:
                cfg[n] = deep_merge(cfg[n], overlay[n])
        for k, v in overlay.items():
            if k not in CFG_NAMES:
                cfg[k] = v
    return cfg, overlay


def new_run_dir(tag):
    rid = f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    d = ROOT / "runs" / rid
    if d.exists():
        raise FileExistsError(f"run {rid} exists; runs are never overwritten")
    for sub in ("config_snapshot", "adapter", "raw_outputs", "presave",
                "metrics", "logs"):
        (d / sub).mkdir(parents=True)
    for n in CFG_NAMES:
        shutil.copy(CFG / f"{n}.yaml", d / "config_snapshot" / f"{n}.yaml")
    return d


def snapshot_overlay(run_dir, overlay_path):
    if overlay_path:
        shutil.copy(overlay_path, run_dir / "config_snapshot" / "OVERLAY.yaml")


def pkg_versions():
    import importlib
    out = {}
    for m in ("torch", "transformers", "trl", "peft", "accelerate", "datasets",
              "numpy", "tokenizers"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception as e:
            out[m] = f"ERR {e}"
    return out


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")

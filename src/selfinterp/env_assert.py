"""Abort if the runtime environment drifts from the pinned stack.

Motivated by a real incident (2026-09-03): after a pod restart, a surviving
.venv silently resolved `import torch` to the system install (2.8.0+cu128)
instead of the pinned 2.5.1+cu124. Nothing flagged it. Any numbers produced in
that state would have been incomparable to A1's, and uninterpretable.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _pins(path=None):
    p = Path(path) if path else ROOT / "requirements-model.txt"
    out = {}
    for line in p.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            k, v = line.split("==", 1)
            out[k.strip().lower().replace("-", "_")] = v.strip()
    return out


def assert_pinned_env(require_cuda=True, verbose=True):
    import importlib
    pins = _pins()
    checked, bad = {}, []
    for mod in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "numpy"):
        want = pins.get(mod)
        if want is None:
            continue
        m = importlib.import_module(mod)
        got = m.__version__
        # torch reports e.g. 2.5.1+cu124; compare the base version
        base = got.split("+")[0]
        checked[mod] = {"want": want, "got": got, "file": getattr(m, "__file__", "?")}
        if base != want:
            bad.append(f"{mod}: pinned {want}, running {got}")
    # the venv must not be resolving into a system dist-packages tree
    import torch
    if "/site-packages/" not in torch.__file__:
        bad.append(f"torch resolved outside a venv site-packages: {torch.__file__}")
    if require_cuda:
        if not torch.cuda.is_available():
            bad.append("CUDA not available")
        elif not torch.cuda.is_bf16_supported():
            bad.append("bf16 not supported on this device")
    if bad:
        raise RuntimeError("ENVIRONMENT DRIFT - refusing to run:\n  " + "\n  ".join(bad))
    if verbose:
        print("[env] pinned stack verified: " +
              ", ".join(f"{k} {v['got']}" for k, v in checked.items()))
        if require_cuda:
            print(f"[env] device: {torch.cuda.get_device_name(0)}, bf16 OK")
    return checked

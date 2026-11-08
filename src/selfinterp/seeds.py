"""Named, reproducible RNG streams.

Streams are addressed by LABEL, not by position, so adding a new stream later
cannot shift any existing draw. Two labels that differ in any character give
independent streams.
"""
import hashlib
import numpy as np


def _key(label: str) -> tuple:
    h = hashlib.sha256(label.encode("utf-8")).digest()
    # 4 x 32-bit words -> spawn_key tuple
    return tuple(int.from_bytes(h[i:i + 4], "big") for i in range(0, 16, 4))


def stream(master_seed: int, label: str) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=_key(label))
    return np.random.default_rng(ss)

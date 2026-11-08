import hashlib, json, platform, subprocess, sys
from pathlib import Path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_record():
    import numpy
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
    }


def write_manifest(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

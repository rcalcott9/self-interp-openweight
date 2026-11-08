"""Target weight vectors.

Paired adapters SHARE a weight vector on their shared contexts (byte-identical)
and are drawn separately elsewhere.

CONSTRAINT (v2): every NONSHARED adapter pair within a context must satisfy
|cos| <= max_abs_cos. Enforced by joint per-context rejection sampling, using
the same named-seed system: attempt 0 uses the base labels (so contexts that
already satisfied the constraint keep their v1 policies unchanged), attempt t>0
appends "|retry{t}". Rejection uses ONLY the target weights -- never any model
behaviour, and never held-out performance.

The resulting nonshared cells are therefore "separately trained, LOW-OVERLAP
policies", not unrestricted independent draws. Rejection truncates the joint
distribution; acceptance statistics are reported in the manifest.
"""
import itertools
import numpy as np
from .seeds import stream

WITHIN_PAIR = {"A": ("A1", "A2"), "B": ("B1", "B2")}


def draw_weights(rng, n_attributes=5, lo=-100.0, hi=100.0, norm=100):
    raw = rng.uniform(lo, hi, size=n_attributes)
    j = int(np.argmax(np.abs(raw)))
    scaled = raw * (norm * np.sign(raw[j])) / raw[j]
    return [int(round(v)) for v in scaled]


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def nonshared_pairs(context, adapters):
    shared = set()
    for pair, members in WITHIN_PAIR.items():
        if context[f"overlap_{pair}"] == "shared":
            shared.add(tuple(sorted(members)))
    return [p for p in itertools.combinations(sorted(adapters), 2)
            if tuple(sorted(p)) not in shared]


def _draw_context(master_seed, context, suffix, n_attributes):
    cid = context["context_id"]
    w = {}
    for pair, members in WITHIN_PAIR.items():
        if context[f"overlap_{pair}"] == "shared":
            v = draw_weights(stream(master_seed,
                                    f"policy|shared_{pair}|{cid}{suffix}"),
                             n_attributes)
            for m in members:
                w[m] = list(v)
        else:
            for m in members:
                w[m] = draw_weights(stream(master_seed,
                                           f"policy|{m}|{cid}{suffix}"),
                                    n_attributes)
    return w


def build_policies(master_seed, contexts, adapters, n_attributes=5,
                   max_abs_cos=0.50, max_attempts=2000):
    pol = {a: {} for a in adapters}
    accept = {}
    for c in contexts:
        cid = c["context_id"]
        ns = nonshared_pairs(c, adapters)
        for t in range(max_attempts):
            suffix = "" if t == 0 else f"|retry{t}"
            w = _draw_context(master_seed, c, suffix, n_attributes)
            worst = max(abs(cosine(w[a], w[b])) for a, b in ns)
            if worst <= max_abs_cos:
                accept[cid] = {"attempts": t + 1, "max_abs_cos_nonshared": worst,
                               "suffix": suffix}
                for a in adapters:
                    pol[a][cid] = w[a]
                break
        else:
            raise RuntimeError(f"policy constraint unsatisfiable in {cid}")
    return pol, accept


def utility(values, attributes, w):
    u = 0.0
    for i, a in enumerate(attributes):
        lo, hi = a["range"]
        u += w[i] * (values[i] - lo) / (hi - lo)
    return u

"""Randomised, balanced, non-blocked assignment of contexts to
(overlap x reliability) cells. Frozen by master_seed."""
from .seeds import stream

CELLS = [
    ("shared", "strong"),   # shared for pair A  -> independent for pair B
    ("shared", "weak"),
    ("independent", "strong"),
    ("independent", "weak"),
]


def _max_run(vals):
    vals = sorted(vals)
    best = run = 1
    for a, b in zip(vals, vals[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


def assign_cells(master_seed, n_contexts, per_cell, max_run):
    rng = stream(master_seed, "design|cell_assignment")
    attempts = 0
    while True:
        attempts += 1
        perm = rng.permutation(n_contexts)
        cells = [sorted((perm[i * per_cell:(i + 1) * per_cell] + 1).tolist())
                 for i in range(len(CELLS))]
        if max(_max_run(c) for c in cells) <= max_run:
            return cells, attempts
        if attempts > 10000:
            raise RuntimeError("no admissible cell assignment found")


def build_design(master_seed, contexts, per_cell=4, max_run=2):
    cells, attempts = assign_cells(master_seed, len(contexts), per_cell, max_run)
    table = {}
    for ci, (a_overlap, reliability) in enumerate(CELLS):
        for num in cells[ci]:
            cid = f"C{num:02d}"
            table[cid] = {
                "overlap_A": a_overlap,
                "overlap_B": "independent" if a_overlap == "shared" else "shared",
                "assigned_consistency": reliability,
            }
    for c in contexts:
        c.update(table[c["context_id"]])
    return contexts, {"cells": cells, "attempts": attempts}

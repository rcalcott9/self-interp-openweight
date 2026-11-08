"""Option-pair generation, utilities, exact-tie rejection, labels, ordering.

Canonical option identity is option_0 / option_1, fixed at draw time.
Presentation order AB puts option_0 at position A; BA puts option_1 at position A.
"""
import hashlib
import numpy as np

from .policies import utility


def draw_option(rng, attributes):
    vals = []
    for a in attributes:
        lo, hi = a["range"]
        vals.append(round(float(rng.uniform(lo, hi)), a["precision"]))
    return vals


def pair_hash(context_id, opt0, opt1):
    """Canonical (order-free) identity of an underlying option pair."""
    a, b = sorted([tuple(opt0), tuple(opt1)])
    return hashlib.sha256(f"{context_id}|{a}|{b}".encode()).hexdigest()[:24]


def is_exact_tie(uA, uB, tol):
    return abs(uA - uB) <= tol * max(1.0, abs(uA), abs(uB))


def draw_pairs(rng, attributes, context_id, n, policies_for_tie_check, tol,
               max_attempts=100, exclude=None):
    """Draw n distinct pairs. Rejects a pair iff it is an EXACT utility tie under
    ANY policy in `policies_for_tie_check` (so every model has a defined target),
    or if its canonical hash is already used. Returns (pairs, n_tie_resampled)."""
    exclude = set() if exclude is None else set(exclude)
    out, ties = [], 0
    while len(out) < n:
        for _ in range(max_attempts):
            o0, o1 = draw_option(rng, attributes), draw_option(rng, attributes)
            h = pair_hash(context_id, o0, o1)
            if h in exclude:
                continue
            tie = False
            for w in policies_for_tie_check:
                if is_exact_tie(utility(o0, attributes, w),
                                utility(o1, attributes, w), tol):
                    tie = True
                    break
            if tie:
                ties += 1
                continue
            exclude.add(h)
            out.append({"option_0": o0, "option_1": o1, "pair_hash": h})
            break
        else:
            raise RuntimeError(f"tie/collision rejection exhausted in {context_id}")
    return out, ties


def target_letter(pair, attributes, w, order):
    """Letter naming the option the TARGET policy prefers, at this presentation."""
    u0 = utility(pair["option_0"], attributes, w)
    u1 = utility(pair["option_1"], attributes, w)
    winner = 0 if u0 > u1 else 1
    if order == "AB":
        return "A" if winner == 0 else "B", u0, u1
    return "A" if winner == 1 else "B", u1, u0


def positioned(pair, order):
    """(values_at_A, values_at_B, underlying_index_at_A, underlying_index_at_B)"""
    if order == "AB":
        return pair["option_0"], pair["option_1"], 0, 1
    return pair["option_1"], pair["option_0"], 1, 0


def balanced_orders(rng, n):
    """Exactly n/2 'AB' and n/2 'BA', shuffled."""
    assert n % 2 == 0
    orders = np.array(["AB"] * (n // 2) + ["BA"] * (n // 2))
    rng.shuffle(orders)
    return orders.tolist()


def flip(letter):
    return "B" if letter == "A" else "A"

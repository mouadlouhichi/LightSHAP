"""Seed aggregation: average each user across seeds, then bootstrap users.

Never treat 5|U| as iid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def average_users_across_seeds(
    per_seed: Sequence[np.ndarray],
) -> np.ndarray:
    """Each array is (n_users,) aligned by user index. Mean over seeds."""
    stacked = np.stack([np.asarray(a, dtype=np.float64) for a in per_seed], axis=0)
    return stacked.mean(axis=0)


def seed_sd(per_seed_scalar: Sequence[float]) -> float:
    arr = np.asarray(list(per_seed_scalar), dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1))


def phi_order_pattern(phi: Mapping[int, float]) -> tuple[int, ...]:
    """Hops sorted by φ descending; ties broken by hop id ascending."""
    items = sorted(phi.items(), key=lambda kv: (-float(kv[1]), int(kv[0])))
    return tuple(int(k) for k, _ in items)


def sign_of(x: float, atol: float = 1e-12) -> int:
    if x > atol:
        return 1
    if x < -atol:
        return -1
    return 0


def sign_agreement(
    phis: Sequence[Mapping[int, float]],
    players: Sequence[int] = (1, 2, 3),
) -> dict[int, float]:
    """Fraction of seeds matching the majority sign, per hop."""
    out: dict[int, float] = {}
    for p in players:
        signs = [sign_of(float(phi[p])) for phi in phis]
        if not signs:
            out[int(p)] = float("nan")
            continue
        # majority
        vals, counts = np.unique(signs, return_counts=True)
        majority = int(vals[int(np.argmax(counts))])
        out[int(p)] = float(np.mean([s == majority for s in signs]))
    return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def mean_pairwise_cosine(vectors: Sequence[Sequence[float]]) -> float:
    vals: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            c = cosine(vectors[i], vectors[j])
            if np.isfinite(c):
                vals.append(c)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def modal_order(phis: Sequence[Mapping[int, float]]) -> tuple[int, ...] | None:
    if not phis:
        return None
    patterns = [phi_order_pattern(p) for p in phis]
    uniq, counts = np.unique(patterns, return_counts=True)
    # np.unique on tuples of ints works via object array; fall back
    from collections import Counter

    c = Counter(patterns)
    return c.most_common(1)[0][0]

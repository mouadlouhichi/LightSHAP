"""BPR negative sampling with the required --neg-pool toggle.

heldout_excluded: exclude train ∪ {val, test}   (main / offline convention)
train_only:       exclude train only            (sensitivity)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from lightshap.constants import NEG_POOL_CHOICES, NEG_POOL_MAIN


def allowed_negatives(
    user: int,
    n_items: int,
    train_items: Mapping[int, set[int]],
    heldout_items: Mapping[int, set[int]] | None,
    neg_pool: str,
) -> np.ndarray:
    if neg_pool not in NEG_POOL_CHOICES:
        raise ValueError(f"unknown neg_pool {neg_pool!r}")
    forbidden = set(train_items.get(user, set()))
    if neg_pool == NEG_POOL_MAIN and heldout_items is not None:
        forbidden |= set(heldout_items.get(user, set()))
    allow = np.array([i for i in range(n_items) if i not in forbidden], dtype=np.int64)
    if allow.size == 0:
        # degenerate tiny graphs: fall back to anything except the train positives
        allow = np.array(
            [i for i in range(n_items) if i not in train_items.get(user, set())],
            dtype=np.int64,
        )
    if allow.size == 0:
        allow = np.arange(n_items, dtype=np.int64)
    return allow


def sample_negatives(
    users: Sequence[int],
    *,
    n_items: int,
    train_items: Mapping[int, set[int]],
    heldout_items: Mapping[int, set[int]] | None,
    neg_pool: str,
    rng: np.random.Generator,
    n_neg: int = 1,
) -> np.ndarray:
    out = np.empty((len(users), n_neg), dtype=np.int64)
    cache: dict[int, np.ndarray] = {}
    for row, u in enumerate(users):
        u = int(u)
        if u not in cache:
            cache[u] = allowed_negatives(u, n_items, train_items, heldout_items, neg_pool)
        pool = cache[u]
        out[row] = rng.choice(pool, size=n_neg, replace=True)
    return out

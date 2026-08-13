"""BPR negative sampling with the required --neg-pool toggle.

heldout_excluded: exclude train ∪ {val, test}   (main / offline convention)
train_only:       exclude train only            (sensitivity)

Sampling is uniform over the allowed item set (rejection sampling, same
distribution as the old per-user Python loop, without the O(I) list build).
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
        allow = np.array(
            [i for i in range(n_items) if i not in train_items.get(user, set())],
            dtype=np.int64,
        )
    if allow.size == 0:
        allow = np.arange(n_items, dtype=np.int64)
    return allow


def build_forbidden_mask(
    n_users: int,
    n_items: int,
    train_items: Mapping[int, set[int]],
    heldout_items: Mapping[int, set[int]] | None,
    neg_pool: str,
) -> np.ndarray:
    """Boolean [U, I] mask of items that must not be drawn as negatives."""
    if neg_pool not in NEG_POOL_CHOICES:
        raise ValueError(f"unknown neg_pool {neg_pool!r}")
    mask = np.zeros((n_users, n_items), dtype=bool)
    for u, items in train_items.items():
        if items:
            mask[int(u), list(items)] = True
    if neg_pool == NEG_POOL_MAIN and heldout_items is not None:
        for u, items in heldout_items.items():
            if items:
                mask[int(u), list(items)] = True
    return mask


def sample_negatives_mask(
    users: Sequence[int] | np.ndarray,
    *,
    forbidden: np.ndarray,
    rng: np.random.Generator,
    n_neg: int = 1,
    max_tries: int = 32,
) -> np.ndarray:
    """Uniform over allowed items via rejection. ``forbidden`` is [U, I] bool."""
    users_a = np.asarray(users, dtype=np.int64)
    n_items = int(forbidden.shape[1])
    b = int(users_a.shape[0])
    neg = rng.integers(0, n_items, size=(b, n_neg), dtype=np.int64)
    for _ in range(max_tries):
        bad = forbidden[users_a[:, None], neg]
        n_bad = int(bad.sum())
        if n_bad == 0:
            return neg
        neg[bad] = rng.integers(0, n_items, size=n_bad, dtype=np.int64)
    # residual collisions: exact draw from the allowed set
    still = forbidden[users_a[:, None], neg]
    rows, cols = np.where(still)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        allow = np.flatnonzero(~forbidden[int(users_a[r])])
        if allow.size == 0:
            allow = np.arange(n_items, dtype=np.int64)
        neg[r, c] = int(rng.choice(allow))
    return neg


def sample_negatives(
    users: Sequence[int],
    *,
    n_items: int,
    train_items: Mapping[int, set[int]],
    heldout_items: Mapping[int, set[int]] | None,
    neg_pool: str,
    rng: np.random.Generator,
    n_neg: int = 1,
    forbidden: np.ndarray | None = None,
) -> np.ndarray:
    if forbidden is None:
        n_users = max(int(u) for u in users) + 1 if users else 0
        n_users = max(n_users, (max(train_items) + 1) if train_items else 0)
        forbidden = build_forbidden_mask(
            n_users, n_items, train_items, heldout_items, neg_pool
        )
    return sample_negatives_mask(users, forbidden=forbidden, rng=rng, n_neg=n_neg)

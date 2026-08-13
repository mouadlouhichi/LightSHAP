"""Seen-item masking via advanced indexing. Dense (U, I) masks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch


def build_dense_seen_mask(
    n_users: int,
    n_items: int,
    pairs: Iterable[tuple[int, int]],
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Dense [U, I] mask. Item indices are 0 .. I-1 (the 'item I-1' convention)."""
    mask = torch.zeros((n_users, n_items), dtype=dtype)
    users: list[int] = []
    items: list[int] = []
    for u, i in pairs:
        users.append(int(u))
        items.append(int(i))
    if users:
        mask[users, items] = True
    return mask


def build_seen_mask_from_lists(
    n_users: int,
    n_items: int,
    user_items: Mapping[int, Sequence[int]],
) -> torch.Tensor:
    pairs = ((u, i) for u, items in user_items.items() for i in items)
    return build_dense_seen_mask(n_users, n_items, pairs)


def apply_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set seen items to -inf. ``mask`` broadcasts to ``scores``."""
    return scores.masked_fill(mask, torch.finfo(torch.float32).min)


def mask_rows(
    scores: torch.Tensor,
    user_idx: torch.Tensor,
    full_mask: torch.Tensor,
) -> torch.Tensor:
    """Advanced-index the dense [U, I] mask for a batch of users."""
    return apply_mask(scores, full_mask.index_select(0, user_idx))


def numpy_seen_sets(
    n_users: int,
    pairs: Iterable[tuple[int, int]],
) -> list[set[int]]:
    sets: list[set[int]] = [set() for _ in range(n_users)]
    for u, i in pairs:
        sets[int(u)].add(int(i))
    return sets

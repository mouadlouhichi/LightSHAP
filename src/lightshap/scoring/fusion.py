"""Frozen-cache fusion. Skip patterns are scoring rules, not architectures."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from lightshap.constants import E0_SLOT, GAME_A_PLAYERS, N_FUSION_SLOTS


def mask_and_mean(layers: Sequence[torch.Tensor], slots: Sequence[int]) -> torch.Tensor:
    """Mode A: E_C = mean_{k in slots} E^k. No learned weights."""
    if not slots:
        raise ValueError("mask_and_mean requires a non-empty slot set")
    stacked = torch.stack([layers[int(k)] for k in slots], dim=0)
    return stacked.mean(dim=0)


def game_a_slots(coalition: Sequence[int]) -> tuple[int, ...]:
    """Game A: E0 is always kept; coalition ⊆ {1,2,3} are extra hops."""
    extra = tuple(sorted(int(k) for k in coalition if int(k) in GAME_A_PLAYERS))
    return (E0_SLOT, *extra)


def game_b_slots(coalition: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(int(k) for k in coalition))


def weighted_sum(layers: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    if len(weights) != len(layers) and len(weights) != N_FUSION_SLOTS:
        raise ValueError("weight length must match layers or 4 slots")
    acc = None
    n = min(len(layers), len(weights))
    for k in range(n):
        term = float(weights[k]) * layers[k]
        acc = term if acc is None else acc + term
    assert acc is not None
    return acc


def inner_product_scores(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
) -> torch.Tensor:
    """GEMM: (B, d) @ (I, d).T → (B, I) float32."""
    return (user_emb @ item_emb.transpose(0, 1)).to(torch.float32)


def split_user_item(
    fused: torch.Tensor,
    n_users: int,
    n_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes are stacked [0, U) users, [U, U+I) items. Item ids are 0..I-1."""
    if fused.shape[0] != n_users + n_items:
        raise ValueError(
            f"fused rows {fused.shape[0]} != U+I = {n_users + n_items}"
        )
    return fused[:n_users], fused[n_users : n_users + n_items]

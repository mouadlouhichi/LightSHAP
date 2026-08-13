"""Game A (and optional Game B) on a frozen layer cache."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from lightshap.constants import (
    E0_SLOT,
    GAME_A_PLAYERS,
    GAME_B_DEFINITION,
    N_GAME_A_COALITIONS,
)
from lightshap.data.bundle import DatasetBundle
from lightshap.experiments.scoring_eval import evaluate_fused
from lightshap.scoring.fusion import game_a_slots, game_b_slots, mask_and_mean
from lightshap.shapley.exact import all_coalitions, exact_shapley, shapley_from_array
from lightshap.shapley.flag import evaluate_game_a_flag
from lightshap.shapley.interactions import leave_one_out, pair_diagnostics, shapley_interaction


def coalition_fused(
    layers: Sequence[torch.Tensor],
    coalition: tuple[int, ...],
    *,
    game: str = "A",
) -> torch.Tensor | None:
    if game == "A":
        slots = game_a_slots(coalition)
        return mask_and_mean(layers, slots)
    slots = game_b_slots(coalition)
    if not slots:
        return None
    return mask_and_mean(layers, slots)


def game_a_values(
    layers: Sequence[torch.Tensor],
    bundle: DatasetBundle,
    *,
    device: str = "cpu",
    split: str = "val",
) -> dict[str, object]:
    """v(C) = NDCG@10(C) - NDCG@10(E0). Empty coalition is E0 (v=0)."""
    players = GAME_A_PLAYERS
    coalitions = all_coalitions(players)
    if len(coalitions) != N_GAME_A_COALITIONS:
        raise RuntimeError(f"expected {N_GAME_A_COALITIONS} coalitions")

    e0_summary, e0_users = evaluate_fused(layers[E0_SLOT], bundle, split, device=device)  # type: ignore[arg-type]
    v_scalar: dict[tuple[int, ...], float] = {}
    v_users: dict[tuple[int, ...], np.ndarray] = {}
    summaries: dict[str, dict[str, float]] = {}
    for c in coalitions:
        fused = coalition_fused(layers, c, game="A")
        assert fused is not None
        summ, per = evaluate_fused(fused, bundle, split, device=device)  # type: ignore[arg-type]
        key = tuple(c)
        v_scalar[key] = float(summ["ndcg@10"] - e0_summary["ndcg@10"])
        v_users[key] = per - e0_users
        summaries[str(key)] = summ
    # empty is identically zero (both are E0)
    v_scalar[()] = 0.0
    v_users[()] = np.zeros_like(e0_users)

    phi = exact_shapley(v_scalar, players)
    phi_u = shapley_from_array(v_users, players)
    inter = shapley_interaction(v_scalar, players)
    loo = leave_one_out(v_scalar, players)
    flag = evaluate_game_a_flag(v_scalar, players)
    return {
        "v": {str(k): float(val) for k, val in v_scalar.items()},
        "v_raw": v_scalar,
        "v_users": v_users,
        "phi": {str(k): float(val) for k, val in phi.items()},
        "phi_raw": phi,
        "phi_users": phi_u,
        "I": {f"{a}-{b}": float(val) for (a, b), val in inter.items() if a < b},
        "LOO": {str(k): float(val) for k, val in loo.items()},
        "pair_diagnostics": {
            f"{a}-{b}": d for (a, b), d in pair_diagnostics(v_scalar, players).items()
        },
        "flag": flag,
        "e0_ndcg": e0_summary,
        "summaries": summaries,
        "split": split,
        "n_coalitions": len(coalitions),
    }


def game_b_values(
    layers: Sequence[torch.Tensor],
    bundle: DatasetBundle,
    *,
    device: str = "cpu",
    split: str = "val",
) -> dict[str, object]:
    """All-slot game. Empty = zero scores (rank by item_id). Different zero."""
    _ = GAME_B_DEFINITION
    players = (0, 1, 2, 3)
    coalitions = all_coalitions(players)
    # empty scores: zeros → lex ranks by item_id
    zero = torch.zeros_like(layers[0])
    empty_summ, empty_users = evaluate_fused(zero, bundle, split, device=device)  # type: ignore[arg-type]
    v_scalar: dict[tuple[int, ...], float] = {(): 0.0}
    for c in coalitions:
        if not c:
            continue
        fused = coalition_fused(layers, c, game="B")
        assert fused is not None
        summ, _per = evaluate_fused(fused, bundle, split, device=device)  # type: ignore[arg-type]
        v_scalar[tuple(c)] = float(summ["ndcg@10"] - empty_summ["ndcg@10"])
    phi = exact_shapley(v_scalar, players)
    return {
        "v": {str(k): float(val) for k, val in v_scalar.items()},
        "phi": {str(k): float(val) for k, val in phi.items()},
        "empty_ndcg": empty_summ,
        "note": "Game B credits are not comparable to Game A.",
        "split": split,
    }


def encode_v_vector(v: dict[tuple[int, ...], float]) -> list[float]:
    """Canonical 8-vector order: empty, then combinations in lexicographic player order."""
    keys = all_coalitions(GAME_A_PLAYERS)
    return [float(v[tuple(k)]) for k in keys]

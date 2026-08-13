"""Val-α: all 286 points on the 0.1 simplex. Exact tie-break."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from lightshap.constants import N_VAL_ALPHA, UNIFORM_4, VAL_ALPHA_GRID


def iter_val_alpha_weights(grid: int = VAL_ALPHA_GRID) -> list[tuple[float, float, float, float]]:
    """w_i = k_i / grid, k_i ∈ ℕ₀, Σ k = grid. Count = C(grid+3, 3) = 286 for grid=10."""
    out: list[tuple[float, float, float, float]] = []
    for k0 in range(grid + 1):
        for k1 in range(grid + 1 - k0):
            for k2 in range(grid + 1 - k0 - k1):
                k3 = grid - k0 - k1 - k2
                out.append((k0 / grid, k1 / grid, k2 / grid, k3 / grid))
    return out


def val_alpha_candidates() -> list[tuple[float, float, float, float]]:
    cands = iter_val_alpha_weights()
    if len(cands) != N_VAL_ALPHA:
        raise RuntimeError(f"expected {N_VAL_ALPHA} candidates, got {len(cands)}")
    return cands


def uniform_l2(w: Sequence[float]) -> float:
    arr = np.asarray(w, dtype=np.float64)
    uni = np.asarray(UNIFORM_4, dtype=np.float64)
    return float(np.linalg.norm(arr - uni))


def break_val_alpha_ties(
    candidates: Sequence[tuple[tuple[float, float, float, float], float]],
) -> tuple[float, float, float, float]:
    """Tie order: max val NDCG → smallest ||w-uniform||_2 → lex (w0, w1, w2)."""
    if not candidates:
        raise ValueError("no val-alpha candidates")
    best_ndcg = max(ndcg for _, ndcg in candidates)
    tied = [w for w, ndcg in candidates if ndcg == best_ndcg]
    if len(tied) == 1:
        return tied[0]
    tied.sort(key=lambda w: (uniform_l2(w), w[0], w[1], w[2]))
    return tied[0]


def select_val_alpha(
    score_fn: Callable[[tuple[float, float, float, float]], float],
) -> dict[str, object]:
    cands = val_alpha_candidates()
    scored = [(w, float(score_fn(w))) for w in cands]
    chosen = break_val_alpha_ties(scored)
    return {
        "weights": list(chosen),
        "n_candidates": len(scored),
        "best_val_ndcg": max(s for _, s in scored),
        "all_scores": [{"w": list(w), "val_ndcg": s} for w, s in scored],
    }

"""Deterministic 4×4 hop-embedding cosine. No subsample, no RNG.

Side-specific: users [0,U) or items [U, U+I).
Drop rows with train-degree 0 on that side.
Row ℓ2-normalize; cosine of the concatenated flattened normalized rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def _row_l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    out = mat / safe
    out[norms[:, 0] == 0.0] = 0.0
    return out


def flattened_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of concatenated flattened *already row-normalized* matrices."""
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def side_matrices(
    layers: Sequence[torch.Tensor] | Sequence[np.ndarray],
    *,
    n_users: int,
    n_items: int,
    side: str,
    train_degree: np.ndarray,
) -> list[np.ndarray]:
    """Return one (n_kept, d) row-normalized matrix per slot."""
    if side not in {"user", "item"}:
        raise ValueError("side must be 'user' or 'item'")
    mats: list[np.ndarray] = []
    keep = train_degree > 0
    if int(keep.sum()) == 0:
        raise ValueError(f"no positive-train-degree rows on side={side}")
    for layer in layers:
        if isinstance(layer, torch.Tensor):
            arr = layer.detach().cpu().numpy().astype(np.float64)
        else:
            arr = np.asarray(layer, dtype=np.float64)
        if side == "user":
            block = arr[:n_users]
        else:
            block = arr[n_users : n_users + n_items]
        if block.shape[0] != train_degree.shape[0]:
            raise ValueError(
                f"{side} rows {block.shape[0]} != degree length {train_degree.shape[0]}"
            )
        kept = block[keep]
        mats.append(_row_l2_normalize(kept))
    return mats


def cosine_matrix_4x4(
    layers: Sequence[torch.Tensor] | Sequence[np.ndarray],
    *,
    n_users: int,
    n_items: int,
    side: str,
    train_degree: np.ndarray,
) -> np.ndarray:
    mats = side_matrices(
        layers, n_users=n_users, n_items=n_items, side=side, train_degree=train_degree
    )
    n = len(mats)
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            c = flattened_cosine(mats[i], mats[j])
            out[i, j] = c
            out[j, i] = c
    return out

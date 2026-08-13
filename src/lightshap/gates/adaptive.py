"""Adaptive fusion weights. β = 1/4. Floor aligned with 0.005.

If v_m < 0.005 → E0-only. No per-group β_m.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from lightshap.constants import (
    ADAPTIVE_BETA,
    ADAPTIVE_Q_DEFINITION,
    N_FUSION_SLOTS,
    STABILITY_FLOOR,
    UNIFORM_4,
)


def attribution_4vector(phi: Mapping[int, float] | Sequence[float]) -> np.ndarray:
    """Map Game-A 3-vector φ to a 4-slot attribution weight.

    Engineering fill (documented in ADAPTIVE_Q_DEFINITION):
    attr[k] = relu(φ_k) for k=1,2,3; attr[0] = mean(relu(φ));
    if all relu are 0, attr = (1,0,0,0). Then L1-normalize.
    """
    _ = ADAPTIVE_Q_DEFINITION
    if isinstance(phi, Mapping):
        extra = np.array(
            [max(float(phi.get(k, 0.0)), 0.0) for k in (1, 2, 3)], dtype=np.float64
        )
    else:
        arr = list(phi)
        if len(arr) == 3:
            extra = np.array([max(float(x), 0.0) for x in arr], dtype=np.float64)
        elif len(arr) == 4:
            extra = np.array([max(float(x), 0.0) for x in arr[1:]], dtype=np.float64)
        else:
            raise ValueError("phi must be length 3 or 4")
    if float(extra.sum()) == 0.0:
        attr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        attr = np.array([float(extra.mean()), extra[0], extra[1], extra[2]], dtype=np.float64)
        attr = attr / attr.sum()
    return attr


def adaptive_weights(
    phi: Mapping[int, float] | Sequence[float],
    v_m: float,
    *,
    beta: float = ADAPTIVE_BETA,
    floor: float = STABILITY_FLOOR,
) -> np.ndarray:
    if v_m < floor:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    attr = attribution_4vector(phi)
    uniform = np.asarray(UNIFORM_4, dtype=np.float64)
    q = (1.0 - beta) * uniform + beta * attr
    s = float(q.sum())
    if s <= 0:
        return uniform.copy()
    return q / s


def assert_valid_weights(q: np.ndarray) -> None:
    if q.shape != (N_FUSION_SLOTS,):
        raise ValueError(f"expected 4 weights, got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("non-finite adaptive weights")
    if np.any(q < -1e-12):
        raise ValueError(f"negative adaptive weights: {q}")
    if abs(float(q.sum()) - 1.0) > 1e-8:
        raise ValueError(f"weights must sum to 1, got {q.sum()}")

"""Spearman with SciPy default average ranks. Used on the 8-vector v(C) only."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


def average_rank(x: Sequence[float]) -> np.ndarray:
    return rankdata(np.asarray(x, dtype=np.float64), method="average")


def spearman_average_ranks(x: Sequence[float], y: Sequence[float]) -> float:
    """Deterministic Spearman ρ with average ranks for ties."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("spearman inputs must share shape")
    rho, _ = spearmanr(a, b)
    return float(rho)


def pairwise_spearman_mean(vectors: Sequence[Sequence[float]]) -> float:
    """Mean pairwise Spearman across seeds (8-vectors)."""
    arrs = [np.asarray(v, dtype=np.float64) for v in vectors]
    rhos: list[float] = []
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            rhos.append(spearman_average_ranks(arrs[i], arrs[j]))
    if not rhos:
        return float("nan")
    return float(np.mean(rhos))

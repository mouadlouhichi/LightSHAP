"""User-level bootstrap. Unit = user, never 5|U|."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_users(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> dict[str, float]:
    """Bootstrap the mean (default) of a per-user array."""
    x = np.asarray(list(values), dtype=np.float64)
    if statistic is None:
        statistic = lambda a: float(np.mean(a))  # noqa: E731
    rng = np.random.default_rng(seed)
    n = x.size
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = statistic(x[idx])
    return {
        "mean": float(statistic(x)),
        "std": float(stats.std(ddof=1)) if n_boot > 1 else 0.0,
        "ci_lo": float(np.quantile(stats, 0.025)),
        "ci_hi": float(np.quantile(stats, 0.975)),
        "n_users": int(n),
        "n_boot": int(n_boot),
        "unit": "user",
    }

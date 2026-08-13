"""Inferential tests. Two-sided Wilcoxon on seed-averaged per-user diffs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import wilcoxon

from lightshap.statistics.holm import holm_adjust


def two_sided_wilcoxon(diff: Sequence[float]) -> dict[str, float | int | str]:
    """Wilcoxon signed-rank, two-sided, on per-user differences.

    Zero differences are handled by SciPy's default (pratt is not used;
    zeros are dropped by the default 'wilcox' zero_method).
    """
    x = np.asarray(list(diff), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "stat": float("nan"),
            "pvalue": float("nan"),
            "n": 0,
            "alternative": "two-sided",
            "test": "wilcoxon",
        }
    # If all zeros, SciPy raises; treat as p=1.
    if np.allclose(x, 0.0):
        return {
            "stat": 0.0,
            "pvalue": 1.0,
            "n": int(x.size),
            "alternative": "two-sided",
            "test": "wilcoxon",
        }
    try:
        res = wilcoxon(x, alternative="two-sided", zero_method="wilcox")
        return {
            "stat": float(res.statistic),
            "pvalue": float(res.pvalue),
            "n": int(x.size),
            "alternative": "two-sided",
            "test": "wilcoxon",
        }
    except ValueError:
        return {
            "stat": float("nan"),
            "pvalue": float("nan"),
            "n": int(x.size),
            "alternative": "two-sided",
            "test": "wilcoxon",
        }


def holm_within_dataset(
    named_pvalues: list[tuple[str, float]],
) -> list[dict[str, float | str]]:
    names = [n for n, _ in named_pvalues]
    p = [float(v) for _, v in named_pvalues]
    adj = holm_adjust(p)
    return [
        {"contrast": names[i], "p_raw": p[i], "p_holm": adj[i]}
        for i in range(len(names))
    ]

"""Holm correction. Applied within a dataset only. No pooling."""

from __future__ import annotations

from collections.abc import Sequence


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values, same order as input."""
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    adj = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        factor = n - rank
        raw = float(pvalues[idx]) * factor
        running = max(running, min(raw, 1.0))
        adj[idx] = running
    # enforce monotonicity in the original sorted order (already via running)
    return adj

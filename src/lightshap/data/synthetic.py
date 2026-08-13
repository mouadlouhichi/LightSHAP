"""Tiny deterministic bipartite graph for end-to-end tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_interactions(
    *,
    n_users: int = 24,
    n_items: int = 20,
    min_degree: int = 7,
    seed: int = 0,
) -> pd.DataFrame:
    """Already 5-core after a temporal LOO split (train ≥ 5).

    Every user gets ``min_degree`` distinct items; leftover item-degree
    holes are filled by extra deterministic edges so every item has
    degree ≥ 5.
    """
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    for u in range(n_users):
        # staggered blocks so items stay covered
        start = (u * 3) % n_items
        base = [(start + k) % n_items for k in range(min_degree)]
        extra = rng.choice(n_items, size=2, replace=False).tolist()
        for i in base + extra:
            edges.add((u, int(i)))
    # enforce item 5-core
    from collections import Counter

    def item_deg() -> Counter[int]:
        return Counter(i for _, i in edges)

    deg = item_deg()
    u_cursor = 0
    for i in range(n_items):
        while deg[i] < 5:
            u = u_cursor % n_users
            u_cursor += 1
            if (u, i) not in edges:
                edges.add((u, i))
                deg[i] += 1
    rows = []
    # timestamps increase with a deterministic function of (u, i)
    for u, i in sorted(edges):
        ts = 1_600_000_000 + 1000 * u + 10 * i + (u * i) % 7
        rows.append({"user_raw": f"U{u}", "item_raw": f"I{i}", "timestamp": int(ts), "rating": 5.0})
    df = pd.DataFrame(rows).sort_values(["user_raw", "timestamp", "item_raw"]).reset_index(drop=True)
    return df

"""Dataset statistics and frozen quartile cuts (once, from train degrees)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuartileCuts:
    """Frozen once from train user degrees. Reused across seeds."""

    edges: tuple[float, float, float]  # 25, 50, 75

    def assign(self, degrees: np.ndarray) -> np.ndarray:
        e = self.edges
        out = np.ones(len(degrees), dtype=np.int64)  # 1..4
        out[degrees > e[0]] = 2
        out[degrees > e[1]] = 3
        out[degrees > e[2]] = 4
        return out

    def to_dict(self) -> dict[str, float]:
        return {"q25": self.edges[0], "q50": self.edges[1], "q75": self.edges[2]}


def compute_quartile_cuts(train_user_degree: np.ndarray) -> QuartileCuts:
    qs = np.quantile(train_user_degree.astype(np.float64), [0.25, 0.50, 0.75])
    return QuartileCuts(edges=(float(qs[0]), float(qs[1]), float(qs[2])))


def dataset_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_users: int,
    n_items: int,
    user_degree: np.ndarray,
    item_degree: np.ndarray,
) -> dict[str, object]:
    n_inter = int(len(train) + len(val) + len(test))
    density = n_inter / max(n_users * n_items, 1)
    return {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_interactions": n_inter,
        "density": float(density),
        "mean_user_degree": float(user_degree.mean()) if len(user_degree) else 0.0,
        "median_user_degree": float(np.median(user_degree)) if len(user_degree) else 0.0,
        "mean_item_degree": float(item_degree.mean()) if len(item_degree) else 0.0,
        "median_item_degree": float(np.median(item_degree)) if len(item_degree) else 0.0,
        "isolated_train_users": int((user_degree == 0).sum()),
        "isolated_train_items": int((item_degree == 0).sum()),
        "graph_sparsity": float(
            (int(user_degree.sum()) * 2) / max((n_users + n_items) ** 2, 1)
        ),
    }

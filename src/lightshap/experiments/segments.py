"""Degree-quartile segments. Cuts frozen once from train_df, reused across seeds."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from lightshap.data.bundle import DatasetBundle
from lightshap.gates.rq3 import extra_deep_mix


def group_phi(
    phi_users: Mapping[int, np.ndarray],
    labels: np.ndarray,
    v_users: np.ndarray,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for g in (1, 2, 3, 4):
        idx = labels == g
        if not np.any(idx):
            out[g] = {"n": 0, "v_m": float("nan"), "phi_1": float("nan"),
                      "phi_2": float("nan"), "phi_3": float("nan"), "D_m": float("nan")}
            continue
        phi_m = {k: float(np.nanmean(arr[idx])) for k, arr in phi_users.items()}
        v_m = float(np.nanmean(v_users[idx]))
        out[g] = {
            "n": int(idx.sum()),
            "v_m": v_m,
            "phi_1": phi_m.get(1, float("nan")),
            "phi_2": phi_m.get(2, float("nan")),
            "phi_3": phi_m.get(3, float("nan")),
            "D_m": extra_deep_mix(phi_m),
        }
    return out


def segment_tables(bundle: DatasetBundle, game: Mapping[str, object]) -> dict[str, object]:
    phi_users = game["phi_users"]  # type: ignore[index]
    v_users = game["v_users"][tuple(sorted((1, 2, 3)))]  # type: ignore[index]
    labels = bundle.user_quartile
    groups = group_phi(phi_users, labels, v_users)  # type: ignore[arg-type]
    return {
        "cuts": bundle.quartiles.to_dict(),
        "groups": {str(k): v for k, v in groups.items()},
        "note": "quartile cuts computed once from frozen train degrees",
    }

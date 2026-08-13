"""RQ1: v(L) size; Spearman of the 8-vector v(C); φ order/signs/cosine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lightshap.experiments.shapley_run import encode_v_vector
from lightshap.shapley.exact import all_coalitions
from lightshap.statistics.aggregate import (
    mean_pairwise_cosine,
    modal_order,
    phi_order_pattern,
    seed_sd,
    sign_agreement,
)
from lightshap.statistics.spearman import pairwise_spearman_mean


def run_rq1(seed_games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``seed_games`` are Game-A result dicts, one per seed (same dataset)."""
    if not seed_games:
        return {"status": "NOT_RUN", "reason": "no seed games"}
    v_vectors = []
    phis = []
    v_L = []
    for g in seed_games:
        raw = g["v_raw"]
        v_vectors.append(encode_v_vector(raw))
        phis.append(g["phi_raw"])
        v_L.append(float(raw[tuple(sorted((1, 2, 3)))]))
    rho = pairwise_spearman_mean(v_vectors)
    orders = [phi_order_pattern(p) for p in phis]
    modal = modal_order(phis)
    signs = sign_agreement(phis)
    vecs = [[float(p[1]), float(p[2]), float(p[3])] for p in phis]
    cos = mean_pairwise_cosine(vecs)
    return {
        "status": "VALID",
        "n_seeds": len(seed_games),
        "v_L_mean": float(sum(v_L) / len(v_L)),
        "v_L_seed_sd": seed_sd(v_L),
        "v_L_per_seed": v_L,
        "v_C_spearman_mean_pairwise": rho,
        "v_C_vectors": v_vectors,
        "coalition_order": [str(c) for c in all_coalitions((1, 2, 3))],
        "phi_order_patterns": [list(o) for o in orders],
        "phi_modal_order": list(modal) if modal else None,
        "phi_sign_agreement": {str(k): v for k, v in signs.items()},
        "phi_mean_pairwise_cosine": cos,
        "phi_vectors": vecs,
        "note": "No Spearman on phi (3-vector is too short). Average ranks for v(C).",
    }

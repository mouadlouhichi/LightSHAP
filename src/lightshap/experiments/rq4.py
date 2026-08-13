"""RQ4: Adaptive + Val-α transfer check. Two-sided Wilcoxon, Holm within dataset.

Engineering fill for the unnamed 3 tests (documented):
  1. adaptive_vs_uniform
  2. val_alpha_vs_uniform
  3. adaptive_vs_val_alpha

Prior: null. Unit: seed-averaged per-user diffs. No pooling. No β_m.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from lightshap.constants import N_HOLM_TESTS_PER_DATASET, RQ4_CONTRASTS, UNIFORM_4
from lightshap.data.bundle import DatasetBundle
from lightshap.experiments.scoring_eval import evaluate_fused
from lightshap.gates.adaptive import adaptive_weights
from lightshap.gates.val_alpha import break_val_alpha_ties, val_alpha_candidates
from lightshap.scoring.fusion import weighted_sum
from lightshap.statistics.aggregate import average_users_across_seeds
from lightshap.statistics.bootstrap import bootstrap_users
from lightshap.statistics.tests import holm_within_dataset, two_sided_wilcoxon


def _eval_weights(
    layers: Sequence[torch.Tensor],
    weights: Sequence[float],
    bundle: DatasetBundle,
    split: str,
    device: str,
) -> tuple[dict[str, float], np.ndarray]:
    fused = weighted_sum(layers, weights)
    return evaluate_fused(fused, bundle, split, device=device)  # type: ignore[arg-type]


def select_val_alpha_on_bundle(
    layers: Sequence[torch.Tensor],
    bundle: DatasetBundle,
    device: str,
) -> dict[str, Any]:
    cands = val_alpha_candidates()
    scored: list[tuple[tuple[float, float, float, float], float]] = []
    for w in cands:
        summ, _ = _eval_weights(layers, w, bundle, "val", device)
        scored.append((w, float(summ["ndcg@10"])))
    chosen = break_val_alpha_ties(scored)
    return {
        "weights": list(chosen),
        "n_candidates": len(scored),
        "best_val_ndcg": max(s for _, s in scored),
    }


def run_rq4_seed(
    *,
    layers: Sequence[torch.Tensor],
    bundle: DatasetBundle,
    game: dict[str, Any],
    device: str,
    beta: float,
    floor: float,
) -> dict[str, Any]:
    phi = game["phi_raw"]
    v_L = float(game["v_raw"][(1, 2, 3)])
    q_adapt = adaptive_weights(phi, v_L, beta=beta, floor=floor)
    # also per-quartile adaptive (reported, but RQ4 tests use global q — no β_m)
    per_q = {}
    from lightshap.experiments.segments import segment_tables

    segs = segment_tables(bundle, game)["groups"]
    for gname, g in segs.items():
        q = adaptive_weights(
            {1: g["phi_1"], 2: g["phi_2"], 3: g["phi_3"]},
            float(g["v_m"]) if g["v_m"] == g["v_m"] else 0.0,
            beta=beta,
            floor=floor,
        )
        per_q[gname] = q.tolist()

    val_a = select_val_alpha_on_bundle(layers, bundle, device)
    uni = list(UNIFORM_4)

    def pack(split: str) -> dict[str, Any]:
        u_s, u_u = _eval_weights(layers, uni, bundle, split, device)
        a_s, a_u = _eval_weights(layers, q_adapt.tolist(), bundle, split, device)
        v_s, v_u = _eval_weights(layers, val_a["weights"], bundle, split, device)
        return {
            "uniform": {"summary": u_s, "per_user": u_u},
            "adaptive": {"summary": a_s, "per_user": a_u},
            "val_alpha": {"summary": v_s, "per_user": v_u},
        }

    return {
        "adaptive_weights": q_adapt.tolist(),
        "per_quartile_weights": per_q,
        "val_alpha": val_a,
        "val": pack("val"),
        "test": pack("test"),
        "v_L": v_L,
    }


def aggregate_rq4(
    seed_rows: Sequence[dict[str, Any]],
    *,
    n_users: int,
) -> dict[str, Any]:
    """Average each user across seeds, then Wilcoxon + Holm + user bootstrap."""
    if not seed_rows:
        return {"status": "NOT_RUN"}

    def stack(method: str) -> np.ndarray:
        arrs = [np.asarray(r["test"][method]["per_user"], dtype=np.float64) for r in seed_rows]
        return average_users_across_seeds(arrs)

    uni = stack("uniform")
    adp = stack("adaptive")
    val = stack("val_alpha")
    contrasts = {
        "adaptive_vs_uniform": adp - uni,
        "val_alpha_vs_uniform": val - uni,
        "adaptive_vs_val_alpha": adp - val,
    }
    assert len(contrasts) == N_HOLM_TESTS_PER_DATASET
    assert tuple(contrasts.keys()) == RQ4_CONTRASTS
    named = []
    tests = {}
    for name, diff in contrasts.items():
        finite = diff[np.isfinite(diff)]
        tw = two_sided_wilcoxon(finite)
        tests[name] = tw
        named.append((name, float(tw["pvalue"]) if tw["pvalue"] == tw["pvalue"] else 1.0))
    holm = holm_within_dataset(named)
    boots = {
        name: bootstrap_users(diff[np.isfinite(diff)])
        for name, diff in contrasts.items()
    }
    return {
        "status": "VALID",
        "n_seeds": len(seed_rows),
        "n_users": n_users,
        "unit": "user (seed-averaged first)",
        "prior": "null",
        "alternative": "two-sided",
        "correction": "Holm within dataset, 3 tests, no pooling",
        "contrasts": RQ4_CONTRASTS,
        "tests": tests,
        "holm": holm,
        "bootstrap": boots,
        "mean_test_ndcg": {
            "uniform": float(np.nanmean(uni)),
            "adaptive": float(np.nanmean(adp)),
            "val_alpha": float(np.nanmean(val)),
        },
    }

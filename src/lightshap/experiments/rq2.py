"""RQ2: flag, LOO vs φ, retrained K vs frozen prefix coalitions. Flag ≠ prune."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lightshap.experiments.scoring_eval import evaluate_fused
from lightshap.scoring.fusion import mask_and_mean


def run_rq2(
    *,
    seed_games: Sequence[dict[str, Any]],
    retrained: Sequence[dict[str, Any]] | None,
    frozen_prefix_test: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    flags = [g["flag"] for g in seed_games]
    loo_vs_phi = []
    for g in seed_games:
        loo_vs_phi.append(
            {
                "LOO": g["LOO"],
                "phi": g["phi"],
                "v": g["v"],
            }
        )
    retrained_table = None
    if retrained:
        retrained_table = [
            {
                "K": r.get("K"),
                "seed": r.get("seed"),
                "val_ndcg@10": r.get("val_ndcg@10"),
                "test_ndcg@10": r.get("test_ndcg@10"),
            }
            for r in retrained
        ]
    return {
        "status": "VALID" if seed_games else "NOT_RUN",
        "flags": flags,
        "any_pair_flagged_any_seed": any(f.get("any_pair_flagged") for f in flags),
        "loo_vs_phi": loo_vs_phi,
        "retrained_K": retrained_table,
        "frozen_prefix_vs_retrained": frozen_prefix_test,
        "note": (
            "The flag does not imply a pair is safe to drop from training. "
            "Retrained K=1,2,3,4 test NDCG is the pruning-relevant control. "
            "Frozen coalition {1} (mean of E0,E1 on the K=3 cache) is not "
            "a K=1 architecture."
        ),
    }


def frozen_prefix_row(
    layers: Sequence[Any],
    bundle: Any,
    retrained_k1_test_ndcg: float | None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare frozen coalition {1} vs retrained K=1 on test NDCG."""
    fused = mask_and_mean(layers, (0, 1))
    summ, _ = evaluate_fused(fused, bundle, "test", device=device)
    return {
        "frozen_coalition_{1}_test_ndcg@10": summ["ndcg@10"],
        "retrained_K1_test_ndcg@10": retrained_k1_test_ndcg,
        "masking_equals_architecture": False,
    }

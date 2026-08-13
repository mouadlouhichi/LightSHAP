"""Ranking metrics. Primary: NDCG@10. LOO ⇒ one relevant item, IDCG=1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from lightshap.constants import PRIMARY_K
from lightshap.scoring.rank import lexsort_topk, ranks_of_targets


def dcg_from_rank(rank: torch.Tensor, k: int = PRIMARY_K) -> torch.Tensor:
    """rank is 1-indexed. 0 if rank > k."""
    rank_f = rank.to(torch.float32)
    return torch.where(
        (rank > 0) & (rank <= k),
        1.0 / torch.log2(rank_f + 1.0),
        torch.zeros_like(rank_f),
    )


def ndcg_at_k_from_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int = PRIMARY_K,
    item_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-user NDCG@k for a single relevant item."""
    ranks = ranks_of_targets(scores, targets, item_ids)
    return dcg_from_rank(ranks, k)


def recall_at_k_from_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int = PRIMARY_K,
    item_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    top = lexsort_topk(scores, item_ids, k)
    return (top == targets.unsqueeze(1)).any(dim=1).to(torch.float32)


def mrr_from_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
    item_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    ranks = ranks_of_targets(scores, targets, item_ids)
    return 1.0 / ranks.to(torch.float32)


def summarize_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    cutoffs: Sequence[int] = (5, 10, 20),
    item_ids: torch.Tensor | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    ranks = ranks_of_targets(scores, targets, item_ids)
    out["mrr"] = float((1.0 / ranks.to(torch.float32)).mean().item())
    for k in cutoffs:
        ndcg = dcg_from_rank(ranks, k)
        rec = (ranks <= k).to(torch.float32)
        out[f"ndcg@{k}"] = float(ndcg.mean().item())
        out[f"recall@{k}"] = float(rec.mean().item())
        out[f"hr@{k}"] = float(rec.mean().item())
    return out


def per_user_ndcg(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int = PRIMARY_K,
    item_ids: torch.Tensor | None = None,
) -> np.ndarray:
    return ndcg_at_k_from_scores(scores, targets, k, item_ids).detach().cpu().numpy()


def mean_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))

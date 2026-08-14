"""Ranking metrics. Primary: NDCG@10. LOO ⇒ one relevant item, IDCG=1."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Optional

import numpy as np
import torch

from lightshap.constants import PRIMARY_K
from lightshap.scoring.rank import lexsort_topk, ranks_of_targets

logger = logging.getLogger(__name__)


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
    logger.debug(
        f"ndcg_at_k: scores={scores.shape}, targets={targets.shape}, k={k}, "
        f"item_ids={'provided' if item_ids is not None else 'None'}"
    )
    ranks = ranks_of_targets(scores, targets, item_ids)
    result = dcg_from_rank(ranks, k)
    logger.debug(
        f"ndcg_at_k: ranks range=[{ranks.min().item():.0f}, {ranks.max().item():.0f}], "
        f"valid={ (ranks > 0).sum().item()}/{ranks.numel()}, ndcg={result.mean().item():.6f}"
    )
    return result


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
    valid = ranks > 0
    valid_count = valid.sum().item()
    total_count = ranks.numel()
    logger.debug(
        f"mrr_from_scores: scores={scores.shape}, targets={targets.shape}, "
        f"valid_ranks={valid_count}/{total_count}, "
        f"rank_range=[{ranks.min().item():.0f}, {ranks.max().item():.0f}]"
    )
    reciprocal = torch.zeros_like(ranks, dtype=torch.float32)
    reciprocal[valid] = 1.0 / ranks[valid].to(torch.float32)
    logger.debug(f"mrr_from_scores: mrr={reciprocal.mean().item():.6f}")
    return reciprocal


def summarize_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    cutoffs: Sequence[int] = (5, 10, 20),
    item_ids: torch.Tensor | None = None,
    prefix: str = "",
) -> dict[str, float]:
    start_time = time.perf_counter()
    out: dict[str, float] = {}
    
    logger.info(
        f"{prefix}summarize_metrics START: scores={list(scores.shape)}, "
        f"targets={list(targets.shape)}, cutoffs={cutoffs}"
    )
    
    # Compute ranks
    t0 = time.perf_counter()
    ranks = ranks_of_targets(scores, targets, item_ids)
    logger.info(f"{prefix}summarize_metrics: ranks_of_targets took {time.perf_counter() - t0:.3f}s")
    
    # Log input statistics
    valid_ranks = ranks > 0
    valid_count = valid_ranks.sum().item()
    total_count = ranks.numel()
    invalid_count = total_count - valid_count
    
    logger.info(
        f"{prefix}summarize_metrics: valid_ranks={valid_count}/{total_count}, "
        f"rank_range=[{ranks.min().item():.0f}, {ranks.max().item():.0f}]"
    )
    
    if invalid_count > 0:
        logger.warning(
            f"{prefix}summarize_metrics: {invalid_count}/{total_count} "
            f"({100*invalid_count/total_count:.1f}%) targets not found in ranking"
        )
    
    # MRR: only count items that were actually ranked (rank > 0)
    t0 = time.perf_counter()
    valid_rank_values = ranks[valid_ranks]
    if valid_rank_values.numel() > 0:
        out["mrr"] = float((1.0 / valid_rank_values.to(torch.float32)).mean().item())
        logger.info(f"{prefix}mrr={out['mrr']:.6f} (from {valid_rank_values.numel()} valid ranks, {time.perf_counter() - t0:.3f}s)")
    else:
        out["mrr"] = 0.0
        logger.warning(f"{prefix}mrr=0.0 (no valid ranks)")
    
    for k in cutoffs:
        t0 = time.perf_counter()
        ndcg = dcg_from_rank(ranks, k)
        rec = (ranks <= k).to(torch.float32)
        ndcg_val = float(ndcg.mean().item())
        rec_val = float(rec.mean().item())
        out[f"ndcg@{k}"] = ndcg_val
        out[f"recall@{k}"] = rec_val
        out[f"hr@{k}"] = rec_val
        logger.info(
            f"{prefix}k={k}: ndcg={ndcg_val:.6f}, recall={rec_val:.6f}, "
            f"hr={rec_val:.6f} ({time.perf_counter() - t0:.3f}s)"
        )
    
    total_time = time.perf_counter() - start_time
    logger.info(f"{prefix}summarize_metrics DONE: total_time={total_time:.3f}s, results={out}")
    
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

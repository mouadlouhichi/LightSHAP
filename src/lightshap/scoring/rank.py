"""Deterministic Top-K: score desc, item_id asc. No epsilon jitter."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def lexsort_topk(
    scores: torch.Tensor,
    item_ids: torch.Tensor | None = None,
    k: int = 10,
) -> torch.Tensor:
    """Return Top-K item indices (into the last dim of ``scores``).

    ``scores``: float32, shape (..., n_items).
    Ties: larger score first, then smaller item_id.
    Implementation matches the spec::

        order = torch.lexsort(torch.stack((item_ids.expand_as(scores), -scores), 0), dim=-1)
    """
    logger.debug(f"lexsort_topk: scores={scores.shape}, k={k}, item_ids={'provided' if item_ids is not None else 'None'}")
    
    if scores.dtype != torch.float32:
        scores = scores.to(torch.float32)
    n_items = scores.shape[-1]
    if item_ids is None:
        item_ids = torch.arange(n_items, device=scores.device)
    if item_ids.shape[-1] != n_items:
        raise ValueError("item_ids last dim must match scores")
    item_ids = item_ids.to(device=scores.device)
    expanded = item_ids.expand_as(scores)
    # torch.lexsort is not available on all builds. Equivalent stable
    # two-key sort: primary = -score (asc) ⇒ score desc; secondary = item_id asc.
    # Spec: order = lexsort(stack((item_ids, -scores))).
    order = _lexsort_score_desc_id_asc(scores, expanded)
    return order[..., :k].contiguous()


def lexsort_full(
    scores: torch.Tensor,
    item_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if scores.dtype != torch.float32:
        scores = scores.to(torch.float32)
    n_items = scores.shape[-1]
    if item_ids is None:
        item_ids = torch.arange(n_items, device=scores.device)
    expanded = item_ids.to(device=scores.device).expand_as(scores)
    return _lexsort_score_desc_id_asc(scores, expanded)


def _lexsort_score_desc_id_asc(scores: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
    """Stable lex sort matching numpy/torch.lexsort((item_id, -score))."""
    # least-significant key first
    by_id = torch.argsort(item_ids, dim=-1, stable=True)
    gathered = scores.gather(-1, by_id)
    by_score = torch.argsort(-gathered, dim=-1, stable=True)
    return by_id.gather(-1, by_score)


def ranks_of_targets(
    scores: torch.Tensor,
    targets: torch.Tensor,
    item_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """1-indexed rank of each target item under lex ordering.

    ``scores`` (B, I), ``targets`` (B,) item indices in [0, I).
    """
    start_time = time.perf_counter()
    batch_size, n_items = scores.shape[-2], scores.shape[-1]
    
    logger.info(
        f"ranks_of_targets START: scores={scores.shape}, targets={targets.shape}, "
        f"batch={batch_size}, items={n_items}"
    )
    
    # Step 1: Full sort
    t0 = time.perf_counter()
    order = lexsort_full(scores, item_ids)
    logger.info(f"ranks_of_targets: lexsort_full took {time.perf_counter() - t0:.3f}s")
    
    # Step 2: Find target positions
    t0 = time.perf_counter()
    matches = order == targets.unsqueeze(1)
    logger.info(f"ranks_of_targets: match comparison took {time.perf_counter() - t0:.3f}s")
    
    # Step 3: argmax
    t0 = time.perf_counter()
    pos = matches.float().argmax(dim=1)
    logger.info(f"ranks_of_targets: argmax took {time.perf_counter() - t0:.3f}s")
    
    # Step 4: Compute hit mask
    hit = matches.any(dim=1)
    ranks = pos + 1
    ranks = torch.where(hit, ranks, torch.full_like(ranks, n_items + 1))
    
    hit_count = hit.sum().item()
    miss_count = len(hit) - hit_count
    
    logger.info(
        f"ranks_of_targets DONE: hit={hit_count}/{len(hit)}, "
        f"rank_range=[{ranks.min().item():.0f}, {ranks.max().item():.0f}], "
        f"total_time={time.perf_counter() - start_time:.3f}s"
    )
    if miss_count > 0:
        logger.warning(
            f"ranks_of_targets: {miss_count}/{len(hit)} targets not found in scores"
        )
    
    return ranks


def ranks_of_targets_batch(
    scores: torch.Tensor,
    targets: torch.Tensor,
    item_ids: torch.Tensor | None = None,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Batch-optimized version of ranks_of_targets for large tensors.
    
    Processes users in batches to avoid memory issues with large item counts.
    """
    start_time = time.perf_counter()
    n_users, n_items = scores.shape[-2], scores.shape[-1]
    
    logger.info(
        f"ranks_of_targets_batch START: scores={scores.shape}, targets={targets.shape}, "
        f"n_users={n_users}, n_items={n_items}, batch_size={batch_size}"
    )
    
    # Flatten if extra dims
    original_shape = scores.shape
    scores = scores.reshape(-1, n_items)
    targets = targets.reshape(-1)
    
    n_batches = (len(scores) + batch_size - 1) // batch_size
    all_ranks = []
    
    for i in range(0, len(scores), batch_size):
        batch_idx = i // batch_size
        t0 = time.perf_counter()
        
        batch_scores = scores[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        # Compute ranks for batch
        order = lexsort_full(batch_scores, item_ids)
        matches = order == batch_targets.unsqueeze(1)
        pos = matches.float().argmax(dim=1)
        hit = matches.any(dim=1)
        ranks = pos + 1
        ranks = torch.where(hit, ranks, torch.full_like(ranks, n_items + 1))
        
        all_ranks.append(ranks)
        
        elapsed = time.perf_counter() - t0
        logger.info(
            f"ranks_of_targets_batch: batch {batch_idx+1}/{n_batches} "
            f"({i+len(batch_scores)}/{len(scores)}) took {elapsed:.3f}s"
        )
    
    result = torch.cat(all_ranks, dim=0)
    if len(original_shape) > 2:
        result = result.reshape(original_shape[:-2] + (-1,))
    
    logger.info(
        f"ranks_of_targets_batch DONE: total_time={time.perf_counter() - start_time:.3f}s"
    )
    
    return result


def numpy_lexsort_topk(
    scores: np.ndarray,
    item_ids: np.ndarray | None = None,
    k: int = 10,
) -> np.ndarray:
    """NumPy equivalent for tests without torch graphs."""
    scores = np.asarray(scores, dtype=np.float32)
    n_items = scores.shape[-1]
    if item_ids is None:
        item_ids = np.arange(n_items, dtype=np.int64)
    if scores.ndim == 1:
        order = np.lexsort((item_ids, -scores))
        return order[:k]
    out = np.empty(scores.shape[:-1] + (k,), dtype=np.int64)
    flat = scores.reshape(-1, n_items)
    for i, row in enumerate(flat):
        out.reshape(-1, k)[i] = np.lexsort((item_ids, -row))[:k]
    return out

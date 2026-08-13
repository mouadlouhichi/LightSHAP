"""Deterministic Top-K: score desc, item_id asc. No epsilon jitter."""

from __future__ import annotations

import numpy as np
import torch


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
    order = lexsort_full(scores, item_ids)
    # position of target in each row
    matches = order == targets.unsqueeze(1)
    # every row must contain the target exactly once if it was not masked out
    pos = matches.float().argmax(dim=1)
    hit = matches.any(dim=1)
    ranks = pos + 1
    ranks = torch.where(hit, ranks, torch.full_like(ranks, scores.shape[-1] + 1))
    return ranks


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

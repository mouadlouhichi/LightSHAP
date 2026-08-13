"""Evaluate a fused embedding table on val or test (full catalog, lex Top-K)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from lightshap.constants import PRIMARY_K
from lightshap.data.bundle import DatasetBundle
from lightshap.scoring.fusion import inner_product_scores, split_user_item
from lightshap.scoring.mask import apply_mask
from lightshap.scoring.metrics import per_user_ndcg, summarize_metrics

Split = Literal["val", "test"]


@torch.no_grad()
def evaluate_fused(
    fused: torch.Tensor,
    bundle: DatasetBundle,
    split: Split,
    *,
    cutoff: int = PRIMARY_K,
    batch_size: int = 256,
    device: str = "cpu",
) -> tuple[dict[str, float], np.ndarray]:
    """Return (summary metrics, per-user NDCG@cutoff aligned to split users)."""
    frame = bundle.val if split == "val" else bundle.test
    mask = bundle.val_seen_mask if split == "val" else bundle.test_seen_mask
    users_np = frame["user_idx"].to_numpy(dtype=np.int64)
    items_np = frame["item_idx"].to_numpy(dtype=np.int64)
    # per-user array over ALL train users (NaN if user missing from split)
    per_user = np.full(bundle.n_users, np.nan, dtype=np.float64)
    summaries: list[dict[str, float]] = []
    weights: list[int] = []
    user_emb, item_emb = split_user_item(fused.to(device), bundle.n_users, bundle.n_items)
    mask = mask.to(device)
    for start in range(0, len(users_np), batch_size):
        sl = slice(start, start + batch_size)
        u = torch.tensor(users_np[sl], dtype=torch.long, device=device)
        t = torch.tensor(items_np[sl], dtype=torch.long, device=device)
        scores = inner_product_scores(user_emb[u], item_emb)
        scores = apply_mask(scores, mask.index_select(0, u))
        summaries.append(summarize_metrics(scores, t, cutoffs=(5, 10, 20)))
        nd = per_user_ndcg(scores, t, k=cutoff)
        for uu, val in zip(users_np[sl], nd, strict=False):
            per_user[int(uu)] = float(val)
        weights.append(int(len(users_np[sl])))
    total = float(sum(weights)) or 1.0
    summary = {
        k: sum(s[k] * w for s, w in zip(summaries, weights, strict=False)) / total
        for k in summaries[0]
    } if summaries else {}
    return summary, per_user

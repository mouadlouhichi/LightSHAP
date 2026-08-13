from __future__ import annotations

import numpy as np
import pytest
import torch

from lightshap.scoring.mask import apply_mask, build_dense_seen_mask
from lightshap.scoring.rank import lexsort_topk, numpy_lexsort_topk, ranks_of_targets


@pytest.mark.unit
@pytest.mark.regression
def test_lex_ties_item_id_ascending() -> None:
    scores = torch.tensor([[0.5, 0.5, 0.5, 0.1]], dtype=torch.float32)
    top = lexsort_topk(scores, k=3)
    # equal 0.5 at items 0,1,2 → order 0,1,2
    assert top.tolist() == [[0, 1, 2]]


@pytest.mark.unit
def test_lex_score_desc_then_id() -> None:
    scores = torch.tensor([[0.2, 0.9, 0.9, 0.1]], dtype=torch.float32)
    top = lexsort_topk(scores, k=2)
    assert top.tolist() == [[1, 2]]  # both 0.9, lower id first


@pytest.mark.unit
def test_numpy_matches_torch() -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(size=(4, 7)).astype(np.float32)
    # plant exact ties
    s[0, 1] = s[0, 3]
    t = torch.tensor(s)
    assert lexsort_topk(t, k=5).numpy().tolist() == numpy_lexsort_topk(s, k=5).tolist()


@pytest.mark.unit
def test_no_jitter_needed_for_equal_scores() -> None:
    scores = torch.ones(1, 5, dtype=torch.float32)
    top = lexsort_topk(scores, k=5)
    assert top.tolist() == [[0, 1, 2, 3, 4]]


@pytest.mark.unit
def test_dense_mask_item_i_minus_1() -> None:
    mask = build_dense_seen_mask(3, 4, [(0, 0), (0, 3), (2, 1)])
    assert mask.shape == (3, 4)
    assert bool(mask[0, 3])  # last item index I-1 = 3
    scores = torch.zeros(3, 4, dtype=torch.float32)
    masked = apply_mask(scores, mask)
    assert masked[0, 3] < -1e10


@pytest.mark.unit
def test_rank_of_target() -> None:
    scores = torch.tensor([[0.1, 0.4, 0.3]], dtype=torch.float32)
    ranks = ranks_of_targets(scores, torch.tensor([2]))
    # order: item1 (0.4), item2 (0.3), item0 (0.1) → target 2 is rank 2
    assert int(ranks[0]) == 2

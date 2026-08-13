from __future__ import annotations

import numpy as np
import pytest

from lightshap.models.negatives import (
    allowed_negatives,
    build_forbidden_mask,
    sample_negatives_mask,
)


@pytest.mark.unit
def test_rejection_never_hits_forbidden() -> None:
    train = {0: {0, 1}, 1: {2}}
    held = {0: {3}, 1: set()}
    mask = build_forbidden_mask(2, 6, train, held, "heldout_excluded")
    rng = np.random.default_rng(0)
    users = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    neg = sample_negatives_mask(users, forbidden=mask, rng=rng, n_neg=4)
    for row, u in enumerate(users):
        for j in neg[row]:
            assert not mask[int(u), int(j)]


@pytest.mark.unit
def test_heldout_toggle_in_mask() -> None:
    train = {0: {0}}
    held = {0: {1}}
    a = build_forbidden_mask(1, 4, train, held, "heldout_excluded")
    b = build_forbidden_mask(1, 4, train, held, "train_only")
    assert a[0, 1]
    assert not b[0, 1]
    assert set(allowed_negatives(0, 4, train, held, "heldout_excluded").tolist()) == {2, 3}

from __future__ import annotations

import numpy as np
import pytest
import torch

from lightshap.checkpoints.store import (
    CheckpointStore,
    training_payload,
    validate_training_checkpoint,
)
from lightshap.data.graph import assert_symmetric_sparse, build_hat_a
from lightshap.data.preprocess import iterative_kcore, reindex, temporal_loo_split
from lightshap.data.stats import compute_quartile_cuts
from lightshap.data.synthetic import make_synthetic_interactions
from lightshap.exceptions import CheckpointError


@pytest.mark.unit
def test_synthetic_is_5core_after_loo() -> None:
    df = make_synthetic_interactions(seed=0)
    core, hist = iterative_kcore(df, k=5)
    assert not core.empty
    assert hist[-1]["users_after"] == hist[-1]["users"]
    train, val, test = temporal_loo_split(core)
    train, val, test, _, _ = reindex(train, val, test)
    # every train user has ≥5 train edges
    deg = train.groupby("user_idx").size()
    assert deg.min() >= 5
    assert set(val["user_idx"]) <= set(train["user_idx"])
    assert set(test["user_idx"]) <= set(train["user_idx"])


@pytest.mark.unit
def test_hat_a_symmetric_train_only() -> None:
    df = make_synthetic_interactions(seed=1)
    core, _ = iterative_kcore(df, 5)
    train, val, test = temporal_loo_split(core)
    train, val, test, _, _ = reindex(train, val, test)
    n_u = int(train["user_idx"].max()) + 1
    n_i = int(train["item_idx"].max()) + 1
    g = build_hat_a(train, n_u, n_i)
    assert_symmetric_sparse(g.hat_a)
    # val/test pairs must not appear as graph edges
    val_pairs = set(zip(val["user_idx"], val["item_idx"], strict=False))
    train_pairs = set(zip(train["user_idx"], train["item_idx"], strict=False))
    assert val_pairs.isdisjoint(train_pairs)


@pytest.mark.unit
def test_quartile_cuts_frozen() -> None:
    deg = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    cuts = compute_quartile_cuts(deg)
    labels = cuts.assign(deg)
    assert set(labels.tolist()) <= {1, 2, 3, 4}
    # same cuts reuse
    labels2 = cuts.assign(deg)
    assert np.array_equal(labels, labels2)


@pytest.mark.unit
def test_checkpoint_atomic_and_hash(tmp_path) -> None:
    store = CheckpointStore(tmp_path)
    payload = training_payload(
        model_state={"w": torch.tensor([1.0])},
        optimizer_state={},
        scheduler_state=None,
        epoch=3,
        best_metric=0.2,
        best_epoch=2,
        patience_counter=1,
        rng_state={"python": None},
        config_hash="abc",
        dataset_hash="def",
        software={"v": 1},
    )
    store.write_torch("model_epoch0003.pt", payload)
    loaded = store.read_torch("model_epoch0003.pt")
    validate_training_checkpoint(loaded, config_hash="abc", dataset_hash="def")
    with pytest.raises(CheckpointError):
        validate_training_checkpoint(loaded, config_hash="zzz")


@pytest.mark.unit
def test_corrupt_checkpoint_rejected(tmp_path) -> None:
    store = CheckpointStore(tmp_path)
    p = tmp_path / "bad.pt"
    p.write_bytes(b"not a torch checkpoint")
    (tmp_path / "bad.pt.meta.json").write_text('{"sha256": "00"}')
    with pytest.raises(CheckpointError):
        store.read_torch("bad.pt")

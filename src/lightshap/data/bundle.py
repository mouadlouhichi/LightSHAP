"""Build, validate, and persist a processed dataset bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lightshap.config import DatasetConfig
from lightshap.data.graph import GraphBundle, assert_symmetric_sparse, build_hat_a
from lightshap.data.preprocess import (
    assert_no_split_leakage,
    implicitize,
    iterative_kcore,
    load_raw_interactions,
    reindex,
    temporal_loo_split,
)
from lightshap.data.stats import QuartileCuts, compute_quartile_cuts, dataset_stats
from lightshap.exceptions import DataIntegrityError
from lightshap.scoring.mask import build_dense_seen_mask
from lightshap.utils.hashing import sha256_json
from lightshap.utils.io import atomic_write_json, ensure_dir


@dataclass
class DatasetBundle:
    name: str
    n_users: int
    n_items: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    graph: GraphBundle
    quartiles: QuartileCuts
    user_quartile: np.ndarray
    stats: dict[str, object]
    core_history: list[dict[str, int]]
    dataset_hash: str
    val_seen_mask: torch.Tensor  # train only
    test_seen_mask: torch.Tensor  # train+val
    train_item_freq: dict[int, int]

    def user_items_train(self) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {u: set() for u in range(self.n_users)}
        for u, i in zip(self.train["user_idx"], self.train["item_idx"], strict=False):
            out[int(u)].add(int(i))
        return out


def build_dataset_bundle(
    cfg: DatasetConfig,
    raw_dir: Path,
    processed_dir: Path,
) -> DatasetBundle:
    raw = load_raw_interactions(cfg, raw_dir)
    implicit = implicitize(raw, cfg.implicit_threshold)
    core, history = iterative_kcore(implicit, k=5)
    if core.empty:
        raise DataIntegrityError(f"{cfg.name}: 5-core is empty")
    train, val, test = temporal_loo_split(core)
    train, val, test, _u_map, _i_map = reindex(train, val, test)
    if train.empty:
        raise DataIntegrityError(f"{cfg.name}: train empty after reindex")
    n_users = int(train["user_idx"].max()) + 1
    n_items = int(train["item_idx"].max()) + 1
    assert_no_split_leakage(train, val, test)
    graph = build_hat_a(train, n_users, n_items)
    assert_symmetric_sparse(graph.hat_a)
    quartiles = compute_quartile_cuts(graph.user_degree)
    user_q = quartiles.assign(graph.user_degree)
    stats = dataset_stats(
        train, val, test, n_users, n_items, graph.user_degree, graph.item_degree
    )
    payload_hash = sha256_json(
        {
            "name": cfg.name,
            "n_users": n_users,
            "n_items": n_items,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "train_pairs": list(
                zip(
                    train["user_idx"].astype(int).tolist(),
                    train["item_idx"].astype(int).tolist(),
                    strict=False,
                )
            )[:50],  # prefix for stability; full hash below
        }
    )
    # full pair hash
    dataset_hash = sha256_json(
        {
            "name": cfg.name,
            "train": list(
                zip(train["user_idx"].tolist(), train["item_idx"].tolist(), strict=False)
            ),
            "val": list(zip(val["user_idx"].tolist(), val["item_idx"].tolist(), strict=False)),
            "test": list(
                zip(test["user_idx"].tolist(), test["item_idx"].tolist(), strict=False)
            ),
        }
    )
    _ = payload_hash
    val_pairs = list(zip(train["user_idx"].tolist(), train["item_idx"].tolist(), strict=False))
    test_pairs = val_pairs + list(
        zip(val["user_idx"].tolist(), val["item_idx"].tolist(), strict=False)
    )
    val_mask = build_dense_seen_mask(n_users, n_items, val_pairs)
    test_mask = build_dense_seen_mask(n_users, n_items, test_pairs)
    freq: dict[int, int] = {}
    for i in train["item_idx"].tolist():
        freq[int(i)] = freq.get(int(i), 0) + 1

    dest = ensure_dir(processed_dir / cfg.name)
    train.to_parquet(dest / "train.parquet") if _has_parquet() else train.to_csv(
        dest / "train.csv", index=False
    )
    # always write csv for portability
    train.to_csv(dest / "train.csv", index=False)
    val.to_csv(dest / "val.csv", index=False)
    test.to_csv(dest / "test.csv", index=False)
    torch.save(
        {
            "hat_a": graph.hat_a,
            "n_users": n_users,
            "n_items": n_items,
            "user_degree": graph.user_degree,
            "item_degree": graph.item_degree,
        },
        dest / "graph.pt",
    )
    atomic_write_json(
        dest / "metadata.json",
        {
            "name": cfg.name,
            "stats": stats,
            "quartiles": quartiles.to_dict(),
            "core_history": history,
            "dataset_hash": dataset_hash,
            "graph": graph.to_meta(),
        },
    )
    np.save(dest / "user_quartile.npy", user_q)
    return DatasetBundle(
        name=cfg.name,
        n_users=n_users,
        n_items=n_items,
        train=train,
        val=val,
        test=test,
        graph=graph,
        quartiles=quartiles,
        user_quartile=user_q,
        stats=stats,
        core_history=history,
        dataset_hash=dataset_hash,
        val_seen_mask=val_mask,
        test_seen_mask=test_mask,
        train_item_freq=freq,
    )


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def load_bundle_from_disk(processed_dir: Path, name: str) -> DatasetBundle:
    dest = processed_dir / name
    meta = dest / "metadata.json"
    if not meta.is_file():
        raise DataIntegrityError(f"processed metadata missing: {meta}")
    import json

    rec = json.loads(meta.read_text())
    train = pd.read_csv(dest / "train.csv")
    val = pd.read_csv(dest / "val.csv")
    test = pd.read_csv(dest / "test.csv")
    g = torch.load(dest / "graph.pt", weights_only=False, map_location="cpu")
    graph = GraphBundle(
        hat_a=g["hat_a"],
        n_users=int(g["n_users"]),
        n_items=int(g["n_items"]),
        n_nodes=int(g["n_users"]) + int(g["n_items"]),
        nnz=int(g["hat_a"]._nnz()),
        user_degree=np.asarray(g["user_degree"]),
        item_degree=np.asarray(g["item_degree"]),
        isolated_users=int((np.asarray(g["user_degree"]) == 0).sum()),
        isolated_items=int((np.asarray(g["item_degree"]) == 0).sum()),
    )
    qdict = rec["quartiles"]
    quartiles = QuartileCuts(edges=(qdict["q25"], qdict["q50"], qdict["q75"]))
    user_q = np.load(dest / "user_quartile.npy")
    val_pairs = list(zip(train["user_idx"].tolist(), train["item_idx"].tolist(), strict=False))
    test_pairs = val_pairs + list(
        zip(val["user_idx"].tolist(), val["item_idx"].tolist(), strict=False)
    )
    n_users, n_items = graph.n_users, graph.n_items
    freq: dict[int, int] = {}
    for i in train["item_idx"].tolist():
        freq[int(i)] = freq.get(int(i), 0) + 1
    return DatasetBundle(
        name=name,
        n_users=n_users,
        n_items=n_items,
        train=train,
        val=val,
        test=test,
        graph=graph,
        quartiles=quartiles,
        user_quartile=user_q,
        stats=rec["stats"],
        core_history=rec["core_history"],
        dataset_hash=rec["dataset_hash"],
        val_seen_mask=build_dense_seen_mask(n_users, n_items, val_pairs),
        test_seen_mask=build_dense_seen_mask(n_users, n_items, test_pairs),
        train_item_freq=freq,
    )

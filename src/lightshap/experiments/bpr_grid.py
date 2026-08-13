"""BPR-MF six-point grid. Selected by val NDCG@10. Locked before interpreting φ."""

from __future__ import annotations

from typing import Any

from lightshap.checkpoints.store import CheckpointStore
from lightshap.config import ExperimentConfig, load_bpr_grid
from lightshap.data.bundle import DatasetBundle
from lightshap.models.bpr import BPRMF
from lightshap.models.train import train_bpr_loop


def run_bpr_grid(
    bundle: DatasetBundle,
    cfg: ExperimentConfig,
    *,
    seed: int,
    device: str,
    ckpt: CheckpointStore,
    config_hash: str,
    software: dict[str, Any],
    resume: bool = True,
) -> dict[str, Any]:
    grid = load_bpr_grid(cfg.project_root / "configs" / "bpr_grid.yaml")
    results: list[dict[str, Any]] = []
    batch = 64 if cfg.profile == "synthetic" else (
        2048 if bundle.name == "ml1m" else 1024
    )
    for idx, spec in enumerate(grid):
        model = BPRMF(bundle.n_users, bundle.n_items, int(spec["d"]))
        prefix = f"{bundle.name}_seed{seed}_bpr{idx}_d{spec['d']}_lr{spec['lr']}"
        tr = train_bpr_loop(
            model,
            bundle,
            kind="bpr",
            lr=float(spec["lr"]),
            reg=float(spec["reg"]),
            batch_size=batch,
            max_epochs=cfg.bpr_max_epochs,
            patience=cfg.bpr_patience,
            neg_pool=cfg.neg_pool,
            seed=seed + 1000 * idx,
            device=device,
            ckpt=ckpt,
            ckpt_prefix=prefix,
            config_hash=config_hash,
            dataset_hash=bundle.dataset_hash,
            software=software,
            resume=resume,
            n_negatives=cfg.n_negatives,
            log_every=cfg.log_every,
            eval_every=cfg.eval_every,
            checkpoint_every_epoch=cfg.checkpoint_every_epoch,
        )
        results.append(
            {
                "index": idx,
                "d": int(spec["d"]),
                "lr": float(spec["lr"]),
                "reg": float(spec["reg"]),
                "val_ndcg@10": tr.best_metric,
                "best_epoch": tr.best_epoch,
                "checkpoint": tr.checkpoint_path,
            }
        )
    best = max(results, key=lambda r: r["val_ndcg@10"])
    return {
        "grid": results,
        "selected": best,
        "selection_rule": "max val NDCG@10",
        "n_configs": len(results),
        "neg_pool": cfg.neg_pool,
        "seed": seed,
        "dataset": bundle.name,
    }

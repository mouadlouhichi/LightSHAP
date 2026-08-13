"""Train the frozen LightGCN and (optionally) retrained K=1..4."""

from __future__ import annotations

from typing import Any

import torch

from lightshap.checkpoints.store import CheckpointStore
from lightshap.config import ExperimentConfig
from lightshap.constants import K_FROZEN
from lightshap.data.bundle import DatasetBundle
from lightshap.models.lightgcn import LightGCN
from lightshap.models.train import train_bpr_loop


def _batch_for(cfg: ExperimentConfig, name: str) -> int:
    if cfg.profile == "synthetic":
        return int(cfg.extra.get("batch_size", 64))
    return 2048 if name == "ml1m" else 1024


def train_lightgcn_k(
    bundle: DatasetBundle,
    cfg: ExperimentConfig,
    *,
    K: int,
    seed: int,
    device: str,
    ckpt: CheckpointStore,
    config_hash: str,
    software: dict[str, Any],
    resume: bool = True,
    fail_after_epoch: int | None = None,
) -> dict[str, Any]:
    hat = bundle.graph.hat_a.to(device if device != "auto" else "cpu")
    model = LightGCN(bundle.n_users, bundle.n_items, cfg.lightgcn.d, K, hat)
    prefix = f"{bundle.name}_seed{seed}_lgcn_K{K}"
    tr = train_bpr_loop(
        model,
        bundle,
        kind="lightgcn",
        lr=cfg.lightgcn.lr,
        reg=cfg.lightgcn.reg,
        batch_size=_batch_for(cfg, bundle.name),
        max_epochs=cfg.lightgcn.max_epochs,
        patience=cfg.lightgcn.patience,
        neg_pool=cfg.neg_pool,
        seed=seed,
        device=device,
        ckpt=ckpt,
        ckpt_prefix=prefix,
        config_hash=config_hash,
        dataset_hash=bundle.dataset_hash,
        software=software,
        resume=resume,
        n_negatives=cfg.n_negatives,
        eval_every=cfg.lightgcn.eval_every,
        fail_after_epoch=fail_after_epoch,
    )
    # reload best and cache layers
    if tr.checkpoint_path:
        payload = ckpt.read_torch(f"{prefix}_best.pt", map_location="cpu")
        cpu_hat = bundle.graph.hat_a.cpu()
        best = LightGCN(bundle.n_users, bundle.n_items, cfg.lightgcn.d, K, cpu_hat)
        best.load_state_dict(payload["model_state"])
        layers = best.cache_layers()
    else:
        layers = model.cache_layers()
    cache_path = ckpt.root / f"{prefix}_layers.pt"
    torch.save({"layers": layers, "K": K, "d": cfg.lightgcn.d}, cache_path)
    return {
        "K": K,
        "seed": seed,
        "dataset": bundle.name,
        "val_ndcg@10": tr.best_metric,
        "best_epoch": tr.best_epoch,
        "last_epoch": tr.last_epoch,
        "checkpoint": tr.checkpoint_path,
        "layers_path": str(cache_path),
        "history": tr.history,
        "frozen_hparams": {
            "K_requested": K,
            "d": cfg.lightgcn.d,
            "lr": cfg.lightgcn.lr,
            "reg": cfg.lightgcn.reg,
            "patience": cfg.lightgcn.patience,
        },
        "is_main_checkpoint": K == K_FROZEN,
    }


def load_cached_layers(path: str | Any) -> list[torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    return list(obj["layers"])

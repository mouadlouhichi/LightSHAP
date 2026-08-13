"""Shared BPR trainer with epoch-level atomic checkpoints and resume."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from lightshap.checkpoints.store import (
    CheckpointStore,
    training_payload,
    validate_training_checkpoint,
)
from lightshap.data.bundle import DatasetBundle
from lightshap.models.negatives import build_forbidden_mask, sample_negatives_mask
from lightshap.scoring.fusion import inner_product_scores
from lightshap.scoring.mask import mask_rows
from lightshap.scoring.metrics import summarize_metrics
from lightshap.utils.seeding import capture_rng_state, restore_rng_state, seed_everything

logger = logging.getLogger("lightshap")

ModelKind = Literal["bpr", "lightgcn"]


@dataclass
class TrainResult:
    best_metric: float
    best_epoch: int
    last_epoch: int
    history: list[dict[str, float]]
    checkpoint_path: str | None
    extra: dict[str, Any]


def _bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos - neg).mean()


@torch.no_grad()
def evaluate_full_catalog(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    user_idx: torch.Tensor,
    target_idx: torch.Tensor,
    seen_mask: torch.Tensor,
    batch_size: int = 256,
    cutoffs: tuple[int, ...] = (10,),
) -> dict[str, float]:
    """score_fn(users) -> (B, I) raw scores (unmasked)."""
    device = user_idx.device
    chunks: list[dict[str, float]] = []
    weights: list[int] = []
    for start in range(0, len(user_idx), batch_size):
        sl = slice(start, start + batch_size)
        users = user_idx[sl]
        targets = target_idx[sl]
        scores = score_fn(users)
        scores = mask_rows(scores, users, seen_mask.to(device))
        chunks.append(summarize_metrics(scores, targets, cutoffs=cutoffs))
        weights.append(int(users.shape[0]))
    out: dict[str, float] = {}
    total = float(sum(weights))
    keys = chunks[0].keys() if chunks else []
    for k in keys:
        out[k] = sum(c[k] * w for c, w in zip(chunks, weights, strict=False)) / total
    return out


def _score_bpr(model: torch.nn.Module, users: torch.Tensor) -> torch.Tensor:
    return model.full_scores(users)


def _score_lightgcn(model: torch.nn.Module, users: torch.Tensor) -> torch.Tensor:
    fused = model.fuse_uniform()
    u = fused[users]
    items = fused[model.n_users :]
    return inner_product_scores(u, items)


def train_bpr_loop(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    *,
    kind: ModelKind,
    lr: float,
    reg: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    neg_pool: str,
    seed: int,
    device: str,
    ckpt: CheckpointStore | None,
    ckpt_prefix: str,
    config_hash: str,
    dataset_hash: str,
    software: dict[str, Any],
    resume: bool = True,
    n_negatives: int = 1,
    log_every: int = 50,
    eval_every: int = 1,
    checkpoint_every_epoch: bool = False,
    fail_after_epoch: int | None = None,
) -> TrainResult:
    """Train with early stopping on val NDCG@10.

    ``fail_after_epoch`` is a test hook: raise after completing that epoch
    (and writing its checkpoint) to simulate a crash.
    """
    seed_everything(seed)
    device_t = torch.device(device if device != "auto" else "cpu")
    model = model.to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_u = bundle.train["user_idx"].to_numpy(dtype=np.int64)
    train_i = bundle.train["item_idx"].to_numpy(dtype=np.int64)
    train_items = bundle.user_items_train()
    heldout: dict[int, set[int]] = {u: set() for u in range(bundle.n_users)}
    for u, i in zip(bundle.val["user_idx"], bundle.val["item_idx"], strict=False):
        heldout[int(u)].add(int(i))
    for u, i in zip(bundle.test["user_idx"], bundle.test["item_idx"], strict=False):
        heldout[int(u)].add(int(i))

    val_users = torch.tensor(bundle.val["user_idx"].to_numpy(), dtype=torch.long, device=device_t)
    val_items = torch.tensor(bundle.val["item_idx"].to_numpy(), dtype=torch.long, device=device_t)
    seen_val = bundle.val_seen_mask.to(device_t)
    forbidden = build_forbidden_mask(
        bundle.n_users,
        bundle.n_items,
        train_items,
        heldout,
        neg_pool,
    )

    start_epoch = 1
    best_metric = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if resume and ckpt is not None:
        latest = ckpt.latest_training(ckpt_prefix)
        if latest is not None:
            payload = ckpt.read_torch(latest.name, map_location=str(device_t))
            try:
                validate_training_checkpoint(
                    payload, config_hash=config_hash, dataset_hash=dataset_hash
                )
            except Exception as exc:
                logger.warning("incompatible checkpoint ignored: %s", exc)
            else:
                model.load_state_dict(payload["model_state"])
                opt.load_state_dict(payload["optimizer_state"])
                restore_rng_state(payload["rng_state"])
                start_epoch = int(payload["epoch"]) + 1
                best_metric = float(payload["best_metric"])
                best_epoch = int(payload["best_epoch"])
                patience_counter = int(payload["patience_counter"])
                history = list(payload.get("extra", {}).get("history", []))
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                logger.info(
                    "resumed %s from epoch %s", ckpt_prefix, payload["epoch"]
                )

    n = len(train_u)
    last_path: str | None = None

    fused_eval: torch.Tensor | None = None

    def score_fn(users: torch.Tensor) -> torch.Tensor:
        nonlocal fused_eval
        if kind == "bpr":
            return _score_bpr(model, users)
        if fused_eval is None:
            fused_eval = model.fuse_uniform()
        u = fused_eval[users]
        items = fused_eval[model.n_users :]
        return inner_product_scores(u, items)

    for epoch in range(start_epoch, max_epochs + 1):
        # Epoch-keyed RNG so resume is deterministic given the sampler.
        rng = np.random.default_rng(seed + epoch * 17)
        model.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            users = torch.tensor(train_u[idx], dtype=torch.long, device=device_t)
            pos = torch.tensor(train_i[idx], dtype=torch.long, device=device_t)
            neg_np = sample_negatives_mask(
                train_u[idx],
                forbidden=forbidden,
                rng=rng,
                n_neg=n_negatives,
            )
            neg = torch.tensor(neg_np[:, 0], dtype=torch.long, device=device_t)
            if kind == "bpr":
                pos_s = model.score(users, pos)
                neg_s = model.score(users, neg)
                l2 = model.l2(users, pos, neg)
            else:
                fused = model.fuse_uniform()
                pos_s = (fused[users] * fused[model.n_users + pos]).sum(-1)
                neg_s = (fused[users] * fused[model.n_users + neg]).sum(-1)
                l2 = model.l2_batch(users, pos, neg)
            loss = _bpr_loss(pos_s, neg_s) + reg * l2 / max(users.shape[0], 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        record: dict[str, float] = {
            "epoch": float(epoch),
            "loss": epoch_loss / max(n_batches, 1),
        }
        did_eval = epoch % max(eval_every, 1) == 0 or epoch == max_epochs
        if did_eval:
            model.eval()
            fused_eval = None
            metrics = evaluate_full_catalog(
                score_fn, val_users, val_items, seen_val, cutoffs=(10,)
            )
            record["val_ndcg@10"] = metrics["ndcg@10"]
            improved = metrics["ndcg@10"] > best_metric + 1e-12
            if improved:
                best_metric = metrics["ndcg@10"]
                best_epoch = epoch
                patience_counter = 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                # patience is counted in epochs since last improvement
                patience_counter = epoch - best_epoch if best_epoch else epoch
        history.append(record)

        should_ckpt = ckpt is not None and (
            checkpoint_every_epoch or did_eval or epoch == max_epochs
        )
        if should_ckpt:
            payload = training_payload(
                model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                optimizer_state=opt.state_dict(),
                scheduler_state=None,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                rng_state=capture_rng_state(),
                config_hash=config_hash,
                dataset_hash=dataset_hash,
                software=software,
                extra={"history": history, "kind": kind},
            )
            path = ckpt.write_torch(f"{ckpt_prefix}_epoch{epoch:04d}.pt", payload)
            last_path = str(path)
            # keep only last 3 epoch ckpts + best
            _prune_old(ckpt, ckpt_prefix, keep=3)

        logger.info(
            "epoch %s %s loss=%.4f val_ndcg@10=%s",
            epoch,
            ckpt_prefix,
            record["loss"],
            record.get("val_ndcg@10"),
        )

        if fail_after_epoch is not None and epoch >= fail_after_epoch:
            raise RuntimeError(f"injected failure after epoch {epoch}")

        if patience_counter >= patience:
            logger.info("early stop at epoch %s (best %s)", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    if ckpt is not None:
        payload = training_payload(
            model_state=best_state,
            optimizer_state=opt.state_dict(),
            scheduler_state=None,
            epoch=best_epoch,
            best_metric=best_metric,
            best_epoch=best_epoch,
            patience_counter=0,
            rng_state=capture_rng_state(),
            config_hash=config_hash,
            dataset_hash=dataset_hash,
            software=software,
            extra={"history": history, "kind": kind, "tag": "best"},
        )
        best_path = ckpt.write_torch(f"{ckpt_prefix}_best.pt", payload)
        last_path = str(best_path)

    return TrainResult(
        best_metric=best_metric,
        best_epoch=best_epoch,
        last_epoch=int(history[-1]["epoch"]) if history else 0,
        history=history,
        checkpoint_path=last_path,
        extra={},
    )


def _prune_old(ckpt: CheckpointStore, prefix: str, keep: int) -> None:
    files = sorted(ckpt.root.glob(f"{prefix}_epoch*.pt"))
    for stale in files[:-keep]:
        try:
            stale.unlink()
            meta = stale.with_suffix(stale.suffix + ".meta.json")
            if meta.exists():
                meta.unlink()
        except OSError:
            pass

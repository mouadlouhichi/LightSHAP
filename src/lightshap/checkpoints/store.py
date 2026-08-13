"""Atomic checkpoint store. Partial writes never become the active checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightshap.exceptions import CheckpointError
from lightshap.utils.hashing import sha256_file
from lightshap.utils.io import atomic_write_json, ensure_dir, read_json
from lightshap.utils.provenance import utc_now

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(root)

    def path_for(self, name: str) -> Path:
        return self.root / name

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.path_for(name)
        wrapped = {**payload, "written_at": utc_now()}
        atomic_write_json(path, wrapped)
        return path

    def read_json(self, name: str) -> dict[str, Any]:
        path = self.path_for(name)
        if not path.is_file():
            raise CheckpointError(f"missing checkpoint {path}")
        data = read_json(path)
        if not isinstance(data, dict):
            raise CheckpointError(f"checkpoint is not an object: {path}")
        return data

    def exists(self, name: str) -> bool:
        return self.path_for(name).is_file()

    def write_torch(self, name: str, payload: dict[str, Any]) -> Path:
        if torch is None:
            raise CheckpointError("torch is required to write model checkpoints")
        path = self.path_for(name)
        ensure_dir(path.parent)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(payload, tmp)
            tmp.replace(path)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            raise CheckpointError(f"failed to write {path}: {exc}") from exc
        meta = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "written_at": utc_now(),
        }
        atomic_write_json(path.with_suffix(path.suffix + ".meta.json"), meta)
        return path

    def read_torch(self, name: str, map_location: str = "cpu") -> dict[str, Any]:
        if torch is None:
            raise CheckpointError("torch is required to read model checkpoints")
        path = self.path_for(name)
        if not path.is_file():
            raise CheckpointError(f"missing checkpoint {path}")
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.is_file():
            meta = read_json(meta_path)
            digest = sha256_file(path)
            if meta.get("sha256") and meta["sha256"] != digest:
                raise CheckpointError(
                    f"checkpoint hash mismatch for {path}: "
                    f"{meta['sha256']} != {digest}"
                )
        try:
            obj = torch.load(path, map_location=map_location, weights_only=False)
        except Exception as exc:
            raise CheckpointError(f"corrupt checkpoint {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise CheckpointError(f"checkpoint payload is not a dict: {path}")
        return obj

    def latest_training(self, prefix: str) -> Path | None:
        candidates = sorted(self.root.glob(f"{prefix}_epoch*.pt"))
        return candidates[-1] if candidates else None


def training_payload(
    *,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any] | None,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    patience_counter: int,
    rng_state: dict[str, Any],
    config_hash: str,
    dataset_hash: str,
    software: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "rng_state": rng_state,
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "software": software,
    }
    if extra:
        payload["extra"] = extra
    return payload


def validate_training_checkpoint(
    payload: dict[str, Any],
    *,
    config_hash: str | None = None,
    dataset_hash: str | None = None,
) -> None:
    required = {
        "model_state",
        "optimizer_state",
        "epoch",
        "best_metric",
        "best_epoch",
        "patience_counter",
        "rng_state",
        "config_hash",
        "dataset_hash",
    }
    missing = required - set(payload)
    if missing:
        raise CheckpointError(f"checkpoint missing keys: {sorted(missing)}")
    if config_hash is not None and payload["config_hash"] != config_hash:
        raise CheckpointError("checkpoint config_hash does not match the current run")
    if dataset_hash is not None and payload["dataset_hash"] != dataset_hash:
        raise CheckpointError("checkpoint dataset_hash does not match the current run")

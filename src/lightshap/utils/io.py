"""Atomic I/O. A partially written file must never become the active artifact."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=path.suffix, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    text = json.dumps(obj, indent=indent, sort_keys=True, default=_json_default)
    atomic_write_text(path, text + "\n")


def atomic_write_yaml(path: Path, obj: Any) -> None:
    text = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    atomic_write_text(path, text)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def file_is_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0

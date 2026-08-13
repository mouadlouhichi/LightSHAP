"""Artifact validation. Existence on disk is not enough."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from lightshap.exceptions import ArtifactError
from lightshap.utils.hashing import sha256_file
from lightshap.utils.io import read_json


def require_file(path: Path, *, min_bytes: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < min_bytes:
        raise ArtifactError(f"missing or empty artifact: {path}")


def require_json_keys(path: Path, keys: list[str]) -> dict[str, Any]:
    require_file(path)
    data = read_json(path)
    if not isinstance(data, dict):
        raise ArtifactError(f"{path} is not a JSON object")
    missing = [k for k in keys if k not in data]
    if missing:
        raise ArtifactError(f"{path} missing keys {missing}")
    return data


def finite_numbers(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        import math

        if math.isnan(obj) or math.isinf(obj):
            # NaN is allowed in per-user arrays that we persist as null; skip
            return
    if isinstance(obj, dict):
        for k, v in obj.items():
            finite_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            finite_numbers(v, f"{path}[{i}]")


def validate_fixture_file(path: Path) -> None:
    data = require_json_keys(path, ["phi", "I"])
    phi = data["phi"]
    expected = [5 / 3, 8 / 3, 11 / 3]
    got = [float(phi["1"]), float(phi["2"]), float(phi["3"])]
    if any(abs(a - b) > 1e-9 for a, b in zip(got, expected, strict=True)):
        raise ArtifactError(f"fixture phi mismatch: {got}")
    for key in ("12", "13", "23"):
        if abs(float(data["I"][key]) - 0.5) > 1e-9:
            raise ArtifactError(f"fixture I_{key} != 1/2")
        if abs(float(data["I"][key]) - 1.0) < 1e-12:
            raise ArtifactError("I12 == 1 is forbidden")


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


Validator = Callable[[Path], None]

"""Run provenance: git, packages, hardware, hashes."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightshap import __version__
from lightshap.config import ExperimentConfig


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def collect_provenance(cfg: ExperimentConfig, run_id: str) -> dict[str, Any]:
    root = cfg.project_root
    dirty = _git(root, "status", "--porcelain")
    packages: dict[str, str] = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "torch",
        "yaml",
        "matplotlib",
    ):
        try:
            mod = __import__(name)
            packages[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            packages[name] = "missing"
    from lightshap.utils.device import resolve_device

    resolved = resolve_device(cfg.device)
    return {
        "run_id": run_id,
        "lightshap_version": __version__,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_git_state": bool(dirty),
        "dirty_files": dirty.splitlines() if dirty else [],
        "config_hash": cfg.config_hash(),
        "profile": cfg.profile,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cpu": platform.processor() or platform.machine(),
        "device": resolved["device"],
        "device_requested": resolved["requested"],
        "cuda_available": resolved["cuda_available"],
        "mps_available": resolved["mps_available"],
        "gpu_name": resolved["gpu_name"],
        "seeds": list(cfg.seeds),
        "created_at": utc_now(),
    }

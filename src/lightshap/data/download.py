"""Dataset download + SHA256. Never fill a hash during the science run."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from lightshap.config import DatasetConfig
from lightshap.exceptions import DataIntegrityError
from lightshap.utils.hashing import sha256_file
from lightshap.utils.io import atomic_write_json, ensure_dir
from lightshap.utils.provenance import utc_now


def download_file(url: str, dest: Path, *, timeout: int = 120) -> Path:
    ensure_dir(dest.parent)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def verify_sha256(path: Path, expected: str | None, *, allow_missing: bool) -> str:
    digest = sha256_file(path)
    if expected:
        if digest.lower() != expected.lower():
            raise DataIntegrityError(
                f"SHA256 mismatch for {path.name}: expected {expected}, got {digest}"
            )
    elif not allow_missing:
        raise DataIntegrityError(
            f"expected SHA256 is empty for {path.name}. "
            "Paste the hash into the dataset yaml before a scientific run. "
            f"Computed digest={digest}"
        )
    return digest


def ensure_raw_dataset(
    cfg: DatasetConfig,
    raw_dir: Path,
    *,
    allow_missing_sha256: bool,
) -> dict[str, object]:
    if cfg.name == "synthetic":
        return {
            "name": "synthetic",
            "path": None,
            "sha256": None,
            "bytes": 0,
            "skipped": True,
            "reason": "synthetic",
        }
    dest = raw_dir / cfg.filename
    download_file(cfg.url, dest)
    digest = verify_sha256(dest, cfg.expected_sha256, allow_missing=allow_missing_sha256)
    rec = {
        "name": cfg.name,
        "url": cfg.url,
        "path": str(dest),
        "filename": cfg.filename,
        "sha256": digest,
        "expected_sha256": cfg.expected_sha256,
        "bytes": dest.stat().st_size,
        "timestamp": utc_now(),
        "verified": bool(cfg.expected_sha256),
    }
    atomic_write_json(raw_dir / f"{cfg.name}_download.json", rec)
    return rec

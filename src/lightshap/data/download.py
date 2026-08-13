"""Dataset download + SHA256.

If the yaml hash is empty, the first successful download *locks* the computed
digest into configs/<dataset>.yaml and continues. Later runs must match.
A mismatch still fails. This avoids dying once per dataset.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from lightshap.config import DatasetConfig
from lightshap.exceptions import DataIntegrityError
from lightshap.utils.hashing import sha256_file
from lightshap.utils.io import atomic_write_json, atomic_write_text, ensure_dir
from lightshap.utils.provenance import utc_now


def download_file(url: str, dest: Path, *, timeout: int = 300) -> Path:
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


def write_sha256_yaml(yaml_path: Path, digest: str) -> None:
    text = yaml_path.read_text(encoding="utf-8") if yaml_path.is_file() else ""
    line = f'sha256: "{digest}"'
    if re.search(r"^sha256:\s*", text, flags=re.M):
        text = re.sub(r"^sha256:\s*.*$", line, text, count=1, flags=re.M)
    else:
        text = text.rstrip() + "\n" + line + "\n"
    atomic_write_text(yaml_path, text)


def verify_sha256(
    path: Path,
    expected: str | None,
    *,
    allow_missing: bool,
    yaml_path: Path | None = None,
    record_if_empty: bool = True,
) -> tuple[str, bool]:
    """Return (digest, recorded_first_seen)."""
    digest = sha256_file(path)
    if expected:
        if digest.lower() != expected.lower():
            raise DataIntegrityError(
                f"SHA256 mismatch for {path.name}: expected {expected}, got {digest}"
            )
        return digest, False
    if record_if_empty and yaml_path is not None:
        write_sha256_yaml(yaml_path, digest)
        return digest, True
    if not allow_missing:
        raise DataIntegrityError(
            f"expected SHA256 is empty for {path.name}. "
            f"Computed digest={digest}. "
            f'Set sha256: "{digest}" in the dataset yaml, then resume.'
        )
    return digest, False


def ensure_raw_dataset(
    cfg: DatasetConfig,
    raw_dir: Path,
    *,
    allow_missing_sha256: bool,
    yaml_path: Path | None = None,
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
    download_file(cfg.url, dest, timeout=600)
    digest, recorded = verify_sha256(
        dest,
        cfg.expected_sha256,
        allow_missing=allow_missing_sha256,
        yaml_path=yaml_path,
        record_if_empty=yaml_path is not None,
    )
    rec = {
        "name": cfg.name,
        "url": cfg.url,
        "path": str(dest),
        "filename": cfg.filename,
        "sha256": digest,
        "expected_sha256": cfg.expected_sha256 or digest,
        "bytes": dest.stat().st_size,
        "timestamp": utc_now(),
        "verified": True,
        "recorded_first_seen": recorded,
    }
    atomic_write_json(raw_dir / f"{cfg.name}_download.json", rec)
    return rec

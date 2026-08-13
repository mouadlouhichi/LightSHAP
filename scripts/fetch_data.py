#!/usr/bin/env python3
"""Download ML-1M and Beauty 2014, print SHA256s to paste into the yaml files.

Does NOT start the science run. Spec: paste hashes before the signed tag.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightshap.config import load_dataset_yaml  # noqa: E402
from lightshap.data.download import download_file  # noqa: E402
from lightshap.utils.hashing import sha256_file  # noqa: E402


def main() -> int:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("ml1m", "beauty"):
        cfg = load_dataset_yaml(name, ROOT)
        dest = raw / cfg.filename
        print(f"== {name} ==")
        print("url", cfg.url)
        print("dest", dest)
        try:
            download_file(cfg.url, dest, timeout=300)
        except Exception as exc:
            print("FAILED", type(exc).__name__, exc)
            continue
        digest = sha256_file(dest)
        print("bytes", dest.stat().st_size)
        print("sha256", digest)
        print(f"Paste into configs/{name}.yaml → sha256: \"{digest}\"")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

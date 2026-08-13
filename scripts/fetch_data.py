#!/usr/bin/env python3
"""Download ML-1M and Beauty 2014, print SHA256s.

Does NOT start the science run. Spec: paste hashes before the signed tag.

    python scripts/fetch_data.py
    python scripts/fetch_data.py --write-yaml   # write hashes into configs/*.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightshap.config import load_dataset_yaml  # noqa: E402
from lightshap.data.download import download_file  # noqa: E402
from lightshap.utils.hashing import sha256_file  # noqa: E402


def write_sha256(yaml_path: Path, digest: str) -> None:
    text = yaml_path.read_text(encoding="utf-8")
    if re.search(r"^sha256:\s*", text, flags=re.M):
        text = re.sub(r'^sha256:\s*.*$', f'sha256: "{digest}"', text, count=1, flags=re.M)
    else:
        text = text.rstrip() + f'\nsha256: "{digest}"\n'
    yaml_path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--write-yaml",
        action="store_true",
        help="Write computed SHA256s into configs/ml1m.yaml and configs/beauty.yaml",
    )
    args = p.parse_args(argv)
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name in ("ml1m", "beauty"):
        cfg = load_dataset_yaml(name, ROOT)
        dest = raw / cfg.filename
        print(f"== {name} ==")
        print("url", cfg.url)
        print("dest", dest)
        try:
            download_file(cfg.url, dest, timeout=600)
        except Exception as exc:
            print("FAILED", type(exc).__name__, exc)
            rc = 1
            continue
        digest = sha256_file(dest)
        print("bytes", dest.stat().st_size)
        print("sha256", digest)
        ypath = ROOT / "configs" / f"{name}.yaml"
        if args.write_yaml:
            write_sha256(ypath, digest)
            print("wrote", ypath)
        else:
            print(f'Paste into configs/{name}.yaml → sha256: "{digest}"')
        print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

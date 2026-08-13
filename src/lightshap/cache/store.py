"""Hash-aware artifact cache. Incompatible entries are never reused."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lightshap.utils.hashing import sha256_file, sha256_json
from lightshap.utils.io import atomic_write_json, ensure_dir, file_is_nonempty, read_json
from lightshap.utils.provenance import utc_now

logger = logging.getLogger("lightshap")


class ArtifactCache:
    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(root)

    def key(self, name: str, deps: dict[str, Any]) -> str:
        return f"{name}-{sha256_json({'name': name, **deps})[:16]}"

    def dir_for(self, name: str, deps: dict[str, Any]) -> Path:
        return self.root / self.key(name, deps)

    def lookup(self, name: str, deps: dict[str, Any], filename: str) -> Path | None:
        d = self.dir_for(name, deps)
        path = d / filename
        meta = d / "cache_meta.json"
        if not (file_is_nonempty(path) and meta.is_file()):
            logger.info("cache miss", extra={"event": {"event": "cache_miss", "name": name}})
            return None
        rec = read_json(meta)
        if rec.get("deps_hash") != sha256_json(deps):
            logger.info(
                "cache invalid (deps)",
                extra={"event": {"event": "cache_invalid", "name": name}},
            )
            return None
        logger.info("cache hit", extra={"event": {"event": "cache_hit", "name": name}})
        return path

    def store(
        self,
        name: str,
        deps: dict[str, Any],
        filename: str,
        src: Path,
    ) -> Path:
        d = ensure_dir(self.dir_for(name, deps))
        dest = d / filename
        dest.write_bytes(src.read_bytes())
        atomic_write_json(
            d / "cache_meta.json",
            {
                "name": name,
                "filename": filename,
                "deps": deps,
                "deps_hash": sha256_json(deps),
                "sha256": sha256_file(dest),
                "stored_at": utc_now(),
            },
        )
        return dest

    def clear(self) -> int:
        n = 0
        if not self.root.exists():
            return 0
        for child in self.root.iterdir():
            if child.is_dir():
                for f in child.rglob("*"):
                    if f.is_file():
                        f.unlink()
                        n += 1
                # remove empty dirs bottom-up
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                child.rmdir()
        return n

#!/usr/bin/env python3
"""Entry point matching the deliverable CLI examples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightshap.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

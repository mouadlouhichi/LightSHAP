#!/usr/bin/env python3
"""Generate and verify the hard-coded Shapley fixtures. Oracle for A.0."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightshap.shapley.fixtures import (  # noqa: E402
    all_fixture_tables,
    verify_halfsplit_is_not_general,
    verify_main_fixture,
    verify_prop2_gadget,
)
from lightshap.shapley.interactions import shapley_interaction  # noqa: E402


def main() -> int:
    out_dir = ROOT / "results" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = all_fixture_tables()
    serial = {name: {str(k): v for k, v in tbl.items()} for name, tbl in tables.items()}
    (out_dir / "games.json").write_text(json.dumps(serial, indent=2) + "\n")
    rec = verify_main_fixture()
    verify_prop2_gadget(tables["prop2_12"], 1, 2)
    verify_prop2_gadget(tables["prop2_23"], 2, 3)
    verify_halfsplit_is_not_general()
    # extra: assert I12 is not 1
    inter = shapley_interaction(tables["main"])
    if abs(inter[(1, 2)] - 1.0) < 1e-12:
        print("FAIL: I12 == 1 (retracted claim)", file=sys.stderr)
        return 1
    payload = {
        "phi": rec,
        "I12": inter[(1, 2)],
        "I13": inter[(1, 3)],
        "I23": inter[(2, 3)],
        "status": "PASS",
    }
    (out_dir / "verified.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

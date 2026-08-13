from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.unit
def test_write_sha256_replaces_empty(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "fetch_data.py"
    spec = importlib.util.spec_from_file_location("fetch_data", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    y = tmp_path / "ml1m.yaml"
    y.write_text('name: ml1m\nsha256: ""\n')
    mod.write_sha256(y, "abc123")
    assert 'sha256: "abc123"' in y.read_text()

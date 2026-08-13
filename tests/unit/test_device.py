from __future__ import annotations

import pytest

from lightshap.utils.device import resolve_device


@pytest.mark.unit
def test_resolve_cpu_explicit() -> None:
    rec = resolve_device("cpu")
    assert rec["device"] == "cpu"
    assert rec["requested"] == "cpu"


@pytest.mark.unit
def test_resolve_auto_is_known() -> None:
    rec = resolve_device("auto")
    assert rec["device"] in {"cpu", "cuda", "mps"}

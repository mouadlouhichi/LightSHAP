from __future__ import annotations

from pathlib import Path

import pytest

from lightshap.constants import (
    ADAPTIVE_BETA,
    FLAG_RELATIVE_TOL,
    FORBIDDEN_I12,
    GAME_B_COSINE_MARGIN,
    LIGHTGCN_FROZEN,
    N_VAL_ALPHA,
    STABILITY_FLOOR,
)
from lightshap.shapley.fixtures import fixture_game
from lightshap.shapley.interactions import shapley_interaction


@pytest.mark.regression
def test_frozen_numeric_constants() -> None:
    assert LIGHTGCN_FROZEN["K"] == 3
    assert LIGHTGCN_FROZEN["d"] == 64
    assert LIGHTGCN_FROZEN["lr"] == 1.0e-3
    assert LIGHTGCN_FROZEN["reg"] == 1.0e-4
    assert LIGHTGCN_FROZEN["patience"] == 20
    assert STABILITY_FLOOR == 0.005
    assert FLAG_RELATIVE_TOL == 0.15
    assert GAME_B_COSINE_MARGIN == 0.05
    assert ADAPTIVE_BETA == 0.25
    assert N_VAL_ALPHA == 286


@pytest.mark.regression
def test_source_never_claims_i12_equals_one() -> None:
    root = Path(__file__).resolve().parents[2] / "src"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # allow the explicit forbidden constant and the rejection message
        if "FORBIDDEN_I12" in text or "I12 == 1" in text or "I_12 = 1" in text:
            continue
        assert "I_12 = 1" not in text


@pytest.mark.regression
def test_fixture_rejects_one() -> None:
    i12 = shapley_interaction(fixture_game())[(1, 2)]
    assert i12 != FORBIDDEN_I12

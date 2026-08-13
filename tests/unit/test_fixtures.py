from __future__ import annotations

import pytest

from lightshap.constants import FIXTURE_I, FIXTURE_PHI, FORBIDDEN_I12
from lightshap.shapley.exact import exact_shapley
from lightshap.shapley.fixtures import (
    all_fixture_tables,
    fixture_game,
    verify_halfsplit_is_not_general,
    verify_main_fixture,
    verify_prop2_gadget,
)
from lightshap.shapley.interactions import leave_one_out, shapley_interaction


@pytest.mark.unit
@pytest.mark.regression
def test_main_fixture_phi_and_i() -> None:
    rec = verify_main_fixture()
    assert rec["phi_1"] == pytest.approx(FIXTURE_PHI[0])
    assert rec["phi_2"] == pytest.approx(FIXTURE_PHI[1])
    assert rec["phi_3"] == pytest.approx(FIXTURE_PHI[2])
    assert rec["I_12"] == pytest.approx(FIXTURE_I)
    assert rec["I_13"] == pytest.approx(FIXTURE_I)
    assert rec["I_23"] == pytest.approx(FIXTURE_I)


@pytest.mark.unit
@pytest.mark.regression
def test_i12_is_not_one() -> None:
    inter = shapley_interaction(fixture_game())
    assert inter[(1, 2)] != pytest.approx(FORBIDDEN_I12)
    assert abs(inter[(1, 2)] - 1.0) > 1e-9


@pytest.mark.unit
def test_prop2_gadgets() -> None:
    tables = all_fixture_tables()
    verify_prop2_gadget(tables["prop2_12"], 1, 2)
    verify_prop2_gadget(tables["prop2_23"], 2, 3)
    loo = leave_one_out(tables["prop2_12"])
    assert loo[1] == pytest.approx(0.0)
    assert loo[2] == pytest.approx(0.0)
    phi = exact_shapley(tables["prop2_12"])
    assert phi[1] == pytest.approx(phi[2])


@pytest.mark.unit
def test_halfsplit_not_general_for_n3() -> None:
    verify_halfsplit_is_not_general()

from __future__ import annotations

import numpy as np
import pytest

from lightshap.shapley.exact import all_coalitions, exact_shapley, shapley_from_array
from lightshap.shapley.flag import evaluate_game_a_flag
from lightshap.shapley.interactions import pair_diagnostics, r_ab


@pytest.mark.unit
def test_eight_coalitions() -> None:
    assert len(all_coalitions((1, 2, 3))) == 8


@pytest.mark.unit
def test_efficiency_additive_game() -> None:
    values = {
        (): 0.0,
        (1,): 1.0,
        (2,): 2.0,
        (3,): 3.0,
        (1, 2): 3.0,
        (1, 3): 4.0,
        (2, 3): 5.0,
        (1, 2, 3): 6.0,
    }
    phi = exact_shapley(values)
    assert sum(phi.values()) == pytest.approx(6.0)
    assert phi[1] == pytest.approx(1.0)
    assert phi[2] == pytest.approx(2.0)
    assert phi[3] == pytest.approx(3.0)


@pytest.mark.unit
def test_efficiency_holds_for_any_set_function() -> None:
    values = {
        (): 0.0,
        (1,): 1.0,
        (2,): 0.0,
        (3,): 0.0,
        (1, 2): 1.0,
        (1, 3): 1.0,
        (2, 3): 0.0,
        (1, 2, 3): 99.0,
    }
    phi = exact_shapley(values)
    assert sum(phi.values()) == pytest.approx(99.0)


@pytest.mark.unit
def test_per_user_mean_matches() -> None:
    n = 5
    values = {
        (): np.zeros(n),
        (1,): np.ones(n),
        (2,): np.ones(n) * 2,
        (3,): np.ones(n) * 3,
        (1, 2): np.ones(n) * 3,
        (1, 3): np.ones(n) * 4,
        (2, 3): np.ones(n) * 5,
        (1, 2, 3): np.ones(n) * 6,
    }
    phi_u = shapley_from_array(values)
    assert phi_u[1].mean() == pytest.approx(1.0)


@pytest.mark.unit
def test_flag_undefined_below_floor() -> None:
    values = {c: 0.001 if c == (1, 2, 3) else 0.0 for c in all_coalitions((1, 2, 3))}
    values[()] = 0.0
    flag = evaluate_game_a_flag(values)
    assert flag["undefined"] is True
    assert flag["any_pair_flagged"] is False
    for rec in flag["pairs"].values():
        assert rec["state"] == "UNDEFINED"
        assert "v_L" in rec["raw"]


@pytest.mark.unit
def test_flag_does_not_replace_undefined_with_zero() -> None:
    values = {c: 0.0 for c in all_coalitions((1, 2, 3))}
    flag = evaluate_game_a_flag(values)
    assert flag["uplift_state"] in {"NONPOSITIVE", "UNDEFINED"}
    assert flag["v_L"] == 0.0


@pytest.mark.unit
def test_r_ab_grand_slice_is_loo() -> None:
    values = {
        (): 0.0,
        (1,): 1.0,
        (2,): 1.0,
        (3,): 1.0,
        (1, 2): 1.5,
        (1, 3): 2.0,
        (2, 3): 2.0,
        (1, 2, 3): 2.2,
    }
    r = r_ab(values, 1, 2)
    # grand-coalition slice = max(|v(N)-v({2,3})|, |v(N)-v({1,3})|)
    grand = max(abs(2.2 - 2.0), abs(2.2 - 2.0))
    assert r >= grand - 1e-12
    diags = pair_diagnostics(values)
    assert (1, 2) in diags

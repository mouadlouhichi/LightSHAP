"""Hard-coded rational fixtures. The generator is the oracle.

φ = (5/3, 8/3, 11/3), I_12 = I_13 = I_23 = 1/2.
If anything claims I_12 = 1, that claim is wrong.
"""

from __future__ import annotations

from collections.abc import Mapping

from lightshap.constants import (
    FIXTURE_I,
    FIXTURE_PHI,
    FIXTURE_V,
    FORBIDDEN_I12,
    GAME_A_PLAYERS,
)
from lightshap.exceptions import MathInvariantError
from lightshap.shapley.exact import Coalition, exact_shapley
from lightshap.shapley.interactions import leave_one_out, shapley_interaction

# Möbius construction (documented):
#   m(∅)=0, m_N=0, m_{ij}=1/2 for every pair,
#   m_i = φ_i - 1/2  ⇒  (7/6, 13/6, 19/6).
# Then v(C) = Σ_{T ⊆ C} m(T).


def fixture_game() -> dict[Coalition, float]:
    return dict(FIXTURE_V)


def prop2_gadget_substitutes() -> dict[Coalition, float]:
    """Exact substitution of players 1 and 2.

    v(C∪{1})=v(C∪{2})=v(C∪{1,2}) for C ⊆ {3}.
    Implies LOO_1 = LOO_2 = 0 and φ_1 = φ_2.
    """
    return {
        (): 0.0,
        (1,): 2.0,
        (2,): 2.0,
        (3,): 1.0,
        (1, 2): 2.0,
        (1, 3): 5.0,
        (2, 3): 5.0,
        (1, 2, 3): 5.0,
    }


def prop2_gadget_halfsplit_counterexample() -> dict[Coalition, float]:
    """Exact substitutes whose n=3 Shapley is *not* the two-player half-split.

    φ_1 = a/3 + (b-c)/6, half-split = (b-c)/2.
    With a=2, b=5, c=1: φ_1 = 4/3 ≠ 2.
    """
    return prop2_gadget_substitutes()


def prop2_gadget_substitutes_23() -> dict[Coalition, float]:
    """Exact substitution of players 2 and 3 (second gadget)."""
    return {
        (): 0.0,
        (1,): 1.0,
        (2,): 3.0,
        (3,): 3.0,
        (1, 2): 4.0,
        (1, 3): 4.0,
        (2, 3): 3.0,
        (1, 2, 3): 4.0,
    }


def _close(a: float, b: float, atol: float = 1e-12) -> bool:
    return abs(a - b) <= atol


def verify_main_fixture(values: Mapping[Coalition, float] | None = None) -> dict[str, float]:
    values = dict(values) if values is not None else fixture_game()
    phi = exact_shapley(values, GAME_A_PLAYERS)
    inter = shapley_interaction(values, GAME_A_PLAYERS)
    got_phi = (phi[1], phi[2], phi[3])
    if not all(_close(g, e) for g, e in zip(got_phi, FIXTURE_PHI, strict=True)):
        raise MathInvariantError(f"fixture phi={got_phi} != {FIXTURE_PHI}")
    for pair in ((1, 2), (1, 3), (2, 3)):
        got = inter[pair]
        if not _close(got, FIXTURE_I):
            raise MathInvariantError(f"fixture I_{pair}={got} != {FIXTURE_I}")
        if _close(got, FORBIDDEN_I12) and FORBIDDEN_I12 != FIXTURE_I:
            raise MathInvariantError("I12 == 1 is a retracted (wrong) claim")
    if _close(inter[(1, 2)], FORBIDDEN_I12):
        raise MathInvariantError("I12 must not equal 1")
    return {
        "phi_1": phi[1],
        "phi_2": phi[2],
        "phi_3": phi[3],
        "I_12": inter[(1, 2)],
        "I_13": inter[(1, 3)],
        "I_23": inter[(2, 3)],
    }


def verify_prop2_gadget(values: Mapping[Coalition, float], a: int, b: int) -> None:
    phi = exact_shapley(dict(values), GAME_A_PLAYERS)
    loo = leave_one_out(dict(values), GAME_A_PLAYERS)
    if not _close(loo[a], 0.0) or not _close(loo[b], 0.0):
        raise MathInvariantError(f"Prop.2 gadget LOO not zero: {loo}")
    if not _close(phi[a], phi[b]):
        raise MathInvariantError(f"Prop.2 gadget phi_{a}={phi[a]} != phi_{b}={phi[b]}")


def verify_halfsplit_is_not_general() -> None:
    values = prop2_gadget_halfsplit_counterexample()
    verify_prop2_gadget(values, 1, 2)
    phi = exact_shapley(values, GAME_A_PLAYERS)
    half = 0.5 * (values[(1, 2, 3)] - values[(3,)])
    if _close(phi[1], half):
        raise MathInvariantError(
            "this gadget was supposed to show the n=3 half-split is not general"
        )


def all_fixture_tables() -> dict[str, dict[Coalition, float]]:
    return {
        "main": fixture_game(),
        "prop2_12": prop2_gadget_substitutes(),
        "prop2_23": prop2_gadget_substitutes_23(),
    }

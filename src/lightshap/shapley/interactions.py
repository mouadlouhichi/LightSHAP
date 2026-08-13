"""Shapley interaction index (2-SII) and LOO / R_ab diagnostics."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from lightshap.constants import GAME_A_PLAYERS, R_AB_DEFINITION
from lightshap.shapley.exact import Coalition, ValueTable


def discrete_derivative(
    values: ValueTable,
    coalition: Coalition,
    i: int,
    j: int,
) -> float:
    c = set(coalition)
    v = values
    return (
        float(v[tuple(sorted(c | {i, j}))])
        - float(v[tuple(sorted(c | {i}))])
        - float(v[tuple(sorted(c | {j}))])
        + float(v[tuple(sorted(c))])
    )


def shapley_interaction(
    values: ValueTable,
    players: Sequence[int] = GAME_A_PLAYERS,
) -> dict[tuple[int, int], float]:
    """
    I_ij = Σ_{C ⊆ N\\{i,j}} |C|!(n-|C|-2)! / (n-1)!  · Δ_{ij} v(C).

    For substitutes I_ij < 0; for complements I_ij > 0.
    Symmetric: I_ij = I_ji. Diagonal is left unused.
    """
    n = len(players)
    out: dict[tuple[int, int], float] = {}
    denom = math.factorial(n - 1)
    for i, j in itertools.combinations(players, 2):
        others = [p for p in players if p not in (i, j)]
        acc = 0.0
        for r in range(len(others) + 1):
            w = math.factorial(r) * math.factorial(n - r - 2) / denom
            for combo in itertools.combinations(others, r):
                acc += w * discrete_derivative(values, tuple(sorted(combo)), i, j)
        out[(int(i), int(j))] = acc
        out[(int(j), int(i))] = acc
    return out


def leave_one_out(
    values: ValueTable,
    players: Sequence[int] = GAME_A_PLAYERS,
) -> dict[int, float]:
    """LOO_i = v(N) - v(N\\{i})."""
    grand = tuple(sorted(players))
    v_n = float(values[grand])
    return {
        int(i): v_n - float(values[tuple(sorted(p for p in players if p != i))])
        for i in players
    }


def r_ab(
    values: ValueTable,
    a: int,
    b: int,
    players: Sequence[int] = GAME_A_PLAYERS,
) -> float:
    """Substitution residual. See R_AB_DEFINITION in constants.py."""
    _ = R_AB_DEFINITION
    others = [p for p in players if p not in (a, b)]
    best = 0.0
    for r in range(len(others) + 1):
        for combo in itertools.combinations(others, r):
            c = set(combo)
            v_ab = float(values[tuple(sorted(c | {a, b}))])
            v_a = float(values[tuple(sorted(c | {a}))])
            v_b = float(values[tuple(sorted(c | {b}))])
            best = max(best, abs(v_ab - v_a), abs(v_ab - v_b))
    return best


def max_interchange_gap(
    values: ValueTable,
    a: int,
    b: int,
    players: Sequence[int] = GAME_A_PLAYERS,
) -> float:
    """max_C |v(C∪a) - v(C∪b)| over C ⊆ N\\{a,b}."""
    others = [p for p in players if p not in (a, b)]
    best = 0.0
    for r in range(len(others) + 1):
        for combo in itertools.combinations(others, r):
            c = set(combo)
            va = float(values[tuple(sorted(c | {a}))])
            vb = float(values[tuple(sorted(c | {b}))])
            best = max(best, abs(va - vb))
    return best


def pair_diagnostics(
    values: ValueTable,
    players: Sequence[int] = GAME_A_PLAYERS,
) -> dict[tuple[int, int], dict[str, float]]:
    inter = shapley_interaction(values, players)
    loo = leave_one_out(values, players)
    v_n = float(values[tuple(sorted(players))])
    out: dict[tuple[int, int], dict[str, float]] = {}
    for a, b in itertools.combinations(players, 2):
        out[(int(a), int(b))] = {
            "I": inter[(a, b)],
            "R": r_ab(values, a, b, players),
            "max_gap": max_interchange_gap(values, a, b, players),
            "LOO_a": loo[a],
            "LOO_b": loo[b],
            "v_L": v_n,
        }
    return out

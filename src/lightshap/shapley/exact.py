"""Exact Shapley value for a small explicit game."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from lightshap.constants import EFFICIENCY_ATOL, GAME_A_PLAYERS
from lightshap.exceptions import MathInvariantError

Coalition = tuple[int, ...]
ValueTable = Mapping[Coalition, float]


def all_coalitions(players: Sequence[int]) -> list[Coalition]:
    out: list[Coalition] = []
    for r in range(len(players) + 1):
        for combo in itertools.combinations(players, r):
            out.append(tuple(combo))
    return out


def shapley_weights(n: int) -> dict[int, float]:
    """Weight as a function of coalition size |C| for player i ∉ C."""
    return {
        s: math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
        for s in range(n)
    }


def exact_shapley(
    values: ValueTable,
    players: Sequence[int] = GAME_A_PLAYERS,
    *,
    check_efficiency: bool = True,
    atol: float = EFFICIENCY_ATOL,
) -> dict[int, float]:
    """φ_i = Σ_{C ⊆ N\\{i}} |C|!(n-|C|-1)! / n!  [v(C∪i) - v(C)]."""
    n = len(players)
    weights = shapley_weights(n)
    phi = {int(p): 0.0 for p in players}
    player_set = set(players)
    for i in players:
        others = [p for p in players if p != i]
        for r in range(n):
            for combo in itertools.combinations(others, r):
                c = tuple(sorted(combo))
                cu = tuple(sorted(combo + (i,)))
                w = weights[len(c)]
                phi[int(i)] += w * (float(values[cu]) - float(values[c]))
    if check_efficiency:
        grand = tuple(sorted(player_set))
        total = sum(phi.values())
        target = float(values[grand]) - float(values.get((), 0.0))
        if not math.isclose(total, target, rel_tol=0.0, abs_tol=atol):
            raise MathInvariantError(
                f"Shapley efficiency failed: sum(phi)={total} != v(N)-v(empty)={target}"
            )
    return phi


def shapley_from_array(
    v: Mapping[Coalition, float] | dict[Coalition, np.ndarray],
    players: Sequence[int] = GAME_A_PLAYERS,
) -> dict[int, np.ndarray]:
    """Vectorized Shapley: each v(C) is an array (e.g. per-user)."""
    sample = next(iter(v.values()))
    if not isinstance(sample, np.ndarray):
        scalar = exact_shapley({k: float(val) for k, val in v.items()}, players)
        return {k: np.asarray([val]) for k, val in scalar.items()}
    n = len(players)
    weights = shapley_weights(n)
    phi = {int(p): np.zeros_like(sample, dtype=np.float64) for p in players}
    for i in players:
        others = [p for p in players if p != i]
        for r in range(n):
            for combo in itertools.combinations(others, r):
                c = tuple(sorted(combo))
                cu = tuple(sorted(combo + (i,)))
                phi[int(i)] = phi[int(i)] + weights[len(c)] * (
                    np.asarray(v[cu], dtype=np.float64) - np.asarray(v[c], dtype=np.float64)
                )
    return phi


def mean_matches_global(
    per_user: Mapping[int, np.ndarray],
    global_phi: Mapping[int, float],
    *,
    atol: float = EFFICIENCY_ATOL,
) -> None:
    for k, arr in per_user.items():
        m = float(np.mean(arr))
        if not math.isclose(m, float(global_phi[k]), rel_tol=0.0, abs_tol=atol):
            raise MathInvariantError(
                f"per-user mean of phi[{k}]={m} != global {global_phi[k]}"
            )


def coalition_key(players: Iterable[int]) -> Coalition:
    return tuple(sorted(int(p) for p in players))


def coalition_to_str(c: Iterable[int]) -> str:
    items = [str(int(x)) for x in c]
    return "-".join(items) if items else "empty"


def parse_coalition_key(s: str) -> Coalition:
    """Parse 'empty', '1-2-3', or the Python tuple repr '(1, 2)'."""
    t = s.strip()
    if t in {"", "empty", "()", "tuple()"}:
        return ()
    if t[0] == "(" and t[-1] == ")":
        inner = t[1:-1].strip()
        if not inner:
            return ()
        return tuple(int(x.strip()) for x in inner.split(",") if x.strip())
    return tuple(int(x) for x in t.split("-") if x != "")

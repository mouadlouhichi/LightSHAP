"""Official Game-A substitution flag.

Operationalization *motivated by* the substitution lemma, not a nested
special case of it. Not a pruning license.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from lightshap.constants import (
    FLAG_RELATIVE_TOL,
    GAME_A_PLAYERS,
    STABILITY_FLOOR,
)
from lightshap.shapley.exact import ValueTable
from lightshap.shapley.interactions import pair_diagnostics


class FlagState(StrEnum):
    DEFINED_POSITIVE = "DEFINED"  # pair flagged as approximate substitutes
    DEFINED_NEGATIVE = "NOT_FLAGGED"
    UNDEFINED = "UNDEFINED"
    NONPOSITIVE = "NONPOSITIVE"
    TINY_POSITIVE = "TINY_POSITIVE"


def classify_uplift(v_l: float, floor: float = STABILITY_FLOOR) -> FlagState:
    if v_l < 0.0 or v_l == 0.0:
        return FlagState.NONPOSITIVE
    if v_l < floor:
        return FlagState.TINY_POSITIVE
    return FlagState.DEFINED_NEGATIVE  # defined enough to evaluate pairs


def evaluate_pair(
    diag: Mapping[str, float],
    *,
    floor: float = STABILITY_FLOOR,
    rel_tol: float = FLAG_RELATIVE_TOL,
) -> dict[str, Any]:
    v_l = float(diag["v_L"])
    raw = {
        "I": float(diag["I"]),
        "R": float(diag["R"]),
        "max_gap": float(diag["max_gap"]),
        "LOO_a": float(diag["LOO_a"]),
        "LOO_b": float(diag["LOO_b"]),
        "v_L": v_l,
    }
    clauses = {
        "I_negative": raw["I"] < 0.0,
        "interchange": (raw["max_gap"] / v_l < rel_tol) if v_l > 0 else False,
        "residual": (raw["R"] / v_l < rel_tol) if v_l > 0 else False,
        "loo": (
            max(abs(raw["LOO_a"]), abs(raw["LOO_b"])) / v_l < rel_tol
            if v_l > 0
            else False
        ),
    }
    if v_l < floor:
        state = FlagState.NONPOSITIVE if v_l <= 0.0 else FlagState.TINY_POSITIVE
        return {
            "state": FlagState.UNDEFINED.value,
            "reason": state.value,
            "flagged": False,
            "clauses": clauses,
            "raw": raw,
        }
    flagged = all(clauses.values())
    return {
        "state": (FlagState.DEFINED_POSITIVE if flagged else FlagState.DEFINED_NEGATIVE).value,
        "reason": None,
        "flagged": flagged,
        "clauses": clauses,
        "raw": raw,
    }


def evaluate_game_a_flag(
    values: ValueTable,
    players: Sequence[int] = GAME_A_PLAYERS,
    *,
    floor: float = STABILITY_FLOOR,
    rel_tol: float = FLAG_RELATIVE_TOL,
) -> dict[str, Any]:
    v_l = float(values[tuple(sorted(players))])
    uplift_state = classify_uplift(v_l, floor)
    diags = pair_diagnostics(values, players)
    pairs: dict[str, Any] = {}
    any_flagged = False
    for (a, b), diag in diags.items():
        rec = evaluate_pair(diag, floor=floor, rel_tol=rel_tol)
        pairs[f"{a}-{b}"] = rec
        any_flagged = any_flagged or bool(rec["flagged"])
    # Co-occurrence of the four clauses (NOT a correlation; N=3 pairs).
    names = ["I_negative", "interchange", "residual", "loo"]
    cooccur = []
    for key, rec in pairs.items():
        row = {"pair": key, **{n: bool(rec["clauses"][n]) for n in names}}
        row["all_four"] = all(row[n] for n in names)
        cooccur.append(row)
    return {
        "v_L": v_l,
        "uplift_state": uplift_state.value,
        "undefined": v_l < floor,
        "any_pair_flagged": any_flagged and v_l >= floor,
        "pairs": pairs,
        "cooccurrence": cooccur,
        "note": (
            "Flag is not a pruning claim. Retrained K is the architecture control. "
            "Motivated by the substitution lemma, not an instance of it."
        ),
    }


def any_pair_flagged(flag_result: Mapping[str, Any]) -> bool:
    return bool(flag_result.get("any_pair_flagged"))

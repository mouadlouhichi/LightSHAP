"""Game-B trigger. Exact A.11 freeze.

Trigger iff:
  * no Game-A pair is flagged, AND
  * there exists a side on which (E0, E2) is strictly the unique maximum
    off-diagonal AND exceeds the second-largest off-diagonal on that same
    side by >= 0.05.
If two off-diagonals tie for max, that side does not fire.
Either side is sufficient.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from lightshap.constants import GAME_B_COSINE_MARGIN


def off_diagonal_entries(mat: np.ndarray) -> list[tuple[int, int, float]]:
    n = mat.shape[0]
    out: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append((i, j, float(mat[i, j])))
    return out


def side_fires(
    cosine: np.ndarray,
    *,
    pair: tuple[int, int] = (0, 2),
    margin: float = GAME_B_COSINE_MARGIN,
) -> dict[str, float | bool | None]:
    entries = off_diagonal_entries(cosine)
    if not entries:
        return {"fires": False, "reason": "empty", "max": None, "second": None}
    values = [e[2] for e in entries]
    max_val = max(values)
    winners = [e for e in entries if e[2] == max_val]
    if len(winners) != 1:
        return {
            "fires": False,
            "reason": "tie_for_max",
            "max": max_val,
            "second": max_val,
            "winners": [(a, b) for a, b, _ in winners],
        }
    w_i, w_j, _ = winners[0]
    a, b = pair
    if {w_i, w_j} != {a, b}:
        return {
            "fires": False,
            "reason": "unique_max_is_not_E0_E2",
            "max": max_val,
            "winner": [w_i, w_j],
        }
    # second-largest among the *other* off-diagonals
    others = [e[2] for e in entries if {e[0], e[1]} != {a, b}]
    if not others:
        second = float("-inf")
    else:
        second = max(others)
    fires = (max_val - second) >= margin
    return {
        "fires": bool(fires),
        "reason": "ok" if fires else "margin_too_small",
        "max": max_val,
        "second": second,
        "margin": float(max_val - second),
        "required": margin,
    }


def game_b_trigger(
    *,
    any_game_a_flag: bool,
    cos_user: np.ndarray,
    cos_item: np.ndarray,
    margin: float = GAME_B_COSINE_MARGIN,
) -> dict[str, object]:
    if any_game_a_flag:
        return {
            "triggered": False,
            "reason": "game_a_pair_flagged",
            "user": None,
            "item": None,
        }
    user = side_fires(cos_user, margin=margin)
    item = side_fires(cos_item, margin=margin)
    triggered = bool(user["fires"] or item["fires"])
    return {
        "triggered": triggered,
        "reason": "side_fire" if triggered else "no_side_met_unique_max_and_margin",
        "user": user,
        "item": item,
        "note": "Game B credits are not comparable to Game A.",
    }


def popularity_rank(train_item_freq: Mapping[int, int], n_items: int) -> list[int]:
    """Train frequency; ties → lower item id."""
    return sorted(
        range(n_items),
        key=lambda i: (-int(train_item_freq.get(i, 0)), i),
    )

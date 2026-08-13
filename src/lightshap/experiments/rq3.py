"""RQ3: Δ = D_Q4 - D_Q1 iff both v_m ≥ 0.005."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lightshap.experiments.segments import segment_tables
from lightshap.gates.rq3 import interpret_D, rq3_delta


def run_rq3(bundle: Any, seed_games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not seed_games:
        return {"status": "NOT_RUN"}
    per_seed = []
    for g in seed_games:
        seg = segment_tables(bundle, g)
        groups = seg["groups"]
        delta = rq3_delta(
            phi_q1={
                2: groups["1"]["phi_2"],
                3: groups["1"]["phi_3"],
            },
            phi_q4={
                2: groups["4"]["phi_2"],
                3: groups["4"]["phi_3"],
            },
            v_q1=groups["1"]["v_m"],
            v_q4=groups["4"]["v_m"],
        )
        per_seed.append({"segments": seg, "delta": delta})
    # confirmatory contrast uses seed-averaged group stats of the first dump;
    # also expose mean Δ when every seed is eligible
    eligible = [p["delta"] for p in per_seed if p["delta"]["eligible"]]
    mean_delta = None
    if eligible and all(p["delta"]["eligible"] for p in per_seed):
        mean_delta = sum(float(p["delta"]["delta"]) for p in per_seed) / len(per_seed)
    return {
        "status": "VALID",
        "per_seed": per_seed,
        "mean_delta_if_all_eligible": mean_delta,
        "interpretation": {
            "negative_D": interpret_D(-1.0),
            "nonnegative_D": interpret_D(1.0),
        },
        "note": "Beauty |U|<1000 ⇒ this RQ3 result is exploratory on Beauty.",
    }

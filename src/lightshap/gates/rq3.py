"""RQ3 mix contrast. Δ only if both extreme groups have v_m ≥ 0.005."""

from __future__ import annotations

from collections.abc import Mapping

from lightshap.constants import STABILITY_FLOOR


def extra_deep_mix(phi: Mapping[int, float]) -> float:
    """D_m = φ_2^(m) + φ_3^(m)."""
    return float(phi.get(2, 0.0)) + float(phi.get(3, 0.0))


def rq3_delta(
    *,
    phi_q1: Mapping[int, float],
    phi_q4: Mapping[int, float],
    v_q1: float,
    v_q4: float,
    floor: float = STABILITY_FLOOR,
) -> dict[str, object]:
    d1 = extra_deep_mix(phi_q1)
    d4 = extra_deep_mix(phi_q4)
    eligible = (v_q1 >= floor) and (v_q4 >= floor)
    out: dict[str, object] = {
        "eligible": eligible,
        "v_Q1": float(v_q1),
        "v_Q4": float(v_q4),
        "D_Q1": d1,
        "D_Q4": d4,
        "floor": floor,
        "negative_D_means": "deep extra hops hurt that group",
    }
    if not eligible:
        out["delta"] = None
        out["claim"] = "uplift too small for a mix comparison"
        out["raw_only"] = True
        return out
    out["delta"] = d4 - d1
    out["claim"] = "delta_defined"
    out["raw_only"] = False
    out["D_over_v_Q1"] = d1 / v_q1
    out["D_over_v_Q4"] = d4 / v_q4
    return out


def interpret_D(d_m: float) -> str:
    if d_m < 0:
        return "deep extra hops hurt that group"
    return "deep extra hops contribute nonnegatively"

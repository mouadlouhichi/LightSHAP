"""Beauty |U|<1000 contingency."""

from __future__ import annotations

from lightshap.constants import BEAUTY_USER_FLOOR, BEAUTY_YEAR, FORBIDDEN_BEAUTY_LABEL


def beauty_status(n_users: int, floor: int = BEAUTY_USER_FLOOR) -> dict[str, object]:
    exploratory = int(n_users) < int(floor)
    return {
        "n_users": int(n_users),
        "floor": int(floor),
        "exploratory": exploratory,
        "rq3_confirmatory_dataset": "ml1m" if exploratory else "both",
        "abstract_template": "B" if exploratory else "A",
        "year": BEAUTY_YEAR,
        "label": f"Amazon Beauty ({BEAUTY_YEAR} 5-core)",
        "forbidden_label": FORBIDDEN_BEAUTY_LABEL,
    }


def abstract_must_not_say_2018(text: str) -> None:
    if FORBIDDEN_BEAUTY_LABEL.lower() in text.lower():
        raise ValueError(f"abstract must never contain {FORBIDDEN_BEAUTY_LABEL!r}")

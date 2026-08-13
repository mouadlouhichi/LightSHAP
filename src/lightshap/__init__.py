"""LightShap: inference-time hop-inclusion attribution on a frozen LightGCN."""

from lightshap.constants import (
    ADAPTIVE_BETA,
    FLAG_RELATIVE_TOL,
    GAME_A_PLAYERS,
    LIGHTGCN_FROZEN,
    N_SEEDS_SCIENTIFIC,
    N_VAL_ALPHA,
    STABILITY_FLOOR,
)

__version__ = "0.1.0"

__all__ = [
    "ADAPTIVE_BETA",
    "FLAG_RELATIVE_TOL",
    "GAME_A_PLAYERS",
    "LIGHTGCN_FROZEN",
    "N_VAL_ALPHA",
    "N_SEEDS_SCIENTIFIC",
    "STABILITY_FLOOR",
    "__version__",
]

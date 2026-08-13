"""Frozen scientific invariants.

These values are the source of truth from
``specs/LightShap_Implementation_Spec (5).md``.  Configuration may *repeat*
them but may not silently override them under ``profile: scientific``.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Estimand
# ---------------------------------------------------------------------------
ESTIMAND: Final[str] = (
    "inference-time extra-hop inclusion on one frozen uniform-K=3, d=64 LightGCN"
)
GAME_A_PLAYERS: Final[tuple[int, ...]] = (1, 2, 3)
E0_SLOT: Final[int] = 0
N_GAME_A_PLAYERS: Final[int] = 3
N_GAME_A_COALITIONS: Final[int] = 8  # 2^3, E0 always on
N_FUSION_SLOTS: Final[int] = 4  # E0..E3
K_FROZEN: Final[int] = 3

# ---------------------------------------------------------------------------
# LightGCN freeze (not searched)
# ---------------------------------------------------------------------------
LIGHTGCN_FROZEN: Final[dict[str, float | int]] = {
    "K": 3,
    "d": 64,
    "lr": 1.0e-3,
    "reg": 1.0e-4,
    "patience": 20,
    "max_epochs": 1000,
}
LIGHTGCN_BATCH: Final[dict[str, int]] = {"ml1m": 2048, "beauty": 1024, "synthetic": 64}

# ---------------------------------------------------------------------------
# BPR-MF six-point grid (A.5). Locked. Not "equal cardinality".
# ---------------------------------------------------------------------------
BPR_GRID: Final[tuple[dict[str, float | int], ...]] = (
    {"d": 32, "lr": 1.0e-3, "reg": 1.0e-4},
    {"d": 64, "lr": 1.0e-3, "reg": 1.0e-4},
    {"d": 128, "lr": 1.0e-3, "reg": 1.0e-4},
    {"d": 32, "lr": 5.0e-4, "reg": 1.0e-4},
    {"d": 64, "lr": 5.0e-4, "reg": 1.0e-4},
    {"d": 128, "lr": 5.0e-4, "reg": 1.0e-4},
)
BPR_MAX_EPOCHS: Final[int] = 1000
BPR_PATIENCE: Final[int] = 20
NEG_POOL_MAIN: Final[str] = "heldout_excluded"
NEG_POOL_SENSITIVITY: Final[str] = "train_only"
NEG_POOL_CHOICES: Final[tuple[str, ...]] = (NEG_POOL_MAIN, NEG_POOL_SENSITIVITY)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
STABILITY_FLOOR: Final[float] = 0.005
FLAG_RELATIVE_TOL: Final[float] = 0.15
FLAG_SENSITIVITY_TOLS: Final[tuple[float, float]] = (0.10, 0.20)
GAME_B_COSINE_MARGIN: Final[float] = 0.05
ADAPTIVE_BETA: Final[float] = 0.25
N_VAL_ALPHA: Final[int] = 286  # compositions of 10 into 4 parts
VAL_ALPHA_GRID: Final[int] = 10
UNIFORM_4: Final[tuple[float, float, float, float]] = (0.25, 0.25, 0.25, 0.25)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
PRIMARY_K: Final[int] = 10
CUTOFFS: Final[tuple[int, ...]] = (5, 10, 20)
N_SEEDS_SCIENTIFIC: Final[int] = 5
RETRAINED_K_VALUES: Final[tuple[int, ...]] = (1, 2, 3, 4)
BEAUTY_USER_FLOOR: Final[int] = 1000
N_HOLM_TESTS_PER_DATASET: Final[int] = 3
RQ4_CONTRASTS: Final[tuple[str, ...]] = (
    "adaptive_vs_uniform",
    "val_alpha_vs_uniform",
    "adaptive_vs_val_alpha",
)
# Engineering fill (spec lists "3 tests" without naming them). Documented in
# experiments/rq4.py and docs/implementation_matrix.md.

# ---------------------------------------------------------------------------
# Fixtures (independently verified; I_12 = 1/2, NOT 1)
# ---------------------------------------------------------------------------
FIXTURE_PHI: Final[tuple[float, float, float]] = (5.0 / 3.0, 8.0 / 3.0, 11.0 / 3.0)
FIXTURE_I: Final[float] = 0.5
FORBIDDEN_I12: Final[float] = 1.0

# Möbius-constructed 3-player game with φ=(5/3,8/3,11/3) and I_ij=1/2.
# m_i = φ_i - 1/2, m_{ij}=1/2, m_N=0.  See shapley/fixtures.py.
FIXTURE_V: Final[dict[tuple[int, ...], float]] = {
    (): 0.0,
    (1,): 7.0 / 6.0,
    (2,): 13.0 / 6.0,
    (3,): 19.0 / 6.0,
    (1, 2): 23.0 / 6.0,
    (1, 3): 29.0 / 6.0,
    (2, 3): 35.0 / 6.0,
    (1, 2, 3): 8.0,
}

# ---------------------------------------------------------------------------
# Data sources (do not silently replace)
# ---------------------------------------------------------------------------
ML1M_URL: Final[str] = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
BEAUTY_URL: Final[str] = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "reviews_Beauty_5.json.gz"
)
BEAUTY_FILENAME: Final[str] = "reviews_Beauty_5.json.gz"
BEAUTY_YEAR: Final[int] = 2014
FORBIDDEN_BEAUTY_LABEL: Final[str] = "2018 All-Beauty"

# ---------------------------------------------------------------------------
# Ranking / scoring
# ---------------------------------------------------------------------------
SCORE_DTYPE: Final[str] = "float32"
# Do NOT add 1e-12 * item_id jitter. Lexicographic ties only.

# ---------------------------------------------------------------------------
# Engineering fills for underspecified formulae (documented, not silent)
# ---------------------------------------------------------------------------
R_AB_DEFINITION: Final[str] = (
    "R_ab = max_{C ⊆ L\\{a,b}} max(|v(C∪{a,b})-v(C∪a)|, |v(C∪{a,b})-v(C∪b)|). "
    "The C = L\\{a,b} slice equals max(|LOO_a|, |LOO_b|), matching the spec's "
    "'LOO is a redundant-by-design restatement of the grand-coalition slice of R_ab'."
)
ADAPTIVE_Q_DEFINITION: Final[str] = (
    "If v_m < 0.005: q = (1,0,0,0). Else: build a 4-vector attr with "
    "attr[0] = mean(relu(φ)) and attr[k] = relu(φ_{k}) for k=1,2,3 "
    "(fallback attr=(1,0,0,0) if all relu(φ)=0); normalize; "
    "q = (1-β)*uniform_4 + β*attr; renormalize. β=1/4. No per-group β_m."
)
GAME_B_DEFINITION: Final[str] = (
    "All-slot game on players {0,1,2,3}. Different zero: empty coalition scores "
    "are the zero vector (ties broken by item_id, so ranking is by item_id). "
    "v_B(C) = NDCG@10(mean_{k in C} E_k) - NDCG@10(zeros). Credits are not "
    "comparable to Game A. Computed only if the A.11 trigger fires."
)

EFFICIENCY_ATOL: Final[float] = 1e-6
NUMERIC_ATOL: Final[float] = 1e-8

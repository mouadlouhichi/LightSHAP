from __future__ import annotations

import numpy as np
import pytest

from lightshap.constants import N_VAL_ALPHA, STABILITY_FLOOR
from lightshap.gates.adaptive import adaptive_weights
from lightshap.gates.beauty import abstract_must_not_say_2018, beauty_status
from lightshap.gates.cosine import cosine_matrix_4x4, flattened_cosine
from lightshap.gates.game_b import game_b_trigger, side_fires
from lightshap.gates.rq3 import rq3_delta
from lightshap.gates.val_alpha import (
    break_val_alpha_ties,
    uniform_l2,
    val_alpha_candidates,
)


@pytest.mark.unit
@pytest.mark.regression
def test_val_alpha_count_is_286() -> None:
    c = val_alpha_candidates()
    assert len(c) == 286 == N_VAL_ALPHA
    assert all(abs(sum(w) - 1.0) < 1e-12 for w in c)


@pytest.mark.unit
def test_val_alpha_tie_uniform_then_lex() -> None:
    # two candidates with identical NDCG
    a = (0.4, 0.2, 0.2, 0.2)
    b = (0.25, 0.25, 0.25, 0.25)
    chosen = break_val_alpha_ties([(a, 0.1), (b, 0.1)])
    assert chosen == b
    # remaining ties: lex (w0,w1,w2)
    c1 = (0.5, 0.3, 0.1, 0.1)
    c2 = (0.5, 0.2, 0.2, 0.1)
    # same L2? force equal distance by using same vector family
    chosen2 = break_val_alpha_ties([(c1, 0.2), (c2, 0.2)])
    # closer to uniform wins; if not, lex
    if abs(uniform_l2(c1) - uniform_l2(c2)) < 1e-15:
        assert chosen2 == min((c1, c2), key=lambda w: (w[0], w[1], w[2]))
    else:
        assert chosen2 == (c1 if uniform_l2(c1) < uniform_l2(c2) else c2)


@pytest.mark.unit
def test_adaptive_floor_e0_only() -> None:
    q = adaptive_weights({1: 0.1, 2: 0.1, 3: 0.1}, v_m=0.004)
    assert q.tolist() == [1.0, 0.0, 0.0, 0.0]
    q2 = adaptive_weights({1: 0.1, 2: 0.2, 3: 0.3}, v_m=STABILITY_FLOOR)
    assert abs(q2.sum() - 1.0) < 1e-12
    assert q2[0] < 1.0


@pytest.mark.unit
def test_rq3_ineligible() -> None:
    rec = rq3_delta(
        phi_q1={2: 0.1, 3: 0.1},
        phi_q4={2: 0.2, 3: 0.2},
        v_q1=0.001,
        v_q4=0.01,
    )
    assert rec["eligible"] is False
    assert rec["delta"] is None
    rec2 = rq3_delta(
        phi_q1={2: -0.1, 3: -0.1},
        phi_q4={2: 0.2, 3: 0.2},
        v_q1=0.01,
        v_q4=0.01,
    )
    assert rec2["eligible"] is True
    assert rec2["D_Q1"] < 0


@pytest.mark.unit
def test_beauty_contingency() -> None:
    st = beauty_status(800)
    assert st["exploratory"] is True
    assert st["abstract_template"] == "B"
    assert st["rq3_confirmatory_dataset"] == "ml1m"
    with pytest.raises(ValueError):
        abstract_must_not_say_2018("we use 2018 All-Beauty")


@pytest.mark.unit
def test_cosine_no_subsample_deterministic() -> None:
    rng = np.random.default_rng(0)
    layers = [rng.normal(size=(10, 4)) for _ in range(4)]
    deg_u = np.array([1, 2, 0, 3, 1])
    a = cosine_matrix_4x4(layers, n_users=5, n_items=5, side="user", train_degree=deg_u)
    b = cosine_matrix_4x4(layers, n_users=5, n_items=5, side="user", train_degree=deg_u)
    assert a.shape == (4, 4)
    assert np.allclose(a, b)
    assert np.allclose(a, a.T, equal_nan=True)
    # dropped the zero-degree row (index 2)
    assert flattened_cosine(np.ones((2, 2)), np.ones((2, 2))) == pytest.approx(1.0)


@pytest.mark.unit
def test_game_b_trigger_boundaries() -> None:
    # 4x4 identity → off-diag all 0, tie for max → no fire
    ident = np.eye(4)
    t = game_b_trigger(any_game_a_flag=False, cos_user=ident, cos_item=ident)
    assert t["triggered"] is False

    # unique max at (0,2) with margin exactly 0.05 (constructed, not 0.85-0.80)
    m = np.eye(4)
    second = 0.80
    m[0, 1] = m[1, 0] = second
    m[0, 2] = m[2, 0] = second + 0.05
    assert side_fires(m)["fires"] is True

    # margin just below 0.05 → no
    m2 = m.copy()
    m2[0, 2] = m2[2, 0] = second + 0.05 - 1e-9
    assert side_fires(m2)["fires"] is False

    # tie for max
    m3 = np.eye(4)
    m3[0, 2] = m3[2, 0] = 0.9
    m3[0, 1] = m3[1, 0] = 0.9
    assert side_fires(m3)["fires"] is False
    assert side_fires(m3)["reason"] == "tie_for_max"

    # unique max is not (E0,E2)
    m4 = np.eye(4)
    m4[1, 2] = m4[2, 1] = 0.99
    m4[0, 2] = m4[2, 0] = 0.5
    assert side_fires(m4)["fires"] is False

    # Game A flag blocks even if cosine would fire
    t2 = game_b_trigger(any_game_a_flag=True, cos_user=m, cos_item=ident)
    assert t2["triggered"] is False
    assert t2["reason"] == "game_a_pair_flagged"

    # either side sufficient
    t3 = game_b_trigger(any_game_a_flag=False, cos_user=ident, cos_item=m)
    assert t3["triggered"] is True

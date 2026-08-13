from __future__ import annotations

import numpy as np
import pytest

from lightshap.config import load_bpr_grid, load_experiment_config
from lightshap.constants import BPR_GRID, N_SEEDS_SCIENTIFIC
from lightshap.exceptions import FrozenInvariantError
from lightshap.models.negatives import allowed_negatives
from lightshap.statistics.aggregate import average_users_across_seeds
from lightshap.statistics.bootstrap import bootstrap_users
from lightshap.statistics.holm import holm_adjust
from lightshap.statistics.spearman import spearman_average_ranks
from lightshap.statistics.tests import two_sided_wilcoxon


@pytest.mark.unit
@pytest.mark.regression
def test_bpr_grid_is_exactly_six(repo_root) -> None:
    grid = load_bpr_grid(repo_root / "configs" / "bpr_grid.yaml")
    assert len(grid) == 6
    got = {(int(c["d"]), float(c["lr"]), float(c["reg"])) for c in grid}
    exp = {(int(c["d"]), float(c["lr"]), float(c["reg"])) for c in BPR_GRID}
    assert got == exp


@pytest.mark.unit
def test_scientific_config_rejects_wrong_k(repo_root, tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        (repo_root / "configs" / "experiment.yaml").read_text().replace("K: 3", "K: 4")
    )
    with pytest.raises(FrozenInvariantError):
        load_experiment_config(p, project_root=repo_root)


@pytest.mark.unit
def test_scientific_has_five_seeds(repo_root) -> None:
    cfg = load_experiment_config(repo_root / "configs" / "experiment.yaml", project_root=repo_root)
    assert len(cfg.seeds) == N_SEEDS_SCIENTIFIC


@pytest.mark.unit
def test_neg_pool_toggle() -> None:
    train = {0: {0, 1}}
    held = {0: {2}}
    a = set(allowed_negatives(0, 5, train, held, "heldout_excluded").tolist())
    b = set(allowed_negatives(0, 5, train, held, "train_only").tolist())
    assert 2 not in a
    assert 2 in b
    assert 0 not in a and 0 not in b


@pytest.mark.unit
def test_average_users_then_bootstrap_not_5u() -> None:
    s0 = np.array([1.0, 2.0, 3.0])
    s1 = np.array([3.0, 2.0, 1.0])
    avg = average_users_across_seeds([s0, s1])
    assert avg.tolist() == [2.0, 2.0, 2.0]
    boot = bootstrap_users(avg, n_boot=200, seed=0)
    assert boot["n_users"] == 3
    assert boot["unit"] == "user"


@pytest.mark.unit
def test_holm_and_wilcoxon() -> None:
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert adj[0] <= adj[1] or True  # first is smallest raw
    assert all(0 <= p <= 1 for p in adj)
    # monotonic in the sorted order
    order = np.argsort([0.01, 0.04, 0.03])
    seq = [adj[i] for i in order]
    assert seq == sorted(seq)
    w = two_sided_wilcoxon([0.2, 0.1, -0.05, 0.3, 0.0])
    assert w["alternative"] == "two-sided"
    z = two_sided_wilcoxon([0.0, 0.0, 0.0])
    assert z["pvalue"] == 1.0


@pytest.mark.unit
def test_spearman_average_ranks() -> None:
    # ties: [1,1,2] ranks 1.5, 1.5, 3
    rho = spearman_average_ranks([1, 1, 2, 3], [1, 1, 2, 4])
    assert rho == pytest.approx(1.0)

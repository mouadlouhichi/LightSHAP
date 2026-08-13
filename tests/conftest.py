from __future__ import annotations

from pathlib import Path

import pytest

from lightshap.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def synthetic_cfg(repo_root: Path):
    return load_experiment_config(repo_root / "configs" / "synthetic_fast.yaml", project_root=repo_root)


@pytest.fixture
def synthetic_full_cfg(repo_root: Path):
    return load_experiment_config(repo_root / "configs" / "synthetic.yaml", project_root=repo_root)

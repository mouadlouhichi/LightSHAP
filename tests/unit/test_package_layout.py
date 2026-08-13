"""Guard against .gitignore swallowing source packages."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "mod",
    [
        "lightshap.checkpoints",
        "lightshap.checkpoints.store",
        "lightshap.cache",
        "lightshap.pipeline",
        "lightshap.pipeline.stages",
        "lightshap.pipeline.runner",
        "lightshap.shapley",
        "lightshap.models",
    ],
)
def test_core_packages_importable(mod: str) -> None:
    importlib.import_module(mod)

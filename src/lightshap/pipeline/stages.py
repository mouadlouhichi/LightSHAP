"""Explicit stage graph. The runner refuses to run a stage with invalid deps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    name: str
    dependencies: tuple[str, ...] = ()
    description: str = ""
    expensive: bool = False
    seed_level: bool = False


STAGES: tuple[StageSpec, ...] = (
    StageSpec("ENVIRONMENT", (), "Python / package / device probe"),
    StageSpec("CONFIG_VALIDATION", ("ENVIRONMENT",), "Load and freeze-check config"),
    StageSpec("DATA_DOWNLOAD", ("CONFIG_VALIDATION",), "Fetch raw files + checksum"),
    StageSpec("DATA_VALIDATION", ("DATA_DOWNLOAD",), "Integrity + schema"),
    StageSpec("DATA_PREPROCESSING", ("DATA_VALIDATION",), "5-core, split, graph, quartiles"),
    StageSpec("FIXTURE_GENERATION", ("CONFIG_VALIDATION",), "Write Shapley gadgets"),
    StageSpec("FIXTURE_TESTS", ("FIXTURE_GENERATION",), "φ=(5/3,8/3,11/3), I=1/2"),
    StageSpec(
        "BPR_GRID",
        ("DATA_PREPROCESSING", "FIXTURE_TESTS"),
        "Six-point BPR-MF grid",
        expensive=True,
        seed_level=True,
    ),
    StageSpec(
        "LIGHTGCN_TRAINING",
        ("DATA_PREPROCESSING", "FIXTURE_TESTS"),
        "Frozen LightGCN + optional retrained K",
        expensive=True,
        seed_level=True,
    ),
    StageSpec(
        "SHAPLEY_COMPUTATION",
        ("LIGHTGCN_TRAINING",),
        "Game A (and Game B if triggered)",
        expensive=True,
        seed_level=True,
    ),
    StageSpec("RQ1", ("SHAPLEY_COMPUTATION",), "v(C) Spearman + φ stability"),
    StageSpec("RQ2", ("SHAPLEY_COMPUTATION",), "Flag, LOO, retrained K"),
    StageSpec("RQ3", ("SHAPLEY_COMPUTATION",), "Δ mix contrast with floor"),
    StageSpec("RQ4", ("SHAPLEY_COMPUTATION",), "Adaptive + Val-α transfer"),
    StageSpec("STATISTICS", ("RQ1", "RQ2", "RQ3", "RQ4"), "Cross-RQ packaging"),
    StageSpec("TABLE_GENERATION", ("STATISTICS",), "JSON/LaTeX tables"),
    StageSpec("FIGURE_GENERATION", ("STATISTICS",), "Figures"),
    StageSpec("FINAL_REPORT", ("TABLE_GENERATION", "FIGURE_GENERATION"), "summary + compliance"),
)

STAGE_INDEX = {s.name: s for s in STAGES}
STAGE_ORDER = [s.name for s in STAGES]


def dependencies_of(name: str) -> tuple[str, ...]:
    return STAGE_INDEX[name].dependencies

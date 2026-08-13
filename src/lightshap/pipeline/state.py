"""Pipeline state machine. Explicit stage states, atomic writes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from lightshap.utils.io import atomic_write_json, read_json
from lightshap.utils.provenance import utc_now


class StageState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INVALIDATED = "INVALIDATED"
    SKIPPED = "SKIPPED"
    NOT_RUN = "NOT_RUN"


class ResultState(StrEnum):
    VALID = "VALID"
    UNDEFINED = "UNDEFINED"
    EXPLORATORY = "EXPLORATORY"
    NOT_RUN = "NOT_RUN"
    FAILED = "FAILED"


@dataclass
class StageRecord:
    name: str
    state: str = StageState.PENDING.value
    started_at: str | None = None
    ended_at: str | None = None
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineState:
    run_id: str
    config_hash: str
    dataset_hashes: dict[str, str] = field(default_factory=dict)
    stages: dict[str, StageRecord] = field(default_factory=dict)
    completed_seeds: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "dataset_hashes": self.dataset_hashes,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "completed_seeds": self.completed_seeds,
            "warnings": self.warnings,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineState:
        stages = {
            k: StageRecord(**v) if not isinstance(v, StageRecord) else v
            for k, v in (d.get("stages") or {}).items()
        }
        return cls(
            run_id=d["run_id"],
            config_hash=d["config_hash"],
            dataset_hashes=d.get("dataset_hashes") or {},
            stages=stages,
            completed_seeds=d.get("completed_seeds") or {},
            warnings=d.get("warnings") or [],
            updated_at=d.get("updated_at") or "",
        )


def state_path(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "pipeline_state.json"


def load_state(run_dir: Path) -> PipelineState | None:
    p = state_path(run_dir)
    if not p.is_file():
        return None
    return PipelineState.from_dict(read_json(p))


def save_state(run_dir: Path, state: PipelineState) -> None:
    state.updated_at = utc_now()
    atomic_write_json(state_path(run_dir), state.to_dict())

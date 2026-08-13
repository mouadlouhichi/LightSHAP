"""Configuration loading and frozen-invariant validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from lightshap.constants import (
    ADAPTIVE_BETA,
    BEAUTY_URL,
    BEAUTY_USER_FLOOR,
    BEAUTY_YEAR,
    BPR_GRID,
    BPR_MAX_EPOCHS,
    BPR_PATIENCE,
    FLAG_RELATIVE_TOL,
    GAME_B_COSINE_MARGIN,
    K_FROZEN,
    LIGHTGCN_FROZEN,
    ML1M_URL,
    N_SEEDS_SCIENTIFIC,
    N_VAL_ALPHA,
    NEG_POOL_CHOICES,
    NEG_POOL_MAIN,
    PRIMARY_K,
    RETRAINED_K_VALUES,
    STABILITY_FLOOR,
)
from lightshap.exceptions import ConfigError, FrozenInvariantError

Profile = Literal["scientific", "synthetic"]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    url: str
    filename: str
    expected_sha256: str | None
    implicit_threshold: float | None  # rating >= t → positive; None = all positive
    batch_size: int
    year: int | None = None


@dataclass(frozen=True)
class LightGCNHParams:
    K: int = 3
    d: int = 64
    lr: float = 1.0e-3
    reg: float = 1.0e-4
    patience: int = 20
    max_epochs: int = 1000
    eval_every: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    profile: Profile
    project_root: Path
    run_name: str
    seeds: tuple[int, ...]
    datasets: tuple[str, ...]
    device: str
    neg_pool: str
    cutoff: int
    cutoffs: tuple[int, ...]
    n_negatives: int
    lightgcn: LightGCNHParams
    bpr_max_epochs: int
    bpr_patience: int
    stability_floor: float
    flag_relative_tol: float
    game_b_margin: float
    adaptive_beta: float
    beauty_user_floor: int
    retrained_k: tuple[int, ...]
    n_val_alpha: int
    run_bpr_grid: bool
    run_retrained_k: bool
    run_game_b_if_triggered: bool
    allow_missing_sha256: bool
    force: bool
    num_workers: int
    log_every: int
    checkpoint_every_epoch: bool
    eval_every: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["project_root"] = str(self.project_root)
        d["seeds"] = list(self.seeds)
        d["datasets"] = list(self.datasets)
        d["cutoffs"] = list(self.cutoffs)
        d["retrained_k"] = list(self.retrained_k)
        return d

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return _sha256_bytes(payload)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists() and (p / "configs").exists():
            return p
    return Path.cwd()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return data


def default_dataset_configs() -> dict[str, DatasetConfig]:
    return {
        "ml1m": DatasetConfig(
            name="ml1m",
            url=ML1M_URL,
            filename="ml-1m.zip",
            expected_sha256=None,
            implicit_threshold=4.0,
            batch_size=2048,
            year=None,
        ),
        "beauty": DatasetConfig(
            name="beauty",
            url=BEAUTY_URL,
            filename="reviews_Beauty_5.json.gz",
            expected_sha256=None,
            implicit_threshold=None,
            batch_size=1024,
            year=BEAUTY_YEAR,
        ),
        "synthetic": DatasetConfig(
            name="synthetic",
            url="",
            filename="",
            expected_sha256=None,
            implicit_threshold=None,
            batch_size=64,
            year=None,
        ),
    }


def validate_frozen(cfg: ExperimentConfig) -> None:
    """Fail early if a scientific freeze is inconsistent."""
    if cfg.neg_pool not in NEG_POOL_CHOICES:
        raise FrozenInvariantError(
            f"neg_pool={cfg.neg_pool!r} not in {NEG_POOL_CHOICES}"
        )
    if cfg.cutoff != PRIMARY_K:
        raise FrozenInvariantError(f"primary cutoff must be {PRIMARY_K}, got {cfg.cutoff}")
    if abs(cfg.stability_floor - STABILITY_FLOOR) > 1e-15:
        raise FrozenInvariantError(
            f"stability_floor must be {STABILITY_FLOOR}, got {cfg.stability_floor}"
        )
    if abs(cfg.flag_relative_tol - FLAG_RELATIVE_TOL) > 1e-15:
        raise FrozenInvariantError(
            f"flag_relative_tol must be {FLAG_RELATIVE_TOL}, got {cfg.flag_relative_tol}"
        )
    if abs(cfg.game_b_margin - GAME_B_COSINE_MARGIN) > 1e-15:
        raise FrozenInvariantError(
            f"game_b_margin must be {GAME_B_COSINE_MARGIN}, got {cfg.game_b_margin}"
        )
    if abs(cfg.adaptive_beta - ADAPTIVE_BETA) > 1e-15:
        raise FrozenInvariantError(
            f"adaptive_beta must be {ADAPTIVE_BETA}, got {cfg.adaptive_beta}"
        )
    if cfg.n_val_alpha != N_VAL_ALPHA:
        raise FrozenInvariantError(f"n_val_alpha must be {N_VAL_ALPHA}")
    if tuple(cfg.retrained_k) != RETRAINED_K_VALUES:
        raise FrozenInvariantError(f"retrained_k must be {RETRAINED_K_VALUES}")
    if cfg.beauty_user_floor != BEAUTY_USER_FLOOR:
        raise FrozenInvariantError(f"beauty_user_floor must be {BEAUTY_USER_FLOOR}")

    if cfg.profile == "scientific":
        if len(cfg.seeds) != N_SEEDS_SCIENTIFIC:
            raise FrozenInvariantError(
                f"scientific profile requires {N_SEEDS_SCIENTIFIC} seeds, got {len(cfg.seeds)}"
            )
        lg = cfg.lightgcn
        expected = LIGHTGCN_FROZEN
        for key, val in expected.items():
            got = getattr(lg, key)
            if isinstance(val, float):
                if abs(float(got) - float(val)) > 1e-15:
                    raise FrozenInvariantError(f"LightGCN.{key} must be {val}, got {got}")
            elif int(got) != int(val):
                raise FrozenInvariantError(f"LightGCN.{key} must be {val}, got {got}")
        if cfg.bpr_max_epochs != BPR_MAX_EPOCHS or cfg.bpr_patience != BPR_PATIENCE:
            raise FrozenInvariantError("BPR patience/epochs deviate from freeze")
        if set(cfg.datasets) - {"ml1m", "beauty"}:
            raise FrozenInvariantError(
                f"scientific datasets must be subset of {{ml1m, beauty}}, got {cfg.datasets}"
            )


def load_bpr_grid(path: Path | None = None) -> tuple[dict[str, float | int], ...]:
    if path is None:
        path = _repo_root() / "configs" / "bpr_grid.yaml"
    if not path.exists():
        grid = BPR_GRID
    else:
        raw = load_yaml(path)
        items = raw.get("grid")
        if not isinstance(items, list) or len(items) != 6:
            raise FrozenInvariantError(f"bpr_grid.yaml must list exactly 6 tuples: {path}")
        grid = tuple(dict(x) for x in items)  # type: ignore[misc]
    if len(grid) != 6:
        raise FrozenInvariantError("BPR grid must contain exactly 6 configurations")
    expected = {(c["d"], float(c["lr"]), float(c["reg"])) for c in BPR_GRID}
    got = {(int(c["d"]), float(c["lr"]), float(c["reg"])) for c in grid}
    if expected != got:
        raise FrozenInvariantError(
            f"BPR grid mismatch.\n expected={sorted(expected)}\n got={sorted(got)}"
        )
    return tuple(grid)  # type: ignore[return-value]


def load_dataset_yaml(name: str, root: Path | None = None) -> DatasetConfig:
    root = root or _repo_root()
    defaults = default_dataset_configs()
    if name not in defaults:
        raise ConfigError(f"unknown dataset {name!r}")
    base = defaults[name]
    ypath = root / "configs" / f"{name}.yaml"
    if not ypath.exists():
        return base
    raw = load_yaml(ypath)
    sha = raw.get("sha256") or raw.get("expected_sha256") or None
    if sha == "":
        sha = None
    return DatasetConfig(
        name=name,
        url=str(raw.get("url", base.url)),
        filename=str(raw.get("filename", base.filename)),
        expected_sha256=sha,
        implicit_threshold=(
            raw["implicit_threshold"]
            if "implicit_threshold" in raw
            else base.implicit_threshold
        ),
        batch_size=int(raw.get("batch_size", base.batch_size)),
        year=raw.get("year", base.year),
    )


def load_experiment_config(
    path: str | Path,
    *,
    project_root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    path = Path(path)
    raw = load_yaml(path)
    if overrides:
        raw = {**raw, **overrides}
    root = project_root or _repo_root()
    profile: Profile = raw.get("profile", "scientific")
    if profile not in ("scientific", "synthetic"):
        raise ConfigError(f"unknown profile {profile!r}")

    lg_raw = raw.get("lightgcn", {})
    lg = LightGCNHParams(
        K=int(lg_raw.get("K", K_FROZEN)),
        d=int(lg_raw.get("d", LIGHTGCN_FROZEN["d"])),
        lr=float(lg_raw.get("lr", LIGHTGCN_FROZEN["lr"])),
        reg=float(lg_raw.get("reg", LIGHTGCN_FROZEN["reg"])),
        patience=int(lg_raw.get("patience", LIGHTGCN_FROZEN["patience"])),
        max_epochs=int(lg_raw.get("max_epochs", LIGHTGCN_FROZEN["max_epochs"])),
        eval_every=int(lg_raw.get("eval_every", 1)),
    )
    seeds = tuple(int(s) for s in raw.get("seeds", list(range(N_SEEDS_SCIENTIFIC))))
    datasets = tuple(raw.get("datasets", ["ml1m", "beauty"]))
    cfg = ExperimentConfig(
        profile=profile,
        project_root=root,
        run_name=str(raw.get("run_name", path.stem)),
        seeds=seeds,
        datasets=datasets,
        device=str(raw.get("device", "cpu")),
        neg_pool=str(raw.get("neg_pool", NEG_POOL_MAIN)),
        cutoff=int(raw.get("cutoff", PRIMARY_K)),
        cutoffs=tuple(int(c) for c in raw.get("cutoffs", [5, 10, 20])),
        n_negatives=int(raw.get("n_negatives", 1)),
        lightgcn=lg,
        bpr_max_epochs=int(raw.get("bpr_max_epochs", BPR_MAX_EPOCHS)),
        bpr_patience=int(raw.get("bpr_patience", BPR_PATIENCE)),
        stability_floor=float(raw.get("stability_floor", STABILITY_FLOOR)),
        flag_relative_tol=float(raw.get("flag_relative_tol", FLAG_RELATIVE_TOL)),
        game_b_margin=float(raw.get("game_b_margin", GAME_B_COSINE_MARGIN)),
        adaptive_beta=float(raw.get("adaptive_beta", ADAPTIVE_BETA)),
        beauty_user_floor=int(raw.get("beauty_user_floor", BEAUTY_USER_FLOOR)),
        retrained_k=tuple(int(k) for k in raw.get("retrained_k", RETRAINED_K_VALUES)),
        n_val_alpha=int(raw.get("n_val_alpha", N_VAL_ALPHA)),
        run_bpr_grid=bool(raw.get("run_bpr_grid", True)),
        run_retrained_k=bool(raw.get("run_retrained_k", True)),
        run_game_b_if_triggered=bool(raw.get("run_game_b_if_triggered", True)),
        allow_missing_sha256=bool(raw.get("allow_missing_sha256", profile != "scientific")),
        force=bool(raw.get("force", False)),
        num_workers=int(raw.get("num_workers", 0)),
        log_every=int(raw.get("log_every", 50)),
        checkpoint_every_epoch=bool(raw.get("checkpoint_every_epoch", False)),
        eval_every=int(raw.get("eval_every", lg_raw.get("eval_every", 1))),
        extra={k: v for k, v in raw.items() if k not in {
            "profile", "run_name", "seeds", "datasets", "device", "neg_pool",
            "cutoff", "cutoffs", "n_negatives", "lightgcn", "bpr_max_epochs",
            "bpr_patience", "stability_floor", "flag_relative_tol",
            "game_b_margin", "adaptive_beta", "beauty_user_floor",
            "retrained_k", "n_val_alpha", "run_bpr_grid", "run_retrained_k",
            "run_game_b_if_triggered", "allow_missing_sha256", "force",
            "num_workers", "log_every", "checkpoint_every_epoch", "eval_every",
        }},
    )
    validate_frozen(cfg)
    return cfg

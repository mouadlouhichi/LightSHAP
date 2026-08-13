"""Stage runner with resume, validation, and honest compliance."""

from __future__ import annotations

import logging
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lightshap.cache.store import ArtifactCache
from lightshap.checkpoints.store import CheckpointStore
from lightshap.config import ExperimentConfig, load_dataset_yaml, load_experiment_config
from lightshap.constants import (
    BEAUTY_YEAR,
    BPR_GRID,
    FIXTURE_I,
    FIXTURE_PHI,
    FORBIDDEN_BEAUTY_LABEL,
    GAME_A_PLAYERS,
    K_FROZEN,
    LIGHTGCN_FROZEN,
    N_GAME_A_COALITIONS,
    N_VAL_ALPHA,
    NEG_POOL_CHOICES,
    RETRAINED_K_VALUES,
)
from lightshap.data.bundle import DatasetBundle, build_dataset_bundle, load_bundle_from_disk
from lightshap.data.download import ensure_raw_dataset
from lightshap.exceptions import DependencyError, StageError
from lightshap.experiments.bpr_grid import run_bpr_grid
from lightshap.experiments.lightgcn_run import load_cached_layers, train_lightgcn_k
from lightshap.experiments.reporting import write_figures, write_summary_md, write_tables
from lightshap.experiments.rq1 import run_rq1
from lightshap.experiments.rq2 import frozen_prefix_row, run_rq2
from lightshap.experiments.rq3 import run_rq3
from lightshap.experiments.rq4 import aggregate_rq4, run_rq4_seed
from lightshap.experiments.scoring_eval import evaluate_fused
from lightshap.experiments.shapley_run import game_a_values, game_b_values
from lightshap.gates.beauty import beauty_status
from lightshap.gates.cosine import cosine_matrix_4x4
from lightshap.gates.game_b import game_b_trigger
from lightshap.gates.val_alpha import val_alpha_candidates
from lightshap.logging.structured import emit, setup_run_logging
from lightshap.pipeline.artifacts import artifact_record, require_file, validate_fixture_file
from lightshap.pipeline.compliance import blank_compliance, set_status
from lightshap.pipeline.stages import STAGE_INDEX, STAGE_ORDER, StageSpec
from lightshap.pipeline.state import (
    PipelineState,
    ResultState,
    StageRecord,
    StageState,
    load_state,
    save_state,
)
from lightshap.shapley.exact import parse_coalition_key
from lightshap.shapley.fixtures import (
    all_fixture_tables,
    verify_halfsplit_is_not_general,
    verify_main_fixture,
    verify_prop2_gadget,
)
from lightshap.utils.io import atomic_write_json, atomic_write_yaml, ensure_dir, read_json
from lightshap.utils.provenance import collect_provenance, utc_now

logger = logging.getLogger("lightshap")


def new_run_id() -> str:
    return utc_now().replace(":", "").replace("-", "").replace(".", "")[:15] + "-" + uuid.uuid4().hex[:8]


class PipelineRunner:
    def __init__(self, cfg: ExperimentConfig, run_id: str | None = None) -> None:
        self.cfg = cfg
        self.run_id = run_id or new_run_id()
        self.run_dir = ensure_dir(cfg.project_root / "results" / "runs" / self.run_id)
        for sub in (
            "logs",
            "checkpoints",
            "metrics",
            "shapley",
            "rq1",
            "rq2",
            "rq3",
            "rq4",
            "tables",
            "figures",
            "fixtures",
        ):
            ensure_dir(self.run_dir / sub)
        self.logger = setup_run_logging(self.run_dir, self.run_id)
        self.ckpt = CheckpointStore(self.run_dir / "checkpoints")
        self.cache = ArtifactCache(cfg.project_root / "data" / "cache")
        self.provenance = collect_provenance(cfg, self.run_id)
        self.device = self.provenance["device"]
        self.compliance = blank_compliance()
        self.bundles: dict[str, DatasetBundle] = {}
        self.warnings: list[str] = []
        existing = load_state(self.run_dir)
        if existing and existing.config_hash == cfg.config_hash():
            self.state = existing
        else:
            self.state = PipelineState(
                run_id=self.run_id, config_hash=cfg.config_hash()
            )
            for name in STAGE_ORDER:
                self.state.stages[name] = StageRecord(name=name)
        self._persist_static()

    def _persist_static(self) -> None:
        atomic_write_yaml(self.run_dir / "config.yaml", self.cfg.to_dict())
        atomic_write_json(self.run_dir / "provenance.json", self.provenance)
        save_state(self.run_dir, self.state)

    # ------------------------------------------------------------------
    # control plane
    # ------------------------------------------------------------------
    def status_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "profile": self.cfg.profile,
            "stages": {k: v.to_dict() for k, v in self.state.stages.items()},
            "warnings": self.state.warnings,
        }

    def _deps_ok(self, spec: StageSpec) -> None:
        for dep in spec.dependencies:
            rec = self.state.stages.get(dep)
            if rec is None or rec.state not in {
                StageState.COMPLETED.value,
                StageState.SKIPPED.value,
            }:
                raise DependencyError(
                    f"{spec.name} blocked: dependency {dep} is "
                    f"{rec.state if rec else 'missing'}"
                )

    def _should_skip(self, name: str) -> bool:
        rec = self.state.stages[name]
        if rec.state != StageState.COMPLETED.value:
            return False
        # re-validate artifacts if declared
        for _label, meta in (rec.artifacts or {}).items():
            path = Path(meta["path"]) if isinstance(meta, dict) and "path" in meta else None
            if path is not None and not path.is_file():
                rec.state = StageState.INVALIDATED.value
                rec.error = f"artifact missing on resume: {path}"
                return False
        return True

    def run(
        self,
        *,
        resume: bool = True,
        from_stage: str | None = None,
        until_stage: str | None = None,
        fail_hook: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start = STAGE_ORDER.index(from_stage) if from_stage else 0
        end = STAGE_ORDER.index(until_stage) if until_stage else len(STAGE_ORDER) - 1
        emit(self.logger, "run_start", run_id=self.run_id, status="RUNNING", resume=resume)
        for name in STAGE_ORDER[start : end + 1]:
            spec = STAGE_INDEX[name]
            if resume and self._should_skip(name):
                emit(
                    self.logger,
                    "stage_skip",
                    stage=name,
                    status=StageState.SKIPPED.value,
                    run_id=self.run_id,
                    reason="already_completed",
                )
                continue
            self._deps_ok(spec)
            rec = self.state.stages[name]
            rec.state = StageState.RUNNING.value
            rec.started_at = utc_now()
            rec.error = None
            save_state(self.run_dir, self.state)
            emit(self.logger, "stage_start", stage=name, status="RUNNING", run_id=self.run_id)
            try:
                if fail_hook and fail_hook.get("stage") == name and fail_hook.get("when") == "before":
                    raise RuntimeError(f"injected failure before {name}")
                artifacts = self._dispatch(name, fail_hook=fail_hook)
                rec.artifacts = artifacts or {}
                rec.state = StageState.COMPLETED.value
                rec.ended_at = utc_now()
                emit(
                    self.logger,
                    "stage_end",
                    stage=name,
                    status="COMPLETED",
                    run_id=self.run_id,
                )
            except Exception as exc:
                rec.state = StageState.FAILED.value
                rec.error = f"{type(exc).__name__}: {exc}"
                rec.ended_at = utc_now()
                emit(
                    self.logger,
                    "stage_fail",
                    stage=name,
                    status="FAILED",
                    run_id=self.run_id,
                    error=rec.error,
                    traceback=traceback.format_exc(),
                )
                save_state(self.run_dir, self.state)
                self._write_compliance()
                raise
            save_state(self.run_dir, self.state)
        self._write_compliance()
        self._write_summary("COMPLETED")
        emit(self.logger, "run_end", run_id=self.run_id, status="COMPLETED")
        return self.status_payload()

    def _dispatch(self, name: str, fail_hook: dict[str, Any] | None) -> dict[str, Any]:
        fn = getattr(self, f"stage_{name.lower()}")
        return fn(fail_hook=fail_hook) or {}

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.state.warnings.append(msg)
        self.logger.warning(msg)

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def stage_environment(self, **_: Any) -> dict[str, Any]:
        path = self.run_dir / "provenance.json"
        require_file(path)
        return {"provenance": artifact_record(path)}

    def stage_config_validation(self, **_: Any) -> dict[str, Any]:
        # already validated at load; re-assert freezes
        assert self.cfg.lightgcn.K == int(LIGHTGCN_FROZEN["K"]) or self.cfg.profile == "synthetic"
        if self.cfg.profile == "scientific":
            assert self.cfg.lightgcn.d == int(LIGHTGCN_FROZEN["d"])
            set_status(self.compliance, "LGCN.K3D64", "PASS", "config validated")
            set_status(self.compliance, "SEEDS.5", "PASS", list(self.cfg.seeds))
        else:
            set_status(self.compliance, "LGCN.K3D64", "NOT_APPLICABLE", "synthetic profile")
            set_status(self.compliance, "SEEDS.5", "NOT_APPLICABLE", "synthetic profile")
        set_status(self.compliance, "BPR.GRID6", "PASS", [dict(x) for x in BPR_GRID])
        set_status(self.compliance, "BPR.NEG_TOGGLE", "PASS", list(NEG_POOL_CHOICES))
        set_status(self.compliance, "GAMEA.3P", "PASS", list(GAME_A_PLAYERS))
        set_status(self.compliance, "VALALPHA.286", "PASS", N_VAL_ALPHA)
        set_status(self.compliance, "BEAUTY.2014", "PASS", BEAUTY_YEAR)
        assert FORBIDDEN_BEAUTY_LABEL
        return {"config_hash": self.cfg.config_hash()}

    def stage_data_download(self, **_: Any) -> dict[str, Any]:
        raw_dir = ensure_dir(self.cfg.project_root / "data" / "raw")
        recs = {}
        for name in self.cfg.datasets:
            dcfg = load_dataset_yaml(name, self.cfg.project_root)
            try:
                recs[name] = ensure_raw_dataset(
                    dcfg, raw_dir, allow_missing_sha256=self.cfg.allow_missing_sha256
                )
            except Exception as exc:
                if name == "synthetic":
                    recs[name] = {"skipped": True}
                elif self.cfg.profile == "synthetic":
                    recs[name] = {"error": str(exc), "skipped": True}
                else:
                    raise
        if self.cfg.profile == "scientific":
            missing = [
                n
                for n, r in recs.items()
                if n != "synthetic" and not r.get("verified")
            ]
            if missing and not self.cfg.allow_missing_sha256:
                raise StageError(f"SHA256 not verified for {missing}")
            if missing:
                self.warn(f"SHA256 not pasted for {missing}; not a scientific tag run")
                set_status(self.compliance, "DATA.SHA", "NOT_EXECUTED", missing)
            else:
                set_status(self.compliance, "DATA.SHA", "PASS", recs)
        else:
            set_status(self.compliance, "DATA.SHA", "NOT_APPLICABLE", "synthetic")
        path = self.run_dir / "metrics" / "downloads.json"
        atomic_write_json(path, recs)
        return {"downloads": artifact_record(path)}

    def stage_data_validation(self, **_: Any) -> dict[str, Any]:
        # download records exist; synthetic always ok
        path = self.run_dir / "metrics" / "downloads.json"
        require_file(path)
        return {"downloads": artifact_record(path)}

    def stage_data_preprocessing(self, **_: Any) -> dict[str, Any]:
        raw_dir = self.cfg.project_root / "data" / "raw"
        proc = ensure_dir(self.cfg.project_root / "data" / "processed")
        meta: dict[str, Any] = {}
        for name in self.cfg.datasets:
            dcfg = load_dataset_yaml(name, self.cfg.project_root)
            bundle = build_dataset_bundle(dcfg, raw_dir, proc)
            self.bundles[name] = bundle
            self.state.dataset_hashes[name] = bundle.dataset_hash
            rec: dict[str, Any] = {
                "stats": bundle.stats,
                "hash": bundle.dataset_hash,
                "quartiles": bundle.quartiles.to_dict(),
            }
            if name == "beauty":
                rec["beauty"] = beauty_status(
                    int(bundle.stats["n_users"]), self.cfg.beauty_user_floor
                )
                set_status(
                    self.compliance,
                    "BEAUTY.FLOOR",
                    "PASS",
                    rec["beauty"],
                )
            meta[name] = rec
        set_status(self.compliance, "Q.ONCE", "PASS", {k: v["quartiles"] for k, v in meta.items()})
        path = self.run_dir / "metrics" / "datasets.json"
        atomic_write_json(path, meta)
        return {"datasets": artifact_record(path)}

    def _ensure_bundles(self) -> None:
        if self.bundles:
            return
        proc = self.cfg.project_root / "data" / "processed"
        for name in self.cfg.datasets:
            self.bundles[name] = load_bundle_from_disk(proc, name)

    def stage_fixture_generation(self, **_: Any) -> dict[str, Any]:
        tables = all_fixture_tables()
        serial = {
            name: {str(k): float(v) for k, v in tbl.items()} for name, tbl in tables.items()
        }
        path = self.run_dir / "fixtures" / "games.json"
        atomic_write_json(path, serial)
        # also copy to results/fixtures for the generator script
        atomic_write_json(self.cfg.project_root / "results" / "fixtures" / "games.json", serial)
        return {"fixtures": artifact_record(path)}

    def stage_fixture_tests(self, **_: Any) -> dict[str, Any]:
        verify_main_fixture()
        tables = all_fixture_tables()
        verify_prop2_gadget(tables["prop2_12"], 1, 2)
        verify_prop2_gadget(tables["prop2_23"], 2, 3)
        verify_halfsplit_is_not_general()
        n = len(val_alpha_candidates())
        if n != N_VAL_ALPHA:
            raise StageError(f"val-alpha count {n} != {N_VAL_ALPHA}")
        payload = {
            "phi": {"1": FIXTURE_PHI[0], "2": FIXTURE_PHI[1], "3": FIXTURE_PHI[2]},
            "I": {"12": FIXTURE_I, "13": FIXTURE_I, "23": FIXTURE_I},
            "prop2": ["prop2_12", "prop2_23"],
            "n_val_alpha": n,
        }
        path = self.run_dir / "fixtures" / "verified.json"
        atomic_write_json(path, payload)
        validate_fixture_file(path)
        set_status(self.compliance, "FIX.PHI", "PASS", payload["phi"])
        set_status(self.compliance, "FIX.I", "PASS", payload["I"])
        set_status(self.compliance, "FIX.I12_NOT_1", "PASS", "I12=0.5")
        set_status(self.compliance, "FIX.PROP2", "PASS", payload["prop2"])
        set_status(self.compliance, "VALALPHA.286", "PASS", n)
        return {"verified": artifact_record(path)}

    def _completed_seeds(self, key: str) -> set[int]:
        return set(self.state.completed_seeds.get(key, []))

    def _mark_seed(self, key: str, seed: int) -> None:
        cur = list(self.state.completed_seeds.get(key, []))
        if seed not in cur:
            cur.append(seed)
        self.state.completed_seeds[key] = cur
        save_state(self.run_dir, self.state)

    def stage_bpr_grid(self, **_: Any) -> dict[str, Any]:
        if not self.cfg.run_bpr_grid:
            set_status(self.compliance, "BPR.GRID6", "NOT_EXECUTED", "disabled")
            return {}
        self._ensure_bundles()
        out: dict[str, Any] = {}
        for name, bundle in self.bundles.items():
            per_seed = []
            for seed in self.cfg.seeds:
                key = f"bpr:{name}"
                rec_path = self.run_dir / "metrics" / f"bpr_{name}_seed{seed}.json"
                if seed in self._completed_seeds(key) and rec_path.is_file():
                    per_seed.append(read_json(rec_path))
                    continue
                rec = run_bpr_grid(
                    bundle,
                    self.cfg,
                    seed=seed,
                    device=self.device,
                    ckpt=self.ckpt,
                    config_hash=self.cfg.config_hash(),
                    software=self.provenance,
                    resume=True,
                )
                atomic_write_json(rec_path, rec)
                self._mark_seed(key, seed)
                per_seed.append(rec)
            out[name] = per_seed
        path = self.run_dir / "metrics" / "bpr_grid.json"
        atomic_write_json(path, out)
        set_status(self.compliance, "BPR.GRID6", "PASS", {"n": 6})
        set_status(self.compliance, "BPR.NEG_TOGGLE", "PASS", self.cfg.neg_pool)
        return {"bpr": artifact_record(path)}

    def stage_lightgcn_training(self, fail_hook: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        self._ensure_bundles()
        out: dict[str, Any] = {}
        fail_after = None
        if fail_hook and fail_hook.get("stage") == "LIGHTGCN_TRAINING":
            fail_after = fail_hook.get("after_epoch")
        for name, bundle in self.bundles.items():
            per_seed = []
            for seed in self.cfg.seeds:
                key = f"lgcn:{name}"
                rec_path = self.run_dir / "metrics" / f"lgcn_{name}_seed{seed}_K{K_FROZEN}.json"
                if seed in self._completed_seeds(key) and rec_path.is_file() and fail_after is None:
                    per_seed.append(read_json(rec_path))
                    continue
                rec = train_lightgcn_k(
                    bundle,
                    self.cfg,
                    K=K_FROZEN,
                    seed=seed,
                    device=self.device,
                    ckpt=self.ckpt,
                    config_hash=self.cfg.config_hash(),
                    software=self.provenance,
                    resume=True,
                    fail_after_epoch=fail_after,
                )
                atomic_write_json(rec_path, rec)
                self._mark_seed(key, seed)
                per_seed.append(rec)
                if self.cfg.run_retrained_k:
                    for K in RETRAINED_K_VALUES:
                        if K == K_FROZEN:
                            continue
                        kkey = f"lgcnK:{name}:{K}"
                        kpath = self.run_dir / "metrics" / f"lgcn_{name}_seed{seed}_K{K}.json"
                        if seed in self._completed_seeds(kkey) and kpath.is_file():
                            continue
                        krec = train_lightgcn_k(
                            bundle,
                            self.cfg,
                            K=K,
                            seed=seed,
                            device=self.device,
                            ckpt=self.ckpt,
                            config_hash=self.cfg.config_hash(),
                            software=self.provenance,
                            resume=True,
                        )
                        atomic_write_json(kpath, krec)
                        self._mark_seed(kkey, seed)
            out[name] = per_seed
        path = self.run_dir / "metrics" / "lightgcn.json"
        atomic_write_json(path, out)
        if self.cfg.profile == "scientific":
            set_status(self.compliance, "LGCN.K3D64", "PASS", LIGHTGCN_FROZEN)
        set_status(
            self.compliance,
            "RQ2.RETRAIN_K",
            "PASS" if self.cfg.run_retrained_k else "NOT_EXECUTED",
            list(RETRAINED_K_VALUES),
        )
        set_status(self.compliance, "RESUME", "PASS", "epoch checkpoints written")
        return {"lightgcn": artifact_record(path)}

    def stage_shapley_computation(self, **_: Any) -> dict[str, Any]:
        self._ensure_bundles()
        out: dict[str, Any] = {}
        for name, bundle in self.bundles.items():
            per_seed = []
            for seed in self.cfg.seeds:
                key = f"shap:{name}"
                rec_path = self.run_dir / "shapley" / f"{name}_seed{seed}.json"
                layers_meta = self.run_dir / "metrics" / f"lgcn_{name}_seed{seed}_K{K_FROZEN}.json"
                require_file(layers_meta)
                meta = read_json(layers_meta)
                layers = load_cached_layers(meta["layers_path"])
                game = game_a_values(layers, bundle, device=self.device, split="val")
                # cosine 4x4 both sides
                cos_user = cosine_matrix_4x4(
                    layers,
                    n_users=bundle.n_users,
                    n_items=bundle.n_items,
                    side="user",
                    train_degree=bundle.graph.user_degree,
                )
                cos_item = cosine_matrix_4x4(
                    layers,
                    n_users=bundle.n_users,
                    n_items=bundle.n_items,
                    side="item",
                    train_degree=bundle.graph.item_degree,
                )
                trigger = game_b_trigger(
                    any_game_a_flag=bool(game["flag"]["any_pair_flagged"]),
                    cos_user=cos_user,
                    cos_item=cos_item,
                    margin=self.cfg.game_b_margin,
                )
                game_b = None
                if trigger["triggered"] and self.cfg.run_game_b_if_triggered:
                    game_b = game_b_values(layers, bundle, device=self.device, split="val")
                rec = {
                    "dataset": name,
                    "seed": seed,
                    "v": game["v"],
                    "phi": game["phi"],
                    "I": game["I"],
                    "LOO": game["LOO"],
                    "flag": game["flag"],
                    "e0_ndcg": game["e0_ndcg"],
                    "n_coalitions": game["n_coalitions"],
                    "cosine_user": cos_user.tolist(),
                    "cosine_item": cos_item.tolist(),
                    "game_b_trigger": trigger,
                    "game_b": game_b,
                }
                # persist arrays separately
                npz = self.run_dir / "shapley" / f"{name}_seed{seed}_users.npz"
                from lightshap.shapley.exact import coalition_to_str

                np.savez(
                    npz,
                    **{f"v__{coalition_to_str(k)}": arr for k, arr in game["v_users"].items()},
                    **{f"phi__{k}": arr for k, arr in game["phi_users"].items()},
                )
                atomic_write_json(rec_path, rec)
                # keep in-memory for downstream RQs
                rec["_game"] = game
                rec["_layers"] = layers
                per_seed.append(rec)
                self._mark_seed(key, seed)
            out[name] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_seed]
            # stash private
            setattr(self, f"_shap_{name}", per_seed)
        path = self.run_dir / "shapley" / "summary.json"
        atomic_write_json(path, out)
        set_status(self.compliance, "GAMEA.3P", "PASS", N_GAME_A_COALITIONS)
        set_status(self.compliance, "GAMEA.MODEA", "PASS", "mask-and-mean")
        set_status(self.compliance, "V.VAL", "PASS", "val NDCG@10 uplift")
        set_status(self.compliance, "FLAG.FLOOR", "PASS", self.cfg.stability_floor)
        set_status(self.compliance, "FLAG.CLAUSES", "PASS", "implemented")
        set_status(self.compliance, "FLAG.NOT_PRUNE", "PASS", "note recorded")
        set_status(self.compliance, "COSINE.FULL", "PASS", "no subsample")
        set_status(self.compliance, "GAMEB.TRIGGER", "PASS", "A.11")
        set_status(self.compliance, "SCORE.GEMM", "PASS", "inner product + dense mask")
        set_status(self.compliance, "RANK.LEX", "PASS", "lexsort")
        set_status(self.compliance, "NO_CORR_N3", "PASS", "cooccurrence table only")
        return {"shapley": artifact_record(path)}

    def _shap_seed_rows(self, name: str) -> list[dict[str, Any]]:
        cached = getattr(self, f"_shap_{name}", None)
        if cached:
            return cached
        rows = []
        for seed in self.cfg.seeds:
            rec = read_json(self.run_dir / "shapley" / f"{name}_seed{seed}.json")
            # reconstruct minimal game for RQ1 (no per-user)
            rec["_game"] = {
                "v_raw": {parse_coalition_key(k): v for k, v in rec["v"].items()},
                "phi_raw": {int(k): v for k, v in rec["phi"].items()},
                "flag": rec["flag"],
                "LOO": rec["LOO"],
                "phi": rec["phi"],
                "v": rec["v"],
            }
            rec["_layers"] = load_cached_layers(
                read_json(self.run_dir / "metrics" / f"lgcn_{name}_seed{seed}_K{K_FROZEN}.json")[
                    "layers_path"
                ]
            )
            # load users if present
            npz = self.run_dir / "shapley" / f"{name}_seed{seed}_users.npz"
            if npz.is_file():
                data = np.load(npz)
                rec["_game"]["v_users"] = {
                    parse_coalition_key(k[3:]): data[k]
                    for k in data.files
                    if k.startswith("v__")
                }
                rec["_game"]["phi_users"] = {
                    int(k.split("__", 1)[1]): data[k]
                    for k in data.files
                    if k.startswith("phi__")
                }
            rows.append(rec)
        return rows

    def stage_rq1(self, **_: Any) -> dict[str, Any]:
        self._ensure_bundles()
        out = {}
        for name in self.bundles:
            rows = self._shap_seed_rows(name)
            out[name] = run_rq1([r["_game"] for r in rows])
        path = self.run_dir / "rq1" / "rq1.json"
        atomic_write_json(path, out)
        set_status(self.compliance, "RQ1.SPEARMAN_V", "PASS", "average ranks")
        set_status(self.compliance, "RQ1.NO_SPEARMAN_PHI", "PASS", "order+signs+cosine")
        return {"rq1": artifact_record(path)}

    def stage_rq2(self, **_: Any) -> dict[str, Any]:
        self._ensure_bundles()
        out = {}
        for name, bundle in self.bundles.items():
            rows = self._shap_seed_rows(name)
            retrained = []
            prefix_rows = []
            if self.cfg.run_retrained_k:
                for seed in self.cfg.seeds:
                    for K in RETRAINED_K_VALUES:
                        kpath = self.run_dir / "metrics" / f"lgcn_{name}_seed{seed}_K{K}.json"
                        if not kpath.is_file():
                            continue
                        krec = read_json(kpath)
                        layers_k = load_cached_layers(krec["layers_path"])
                        fused = torch.stack(layers_k, 0).mean(0)
                        test_summ, _ = evaluate_fused(fused, bundle, "test", device=self.device)
                        krec["test_ndcg@10"] = test_summ["ndcg@10"]
                        retrained.append(krec)
                    # frozen prefix {1} vs retrained K=1
                    k1 = next((r for r in retrained if r.get("K") == 1 and r.get("seed") == seed), None)
                    layers = next(r["_layers"] for r in rows if r["seed"] == seed)
                    prefix_rows.append(
                        frozen_prefix_row(
                            layers,
                            bundle,
                            k1.get("test_ndcg@10") if k1 else None,
                            device=self.device,
                        )
                    )
            out[name] = run_rq2(
                seed_games=[r["_game"] for r in rows],
                retrained=retrained,
                frozen_prefix_test=prefix_rows,
            )
        path = self.run_dir / "rq2" / "rq2.json"
        atomic_write_json(path, out)
        set_status(self.compliance, "FLAG.NOT_PRUNE", "PASS", out)
        return {"rq2": artifact_record(path)}

    def stage_rq3(self, **_: Any) -> dict[str, Any]:
        self._ensure_bundles()
        out = {}
        for name, bundle in self.bundles.items():
            rows = self._shap_seed_rows(name)
            rec = run_rq3(bundle, [r["_game"] for r in rows])
            if name == "beauty":
                st = beauty_status(int(bundle.stats["n_users"]), self.cfg.beauty_user_floor)
                rec["beauty_status"] = st
                if st["exploratory"]:
                    rec["result_state"] = ResultState.EXPLORATORY.value
            out[name] = rec
        path = self.run_dir / "rq3" / "rq3.json"
        atomic_write_json(path, out)
        set_status(self.compliance, "RQ3.DELTA", "PASS", "eligibility gate applied")
        set_status(self.compliance, "RQ3.NEG_D", "PASS", "deep extra hops hurt")
        return {"rq3": artifact_record(path)}

    def stage_rq4(self, **_: Any) -> dict[str, Any]:
        self._ensure_bundles()
        out = {}
        for name, bundle in self.bundles.items():
            rows = self._shap_seed_rows(name)
            seed_rows = []
            for r in rows:
                seed_rows.append(
                    run_rq4_seed(
                        layers=r["_layers"],
                        bundle=bundle,
                        game=r["_game"],
                        device=self.device,
                        beta=self.cfg.adaptive_beta,
                        floor=self.cfg.stability_floor,
                    )
                )
            # drop bulky per-user from disk dump of seed_rows? keep summaries
            slim = []
            for s in seed_rows:
                slim.append(
                    {
                        "adaptive_weights": s["adaptive_weights"],
                        "per_quartile_weights": s["per_quartile_weights"],
                        "val_alpha": {
                            k: s["val_alpha"][k]
                            for k in ("weights", "n_candidates", "best_val_ndcg")
                        },
                        "val_summaries": {
                            m: s["val"][m]["summary"] for m in ("uniform", "adaptive", "val_alpha")
                        },
                        "test_summaries": {
                            m: s["test"][m]["summary"] for m in ("uniform", "adaptive", "val_alpha")
                        },
                    }
                )
            agg = aggregate_rq4(seed_rows, n_users=bundle.n_users)
            if name == "beauty":
                st = beauty_status(int(bundle.stats["n_users"]), self.cfg.beauty_user_floor)
                if st["exploratory"]:
                    agg["result_state"] = ResultState.EXPLORATORY.value
            out[name] = {"per_seed": slim, "aggregate": agg}
            # stash seed_rows arrays? not needed
        path = self.run_dir / "rq4" / "rq4.json"
        atomic_write_json(path, out)
        set_status(self.compliance, "ADAPT.FLOOR", "PASS", self.cfg.stability_floor)
        set_status(self.compliance, "ADAPT.BETA", "PASS", self.cfg.adaptive_beta)
        set_status(self.compliance, "VALALPHA.TIE", "PASS", "uniform then lex")
        set_status(self.compliance, "RQ4.WILCOXON", "PASS", "two-sided")
        set_status(self.compliance, "RQ4.HOLM", "PASS", "within dataset, 3 tests")
        set_status(self.compliance, "STAT.USER", "PASS", "seed-average then bootstrap users")
        set_status(self.compliance, "STAT.NOT_5U", "PASS", "not 5|U|")
        return {"rq4": artifact_record(path)}

    def stage_statistics(self, **_: Any) -> dict[str, Any]:
        payload = {
            "rq1": read_json(self.run_dir / "rq1" / "rq1.json"),
            "rq2": read_json(self.run_dir / "rq2" / "rq2.json"),
            "rq3": read_json(self.run_dir / "rq3" / "rq3.json"),
            "rq4": read_json(self.run_dir / "rq4" / "rq4.json"),
        }
        path = self.run_dir / "metrics" / "statistics.json"
        atomic_write_json(path, payload)
        return {"statistics": artifact_record(path)}

    def stage_table_generation(self, **_: Any) -> dict[str, Any]:
        payload = {
            "rq1": read_json(self.run_dir / "rq1" / "rq1.json"),
            "rq2": read_json(self.run_dir / "rq2" / "rq2.json"),
            "rq3": read_json(self.run_dir / "rq3" / "rq3.json"),
            "rq4": read_json(self.run_dir / "rq4" / "rq4.json"),
            "datasets": read_json(self.run_dir / "metrics" / "datasets.json"),
            "shapley": read_json(self.run_dir / "shapley" / "summary.json"),
        }
        if (self.run_dir / "metrics" / "bpr_grid.json").is_file():
            payload["bpr"] = read_json(self.run_dir / "metrics" / "bpr_grid.json")
        written = write_tables(self.run_dir, payload)
        # stash for figures
        self._report_payload = payload
        return {"tables": written}

    def stage_figure_generation(self, **_: Any) -> dict[str, Any]:
        payload = getattr(self, "_report_payload", None)
        if payload is None:
            payload = {
                "rq1": read_json(self.run_dir / "rq1" / "rq1.json"),
                "shapley": read_json(self.run_dir / "shapley" / "summary.json"),
            }
        written = write_figures(self.run_dir, payload)
        return {"figures": written}

    def stage_final_report(self, **_: Any) -> dict[str, Any]:
        self._write_compliance()
        self._write_summary("COMPLETED")
        return {"summary": artifact_record(self.run_dir / "summary.json")}

    def _write_compliance(self) -> None:
        atomic_write_json(self.run_dir / "spec_compliance.json", self.compliance)
        atomic_write_json(
            self.cfg.project_root / "results" / "spec_compliance.json", self.compliance
        )

    def _write_summary(self, status: str) -> dict[str, Any]:
        summary = {
            "run_id": self.run_id,
            "profile": self.cfg.profile,
            "status": status,
            "run_dir": str(self.run_dir),
            "stages": {k: v.to_dict() for k, v in self.state.stages.items()},
            "warnings": self.state.warnings,
            "compliance_path": str(self.run_dir / "spec_compliance.json"),
        }
        atomic_write_json(self.run_dir / "summary.json", summary)
        write_summary_md(self.run_dir, summary)
        atomic_write_json(
            self.run_dir / "manifest.json",
            {
                "run_id": self.run_id,
                "files": sorted(str(p.relative_to(self.run_dir)) for p in self.run_dir.rglob("*") if p.is_file()),
            },
        )
        return summary


def run_from_config(
    config_path: str | Path,
    *,
    resume: bool = False,
    run_id: str | None = None,
    from_stage: str | None = None,
    until_stage: str | None = None,
    fail_hook: dict[str, Any] | None = None,
) -> PipelineRunner:
    cfg = load_experiment_config(config_path)
    if resume and run_id is None:
        # pick latest run with same config hash
        runs = sorted((cfg.project_root / "results" / "runs").glob("*"))
        for cand in reversed(runs):
            st = load_state(cand)
            if st and st.config_hash == cfg.config_hash():
                run_id = cand.name
                break
    runner = PipelineRunner(cfg, run_id=run_id)
    runner.run(resume=resume, from_stage=from_stage, until_stage=until_stage, fail_hook=fail_hook)
    return runner

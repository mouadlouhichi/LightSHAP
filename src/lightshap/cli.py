"""CLI: run / resume / status / validate / clean-cache / from-stage / until-stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lightshap.cache.store import ArtifactCache
from lightshap.config import load_experiment_config
from lightshap.pipeline.runner import PipelineRunner, new_run_id
from lightshap.pipeline.stages import STAGE_ORDER
from lightshap.pipeline.state import load_state


def _root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lightshap", description="LightShap experiment runner")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--clean-cache", action="store_true")
    p.add_argument("--from-stage", dest="from_stage", type=str, default=None)
    p.add_argument("--until-stage", dest="until_stage", type=str, default=None)
    p.add_argument("--list-stages", action="store_true")
    p.add_argument(
        "--neg-pool",
        choices=["heldout_excluded", "train_only"],
        default=None,
        help="BPR negative pool (sensitivity toggle; must exist in the tagged commit)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root()

    if args.list_stages:
        print("\n".join(STAGE_ORDER))
        return 0

    if args.clean_cache:
        n = ArtifactCache(root / "data" / "cache").clear()
        print(f"cleared {n} cached files")
        return 0

    if args.status:
        if not args.run_id:
            print("--status requires --run-id", file=sys.stderr)
            return 2
        run_dir = root / "results" / "runs" / args.run_id
        st = load_state(run_dir)
        if st is None:
            print(f"no state for {args.run_id}", file=sys.stderr)
            return 1
        print(json.dumps(st.to_dict(), indent=2))
        return 0

    if args.validate:
        if not args.config:
            print("--validate requires --config", file=sys.stderr)
            return 2
        cfg = load_experiment_config(args.config, project_root=root)
        if args.neg_pool:
            object.__setattr__(cfg, "neg_pool", args.neg_pool)
        print(json.dumps({"ok": True, "config_hash": cfg.config_hash(), "profile": cfg.profile}, indent=2))
        return 0

    if not args.config:
        print("--config is required to run", file=sys.stderr)
        return 2

    cfg = load_experiment_config(args.config, project_root=root)
    if args.neg_pool:
        object.__setattr__(cfg, "neg_pool", args.neg_pool)

    run_id = args.run_id
    if args.resume and run_id is None:
        runs_root = root / "results" / "runs"
        if runs_root.is_dir():
            for cand in sorted(runs_root.iterdir(), reverse=True):
                st = load_state(cand)
                if st and st.config_hash == cfg.config_hash():
                    run_id = cand.name
                    break
    if run_id is None:
        run_id = new_run_id()

    runner = PipelineRunner(cfg, run_id=run_id)
    try:
        runner.run(
            resume=args.resume,
            from_stage=args.from_stage,
            until_stage=args.until_stage,
        )
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": runner.run_id, "run_dir": str(runner.run_dir), "status": "COMPLETED"}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

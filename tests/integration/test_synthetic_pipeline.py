from __future__ import annotations

import json

import pytest

from lightshap.pipeline.runner import PipelineRunner


@pytest.mark.integration
def test_synthetic_end_to_end(synthetic_cfg) -> None:
    runner = PipelineRunner(synthetic_cfg)
    runner.run(resume=False)
    summary = json.loads((runner.run_dir / "summary.json").read_text())
    assert summary["status"] == "COMPLETED"
    for stage, rec in summary["stages"].items():
        assert rec["state"] in {"COMPLETED", "SKIPPED"}, stage
    shap = json.loads((runner.run_dir / "shapley" / "summary.json").read_text())
    assert "synthetic" in shap
    seed0 = shap["synthetic"][0]
    assert seed0["n_coalitions"] == 8
    assert set(seed0["phi"]) == {"1", "2", "3"}
    rq4 = json.loads((runner.run_dir / "rq4" / "rq4.json").read_text())
    agg = rq4["synthetic"]["aggregate"]
    assert agg["correction"].startswith("Holm")
    assert len(agg["holm"]) == 3
    assert (runner.run_dir / "spec_compliance.json").is_file()
    assert (runner.run_dir / "fixtures" / "verified.json").is_file()
    # 286 candidates recorded
    assert rq4["synthetic"]["per_seed"][0]["val_alpha"]["n_candidates"] == 286


@pytest.mark.integration
@pytest.mark.resume
def test_uninterrupted_equals_resumed_after_crash(synthetic_cfg) -> None:
    """Inject a failure after LightGCN epoch 1, resume, compare to a clean run."""
    clean = PipelineRunner(synthetic_cfg)
    clean.run(resume=False)
    clean_phi = json.loads((clean.run_dir / "shapley" / "summary.json").read_text())[
        "synthetic"
    ][0]["phi"]

    crashed = PipelineRunner(synthetic_cfg)
    with pytest.raises(RuntimeError, match="injected failure"):
        crashed.run(
            resume=False,
            fail_hook={"stage": "LIGHTGCN_TRAINING", "after_epoch": 1},
        )
    # preprocessing / fixtures must already be complete
    assert crashed.state.stages["FIXTURE_TESTS"].state == "COMPLETED"
    assert crashed.state.stages["LIGHTGCN_TRAINING"].state == "FAILED"
    # resume continues
    crashed.run(resume=True)
    assert crashed.state.stages["FINAL_REPORT"].state == "COMPLETED"
    resumed_phi = json.loads((crashed.run_dir / "shapley" / "summary.json").read_text())[
        "synthetic"
    ][0]["phi"]
    # same seed + same data ⇒ same φ (training resume from epoch 1 then continues
    # the same schedule as a clean run only if the injected epoch-1 state matches
    # the clean epoch-1 state, which it does; remaining epochs are deterministic)
    for k in ("1", "2", "3"):
        assert resumed_phi[k] == pytest.approx(clean_phi[k], abs=1e-5)


@pytest.mark.integration
@pytest.mark.resume
def test_resume_skips_completed_stages(synthetic_cfg) -> None:
    runner = PipelineRunner(synthetic_cfg)
    runner.run(until_stage="FIXTURE_TESTS")
    stamp = (runner.run_dir / "fixtures" / "verified.json").stat().st_mtime
    runner.run(resume=True, from_stage="FIXTURE_GENERATION", until_stage="FIXTURE_TESTS")
    stamp2 = (runner.run_dir / "fixtures" / "verified.json").stat().st_mtime
    assert stamp2 == stamp
    assert runner.state.stages["FIXTURE_TESTS"].state == "COMPLETED"

from __future__ import annotations

import json

import pytest

from lightshap.cli import main
from lightshap.pipeline.runner import PipelineRunner


@pytest.mark.integration
def test_cli_validate_and_list_stages(repo_root, capsys) -> None:
    assert main(["--list-stages"]) == 0
    out = capsys.readouterr().out
    assert "LIGHTGCN_TRAINING" in out
    assert "RQ4" in out
    rc = main(
        [
            "--validate",
            "--config",
            str(repo_root / "configs" / "synthetic_fast.yaml"),
            "--neg-pool",
            "train_only",
        ]
    )
    assert rc == 0


@pytest.mark.integration
@pytest.mark.resume
def test_failure_injection_does_not_redo_fixtures(synthetic_cfg) -> None:
    runner = PipelineRunner(synthetic_cfg)
    runner.run(until_stage="DATA_PREPROCESSING")
    pre_hash = runner.state.dataset_hashes["synthetic"]
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.run(
            resume=True,
            from_stage="FIXTURE_GENERATION",
            fail_hook={"stage": "LIGHTGCN_TRAINING", "after_epoch": 1},
        )
    assert runner.state.stages["DATA_PREPROCESSING"].state == "COMPLETED"
    assert runner.state.dataset_hashes["synthetic"] == pre_hash
    runner.run(resume=True)
    assert runner.state.stages["FINAL_REPORT"].state == "COMPLETED"
    compliance = json.loads((runner.run_dir / "spec_compliance.json").read_text())
    assert compliance["FIX.PHI"]["status"] == "PASS"
    assert compliance["FIX.I12_NOT_1"]["status"] == "PASS"
    assert compliance["VALALPHA.286"]["status"] == "PASS"

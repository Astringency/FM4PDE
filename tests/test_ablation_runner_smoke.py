from pathlib import Path

from sampling.runner import run_from_config_path


def test_runner_dry_run_writes_metrics(tmp_path):
    result = run_from_config_path(
        "configs/ablations/smoke.yaml",
        overrides={"dry_run": True, "output_dir": str(tmp_path)},
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "metrics_final.json").exists()

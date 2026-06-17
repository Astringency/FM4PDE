import pytest

from fm4pde_ablation.config import AblationConfig, VALID_SENSOR_MODES, load_config, parse_cli_overrides


def test_load_smoke_config():
    cfg = load_config("configs/ablations/smoke.yaml", overrides={"dry_run": True})
    assert cfg.pde == "poisson"
    assert cfg.task == "forward"
    assert cfg.num_steps == 5
    assert cfg.loss_state == "endpoint"


def test_cli_overrides():
    overrides = parse_cli_overrides(["num_steps=20", "--clip-threshold=10", "dry_run=true"])
    cfg = load_config("configs/ablations/smoke.yaml", overrides=overrides)
    assert cfg.num_steps == 20
    assert cfg.clip_threshold == 10
    assert cfg.dry_run is True


def test_invalid_task_guidance_conflict():
    cfg = AblationConfig(task="forward", guidance_components="sol_obs_only")
    with pytest.raises(ValueError):
        cfg.validate()


def test_time_varying_sensor_mode_is_legacy_alias():
    cfg = AblationConfig(sensor_mode="time_varying")
    with pytest.warns(DeprecationWarning):
        cfg.validate()
    assert "time_varying" not in VALID_SENSOR_MODES
    assert cfg.sensor_mode == "per_sample_random"

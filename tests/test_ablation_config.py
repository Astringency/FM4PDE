import pytest

from sampling.config import AblationConfig, VALID_RESIDUAL_MODES, VALID_SENSOR_MODES, load_config, parse_cli_overrides


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


def test_residual_mode_validation_and_aliases():
    cfg = AblationConfig(residual_mode="two_time_level")
    cfg.validate()
    assert cfg.residual_mode == "endpoint_secant"
    assert "hermite_bridge" in VALID_RESIDUAL_MODES

    bad = AblationConfig(residual_mode="near_endpoint_temporal")
    with pytest.raises(ValueError, match="num_near_endpoint_obs"):
        bad.validate()

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


def test_time_varying_sensor_mode_is_rejected():
    cfg = AblationConfig(sensor_mode="time_varying")
    with pytest.raises(ValueError, match="sensor_mode"):
        cfg.validate()
    assert "time_varying" not in VALID_SENSOR_MODES


def test_residual_mode_validation():
    cfg = AblationConfig(residual_mode="endpoint_secant")
    cfg.validate()
    assert cfg.residual_mode == "endpoint_secant"
    assert "hermite_bridge" in VALID_RESIDUAL_MODES
    assert "full_trajectory_fd" in VALID_RESIDUAL_MODES

    alias = AblationConfig(residual_mode="two_time_level")
    with pytest.raises(ValueError, match="residual_mode"):
        alias.validate()

    bad = AblationConfig(residual_mode="near_endpoint_temporal")
    with pytest.raises(ValueError, match="num_near_endpoint_obs"):
        bad.validate()


def test_model_profile_validation():
    cfg = AblationConfig(model_profile="legacy_base")
    cfg.validate()

    bad = AblationConfig(model_profile="old_default")
    with pytest.raises(ValueError, match="model_profile"):
        bad.validate()


def test_load_config_rejects_old_schema(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text(
        "data:\n"
        "  name: heat\n"
        "generate:\n"
        "  num_steps: 10\n"
        "model: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="old data/generate/model schema"):
        load_config(path)


def test_deprecated_loss_type_is_rejected(tmp_path):
    path = tmp_path / "bad_loss.yaml"
    path.write_text(
        "pde: poisson\n"
        "task: forward\n"
        "loss_type: l2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loss_type"):
        load_config(path)


def test_deprecated_loss_type_extra_is_rejected():
    cfg = AblationConfig()
    cfg.extra["loss_type"] = "l2"
    with pytest.raises(ValueError, match="loss_type"):
        cfg.validate()


def test_residual_mode_changes_ablation_name():
    hermite = AblationConfig(residual_mode="hermite_bridge").resolved_ablation_name()
    secant = AblationConfig(residual_mode="endpoint_secant").resolved_ablation_name()
    assert hermite != secant
    assert "res-hermite" in hermite
    assert "res-secant" in secant


def test_boundary_condition_changes_ablation_name():
    periodic = AblationConfig(boundary_condition_mode="periodic").resolved_ablation_name()
    neumann = AblationConfig(boundary_condition_mode="neumann_zero").resolved_ablation_name()
    assert periodic != neumann
    assert "bc-periodic" in periodic
    assert "bc-neumann_zero" in neumann


def test_near_endpoint_obs_changes_ablation_name():
    one = AblationConfig(residual_mode="near_endpoint_temporal", num_near_endpoint_obs=1).resolved_ablation_name()
    two = AblationConfig(residual_mode="near_endpoint_temporal", num_near_endpoint_obs=2).resolved_ablation_name()
    assert one != two
    assert "res-near1" in one
    assert "res-near2" in two

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


def test_runtime_metadata_is_not_serialized_as_input_config():
    cfg = load_config("configs/ablations/smoke.yaml")
    cfg.runtime_metadata["effective_guidance"] = "obs_only"

    assert "runtime_metadata" not in cfg.asdict()


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

    invalid = AblationConfig(residual_mode="two_time_level")
    with pytest.raises(ValueError, match="residual_mode"):
        invalid.validate()

    bad = AblationConfig(residual_mode="near_endpoint_temporal")
    with pytest.raises(ValueError, match="only supported for temporal endpoint PDEs"):
        bad.validate()


def test_sampling_rejects_residual_modes_that_require_ground_truth_time_fields():
    for pde in (
        "heat",
        "wave",
        "advection_diffusion",
        "reaction_diffusion",
        "shallow_water",
        "nsnonbounded",
    ):
        near = AblationConfig(pde=pde, residual_mode="near_endpoint_temporal", num_obs=1)
        near.validate()

    for pde in ("poisson", "burger"):
        near = AblationConfig(pde=pde, residual_mode="near_endpoint_temporal", num_obs=1)
        with pytest.raises(ValueError, match="only supported for temporal endpoint PDEs"):
            near.validate()

    full_trajectory = AblationConfig(pde="heat", residual_mode="full_trajectory_fd")
    with pytest.raises(ValueError, match="Only Burgers"):
        full_trajectory.validate()

    full_time_space = AblationConfig(pde="heat", residual_mode="full_time_space")
    with pytest.raises(ValueError, match="Only Burgers"):
        full_time_space.validate()

    burger = AblationConfig(pde="burger", residual_mode="full_trajectory_fd")
    burger.validate()


def test_model_profile_validation():
    cfg = AblationConfig(model_profile="recommended")
    cfg.validate()

    bad = AblationConfig(model_profile="unsupported")
    with pytest.raises(ValueError, match="model_profile"):
        bad.validate()


def test_load_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "unknown.yaml"
    path.write_text(
        "data:\n"
        "  name: heat\n"
        "generate:\n"
        "  num_steps: 10\n"
        "model: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown config fields"):
        load_config(path)


def test_unknown_loss_field_is_rejected(tmp_path):
    path = tmp_path / "bad_loss.yaml"
    path.write_text(
        "pde: poisson\n"
        "task: forward\n"
        "loss_type: l2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loss_type"):
        load_config(path)


def test_observation_guidance_reduction_is_explicit_and_named_in_ablation():
    mse = AblationConfig(obs_guidance_reduction="mse")
    mse.validate()
    assert "obsred" not in mse.resolved_ablation_name()

    l2 = AblationConfig(obs_guidance_reduction="l2_norm")
    l2.validate()
    assert "obsred-l2_norm" in l2.resolved_ablation_name()

    with pytest.raises(ValueError, match="obs_guidance_reduction"):
        AblationConfig(obs_guidance_reduction="relative_l2").validate()


def test_pde_guidance_gate_validation_and_ablation_name():
    cfg = AblationConfig(
        pde_guidance_start_ratio=0.5,
        pde_guidance_ramp_ratio=0.1,
    )
    cfg.validate()
    assert "pdegate-s0.5-r0.1" in cfg.resolved_ablation_name()
    assert "pdegate" not in AblationConfig().resolved_ablation_name()

    with pytest.raises(ValueError, match="pde_guidance_start_ratio"):
        AblationConfig(pde_guidance_start_ratio=-0.1).validate()
    with pytest.raises(ValueError, match="pde_guidance_ramp_ratio"):
        AblationConfig(pde_guidance_ramp_ratio=1.1).validate()
    with pytest.raises(ValueError, match="must be <= 1"):
        AblationConfig(
            pde_guidance_start_ratio=0.8,
            pde_guidance_ramp_ratio=0.3,
        ).validate()


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
    one = AblationConfig(pde="heat", residual_mode="near_endpoint_temporal", num_obs=1).resolved_ablation_name()
    two = AblationConfig(pde="heat", residual_mode="near_endpoint_temporal", num_obs=2).resolved_ablation_name()
    assert one != two
    assert "res-near-aligned-random1" in one
    assert "res-near-aligned-random2" in two


def test_removed_independent_near_endpoint_mask_config_is_rejected(tmp_path):
    path = tmp_path / "removed_near.yaml"
    path.write_text(
        "pde: heat\nresidual_mode: near_endpoint_temporal\nnum_near_endpoint_obs: 4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown config fields"):
        load_config(path)


def test_sensor_column_has_separate_explicit_budget():
    with pytest.raises(ValueError, match="num_sensor_columns"):
        AblationConfig(sensor_mode="sensor_column").validate()
    cfg = AblationConfig(sensor_mode="sensor_column", num_sensor_columns=3)
    cfg.validate()


def test_next_state_direct_requires_x_next_loss_state():
    with pytest.raises(ValueError, match="loss_state='x_next'"):
        AblationConfig(loss_state="endpoint", gradient_target="next_state_direct").validate()

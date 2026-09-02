from pathlib import Path

from sampling.config import load_config


MAIN_INVERSE_PARAMS = {
    "darcy": (0.0, 128_000_000_000.0, 0.03, 90.0),
    "poisson": (0.0, 5_760_000_000.0, 0.3, 50.0),
    "helmholtz": (0.0, 7_680_000_000.0, 0.03, 50.0),
    "nsnonbounded": (0.0, 240_000.0, 0.3, 100.0),
    "reaction_diffusion": (0.0, 4_000_000.0, 1.0, 50.0),
    "shallow_water": (0.0, 800_000.0, 0.0, 50.0),
    "heat": (0.0, 100_000.0, 0.0, 50.0),
    "wave": (0.0, 100_000.0, 0.1, 50.0),
    "advection_diffusion": (0.0, 1_000_000.0, 0.0, 50.0),
    "steady_heat_conduction": (0.0, 1_000_000.0, 0.0, 50.0),
}


def test_non_burger_main_configs_use_finite_inverse_tuning_params():
    for pde, expected_params in MAIN_INVERSE_PARAMS.items():
        config = load_config(f"configs/main/inverse/{pde}.yaml")
        assert config.task == "inverse"
        assert config.sensor_mode == "random"
        assert config.sampler_phase == "stochastic"
        assert config.num_steps == 100
        assert config.clip_mode == "global_norm"
        assert (
            config.zeta_obs_a,
            config.zeta_obs_u,
            config.zeta_pde,
            config.clip_threshold,
        ) == expected_params


def test_burger_main_config_retains_tuned_random_and_column_settings():
    config = load_config("configs/main/both/burger.yaml")
    assert config.task == "both"
    assert config.sensor_mode == "random"
    assert config.num_sensor_columns == 16
    assert config.sampler_phase == "stochastic"
    assert config.num_steps == 100
    assert config.clip_mode == "global_norm"
    assert config.clip_threshold == 50.0
    assert (config.zeta_obs_a, config.zeta_obs_u, config.zeta_pde) == (
        0.0,
        409_600.0,
        10.0,
    )


def test_main_config_tree_has_task_specific_files_and_burger_only_in_both():
    non_burger_pdes = set(MAIN_INVERSE_PARAMS)
    for task in ("both", "forward", "inverse"):
        task_paths = list(Path(f"configs/main/{task}").glob("*.yaml"))
        task_pdes = {path.stem for path in task_paths}
        assert non_burger_pdes <= task_pdes
        assert all(load_config(path).task == task for path in task_paths)
    assert Path("configs/main/both/burger.yaml").is_file()
    assert not Path("configs/main/forward/burger.yaml").exists()
    assert not Path("configs/main/inverse/burger.yaml").exists()

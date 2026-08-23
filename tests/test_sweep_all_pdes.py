from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

from sampling.config import load_config
from sampling.sweep import expand_grid


GRID = "configs/ablations/all_internal_ablation_grid.yaml"
FOCUSED_GRID = "configs/ablations/all_ablation_grid.yaml"
ALL_PDES = {
    "darcy", "poisson", "helmholtz", "nsnonbounded", "burger",
    "reaction_diffusion", "shallow_water", "heat", "wave",
    "advection_diffusion", "steady_heat_conduction",
}
TEMPORAL_PDES = {
    "heat", "wave", "advection_diffusion", "reaction_diffusion",
    "shallow_water", "nsnonbounded",
}
EXPECTED_PER_PDE = {
    "guidance_components": 12,
    "loss_state_by_sampler": 6,
    "sampler_phase": 8,
    "time_grid_by_sampler": 12,
    "num_steps_by_sampler": 28,
    "step_method_by_sampler": 8,
    "sensor_sparsity": 5,
    "sensor_mode": 5,
    "noise_robustness": 4,
    "statistics_stability": 5,
}


def _jobs(group: str, pde: str = "poisson"):
    return expand_grid(GRID, selected_groups={group}, selected_pdes={pde})


def test_formal_grid_has_exact_groups_and_job_counts():
    jobs = expand_grid(GRID)
    counts = Counter(params["ablation_group"] for _, params in jobs)
    expected = {group: count * len(ALL_PDES) for group, count in EXPECTED_PER_PDE.items()}
    expected["temporal_residual_mode"] = 3 * len(TEMPORAL_PDES)
    assert counts == expected
    assert len(jobs) == 1041
    assert len(expand_grid(FOCUSED_GRID)) == 111


def test_guidance_components_cross_tasks_and_components():
    combinations = {(p["task"], p["guidance_components"]) for _, p in _jobs("guidance_components")}
    assert combinations == {
        (task, component)
        for task in ("both", "forward", "inverse")
        for component in ("noguide", "obs_only", "pde_only", "obs_pde")
    }


def test_loss_state_and_sampler_phase_matrices():
    assert {(p["loss_state"], p["sampler_phase"]) for _, p in _jobs("loss_state_by_sampler")} == {
        (state, phase)
        for state in ("xt", "x_next", "endpoint")
        for phase in ("deterministic", "stochastic")
    }
    assert {(p["sampler_phase"], p["switch_ratio"]) for _, p in _jobs("sampler_phase")} == {
        ("deterministic", 0.5),
        ("stochastic", 0.5),
        ("hybrid_d2s", 0.2),
        ("hybrid_d2s", 0.5),
        ("hybrid_d2s", 0.8),
        ("hybrid_s2d", 0.2),
        ("hybrid_s2d", 0.5),
        ("hybrid_s2d", 0.8),
    }


def test_poisson_main_inverse_tuning_and_ablation_baseline_are_locked():
    expected_baseline = (50000.0, 90000000.0, 0.1)
    main = load_config("configs/main/inverse/poisson.yaml")
    assert (main.zeta_obs_a, main.zeta_obs_u, main.zeta_pde) == (0.0, 360000000.0, 0.3)
    for task in ("forward", "inverse", "both"):
        cfg = load_config(f"configs/ablations/base/poisson_{task}.yaml")
        assert (cfg.zeta_obs_a, cfg.zeta_obs_u, cfg.zeta_pde) == expected_baseline

    resolved = {}
    for path, overrides in _jobs("loss_state_by_sampler"):
        cfg = load_config(path, overrides=overrides)
        resolved[(cfg.loss_state, cfg.sampler_phase)] = (
            cfg.zeta_obs_a,
            cfg.zeta_obs_u,
            cfg.zeta_pde,
        )
    assert resolved[("x_next", "stochastic")] == (600000.0, 2000000000.0, 1.0)
    assert all(
        zeta == expected_baseline
        for variant, zeta in resolved.items()
        if variant != ("x_next", "stochastic")
    )

    darcy = expand_grid(GRID, selected_groups={"loss_state_by_sampler"}, selected_pdes={"darcy"})
    assert all("zeta_obs_a" not in overrides for _, overrides in darcy)

    forced = expand_grid(
        GRID,
        selected_groups={"loss_state_by_sampler"},
        selected_pdes={"poisson"},
        global_overrides={"zeta_obs_a": 123.0},
    )
    assert all(overrides["zeta_obs_a"] == 123.0 for _, overrides in forced)


def test_time_discretization_is_three_independent_ablations():
    phases = {"deterministic", "stochastic", "hybrid_d2s", "hybrid_s2d"}
    grid_jobs = _jobs("time_grid_by_sampler")
    assert {(p["time_grid"], p["sampler_phase"]) for _, p in grid_jobs} == {
        (grid, phase) for grid in ("uniform", "geometric", "cosine") for phase in phases
    }
    assert all(p["num_steps"] == 100 and p["step_method"] == "euler" for _, p in grid_jobs)

    step_jobs = _jobs("num_steps_by_sampler")
    assert {(p["num_steps"], p["sampler_phase"]) for _, p in step_jobs} == {
        (steps, phase) for steps in (10, 50, 100, 200, 500, 1000, 2000) for phase in phases
    }
    assert all(p["time_grid"] == "uniform" and p["step_method"] == "euler" for _, p in step_jobs)

    method_jobs = _jobs("step_method_by_sampler")
    assert {(p["step_method"], p["sampler_phase"]) for _, p in method_jobs} == {
        (method, phase) for method in ("euler", "midpoint") for phase in phases
    }
    assert all(p["time_grid"] == "uniform" and p["num_steps"] == 100 for _, p in method_jobs)
    assert all(p["switch_ratio"] == 0.5 for _, p in grid_jobs + step_jobs + method_jobs)


def test_observation_and_stability_groups_use_locked_baselines():
    sparsity = _jobs("sensor_sparsity")
    assert {p["num_obs"] for _, p in sparsity} == {50, 100, 250, 500, 1000}
    assert all(p["sampler_phase"] == "stochastic" for _, p in sparsity)

    assert {p["sensor_mode"] for _, p in _jobs("sensor_mode")} == {
        "random", "fixed", "grid", "sensor_column", "per_sample_random"
    }
    assert {p["noise_level"] for _, p in _jobs("noise_robustness")} == {0.0, 0.01, 0.05, 0.10}

    stability = _jobs("statistics_stability")
    assert [(p["sample_seed"], p["mask_seed"]) for _, p in stability] == [
        (0, 0), (1, 11), (2, 22), (3, 33), (4, 44)
    ]
    assert all(p["offset"] == 0 and p["noise_seed"] == 0 for _, p in stability)


def test_temporal_residual_modes_are_comparable_and_scoped():
    jobs = expand_grid(GRID, selected_groups={"temporal_residual_mode"})
    assert {load_config(path).pde for path, _ in jobs} == TEMPORAL_PDES
    assert {p["residual_mode"] for _, p in jobs} == {
        "endpoint_secant", "hermite_bridge", "near_endpoint_temporal"
    }
    assert all(p["num_obs"] == 500 and p["sampler_phase"] == "stochastic" for _, p in jobs)


def test_pde_filter_and_global_overrides():
    jobs = expand_grid(
        GRID,
        selected_groups={"guidance_components"},
        selected_pdes={"poisson", "heat"},
        global_overrides={"output_dir": "custom", "device": "cpu", "batch_size": 3},
    )
    assert len(jobs) == 24
    assert {load_config(path).pde for path, _ in jobs} == {"poisson", "heat"}
    assert all(
        p["output_dir"] == "custom" and p["device"] == "cpu" and p["batch_size"] == 3
        for _, p in jobs
    )


def test_sweep_cli_lists_selected_jobs_and_rejects_unknown_filters():
    result = subprocess.run(
        [
            sys.executable, "-m", "sampling.sweep", "--grid", GRID,
            "--pde", "poisson", "--group", "guidance_components", "--list",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert len(result.stdout.splitlines()) == 12
    assert all("poisson_both.yaml" in line for line in result.stdout.splitlines())

    bad = subprocess.run(
        [sys.executable, "-m", "sampling.sweep", "--grid", GRID, "--pde", "not_a_pde", "--list"],
        check=False,
        text=True,
        stderr=subprocess.PIPE,
    )
    assert bad.returncode != 0
    assert "Unknown PDE" in bad.stderr


def test_shell_wrapper_plan_only_filters_pde_and_group():
    env = os.environ.copy()
    env.update({"PDE_LIST": "poisson", "PLAN_ONLY": "true"})
    result = subprocess.run(
        ["bash", "scripts/run_ablations.sh", "guidance_components"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        env=env,
    )
    assert len(result.stdout.splitlines()) == 12
    assert all("poisson_both.yaml" in line for line in result.stdout.splitlines())


def test_topic_grids_match_focused_grid_subsets():
    topic_files = {
        "configs/ablations/main_guidance_components.yaml": {"guidance_components"},
        "configs/ablations/main_loss_state.yaml": {"loss_state_by_sampler"},
        "configs/ablations/main_sampler_phase.yaml": {"sampler_phase"},
        "configs/ablations/main_steps_timegrid.yaml": {
            "time_grid_by_sampler", "num_steps_by_sampler", "step_method_by_sampler",
        },
        "configs/ablations/main_sensor_noise.yaml": {
            "sensor_sparsity", "sensor_mode", "noise_robustness",
        },
        "configs/ablations/main_temporal_residual.yaml": {"temporal_residual_mode"},
        "configs/ablations/main_statistics_stability.yaml": {"statistics_stability"},
    }
    for topic_file, groups in topic_files.items():
        topic = expand_grid(topic_file)
        focused = expand_grid(FOCUSED_GRID, selected_groups=groups)
        topic_variants = [{k: v for k, v in params.items() if k != "ablation_name"} for _, params in topic]
        focused_variants = [{k: v for k, v in params.items() if k != "ablation_name"} for _, params in focused]
        assert topic_variants == focused_variants


def test_every_formal_job_validates_and_checkpoints_exist():
    checked_paths = set()
    for path, overrides in expand_grid(GRID):
        cfg = load_config(path, overrides=overrides)
        assert cfg.model_profile == "recommended"
        checked_paths.add(path)
    for path in checked_paths:
        checkpoint = Path(load_config(path).checkpoint_path)
        assert checkpoint.is_file(), f"Missing checkpoint for {path}: {checkpoint}"


def test_removed_experiment_definitions_are_absent():
    for name in (
        "main_guidance_schedule.yaml",
        "main_clipping.yaml",
        "main_pde_residual_region.yaml",
        "main_zeta_sensitivity.yaml",
    ):
        assert not Path("configs/ablations", name).exists()

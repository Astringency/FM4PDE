from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sampling.config import dump_yaml, load_config, load_yaml_file
from sampling.data import finalize_ground_truth_config
from sampling.sweep import expand_grid, find_matching_completed_run


GRID = "configs/ablations/all_internal_ablation_grid.yaml"
FOCUSED_GRID = "configs/ablations/all_ablation_grid.yaml"
DETERMINISTIC_SAFE_GRID = "configs/ablations/main_deterministic_safe.yaml"
POISSON_SAMPLER_COMPARISON_GRID = "configs/ablations/poisson_sampler_comparison.yaml"
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
    expected["guidance_components"] -= 8
    expected["deterministic_endpoint_bt"] = 27
    expected["temporal_residual_mode"] = 3 * len(TEMPORAL_PDES)
    assert counts == expected
    assert len(jobs) == 1060
    assert len(expand_grid(FOCUSED_GRID)) == 138


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


def test_poisson_deterministic_endpoint_bt_matrix_is_paired_and_endpoint_guided():
    jobs = _jobs("deterministic_endpoint_bt")
    assert len(jobs) == 27
    assert {path for path, _ in jobs} == {"configs/main/both/poisson.yaml"}
    assert all(
        params["task"] == "both"
        and params["guidance_components"] == "obs_pde"
        and params["loss_state"] == "endpoint"
        and params["time_grid"] == "uniform"
        and params["num_steps"] == 100
        for _, params in jobs
    )
    assert sum(params["sampler_phase"] == "stochastic" for _, params in jobs) == 1
    deterministic = [params for _, params in jobs if params["sampler_phase"] == "deterministic"]
    for mode in ("single_step", "rollout"):
        variants = [params for params in deterministic if params["deterministic_endpoint_mode"] == mode]
        assert len(variants) == {"single_step": 11, "rollout": 15}[mode]
        assert {params["deterministic_bt_mode"] for params in variants} == {
            "legacy", "zero_at_t0", "t_next", "clipped", "clipped_zero_at_t0",
            "stochastic_like", "capped_stochastic_like"
        }
        assert all(
            params["deterministic_rollout_checkpoint"] is (mode == "rollout")
            for params in variants
        )
        capped = [
            params for params in variants
            if params["deterministic_bt_mode"] == "capped_stochastic_like"
        ]
        assert len(capped) == {"single_step": 2, "rollout": 5}[mode]
        assert {params["zeta_pde"] for params in capped} == {0.1, 3.0}
        expected_caps = {0.0075} if mode == "single_step" else {0.001, 0.003, 0.005}
        assert {params["deterministic_bt_max_scale"] for params in capped} == expected_caps


def test_safe_deterministic_transfer_grid_compares_both_endpoint_predictors():
    jobs = expand_grid(DETERMINISTIC_SAFE_GRID, selected_pdes={"poisson"})
    assert len(jobs) == 3
    by_group = Counter(params["ablation_group"] for _, params in jobs)
    assert by_group == {"deterministic_safe_single": 2, "deterministic_safe_rollout": 1}

    stochastic = [params for _, params in jobs if params["sampler_phase"] == "stochastic"]
    deterministic = [params for _, params in jobs if params["sampler_phase"] == "deterministic"]
    assert len(stochastic) == 1
    assert {params["deterministic_endpoint_mode"] for params in deterministic} == {
        "single_step",
        "rollout",
    }
    assert all(
        params["deterministic_bt_mode"] == "clipped_zero_at_t0"
        for params in deterministic
    )
    assert all(
        params["deterministic_bt_max_scale"] == pytest.approx(0.0125)
        for params in deterministic
    )
    assert all(
        params["deterministic_guidance_start_ratio"] == pytest.approx(0.02)
        for params in deterministic
    )
    assert all(
        params["deterministic_guidance_ramp_ratio"] == pytest.approx(0.04)
        for params in deterministic
    )
    assert all(
        params["deterministic_correction_max_rms"] == pytest.approx(0.02)
        for params in deterministic
    )
    assert all("zeta_pde" not in params for params in deterministic)
    assert "zeta_pde" not in stochastic[0]
    assert all(
        load_config(path, overrides=params).zeta_pde == pytest.approx(0.1)
        for path, params in jobs
    )


def test_poisson_sampler_comparison_crosses_tasks_and_sampler_phases():
    jobs = expand_grid(POISSON_SAMPLER_COMPARISON_GRID)

    assert len(jobs) == 12
    assert {(params["task"], params["sampler_phase"]) for _, params in jobs} == {
        (task, phase)
        for task in ("both", "forward", "inverse")
        for phase in ("stochastic", "deterministic", "hybrid_d2s", "hybrid_s2d")
    }
    assert all(params["switch_ratio"] == pytest.approx(0.5) for _, params in jobs)
    expected_zeta_pde = {"both": 0.1, "forward": 0.1, "inverse": 0.3}
    assert all(
        load_config(path, overrides=params).zeta_pde
        == pytest.approx(expected_zeta_pde[params["task"]])
        for path, params in jobs
    )
    assert all(params["deterministic_bt_mode"] == "clipped_zero_at_t0" for _, params in jobs)
    assert all(params["deterministic_bt_max_scale"] == pytest.approx(0.0125) for _, params in jobs)


def test_poisson_ablations_inherit_task_specific_main_tuning():
    expected_by_task = {
        "both": (50000.0, 90000000.0, 0.1),
        "forward": (50000.0, 90000000.0, 0.1),
        "inverse": (0.0, 5760000000.0, 0.3),
    }
    guidance_jobs = _jobs("guidance_components")
    for path, overrides in guidance_jobs:
        task = overrides["task"]
        assert path == f"configs/main/{task}/poisson.yaml"
        cfg = load_config(path, overrides=overrides)
        assert (cfg.zeta_obs_a, cfg.zeta_obs_u, cfg.zeta_pde) == expected_by_task[task]
        assert cfg.data_path.endswith("/poisson/poisson_test_10000-128-128_id.mat")

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
        zeta == expected_by_task["both"]
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

    forced_task = expand_grid(
        GRID,
        selected_groups={"sampler_phase"},
        selected_pdes={"poisson"},
        global_overrides={"task": "inverse"},
    )
    assert all(path == "configs/main/inverse/poisson.yaml" for path, _ in forced_task)
    assert all(
        load_config(path, overrides=overrides).zeta_obs_u == 5760000000.0
        for path, overrides in forced_task
    )


def test_burger_has_only_both_task_jobs():
    jobs = expand_grid(
        GRID,
        selected_groups={"guidance_components"},
        selected_pdes={"burger"},
    )
    assert len(jobs) == 4
    assert {overrides["task"] for _, overrides in jobs} == {"both"}
    assert {path for path, _ in jobs} == {"configs/main/both/burger.yaml"}
    assert {
        (
            load_config(path, overrides=overrides).zeta_obs_a,
            load_config(path, overrides=overrides).zeta_obs_u,
            load_config(path, overrides=overrides).zeta_pde,
        )
        for path, overrides in jobs
    } == {(0.0, 409600.0, 10.0)}


def test_resume_reuses_only_matching_complete_successful_runs(tmp_path):
    config_path, overrides = _jobs("guidance_components")[0]
    overrides = {**overrides, "output_dir": str(tmp_path), "device": "cuda:1"}
    cfg = finalize_ground_truth_config(load_config(config_path, overrides=overrides))
    run_dir = (
        tmp_path
        / cfg.pde
        / cfg.task
        / cfg.resolved_ablation_name()
        / "completed"
    )
    run_dir.mkdir(parents=True)
    saved = cfg.asdict()
    saved["device"] = "cuda:0"
    (run_dir / "resolved_config.yaml").write_text(dump_yaml(saved), encoding="utf-8")
    (run_dir / "metrics_final.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "pde_residual_status": "reliable",
                "num_samples": cfg.batch_size,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "result.pt").write_bytes(b"result")
    with (run_dir / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index"])
        writer.writeheader()
        writer.writerow({"sample_index": 0})

    assert find_matching_completed_run(cfg) == run_dir

    changed = finalize_ground_truth_config(load_config(config_path, overrides=overrides))
    changed.zeta_pde += 1.0
    assert find_matching_completed_run(changed) is None

    (run_dir / "result.pt").unlink()
    assert find_matching_completed_run(cfg) is None


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
    lines = result.stdout.splitlines()
    assert sum("configs/main/both/poisson.yaml" in line for line in lines) == 4
    assert sum("configs/main/forward/poisson.yaml" in line for line in lines) == 4
    assert sum("configs/main/inverse/poisson.yaml" in line for line in lines) == 4

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
    lines = result.stdout.splitlines()
    assert sum("configs/main/both/poisson.yaml" in line for line in lines) == 4
    assert sum("configs/main/forward/poisson.yaml" in line for line in lines) == 4
    assert sum("configs/main/inverse/poisson.yaml" in line for line in lines) == 4


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
        "configs/ablations/main_deterministic_endpoint_bt.yaml": {"deterministic_endpoint_bt"},
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


def test_formal_grids_do_not_reference_duplicated_ablation_bases():
    assert not list(Path("configs/ablations/base").glob("*.yaml"))
    for grid_path in Path("configs/ablations").glob("*.yaml"):
        assert "configs/ablations/base/" not in grid_path.read_text(encoding="utf-8")

    suite = load_yaml_file("configs/ablations/formal_suite.yaml")
    inherited_overrides = [suite.get("common_overrides", {})]
    inherited_overrides.extend(suite.get("pde_overrides", {}).values())
    assert all(
        not {"zeta_obs_a", "zeta_obs_u", "zeta_pde"}.intersection(overrides)
        for overrides in inherited_overrides
    )


def test_sweep_requires_main_config_source(tmp_path):
    grid = tmp_path / "grid.yaml"
    grid.write_text("pdes: [poisson]\ngroups:\n  check: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="main_config_root"):
        expand_grid(str(grid))


def test_removed_experiment_definitions_are_absent():
    for name in (
        "main_guidance_schedule.yaml",
        "main_clipping.yaml",
        "main_pde_residual_region.yaml",
        "main_zeta_sensitivity.yaml",
    ):
        assert not Path("configs/ablations", name).exists()

import csv
import json

import pytest

from sampling.aggregate import aggregate_root
from sampling.config import dump_yaml


def _write_run(root, name, rel_l2_a, rel_l2_u, config_updates=None):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    config = {
        "pde": "poisson",
        "task": "both",
        "ablation_name": "same",
        "guidance_components": "obs_pde",
        "loss_state": "endpoint",
        "sampler_phase": "stochastic",
        "switch_ratio": 0.5,
        "sensor_mode": "random",
        "num_obs": 10,
        "noise_level": 0.0,
        "zeta_obs_a": 1,
        "zeta_obs_u": 1,
        "zeta_pde": 1,
        "num_steps": 2,
        "time_grid": "uniform",
        "step_method": "euler",
        "ablation_group": "group",
    }
    for key, value in (config_updates or {}).items():
        config[key] = value
    (run_dir / "resolved_config.yaml").write_text(dump_yaml(config), encoding="utf-8")
    metrics = {
        "rel_l2_a": rel_l2_a,
        "rel_l2_u": rel_l2_u,
        "obs_rel_l2_a": rel_l2_a,
        "obs_rel_l2_u": rel_l2_u,
        "L_obs_a": rel_l2_a + 0.1,
        "L_obs_u": rel_l2_u + 0.2,
        "clean_L_obs_a": rel_l2_a,
        "clean_L_obs_u": rel_l2_u,
        "L_pde": 0.5,
        "pde_residual_norm": 0.25,
        "wall_clock_time": 10.0,
        "status": "ok",
        "pde_residual_status": "reliable",
    }
    (run_dir / "metrics_final.json").write_text(json.dumps(metrics), encoding="utf-8")
    with (run_dir / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    with (run_dir / "metrics_step.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 0, "rel_l2_a": rel_l2_a, "rel_l2_u": rel_l2_u, "status": "ok", "pde_residual_status": "reliable"}) + "\n")


def test_aggregate_outputs_statistics(tmp_path):
    _write_run(tmp_path, "run1", 1.0, 2.0)
    _write_run(tmp_path, "run2", 3.0, 4.0)

    outputs = aggregate_root(tmp_path)

    grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))
    run_grouped = list(csv.DictReader(outputs["run_seed_grouped"].open(encoding="utf-8")))
    assert len(grouped) == 1
    assert len(run_grouped) == 1
    row = grouped[0]
    assert float(row["rel_l2_a_mean"]) == pytest.approx(2.0)
    assert float(row["rel_l2_a_std"]) == pytest.approx(2**0.5)
    assert int(row["rel_l2_a_n"]) == 2
    assert float(row["rel_l2_a_sem"]) == pytest.approx(1.0)
    assert float(row["rel_l2_a_ci95"]) == pytest.approx(1.96)
    assert float(row["rel_l2_a_median"]) == pytest.approx(2.0)
    assert float(row["rel_l2_a_p90"]) == pytest.approx(2.8)
    assert float(row["rel_l2_a_min"]) == pytest.approx(1.0)
    assert float(row["rel_l2_a_max"]) == pytest.approx(3.0)
    assert float(run_grouped[0]["L_obs_a_mean"]) == pytest.approx(2.1)
    assert float(run_grouped[0]["L_obs_u_mean"]) == pytest.approx(3.2)
    assert float(run_grouped[0]["L_pde_mean"]) == pytest.approx(0.5)

    curves = list(csv.DictReader(outputs["curves"].open(encoding="utf-8")))
    assert len(curves) == 1
    assert float(curves[0]["rel_l2_u_mean"]) == pytest.approx(3.0)


def test_latest_outputs_exclude_superseded_runs_without_dropping_history(tmp_path):
    _write_run(tmp_path, "run1", 1.0, 2.0)
    _write_run(tmp_path, "run2", 3.0, 4.0)
    # Make the intended current result deterministic even on coarse-mtime file systems.
    metrics_path = tmp_path / "run2" / "metrics_final.json"
    metrics_path.touch()

    outputs = aggregate_root(tmp_path)

    history = list(csv.DictReader(outputs["raw"].open(encoding="utf-8")))
    latest = list(csv.DictReader(outputs["latest"].open(encoding="utf-8")))
    excluded = list(csv.DictReader(outputs["excluded"].open(encoding="utf-8")))
    latest_samples = list(csv.DictReader(outputs["latest_sample_raw"].open(encoding="utf-8")))
    latest_sample_grouped = list(csv.DictReader(outputs["latest_grouped"].open(encoding="utf-8")))
    latest_grouped = list(csv.DictReader(outputs["latest_run_seed_grouped"].open(encoding="utf-8")))
    latest_curves = list(csv.DictReader(outputs["latest_curves"].open(encoding="utf-8")))
    report_metrics = list(csv.DictReader(outputs["report_metrics"].open(encoding="utf-8")))

    assert len(history) == 2
    assert len(latest) == 1
    assert latest[0]["run_dir"].endswith("run2")
    assert len(excluded) == 1
    assert excluded[0]["exclusion_reason"] == "superseded_by_newer_successful_run"
    assert excluded[0]["excluded_run_dir"].endswith("run1")
    assert len(latest_samples) == 1
    assert latest_samples[0]["run_dir"].endswith("run2")
    assert float(latest_sample_grouped[0]["rel_l2_a_mean"]) == pytest.approx(3.0)
    assert float(latest_grouped[0]["rel_l2_a_mean"]) == pytest.approx(3.0)
    assert int(latest_grouped[0]["rel_l2_a_n"]) == 1
    assert len(latest_curves) == 1
    assert float(latest_curves[0]["rel_l2_u_mean"]) == pytest.approx(4.0)
    assert len(report_metrics) == 1
    assert list(report_metrics[0]) == [
        "pde",
        "task",
        "ablation_group",
        "ablation_name",
        "sample_seed",
        "guidance_components",
        "loss_state",
        "sampler_phase",
        "switch_ratio",
        "time_grid",
        "num_steps",
        "step_method",
        "deterministic_endpoint_mode",
        "deterministic_bt_mode",
        "deterministic_guidance_coeff",
        "deterministic_bt_max_scale",
        "deterministic_guidance_start_ratio",
        "deterministic_guidance_ramp_ratio",
        "deterministic_correction_max_rms",
        "deterministic_numerical_guard",
        "sensor_mode",
        "num_obs",
        "noise_level",
        "residual_mode",
        "resolved_residual_mode",
        "rel_l2_a",
        "rel_l2_u",
        "L_obs_a",
        "L_obs_u",
        "L_pde",
        "run_dir",
        "metrics_path",
    ]
    assert float(report_metrics[0]["rel_l2_a"]) == pytest.approx(3.0)
    assert float(report_metrics[0]["rel_l2_u"]) == pytest.approx(4.0)
    assert float(report_metrics[0]["L_obs_a"]) == pytest.approx(3.1)
    assert float(report_metrics[0]["L_obs_u"]) == pytest.approx(4.2)
    assert float(report_metrics[0]["L_pde"]) == pytest.approx(0.5)


def test_failed_pde_evaluations_remain_in_raw_output_but_are_not_aggregated(tmp_path):
    _write_run(tmp_path, "ok", 1.0, 2.0)
    _write_run(tmp_path, "failed", 100.0, 200.0)
    metrics_path = tmp_path / "failed" / "metrics_final.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    # A run-level success status does not override a failed PDE evaluation.
    metrics["status"] = "ok"
    metrics["pde_residual_status"] = "error"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    outputs = aggregate_root(tmp_path)
    raw = list(csv.DictReader(outputs["raw"].open(encoding="utf-8")))
    grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))

    assert len(raw) == 2
    assert len(grouped) == 1
    assert float(grouped[0]["rel_l2_a_mean"]) == pytest.approx(1.0)
    assert int(grouped[0]["rel_l2_a_n"]) == 1


def test_statistics_stability_groups_same_offset_across_seeds(tmp_path):
    for idx, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        _write_run(
            tmp_path,
            f"run{idx}",
            value,
            value,
            {
                "ablation_name": f"statistics_stability_poisson_{idx:03d}",
                "sample_seed": idx,
                "mask_seed": idx * 11,
                "noise_seed": 0,
                "offset": 0,
                "batch_size": 1,
                "noise_level": 0.0,
                "sensor_mode": "per_sample_random",
                    "ablation_group": "statistics_stability",
            },
        )

    outputs = aggregate_root(tmp_path)
    raw = list(csv.DictReader(outputs["raw"].open(encoding="utf-8")))
    grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))

    assert len(raw) == 5
    assert {row["ablation_name"] for row in raw} == {
        "statistics_stability_poisson_000",
        "statistics_stability_poisson_001",
        "statistics_stability_poisson_002",
        "statistics_stability_poisson_003",
        "statistics_stability_poisson_004",
    }
    assert {row["sample_seed"] for row in raw} == {"0", "1", "2", "3", "4"}
    assert {row["mask_seed"] for row in raw} == {"0", "11", "22", "33", "44"}
    assert {row["noise_seed"] for row in raw} == {"0"}
    assert {row["offset"] for row in raw} == {"0"}
    assert len(grouped) == 1
    assert grouped[0]["ablation_family"] == "poisson|both|statistics_stability"
    assert int(grouped[0]["rel_l2_a_n"]) == 5
    for raw_only_key in ("sample_seed", "mask_seed", "noise_seed", "offset", "batch_size", "ablation_name"):
        assert raw_only_key not in grouped[0]


def test_residual_mode_is_grouped_separately(tmp_path):
    _write_run(tmp_path, "endpoint", 1.0, 1.0, {"residual_mode": "endpoint_secant"})
    _write_run(tmp_path, "hermite", 2.0, 2.0, {"residual_mode": "hermite_bridge"})

    outputs = aggregate_root(tmp_path)
    grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))

    assert len(grouped) == 2
    assert {row["residual_mode"] for row in grouped} == {"endpoint_secant", "hermite_bridge"}


def test_gradient_target_is_grouped_separately(tmp_path):
    _write_run(tmp_path, "chain", 1.0, 1.0, {"gradient_target": "current_state_chain_rule"})
    _write_run(tmp_path, "direct", 2.0, 2.0, {"gradient_target": "loss_state_direct"})

    outputs = aggregate_root(tmp_path)
    grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))

    assert len(grouped) == 2
    assert {row["gradient_target"] for row in grouped} == {"current_state_chain_rule", "loss_state_direct"}


def test_stochastic_guidance_time_is_grouped_separately(tmp_path):
    _write_run(tmp_path, "t", 1.0, 1.0, {"stochastic_guidance_time": "t"})
    _write_run(tmp_path, "t_next", 2.0, 2.0, {"stochastic_guidance_time": "t_next"})

    outputs = aggregate_root(tmp_path)
    grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))

    assert len(grouped) == 2
    assert {row["stochastic_guidance_time"] for row in grouped} == {"t", "t_next"}


def test_list_group_values_are_stably_serialized(tmp_path):
    _write_run(tmp_path, "list", 1.0, 1.0, {"hermite_collocation_times": [0.25, 0.5, 0.75]})
    _write_run(tmp_path, "string", 2.0, 2.0, {"hermite_collocation_times": "[0.25,0.5,0.75]"})

    outputs = aggregate_root(tmp_path)
    grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))

    assert len(grouped) == 1
    assert grouped[0]["hermite_collocation_times"] == "[0.25,0.5,0.75]"
    assert int(grouped[0]["rel_l2_a_n"]) == 2


def test_cross_run_grouped_metrics_are_sample_weighted_with_separate_run_stats(tmp_path):
    _write_run(tmp_path, "small", 1.0, 1.0)
    _write_run(tmp_path, "large", 3.0, 3.0)
    for name, values in (("small", [1.0]), ("large", [3.0, 3.0, 3.0])):
        path = tmp_path / name / "metrics_per_sample.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_index", "sample_id", "rel_l2_a", "rel_l2_u",
                        "obs_rel_l2_a", "obs_rel_l2_u", "pde_residual_norm",
                        "status", "pde_residual_status",
                ],
            )
            writer.writeheader()
            for index, value in enumerate(values):
                writer.writerow(
                    {
                        "sample_index": index,
                        "sample_id": f"{name}-{index}",
                        "rel_l2_a": value,
                        "rel_l2_u": value,
                        "obs_rel_l2_a": value,
                        "obs_rel_l2_u": value,
                            "pde_residual_norm": value,
                            "status": "ok",
                            "pde_residual_status": "reliable",
                    }
                )

    outputs = aggregate_root(tmp_path)
    sample_grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))[0]
    run_grouped = list(csv.DictReader(outputs["run_seed_grouped"].open(encoding="utf-8")))[0]
    assert float(sample_grouped["rel_l2_a_mean"]) == pytest.approx(2.5)
    assert int(sample_grouped["rel_l2_a_n"]) == 4
    assert float(run_grouped["rel_l2_a_mean"]) == pytest.approx(2.0)
    assert int(run_grouped["rel_l2_a_n"]) == 2

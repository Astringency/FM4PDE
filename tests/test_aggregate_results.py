import csv
import json

import pytest

from fm4pde_ablation.aggregate import aggregate_root
from fm4pde_ablation.config import dump_yaml


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
        "extra": {"ablation_group": "group"},
    }
    for key, value in (config_updates or {}).items():
        if key == "extra":
            config["extra"].update(value)
        else:
            config[key] = value
    (run_dir / "resolved_config.yaml").write_text(dump_yaml(config), encoding="utf-8")
    metrics = {
        "rel_l2_a": rel_l2_a,
        "rel_l2_u": rel_l2_u,
        "obs_rel_l2_a": rel_l2_a,
        "obs_rel_l2_u": rel_l2_u,
        "clean_L_obs_a": rel_l2_a,
        "clean_L_obs_u": rel_l2_u,
        "L_pde": 0.5,
        "pde_residual_norm": 0.25,
        "wall_clock_time": 10.0,
    }
    (run_dir / "metrics_final.json").write_text(json.dumps(metrics), encoding="utf-8")
    with (run_dir / "metrics_step.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 0, "rel_l2_a": rel_l2_a, "rel_l2_u": rel_l2_u}) + "\n")


def test_aggregate_outputs_statistics(tmp_path):
    _write_run(tmp_path, "run1", 1.0, 2.0)
    _write_run(tmp_path, "run2", 3.0, 4.0)

    outputs = aggregate_root(tmp_path)

    grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))
    assert len(grouped) == 1
    row = grouped[0]
    assert float(row["rel_l2_a_mean"]) == pytest.approx(2.0)
    assert float(row["rel_l2_a_std"]) == pytest.approx(2**0.5)
    assert int(row["rel_l2_a_n"]) == 2
    assert float(row["rel_l2_a_sem"]) == pytest.approx(1.0)
    assert float(row["rel_l2_a_ci95"]) == pytest.approx(1.96)
    assert float(row["rel_l2_a_median"]) == pytest.approx(2.0)
    assert float(row["rel_l2_a_min"]) == pytest.approx(1.0)
    assert float(row["rel_l2_a_max"]) == pytest.approx(3.0)

    curves = list(csv.DictReader(outputs["curves"].open(encoding="utf-8")))
    assert len(curves) == 1
    assert float(curves[0]["rel_l2_u_mean"]) == pytest.approx(3.0)


def test_statistics_seed_offset_groups_across_names_and_seeds(tmp_path):
    for idx, value in enumerate([1.0, 2.0, 3.0]):
        _write_run(
            tmp_path,
            f"run{idx}",
            value,
            value,
            {
                "ablation_name": f"statistics_seed_offset_poisson_{idx:03d}",
                "sample_seed": idx,
                "mask_seed": idx * 11,
                "noise_seed": idx * 101,
                "offset": idx,
                "batch_size": 1,
                "noise_level": 0.01,
                "sensor_mode": "per_sample_random",
                "extra": {"ablation_group": "statistics_seed_offset"},
            },
        )

    outputs = aggregate_root(tmp_path)
    raw = list(csv.DictReader(outputs["raw"].open(encoding="utf-8")))
    grouped = list(csv.DictReader(outputs["grouped"].open(encoding="utf-8")))

    assert len(raw) == 3
    assert {row["ablation_name"] for row in raw} == {
        "statistics_seed_offset_poisson_000",
        "statistics_seed_offset_poisson_001",
        "statistics_seed_offset_poisson_002",
    }
    assert len(grouped) == 1
    assert grouped[0]["ablation_family"] == "poisson|both|statistics_seed_offset"
    assert int(grouped[0]["rel_l2_a_n"]) == 3
    for raw_only_key in ("sample_seed", "mask_seed", "noise_seed", "offset", "batch_size", "ablation_name"):
        assert raw_only_key not in grouped[0]

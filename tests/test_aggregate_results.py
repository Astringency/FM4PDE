import csv
import json

import pytest

from fm4pde_ablation.aggregate import aggregate_root
from fm4pde_ablation.config import dump_yaml


def _write_run(root, name, rel_l2_a, rel_l2_u):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    config = {
        "pde": "poisson",
        "task": "both",
        "ablation_name": "same",
        "guidance_components": "obs_pde",
        "loss_state": "endpoint",
        "sampler_phase": "stochastic",
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

import os
import subprocess

from scripts.tuning.select_inverse_params import select_rows


def _run_plan(script: str, updates: dict[str, str]) -> list[str]:
    env = os.environ.copy()
    env.update({"PLAN_ONLY": "true", "AGGREGATE": "false", **updates})
    result = subprocess.run(
        ["bash", script],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        env=env,
    )
    return [line for line in result.stdout.splitlines() if line.startswith("PLAN ")]


def test_burger_tuning_plan_expands_the_requested_cross_product():
    lines = _run_plan(
        "scripts/tuning/run_burger_tuning.sh",
        {
            "STAGE": "screen",
            "DEVICE_LIST": "cuda:0 cuda:1",
            "SENSOR_MODE_LIST": "random sensor_column",
            "SAMPLER_LIST": "stochastic deterministic",
            "ZETA_OBS_U_LIST": "3200 6400",
            "ZETA_PDE_LIST": "1 10",
            "OFFSET_LIST": "0 1000",
            "MASK_SEED_LIST": "0 11",
            "CLIP_MODE_LIST": "global_norm per_component_norm",
            "CLIP_THRESHOLD_LIST": "50 100",
            "BATCH_SIZE": "2",
        },
    )

    assert len(lines) == 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    assert {line.rsplit("device=", 1)[1] for line in lines} == {"cuda:0", "cuda:1"}


def test_inverse_debug_plan_covers_ten_non_burger_equations():
    lines = _run_plan(
        "scripts/tuning/run_inverse_debug.sh",
        {"DEVICE_LIST": "cuda:0 cuda:1"},
    )

    assert len(lines) == 10
    assert all("pde=burger" not in line for line in lines)


def test_inverse_debug_preflight_reports_missing_data_without_aggregation(tmp_path):
    output_dir = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "PDE_LIST": "darcy",
            "PDE_DATA_ROOT": str(tmp_path / "missing-data"),
            "OUTPUT_DIR": str(output_dir),
            "DEVICE_LIST": "cuda:0",
            "MAX_PARALLEL_TASKS": "1",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/tuning/run_inverse_debug.sh"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2
    assert "MISSING DATA   darcy:" in result.stderr
    assert "Set PDE_DATA_ROOT" in result.stderr
    assert "Summary:" not in result.stdout
    assert not (output_dir / "summary").exists()


def test_inverse_tuning_stabilize_plan_uses_two_offsets_and_nine_configs():
    lines = _run_plan(
        "scripts/tuning/run_inverse_tuning.sh",
        {
            "STAGE": "stabilize",
            "PDE_LIST": "heat",
            "DEVICE_LIST": "cuda:0 cuda:1",
            "OFFSET_LIST": "0 1000",
        },
    )

    assert len(lines) == 18
    assert {line.split(" offset=", 1)[1].split()[0] for line in lines} == {
        "0",
        "1000",
    }
    assert all("batch=2" in line and "pde=heat" in line for line in lines)


def test_inverse_tuning_refine_default_plan_has_96_jobs():
    lines = _run_plan(
        "scripts/tuning/run_inverse_tuning.sh",
        {
            "STAGE": "refine",
            "DEVICE_LIST": "cuda:0 cuda:1",
            "OFFSET_LIST": "0 1000",
        },
    )

    assert len(lines) == 96
    assert {line.split("pde=", 1)[1].split()[0] for line in lines} == {
        "darcy",
        "poisson",
        "helmholtz",
        "nsnonbounded",
    }


def test_inverse_selector_rejects_incomplete_configs_and_ranks_robust_error():
    base = {
        "pde": "darcy",
        "clip_mode": "global_norm",
        "clip_threshold": "50",
        "rel_l2_a_median": "0.1",
        "rel_l2_u_mean": "0.02",
        "pde_residual_norm_mean": "0.5",
    }
    rows = [
        {
            **base,
            "zeta_obs_u": "1",
            "zeta_pde": "0.1",
            "rel_l2_a_n": "4",
            "rel_l2_a_mean": "0.10",
            "rel_l2_a_p90": "0.15",
            "rel_l2_a_max": "0.20",
        },
        {
            **base,
            "zeta_obs_u": "2",
            "zeta_pde": "0.1",
            "rel_l2_a_n": "4",
            "rel_l2_a_mean": "0.11",
            "rel_l2_a_p90": "0.12",
            "rel_l2_a_max": "0.13",
        },
        {
            **base,
            "zeta_obs_u": "3",
            "zeta_pde": "0.1",
            "rel_l2_a_n": "2",
            "rel_l2_a_mean": "0.01",
            "rel_l2_a_p90": "0.01",
            "rel_l2_a_max": "0.01",
        },
    ]

    selected = select_rows(rows, expected_n=4, top_k=2)

    assert [row["zeta_obs_u"] for row in selected] == ["2", "1"]

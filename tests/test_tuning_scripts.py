import os
import subprocess


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
            "CLIP_MODE_LIST": "global_norm per_sample_norm",
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

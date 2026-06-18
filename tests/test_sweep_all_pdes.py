import subprocess
import sys

from sampling.sweep import expand_grid


def test_all_internal_grid_lists_and_uses_group_base_configs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sampling.sweep",
            "--grid",
            "configs/ablations/all_internal_ablation_grid.yaml",
            "--list",
            "--limit",
            "3",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "configs/ablations/base/darcy.yaml" in result.stdout

    jobs = expand_grid("configs/ablations/all_internal_ablation_grid.yaml")
    paths = {path for path, _ in jobs}
    assert "configs/ablations/base/heat.yaml" in paths
    assert "configs/ablations/base/nsnonbounded.yaml" in paths
    assert any(
        path.endswith("nsnonbounded.yaml") and params["ablation_group"] == "time_dependent_residual_mode"
        for path, params in jobs
    )
    main_groups = {
        "guidance_components",
        "sensor_sparsity",
        "sensor_mode",
        "noise_robustness",
        "zeta_sensitivity",
        "time_grid_steps",
        "clipping",
        "residual_region",
        "statistics_seed_offset",
    }
    ns_main_groups = {
        params["ablation_group"]
        for path, params in jobs
        if path.endswith("nsnonbounded.yaml") and params.get("ablation_group") in main_groups
    }
    assert main_groups <= ns_main_groups
    assert not any(params["ablation_group"] == "ns_observation_only" for _, params in jobs)
    assert all(params.get("sensor_mode") != "time_varying" for _, params in jobs)
    assert all(params.get("loss_state") != "denoised_endpoint" for _, params in jobs)
    main_guidance = [
        params
        for _, params in jobs
        if params.get("ablation_group") == "guidance_components"
    ]
    assert main_guidance
    assert all(params.get("guidance_components") != "both_obs" for params in main_guidance)
    assert any(params.get("sensor_mode") == "per_sample_random" for _, params in jobs)

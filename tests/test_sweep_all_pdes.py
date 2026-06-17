import subprocess
import sys

from fm4pde_ablation.sweep import expand_grid


def test_all_internal_grid_lists_and_uses_group_base_configs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fm4pde_ablation.sweep",
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
    assert any(path.endswith("nsnonbounded.yaml") and params["ablation_group"] == "ns_observation_only" for path, params in jobs)
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

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from sampling.config import AblationConfig, save_resolved_config


def make_run_dir(config: AblationConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(config.output_dir) / config.pde / config.task / config.resolved_ablation_name()
    run_dir = base / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def write_run_metadata(
    config: AblationConfig,
    run_dir: str | os.PathLike[str],
    ground_truth_metadata: dict[str, Any] | None = None,
    residual_metadata: dict[str, Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> None:
    run_dir = Path(run_dir)
    save_resolved_config(config, run_dir)
    ground_truth_metadata = ground_truth_metadata or {}
    metadata = {
        "git_commit": git_commit_hash(),
        "checkpoint_path": config.checkpoint_path,
        "data_path": config.data_path,
        "data_config_path": config.data_config_path,
        "pde": config.pde,
        "task": config.task,
        "channel_names": ground_truth_metadata.get("channel_names", []),
        "pde_params_keys": ground_truth_metadata.get("pde_params_keys", []),
        "pde_params_sources": ground_truth_metadata.get("pde_params_sources", {}),
        "near_endpoint_temporal": ground_truth_metadata.get("near_endpoint_temporal", {}),
        "boundary_condition": ground_truth_metadata.get("boundary_condition"),
        "boundary_condition_source": ground_truth_metadata.get("boundary_condition_source"),
        "scalar_params_loaded": ground_truth_metadata.get("scalar_params_loaded", False),
        "mask_seed": config.mask_seed,
        "noise_seed": config.noise_seed,
        "sample_seed": config.sample_seed,
        "offset": config.offset,
        "batch_size": config.batch_size,
        "guidance": {
            "guidance_components": config.guidance_components,
            "guidance_schedule": config.guidance_schedule,
            "zeta_obs_a": config.zeta_obs_a,
            "zeta_obs_u": config.zeta_obs_u,
            "zeta_pde": config.zeta_pde,
            "clip_mode": config.clip_mode,
            "clip_threshold": config.clip_threshold,
            "pde_residual_region": config.pde_residual_region,
            "gradient_target": config.gradient_target,
            "stochastic_guidance_time": config.stochastic_guidance_time,
            "enforce_boundary_conditions": config.enforce_boundary_conditions,
            "enforce_initial_conditions": config.enforce_initial_conditions,
            "boundary_condition_mode": config.boundary_condition_mode,
            "initial_condition_mode": config.initial_condition_mode,
            "bc_weight": config.bc_weight,
            "ic_weight": config.ic_weight,
            "endpoint_bc_weight": config.endpoint_bc_weight,
            "boundary_residual_normalization": config.boundary_residual_normalization,
            "allow_unknown_boundary_conditions": config.allow_unknown_boundary_conditions,
            "legacy_ignore_boundary": config.legacy_ignore_boundary,
        },
        "sampler": {
            "sampler_phase": config.sampler_phase,
            "switch_ratio": config.switch_ratio,
            "loss_state": config.loss_state,
            "time_grid": config.time_grid,
            "num_steps": config.num_steps,
            "step_method": config.step_method,
        },
        "sensor": {
            "num_obs": config.num_obs,
            "sensor_mode": config.sensor_mode,
            "shared_mask": config.shared_mask,
            "noise_level": config.noise_level,
            "noise_level_coef": config.noise_level_coef,
            "noise_level_sol": config.noise_level_sol,
        },
        "residual": residual_metadata or {},
        "checkpoint": checkpoint_metadata or {},
        "legacy_checkpoint_boundary_metadata_missing": not bool((checkpoint_metadata or {}).get("boundary_condition") or (checkpoint_metadata or {}).get("boundary_condition_mode")),
        "model": {
            "runtime_requested_model_profile": config.model_profile,
            "checkpoint_model_profile": (checkpoint_metadata or {}).get("model_profile"),
            "selected_model_profile": (checkpoint_metadata or {}).get("selected_model_profile"),
            "selected_architecture_family": (checkpoint_metadata or {}).get("selected_architecture_family"),
            "selected_model_config_metadata": (checkpoint_metadata or {}).get("selected_model_config_metadata"),
        },
        "device": config.device,
        "dtype": config.dtype,
        "torch": torch_runtime_metadata(),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_torch(path: str | os.PathLike[str], payload: Any) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def torch_runtime_metadata() -> dict[str, Any]:
    try:
        import torch

        return {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception:
        return {"version": "unavailable", "cuda_version": None, "cuda_available": False, "device_count": 0}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    try:
        return value.item()
    except Exception:
        return str(value)

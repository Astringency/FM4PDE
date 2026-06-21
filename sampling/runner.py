from __future__ import annotations

import argparse
import random
import time
import warnings
from pathlib import Path
from typing import Any

from sampling.config import AblationConfig, load_config, normalize_residual_mode, parse_cli_overrides
from sampling.data import finalize_ground_truth_config, load_ground_truth
from sampling.guidance import apply_guidance_update, compute_guidance_gradient, make_zeta_schedule
from sampling.logging import make_run_dir, save_torch, write_run_metadata
from sampling.losses import ObservationTargets, compute_guidance_losses, guidance_component_flags
from sampling.masks import make_pair_masks
from sampling.metrics import append_jsonl, final_metrics, step_metrics, write_csv, write_json
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.noise import add_observation_noise
from sampling.pde_residuals import residual_status
from sampling.sampler_wrappers import phase_for_step, sampler_step
from sampling.state import SplitState, standardized_to_physical_state
from sampling.time_grid import affine_coefficients, make_time_grid, scheduler_coefficients


def run_single_ablation(config: AblationConfig) -> dict[str, Any]:
    config = finalize_ground_truth_config(config)
    config.validate()
    _disable_unreliable_pde_guidance(config)

    if config.dry_run and not _torch_available():
        run_dir = make_run_dir(config)
        write_run_metadata(config, run_dir, ground_truth_metadata={}, residual_metadata=_residual_metadata_for_config(config))
        result = {
            "status": "dry_run_no_torch",
            "reason": "torch is not installed in the active interpreter",
            "run_dir": str(run_dir),
            "ablation_name": config.resolved_ablation_name(),
        }
        write_json(run_dir / "metrics_final.json", result)
        write_csv(run_dir / "summary.csv", [result])
        return result

    import torch

    _set_seed(config.sample_seed)
    device = _resolve_device(config.device)
    config.device = str(device)
    gt = load_ground_truth(config)
    config.img_channels = int(gt.pair.shape[1])
    run_dir = make_run_dir(config)
    write_run_metadata(config, run_dir, ground_truth_metadata=gt.metadata, residual_metadata=_residual_metadata_for_config(config))
    masks = make_pair_masks(
        gt.coef.shape,
        gt.sol.shape,
        config.num_obs,
        config.sensor_mode,
        config.shared_mask,
        config.mask_seed,
        device=device,
        dtype=gt.coef.dtype,
    )
    noise_a = add_observation_noise(
        gt.coef,
        masks.coef,
        config.noise_level if config.noise_level_coef is None else config.noise_level_coef,
        seed=config.noise_seed,
    )
    noise_u = add_observation_noise(
        gt.sol,
        masks.sol,
        config.noise_level if config.noise_level_sol is None else config.noise_level_sol,
        seed=config.noise_seed + 1,
    )
    observations = ObservationTargets(
        coef_clean=noise_a.clean,
        sol_clean=noise_u.clean,
        coef_noisy=noise_a.noisy,
        sol_noisy=noise_u.noisy,
    )
    save_torch(run_dir / "masks.pt", {"coef": masks.coef, "sol": masks.sol, "metadata": masks.metadata})

    if config.dry_run:
        net = _ZeroVelocityModel()
        normalizer = _identity_normalizer(gt)
        checkpoint_payload: dict[str, Any] = {}
    else:
        net, normalizer, checkpoint_payload = load_fm4pde_checkpoint_bundle(
            config.checkpoint_path,
            config.pde,
            device=device,
            wrap=True,
            model_profile=config.model_profile,
        )
    checkpoint_metadata = _checkpoint_metadata(checkpoint_payload)
    write_run_metadata(
        config,
        run_dir,
        ground_truth_metadata=gt.metadata,
        residual_metadata=_residual_metadata_for_config(config),
        checkpoint_metadata=checkpoint_metadata,
    )
    _check_sampling_channels(gt, normalizer, checkpoint_payload)

    grid = make_time_grid(config.time_grid, config.num_steps, device=device, eta=config.time_grid_eta)
    x_next = torch.randn(
        config.batch_size,
        int(gt.pair.shape[1]),
        int(gt.pair.shape[2]),
        int(gt.pair.shape[3]),
        device=device,
        dtype=gt.pair.dtype,
    )

    rows: list[dict[str, Any]] = []
    intermediates = []
    start = time.time()
    for step in range(config.num_steps):
        step_start = time.time()
        phase = phase_for_step(config.sampler_phase, config.switch_ratio, step, config.num_steps)
        x_cur = x_next.detach().clone()
        if _has_guidance(config):
            x_cur.requires_grad_(True)
        t = grid[step]
        t_next = grid[step + 1]
        step_out = sampler_step(
            net=net,
            x_cur=x_cur,
            t=t,
            t_next=t_next,
            phase=phase,
            step_method=config.step_method,
            loss_state=config.loss_state,
            device=device,
        )
        phys_loss = _physical_from_model_state(step_out.x_loss_state, config, normalizer)
        losses = compute_guidance_losses(phys_loss, gt, masks, config, observations)
        coeffs = scheduler_coefficients(t, scheduler="CondOT")
        affine = affine_coefficients(coeffs, training="velocity")
        schedule = make_zeta_schedule(config, t, t_next, affine.b_t)

        gradient = None
        guided_next = step_out.x_raw_next
        if _has_guidance(config):
            gradient_input = _gradient_target_tensor(config, x_cur, step_out)
            gradient = compute_guidance_gradient(losses, gradient_input, schedule, config)
            guided_next = apply_guidance_update(step_out.x_raw_next, gradient, step_out, schedule, config)
        x_next = guided_next.detach()
        if config.empty_cache_each_step and device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            phys_eval = _physical_from_model_state(x_next, config, normalizer)
            eval_losses = compute_guidance_losses(phys_eval, gt, masks, config, observations)
        row = step_metrics(step, step_out, losses, eval_losses, gradient, phys_eval, gt, masks, time.time() - step_start)
        row.update(
            {
                "zeta_obs_a_t": _scalar(schedule.zeta_obs_a_t),
                "zeta_obs_u_t": _scalar(schedule.zeta_obs_u_t),
                "zeta_pde_t": _scalar(schedule.zeta_pde_t),
                "guidance_schedule_factor": _scalar(schedule.metadata.get("factor", 1.0)),
                "bt": _scalar(schedule.bt),
            }
        )
        rows.append(row)
        append_jsonl(run_dir / "metrics_step.jsonl", row)
        if config.save_intermediate:
            intermediates.append(
                {
                    "x_raw_current": step_out.x_raw_current.detach().cpu(),
                    "x_raw_next": step_out.x_raw_next.detach().cpu(),
                    "x_endpoint": step_out.x_endpoint.detach().cpu(),
                    "x_loss_state": step_out.x_loss_state.detach().cpu(),
                    "phase": step_out.phase,
                    "loss_state": step_out.loss_state,
                    "t": _scalar(step_out.t),
                    "t_next": _scalar(step_out.t_next),
                    "step_size": _scalar(step_out.step_size),
                    "wall_time": step_out.wall_time,
                }
            )

    final_phys = _physical_from_model_state(x_next, config, normalizer)
    final = final_metrics(rows)
    final.update(
        {
            "status": "ok",
            "run_dir": str(run_dir),
            "wall_clock_time": time.time() - start,
            "synthetic_data": bool(gt.metadata.get("synthetic", False)),
            "pde_residual_status": rows[-1].get("pde_residual_status", residual_status(config.pde)) if rows else residual_status(config.pde),
            "gradient_target": config.gradient_target,
            "stochastic_guidance_time": config.stochastic_guidance_time,
        }
    )
    write_json(run_dir / "metrics_final.json", final)
    write_csv(run_dir / "curves.csv", rows)
    write_csv(run_dir / "summary.csv", [final])
    save_torch(
        run_dir / "result.pt",
        {
            "coef_final": final_phys.coef.detach().cpu(),
            "sol_final": final_phys.sol.detach().cpu(),
            "coef_ground_truth": gt.coef.detach().cpu(),
            "sol_ground_truth": gt.sol.detach().cpu(),
            "masks": {"coef": masks.coef.detach().cpu(), "sol": masks.sol.detach().cpu(), "metadata": masks.metadata},
            "pde_params": _sanitize_pde_params_for_artifact(gt.pde_params, config),
            "metrics": final,
            "intermediate": intermediates,
            "ground_truth_metadata": gt.metadata,
            "normalizer": normalizer.state_dict() if normalizer is not None else None,
            "checkpoint_metadata": _to_cpu_recursive(checkpoint_metadata),
            "config": config.asdict(),
        },
    )
    return final


def run_from_config_path(config_path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, overrides=overrides)
    return run_single_ablation(cfg)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single FM4PDE ablation.")
    parser.add_argument("--config", required=True, help="Flat AblationConfig YAML.")
    parser.add_argument("--override", action="append", default=[], help="Override key=value. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and run without loading the checkpoint.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    overrides = parse_cli_overrides(args.override)
    if args.dry_run:
        overrides["dry_run"] = True
    result = run_from_config_path(args.config, overrides=overrides)
    print(result)
    return result


def _physical_from_model_state(x_model: Any, config: AblationConfig, normalizer: Any | None) -> SplitState:
    return standardized_to_physical_state(
        x_model,
        pde=config.pde,
        img_channels=config.img_channels,
        normalizer=normalizer,
    )


def _gradient_target_tensor(config: AblationConfig, x_cur: Any, step_out: Any) -> Any:
    if config.gradient_target == "current_state_chain_rule":
        return x_cur
    if config.gradient_target == "loss_state_direct":
        return step_out.x_loss_state
    if config.gradient_target == "next_state_direct":
        return step_out.x_raw_next
    raise ValueError(f"Unknown gradient_target={config.gradient_target!r}")


def _identity_normalizer(gt: Any) -> Any:
    from data.transform import PDEStandardizer

    return PDEStandardizer.identity(
        int(gt.pair.shape[1]),
        channel_names=list(gt.channel_names_coef) + list(gt.channel_names_sol),
        pde=gt.pde,
    )


def _checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "epoch": payload.get("epoch"),
        "data_shape": payload.get("data_shape"),
        "num_channels": payload.get("num_channels"),
        "normalization": payload.get("normalization"),
        "checkpoint_schema_version": payload.get("checkpoint_schema_version"),
        "use_ema": payload.get("use_ema"),
        "has_ema": payload.get("has_ema"),
        "selected_inference_weight": payload.get("selected_inference_weight"),
        "model_profile": payload.get("model_profile"),
        "model_config_metadata": payload.get("model_config_metadata"),
        "runtime_requested_model_profile": payload.get("runtime_requested_model_profile"),
        "selected_model_profile": payload.get("selected_model_profile"),
        "selected_model_config_metadata": payload.get("selected_model_config_metadata"),
        "selected_architecture_family": payload.get("selected_architecture_family"),
        "data_metadata": payload.get("data_metadata"),
    }


def _check_sampling_channels(gt: Any, normalizer: Any | None, payload: dict[str, Any]) -> None:
    expected = int(gt.pair.shape[1])
    if normalizer is not None and int(normalizer.mean.shape[1]) != expected:
        raise ValueError(
            f"Checkpoint normalizer has {int(normalizer.mean.shape[1])} channels, "
            f"but ground-truth pair has {expected}. If this checkpoint was trained with scalar PDE "
            "parameters materialized as constant fields, retrain it with the current sample-level "
            "pde_params channel definition."
        )
    if payload.get("num_channels") is not None and int(payload["num_channels"]) != expected:
        raise ValueError(
            f"Checkpoint num_channels={payload['num_channels']} but ground-truth pair has {expected}. "
            "Checkpoints using old scalar-parameter channels must be retrained with the current "
            "sample-level pde_params channel definition."
        )


def _has_guidance(config: AblationConfig) -> bool:
    flags = guidance_component_flags(config.guidance_components, config.task)
    return flags["obs_a"] or flags["obs_u"] or flags["pde"]


def _disable_unreliable_pde_guidance(config: AblationConfig) -> None:
    if not guidance_component_flags(config.guidance_components, config.task).get("pde", False):
        return
    if residual_status(config.pde) != "disabled":
        return
    requested = config.guidance_components
    mapping = {
        "pde_only": "noguide",
        "obs_pde": "obs_only",
    }
    effective = mapping.get(requested)
    if effective is None:
        config.zeta_pde = 0.0
        return
    warnings.warn(
        f"PDE guidance is disabled for pde={config.pde!r}; mapping guidance_components={requested!r} to {effective!r}.",
        RuntimeWarning,
        stacklevel=2,
    )
    config.extra["guidance_components_requested"] = requested
    config.extra["guidance_components_effective"] = effective
    config.guidance_components = effective
    config.zeta_pde = 0.0


def _residual_metadata_for_config(config: AblationConfig) -> dict[str, Any]:
    requested_mode = normalize_residual_mode(config.residual_mode)
    from data.specs import get_pde_spec

    spec = get_pde_spec(config.pde)
    if spec.residual_family == "temporal_endpoint":
        resolved_mode = "hermite_bridge" if requested_mode == "auto" else requested_mode
    elif spec.residual_family == "full_time_space":
        resolved_mode = "full_time_space"
    elif config.pde == "steady_heat_conduction":
        resolved_mode = "static_nonlinear_boundary"
    elif spec.residual_family == "static":
        resolved_mode = "static"
    else:
        raise ValueError(f"Unsupported residual_family={spec.residual_family!r} for {config.pde}")
    if resolved_mode == "hermite_bridge":
        temporal_derivative_mode = "hermite_bridge"
        endpoint_only = True
        uses_generated_trajectory = False
        uses_extra_temporal_observations = False
    elif resolved_mode == "near_endpoint_temporal":
        temporal_derivative_mode = "near_endpoint_sparse_fd"
        endpoint_only = False
        uses_generated_trajectory = False
        uses_extra_temporal_observations = True
    elif resolved_mode == "endpoint_secant":
        temporal_derivative_mode = "endpoint_secant"
        endpoint_only = True
        uses_generated_trajectory = False
        uses_extra_temporal_observations = False
    elif resolved_mode in {"full_time_space", "full_trajectory_fd"}:
        temporal_derivative_mode = "full_fd"
        endpoint_only = False
        uses_generated_trajectory = True
        uses_extra_temporal_observations = False
    else:
        temporal_derivative_mode = "none"
        endpoint_only = False
        uses_generated_trajectory = False
        uses_extra_temporal_observations = False
    metadata = {
        "pde": config.pde,
        "residual_status": residual_status(config.pde),
        "residual_family": "full_trajectory" if resolved_mode == "full_trajectory_fd" else spec.residual_family,
        "temporal_derivative_mode": temporal_derivative_mode,
        "endpoint_only": endpoint_only,
        "uses_generated_trajectory": uses_generated_trajectory,
        "uses_extra_temporal_observations": uses_extra_temporal_observations,
        "residual_mode": config.residual_mode,
        "resolved_residual_mode": resolved_mode,
        "zeta_pde": config.zeta_pde,
        "hermite_collocation_times": config.hermite_collocation_times,
        "hermite_num_collocation": config.hermite_num_collocation,
        "hermite_include_integral_residual": config.hermite_include_integral_residual,
        "hermite_integral_weight": config.hermite_integral_weight,
        "num_near_endpoint_obs": config.num_near_endpoint_obs,
        "near_endpoint_sensor_mode": config.near_endpoint_sensor_mode,
        "near_endpoint_mask_seed": config.near_endpoint_mask_seed,
        "near_endpoint_shared_mask": config.near_endpoint_shared_mask,
    }
    return metadata


def _to_cpu_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_cpu_recursive(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_recursive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_recursive(item) for item in value)
    try:
        return value.detach().cpu()
    except Exception:
        return value


def _sanitize_pde_params_for_artifact(params: dict[str, Any] | None, config: AblationConfig | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    residual_mode = normalize_residual_mode(config.residual_mode) if config is not None else "auto"
    save_intermediate = bool(getattr(config, "save_intermediate", False)) if config is not None else False
    for key, value in (params or {}).items():
        if key == "near_endpoint_temporal" and isinstance(value, dict):
            out[key] = _sanitize_near_endpoint_temporal(value)
        elif key in {"trajectory", "full_trajectory"}:
            if residual_mode == "full_trajectory_fd" and save_intermediate:
                saved = _to_cpu_recursive(value)
                out[key] = saved
                out.setdefault("artifact_metadata", {})[key] = {"large_artifact": True, "full_trajectory_saved": True}
            else:
                out.setdefault("artifact_metadata", {})[key] = {
                    "full_trajectory_saved": False,
                    "large_artifact": False,
                    "reason": "omitted unless residual_mode='full_trajectory_fd' and save_intermediate=true",
                }
        else:
            out[key] = _to_cpu_recursive(value)
    return out


def _sanitize_near_endpoint_temporal(value: dict[str, Any]) -> dict[str, Any]:
    import torch

    sanitized: dict[str, Any] = {}
    q_dt = value.get("q_dt")
    q_t_minus = value.get("q_T_minus_dt")
    mask_0 = value.get("mask_0")
    mask_t = value.get("mask_T")
    if q_dt is not None and mask_0 is not None:
        sanitized["q_dt_obs"] = (torch.as_tensor(q_dt).detach().cpu() * torch.as_tensor(mask_0).detach().cpu())
    if q_t_minus is not None and mask_t is not None:
        sanitized["q_T_minus_dt_obs"] = (torch.as_tensor(q_t_minus).detach().cpu() * torch.as_tensor(mask_t).detach().cpu())
    for key in ("mask_0", "mask_T", "dt"):
        if key in value:
            sanitized[key] = _to_cpu_recursive(value[key])
    metadata = dict(value.get("metadata", {}) if isinstance(value.get("metadata", {}), dict) else {})
    metadata.update(
        {
            "full_near_endpoint_frames_saved": False,
            "sparse_temporal_observations_saved": True,
        }
    )
    sanitized["metadata"] = _to_cpu_recursive(metadata)
    return sanitized


def _resolve_device(device: str) -> Any:
    import torch

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return target


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def _scalar(value: Any) -> float:
    try:
        return float(value.detach().cpu())
    except Exception:
        return float(value)


class _ZeroVelocityModel:
    def __call__(self, x: Any, t: Any, **extras: Any) -> Any:
        return x * 0.0


if __name__ == "__main__":
    main()

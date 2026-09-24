from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any

from sampling.config import AblationConfig, load_config, normalize_residual_mode, parse_cli_overrides
from sampling.data import attach_near_endpoint_observations, finalize_ground_truth_config, load_ground_truth
from sampling.guidance import apply_guidance_update, compute_guidance_gradient, make_zeta_schedule
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

from sampling.logging import make_run_dir, save_torch, write_run_metadata
from sampling.losses import ObservationTargets, compute_guidance_losses, guidance_component_flags
from sampling.masks import make_pair_masks
from sampling.metrics import append_jsonl, final_metrics, per_sample_metrics, step_metrics, write_csv, write_json
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.noise import add_observation_noise
from sampling.pde_residuals import residual_status
from sampling.sampler_wrappers import phase_for_step, sampler_step
from sampling.state import SplitState, standardized_to_physical_state
from sampling.time_grid import affine_coefficients, make_time_grid, scheduler_coefficients


def run_single_ablation(
    config: AblationConfig,
    checkpoint_bundle: tuple[Any, Any, dict[str, Any]] | None = None,
    *,
    ground_truth: Any | None = None,
    observation_masks: Any | None = None,
) -> dict[str, Any]:
    config = finalize_ground_truth_config(config)
    config.validate()
    _disable_unreliable_pde_guidance(config)
    _write_sweep_progress(config, status="initializing", step=0, force=True)

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
        _write_sweep_progress(
            config,
            status="complete",
            step=0,
            run_dir=run_dir,
            metrics=result,
            force=True,
        )
        return result

    import torch

    _set_seed(config.sample_seed)
    device = _resolve_device(config.device)
    config.device = str(device)
    gt = ground_truth if ground_truth is not None else load_ground_truth(config)
    config.img_channels = int(gt.pair.shape[1])
    masks = observation_masks if observation_masks is not None else make_pair_masks(
        gt.coef.shape,
        gt.sol.shape,
        config.num_obs,
        config.sensor_mode,
        config.shared_mask,
        config.mask_seed,
        device=device,
        dtype=gt.coef.dtype,
        num_sensor_columns=config.num_sensor_columns,
    )
    gt = attach_near_endpoint_observations(config, gt, masks)
    run_dir = make_run_dir(config)
    _write_sweep_progress(config, status="initializing", step=0, run_dir=run_dir, force=True)
    write_run_metadata(config, run_dir, ground_truth_metadata=gt.metadata, residual_metadata=_residual_metadata_for_config(config))
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
    elif checkpoint_bundle is None:
        net, normalizer, checkpoint_payload = load_fm4pde_checkpoint_bundle(
            config.checkpoint_path,
            config.pde,
            device=device,
            wrap=True,
            model_profile=config.model_profile,
        )
    else:
        net, normalizer, checkpoint_payload = checkpoint_bundle
    checkpoint_metadata = _checkpoint_metadata(checkpoint_payload)
    scalar_extra, scalar_conditioning_metadata = _scalar_conditioning_for_sampling(
        checkpoint_payload=checkpoint_payload,
        gt=gt,
        config=config,
        device=device,
    )
    class_extra, class_conditioning_metadata = _class_conditioning_for_sampling(
        checkpoint_payload=checkpoint_payload,
        pde=config.pde,
        batch_size=config.batch_size,
        device=device,
        cfg_scale=config.cfg_scale,
    )
    combined_extra = {**class_extra, **(scalar_extra or {})}
    model_extra = combined_extra or None
    checkpoint_metadata["class_conditioning"] = class_conditioning_metadata
    write_run_metadata(
        config,
        run_dir,
        ground_truth_metadata=gt.metadata,
        residual_metadata=_residual_metadata_for_config(config),
        checkpoint_metadata=checkpoint_metadata,
        scalar_conditioning_metadata=scalar_conditioning_metadata,
    )
    if config.model_gradient_checkpointing and not config.dry_run:
        base_model = getattr(net, "model", net)
        for module in base_model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True
    _check_sampling_channels(gt, normalizer, checkpoint_payload)

    grid = make_time_grid(config.time_grid, config.num_steps, device=device, eta=config.time_grid_eta)
    x_next = _sample_initial_noise(config, gt, device)

    rows: list[dict[str, Any]] = []
    per_sample_curve_rows: list[dict[str, Any]] = []
    intermediates = []
    start = time.time()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("rel₂(u)={task.fields[rel_u]}"),
        TextColumn("rel₂(a)={task.fields[rel_a]}"),
        TextColumn("L_pde={task.fields[pde_loss]}"),
        TextColumn("[{task.fields[phase]}]"),
        transient=True,
        expand=False,
    ) as progress:
        task_id = progress.add_task(
            "sampling",
            total=config.num_steps,
            rel_u="---",
            rel_a="---",
            pde_loss="---",
            phase="init",
        )
        for step in range(config.num_steps):
            step_start = time.time()
            phase = phase_for_step(config.sampler_phase, config.switch_ratio, step, config.num_steps)
            x_cur = x_next.detach().clone()
            if _has_guidance(config) and config.gradient_target != "proposal_state_chain_rule":
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
                model_extra=model_extra,
                stochastic_noise_source_batch_size=config.initial_noise_source_batch_size,
                stochastic_noise_source_indices=config.initial_noise_source_indices or None,
                deterministic_endpoint_mode=config.deterministic_endpoint_mode,
                deterministic_endpoint_time_grid=grid[step:],
                deterministic_rollout_checkpoint=config.deterministic_rollout_checkpoint,
                gradient_target=config.gradient_target,
            )
            phys_loss = _physical_from_model_state(step_out.x_loss_state, config, normalizer)
            losses = compute_guidance_losses(phys_loss, gt, masks, config, observations)
            calibration = _calibrate_l2_observation_zeta(config, losses, step=step)
            if calibration:
                config.runtime_metadata["obs_l2_step0_calibration"] = calibration
                write_run_metadata(
                    config,
                    run_dir,
                    ground_truth_metadata=gt.metadata,
                    residual_metadata=_residual_metadata_for_config(config),
                    checkpoint_metadata=checkpoint_metadata,
                    scalar_conditioning_metadata=scalar_conditioning_metadata,
                )
            coeffs = scheduler_coefficients(t, scheduler="CondOT")
            affine = affine_coefficients(coeffs, training="velocity")
            schedule = make_zeta_schedule(config, t, t_next, affine.b_t, step=step)

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
                    "pde_guidance_factor": _scalar(schedule.metadata.get("pde_guidance_factor", 1.0)),
                    "bt": _scalar(schedule.bt),
                }
            )
            if config.save_per_sample_curves:
                per_sample_curve_rows.extend(
                    per_sample_metrics(
                        phys_eval,
                        gt,
                        masks,
                        eval_losses,
                        step=step,
                    )
                )
            # Standard step artifacts contain batch means only. This keeps
            # metrics_final.json and curves.csv compact for large batches.
            row = {key: value for key, value in row.items() if not key.endswith("_per_sample")}
            rows.append(row)
            append_jsonl(run_dir / "metrics_step.jsonl", row)
            _write_sweep_progress(
                config,
                status="running",
                step=step + 1,
                run_dir=run_dir,
                metrics=row,
            )
            progress.update(
                task_id,
                advance=1,
                rel_u=f"{row["rel_l2_u"]:.4g}",
                rel_a=f"{row["rel_l2_a"]:.4g}",
                pde_loss="n/a" if row["L_pde"] is None else f"{row['L_pde']:.2e}",
                phase=phase,
            )
            if config.save_intermediate:
                intermediates.append(
                    {
                        "x_raw_current": step_out.x_raw_current.detach().cpu(),
                        "x_raw_next": step_out.x_raw_next.detach().cpu(),
                        "x_endpoint": step_out.x_endpoint.detach().cpu(),
                        "x_loss_state": step_out.x_loss_state.detach().cpu(),
                        "phase": step_out.phase,
                        "loss_state": step_out.loss_state,
                        "endpoint_prediction_mode": step_out.endpoint_prediction_mode,
                        "endpoint_model_evaluations": step_out.endpoint_model_evaluations,
                        "t": _scalar(step_out.t),
                        "t_next": _scalar(step_out.t_next),
                        "step_size": _scalar(step_out.step_size),
                        "wall_time": step_out.wall_time,
                    }
                )

    final_phys = _physical_from_model_state(x_next, config, normalizer)
    with torch.no_grad():
        final_eval_losses = compute_guidance_losses(final_phys, gt, masks, config, observations)
    sample_rows = per_sample_metrics(final_phys, gt, masks, final_eval_losses)
    final = final_metrics(rows)
    final_residual_status = (
        rows[-1].get("pde_residual_status", residual_status(config.pde))
        if rows
        else residual_status(config.pde)
    )
    pde_eval_error_count = sum(row.get("pde_residual_status") == "error" for row in rows)
    final.update(
        {
            "status": "pde_eval_error" if pde_eval_error_count else "ok",
            "run_dir": str(run_dir),
            "wall_clock_time": time.time() - start,
            "synthetic_data": bool(gt.metadata.get("synthetic", False)),
            "pde_residual_status": final_residual_status,
            "pde_eval_error_count": pde_eval_error_count,
            "gradient_target": config.gradient_target,
            "stochastic_guidance_time": config.stochastic_guidance_time,
            "num_samples": len(sample_rows),
            "per_sample_metrics_file": "metrics_per_sample.csv",
            "per_sample_curve_file": (
                "metrics_step_per_sample.csv" if config.save_per_sample_curves else None
            ),
        }
    )
    write_json(run_dir / "metrics_final.json", final)
    write_csv(run_dir / "curves.csv", rows)
    write_csv(run_dir / "metrics_per_sample.csv", sample_rows)
    if config.save_per_sample_curves:
        write_csv(run_dir / "metrics_step_per_sample.csv", per_sample_curve_rows)
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
            "scalar_conditioning": _to_cpu_recursive(
                {
                    "metadata": scalar_conditioning_metadata,
                    "standardized": (
                        model_extra["scalar_conditioning"].detach().cpu()
                        if model_extra and "scalar_conditioning" in model_extra
                        else None
                    ),
                }
            ),
            "config": config.asdict(),
        },
    )

    if getattr(config, "save_plots", False):
        _save_visualization(run_dir, config.pde)

    _write_sweep_progress(
        config,
        status="complete",
        step=config.num_steps,
        run_dir=run_dir,
        metrics=final,
        force=True,
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
    parser.add_argument("--vis", action="store_true", help="Plot ground truth vs prediction after sampling.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    overrides = parse_cli_overrides(args.override)
    if args.dry_run:
        overrides["dry_run"] = True
    if args.vis:
        overrides["save_plots"] = True
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


def _sample_initial_noise(config: AblationConfig, ground_truth: Any, device: Any) -> Any:
    """Sample the initial latent, optionally selecting rows from a larger reproducibility batch."""
    import torch

    source_batch_size = int(
        config.batch_size
        if config.initial_noise_source_batch_size is None
        else config.initial_noise_source_batch_size
    )
    source_indices_raw = config.initial_noise_source_indices or None
    if source_indices_raw is None:
        source_indices = list(range(int(config.batch_size)))
    elif isinstance(source_indices_raw, int):
        source_indices = [int(source_indices_raw)]
    else:
        source_indices = [int(value) for value in source_indices_raw]
    if source_batch_size < int(config.batch_size):
        raise ValueError("initial_noise_source_batch_size must be at least batch_size")
    if len(source_indices) != int(config.batch_size):
        raise ValueError("initial_noise_source_indices must contain exactly batch_size entries")
    if any(index < 0 or index >= source_batch_size for index in source_indices):
        raise ValueError("initial_noise_source_indices values must lie inside the source batch")

    source = torch.randn(
        source_batch_size,
        int(ground_truth.pair.shape[1]),
        int(ground_truth.pair.shape[2]),
        int(ground_truth.pair.shape[3]),
        device=device,
        dtype=ground_truth.pair.dtype,
    )
    if source_batch_size == int(config.batch_size) and source_indices == list(range(int(config.batch_size))):
        return source
    index = torch.as_tensor(source_indices, dtype=torch.long, device=device)
    return source.index_select(0, index)


def _calibrate_l2_observation_zeta(
    config: AblationConfig,
    losses: Any,
    *,
    step: int,
) -> dict[str, Any]:
    """Match batch-1 L2 observation-gradient scale to configured MSE-reference zeta values."""
    if step != 0 or config.obs_guidance_reduction != "l2_norm":
        return {}
    reference_fields = {
        "a": "obs_l2_reference_mse_zeta_a",
        "u": "obs_l2_reference_mse_zeta_u",
    }
    if not any(getattr(config, field) is not None for field in reference_fields.values()):
        return {}
    if int(config.batch_size) != 1:
        raise ValueError("Exact step-0 L2/MSE zeta calibration currently requires batch_size=1")

    flags = guidance_component_flags(config.guidance_components, config.task)
    counts = losses.metadata.get("obs_counts", {})
    calibration: dict[str, Any] = {
        "method": "exact_batch1_step0_gradient_scale_match",
        "formula": "zeta_l2=zeta_mse*2*sqrt(masked_mse)/sqrt(observed_entries)",
    }
    for side, count_key in (("a", "coef"), ("u", "sol")):
        reference_field = reference_fields[side]
        reference_value = getattr(config, reference_field)
        if reference_value is None or not flags[f"obs_{side}"]:
            continue
        reference_zeta = float(reference_value)
        mse = float(getattr(losses, f"L_obs_{side}").detach().cpu())
        observed_entries = float(counts.get(count_key, 0.0))
        if mse <= 0.0 or observed_entries <= 0.0:
            raise ValueError(
                f"Cannot calibrate L2 observation zeta for side {side}: "
                f"masked_mse={mse}, observed_entries={observed_entries}"
            )
        factor = observed_entries**0.5 / (2.0 * mse**0.5)
        calibrated_zeta = reference_zeta / factor
        setattr(config, f"zeta_obs_{side}", calibrated_zeta)
        calibration[side] = {
            "reference_mse_zeta": reference_zeta,
            "step0_masked_mse": mse,
            "observed_entries": observed_entries,
            "l2_to_mse_gradient_factor": factor,
            "calibrated_l2_zeta": calibrated_zeta,
            "weighted_gradient_ratio_l2_over_mse": calibrated_zeta * factor / reference_zeta,
        }
    return calibration


def _gradient_target_tensor(config: AblationConfig, x_cur: Any, step_out: Any) -> Any:
    if config.gradient_target == "current_state_chain_rule":
        return x_cur
    if config.gradient_target == "loss_state_direct":
        return step_out.x_loss_state
    if config.gradient_target in {"next_state_direct", "proposal_state_chain_rule"}:
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
        "legacy_compatibility": payload.get("legacy_compatibility"),
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


def _scalar_conditioning_for_sampling(
    *,
    checkpoint_payload: dict[str, Any],
    gt: Any,
    config: AblationConfig,
    device: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from data.scalar_conditioning import (
        scalar_conditioning_metadata_from_checkpoint_payload,
        standardize_scalar_conditioning,
    )

    metadata = scalar_conditioning_metadata_from_checkpoint_payload(checkpoint_payload)
    if not bool(metadata.get("enabled", False)):
        return None, {
            **metadata,
            "source": None,
            "standardization": metadata.get("normalization"),
        }

    scalar = standardize_scalar_conditioning(
        pde_params=gt.pde_params,
        metadata=metadata,
        expected_sample_count=int(config.batch_size),
        source_name="data_aligned_ground_truth pde_params",
        device=device,
        dtype=gt.pair.dtype,
    )
    return {"scalar_conditioning": scalar}, {
        **metadata,
        "source": "data_aligned_ground_truth",
        "standardization": metadata.get("normalization"),
        "batch_size": int(config.batch_size),
        "offset": int(config.offset),
        "standardized_shape": [int(dim) for dim in scalar.shape],
    }


def _class_conditioning_for_sampling(
    *,
    checkpoint_payload: dict[str, Any],
    pde: str,
    batch_size: int,
    device: Any,
    cfg_scale: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    model_config = checkpoint_payload.get("model_config", {})
    num_classes = model_config.get("num_classes") if isinstance(model_config, dict) else None
    if num_classes is None:
        return {}, {"enabled": False, "cfg_scale": 1.0, "pde_label_mapping": None}
    metadata = checkpoint_payload.get("model_config_metadata", {})
    data_metadata = checkpoint_payload.get("data_metadata", {})
    mapping = metadata.get("pde_label_mapping") if isinstance(metadata, dict) else None
    if not isinstance(mapping, dict) and isinstance(data_metadata, dict):
        mapping = data_metadata.get("pde_label_mapping")
    if not isinstance(mapping, dict):
        raise ValueError(
            "Joint checkpoint has a category layer but no contiguous pde_label_mapping. "
            "The checkpoint is invalid and must be retrained."
        )
    normalized = {str(name): int(index) for name, index in mapping.items()}
    expected_indices = list(range(int(num_classes)))
    if sorted(normalized.values()) != expected_indices:
        raise ValueError(
            f"Joint checkpoint pde_label_mapping must be contiguous {expected_indices}, got {normalized}"
        )
    if pde not in normalized:
        raise ValueError(f"PDE {pde!r} is absent from joint checkpoint mapping {normalized}")
    labels = torch.full(
        (int(batch_size),),
        int(normalized[pde]),
        dtype=torch.long,
        device=device,
    )
    return {
        "label": labels,
        "_cfg_scale": float(cfg_scale),
        "_cfg_null_label": int(num_classes),
    }, {
        "enabled": True,
        "cfg_scale": float(cfg_scale),
        "conditional_label": int(normalized[pde]),
        "null_label": int(num_classes),
        "pde_label_mapping": normalized,
        "unconditional_drops": ["pde_label"],
        "unconditional_retains": ["scalar_conditioning"],
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
            "Retrain with the current sample-level pde_params channel definition."
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
    config.runtime_metadata["guidance_components_requested"] = requested
    config.runtime_metadata["guidance_components_effective"] = effective
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
    uses_sparse_near_endpoint_exception = resolved_mode == "near_endpoint_temporal"
    metadata = {
        "pde": config.pde,
        "residual_status": residual_status(config.pde),
        "residual_family": "full_trajectory" if resolved_mode == "full_trajectory_fd" else spec.residual_family,
        "temporal_derivative_mode": temporal_derivative_mode,
        "endpoint_only": endpoint_only,
        "uses_generated_trajectory": uses_generated_trajectory,
        "uses_extra_temporal_observations": uses_extra_temporal_observations,
        "field_input_sources": {"coef": "model_output", "sol": "model_output"},
        "uses_ground_truth_fields": uses_sparse_near_endpoint_exception,
        "uses_ground_truth_endpoint_fields": False,
        "ground_truth_field_exception": (
            "near_endpoint_temporal_sparse_observations"
            if uses_sparse_near_endpoint_exception
            else None
        ),
        "auxiliary_field_input_sources": (
            {
                "q_dt": "sparse_ground_truth_observations",
                "q_T_minus_dt": "sparse_ground_truth_observations",
            }
            if uses_sparse_near_endpoint_exception
            else {}
        ),
        "residual_mode": config.residual_mode,
        "resolved_residual_mode": resolved_mode,
        "zeta_pde": config.zeta_pde,
        "hermite_collocation_times": config.hermite_collocation_times,
        "hermite_num_collocation": config.hermite_num_collocation,
        "hermite_include_integral_residual": config.hermite_include_integral_residual,
        "hermite_integral_weight": config.hermite_integral_weight,
        "near_endpoint_mask_alignment": (
            {"q_dt": "coef/q0", "q_T_minus_dt": "sol/qT"}
            if uses_sparse_near_endpoint_exception
            else None
        ),
        "sensor_mode": config.sensor_mode,
        "num_obs": config.num_obs,
        "num_sensor_columns": config.num_sensor_columns,
        "ns_operator_mode": config.ns_operator_mode,
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


_SWEEP_PROGRESS_LAST_WRITE: dict[str, float] = {}


def _write_sweep_progress(
    config: AblationConfig,
    *,
    status: str,
    step: int,
    run_dir: Path | None = None,
    metrics: dict[str, Any] | None = None,
    force: bool = False,
) -> None:
    """Publish throttled, atomic runner progress for the sweep dashboard."""
    raw_path = os.environ.get("FM4PDE_SWEEP_PROGRESS_FILE")
    if not raw_path:
        return
    now = time.monotonic()
    if not force and now - _SWEEP_PROGRESS_LAST_WRITE.get(raw_path, 0.0) < 0.5:
        return
    path = Path(raw_path)
    payload = {
        "status": status,
        "pde": config.pde,
        "task": config.task,
        "sampler": config.sampler_phase,
        "sensor_mode": config.sensor_mode,
        "offset": int(config.offset),
        "batch_size": int(config.batch_size),
        "step": int(step),
        "num_steps": int(config.num_steps),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "rel_l2_a": (metrics or {}).get("rel_l2_a"),
        "rel_l2_u": (metrics or {}).get("rel_l2_u"),
        "L_pde": (metrics or {}).get("L_pde"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    _SWEEP_PROGRESS_LAST_WRITE[raw_path] = now


class _ZeroVelocityModel:
    def __call__(self, x: Any, t: Any, **extras: Any) -> Any:
        return x * 0.0


def _save_visualization(run_dir: Path, pde: str) -> None:
    """Plot ground truth vs prediction from result.pt into run_dir/figures/."""
    from plot.plot import plot_from_result_pt

    result_pt = run_dir / "result.pt"
    if not result_pt.exists():
        return
    output_path = run_dir / "figures" / f"{pde}_prediction.png"
    try:
        plot_from_result_pt(str(result_pt), str(output_path))
    except Exception as ex:
        print(f"[vis] Failed to plot: {ex}")


if __name__ == "__main__":
    main()

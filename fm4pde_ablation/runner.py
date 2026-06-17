from __future__ import annotations

import argparse
import os
import pickle
import random
import time
import warnings
from pathlib import Path
from typing import Any

from fm4pde_ablation.config import AblationConfig, load_config, parse_cli_overrides, str2bool
from fm4pde_ablation.data import finalize_ground_truth_config, load_ground_truth
from fm4pde_ablation.guidance import apply_guidance_update, compute_guidance_gradient, make_zeta_schedule
from fm4pde_ablation.logging import make_run_dir, save_torch, write_run_metadata
from fm4pde_ablation.losses import ObservationTargets, compute_guidance_losses, guidance_component_flags
from fm4pde_ablation.masks import make_pair_masks
from fm4pde_ablation.metrics import append_jsonl, final_metrics, step_metrics, write_csv, write_json
from fm4pde_ablation.model_io import load_fm4pde_checkpoint_bundle
from fm4pde_ablation.noise import add_observation_noise
from fm4pde_ablation.pde_residuals import residual_status
from fm4pde_ablation.sampler_wrappers import phase_for_step, sampler_step
from fm4pde_ablation.state import SplitState, standardized_to_physical_state
from fm4pde_ablation.time_grid import affine_coefficients, make_time_grid, scheduler_coefficients


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
            config.checkpoint_path, config.pde, device=device, wrap=True
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
            gradient = compute_guidance_gradient(losses, step_out.x_loss_state, schedule, config)
            guided_next = apply_guidance_update(step_out.x_raw_next, gradient, step_out, schedule, config)
        x_next = guided_next.detach()
        if config.empty_cache_each_step and device.type == "cuda":
            torch.cuda.empty_cache()

        phys_eval = _physical_from_model_state(x_next, config, normalizer)
        row = step_metrics(step, step_out, losses, gradient, phys_eval, gt, masks, time.time() - step_start)
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
            "pde_params": _cpu_pde_params(gt.pde_params),
            "metrics": final,
            "intermediate": intermediates,
            "ground_truth_metadata": gt.metadata,
            "normalizer": normalizer.state_dict() if normalizer is not None else None,
            "checkpoint_metadata": _checkpoint_metadata(checkpoint_payload),
            "config": config.asdict(),
        },
    )
    if config.legacy_pickle:
        with (run_dir / "legacy_results.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "obs_index": {"known_index_a": masks.coef.detach().cpu(), "known_index_u": masks.sol.detach().cpu()},
                    "coef_final": final_phys.coef.detach().cpu(),
                    "sol_final": final_phys.sol.detach().cpu(),
                    "loss": rows,
                    "intermediate": intermediates,
                    "time": final["wall_clock_time"],
                    "config": config.asdict(),
                },
                handle,
            )
    return final


def run_from_config_path(config_path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, overrides=overrides)
    return run_single_ablation(cfg)


def run_from_legacy_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    overrides: dict[str, Any] = {
        "pde": args.pdetype,
        "task": args.problem,
        "legacy_mode": args.mode,
        "k": args.k,
    }
    if args.num_steps:
        overrides["num_steps"] = args.num_steps
    if args.num_obs:
        overrides["num_obs"] = args.num_obs
    if args.sampler:
        if args.hybrid and args.sampler == "deterministic":
            overrides["sampler_phase"] = "hybrid_d2s"
            overrides["switch_ratio"] = 0.5
        elif args.hybrid and args.sampler == "stochastic":
            overrides["sampler_phase"] = "hybrid_s2d"
            overrides["switch_ratio"] = 0.5
        else:
            overrides["sampler_phase"] = args.sampler
    if args.guide is not None and not str2bool(args.guide):
        overrides["guidance_components"] = "noguide"
    if not args.pdeguide:
        overrides["zeta_pde"] = 0.0
    if args.dry_run:
        overrides["dry_run"] = True
    overrides["extra"] = {
        "legacy_remark": args.remark,
        "legacy_dt_sampler": args.dt_sampler,
        "legacy_perturb": args.perturb,
        "legacy_perturb_rate": args.perturb_rate,
        "legacy_lr_decay": args.lr_decay,
        "legacy_freq_decay": args.freq_decay,
    }
    results = []
    for i in range(int(args.batch)):
        cfg = load_config(args.config, overrides=overrides)
        cfg.offset += i
        if args.problem == "forward" and cfg.guidance_components in {"obs_pde", "obs_only"}:
            pass
        if args.mode == "full":
            cfg.num_obs = cfg.img_resolution * cfg.img_resolution
            cfg.sensor_mode = "fixed"
        results.append(run_single_ablation(cfg))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single FM4PDE ablation.")
    parser.add_argument("--config", required=True, help="Ablation YAML or legacy PDE YAML.")
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
        legacy_minmax=getattr(config, "legacy_minmax", False),
    )


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
    flags = guidance_component_flags(config.guidance_components, config.task)
    if config.pde == "nsnonbounded" and flags["pde"] and float(config.zeta_pde) != 0.0:
        warnings.warn(
            "nsnonbounded PDE guidance is disabled because no reliable NS residual is implemented; setting zeta_pde=0.",
            RuntimeWarning,
            stacklevel=2,
        )
        config.zeta_pde = 0.0


def _residual_metadata_for_config(config: AblationConfig) -> dict[str, Any]:
    return {
        "pde": config.pde,
        "residual_status": residual_status(config.pde),
        "residual_mode": config.residual_mode,
        "zeta_pde": config.zeta_pde,
    }


def _cpu_pde_params(params: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in (params or {}).items():
        try:
            out[key] = value.detach().cpu()
        except Exception:
            out[key] = value
    return out


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

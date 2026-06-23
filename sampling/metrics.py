from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def relative_l2(pred: Any, target: Any, eps: float = 1e-12) -> float:
    import torch

    return float((torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target).clamp_min(eps)).detach().cpu())


def obs_relative_l2(pred: Any, target: Any, mask: Any, eps: float = 1e-12) -> float:
    import torch

    diff = (pred - target) * mask
    denom = torch.linalg.vector_norm(target * mask).clamp_min(eps)
    return float((torch.linalg.vector_norm(diff) / denom).detach().cpu())


def pde_residual_norm(residual: Any | None, eps: float = 1e-12) -> float:
    import torch

    if residual is None:
        return 0.0
    return float((torch.linalg.vector_norm(residual) / max(residual.numel(), 1)).detach().cpu())


def step_metrics(
    step: int,
    step_output: Any,
    guidance_losses: Any,
    eval_losses: Any,
    gradient: Any | None,
    phys_state: Any,
    ground_truth: Any,
    masks: Any,
    wall_time: float,
) -> dict[str, Any]:
    pde_meta = eval_losses.metadata.get("pde", {})
    guidance_pde_meta = guidance_losses.metadata.get("pde", {})
    channels = pde_meta.get("residual_channels", {}) or {}
    loss_reduction = eval_losses.metadata.get("loss_reduction", {})
    obs_counts = eval_losses.metadata.get("obs_counts", {})
    component_losses = pde_meta.get("component_losses", {}) or {}
    guidance_component_losses = guidance_pde_meta.get("component_losses", {}) or {}
    row = {
        "step": step,
        "t": _scalar(step_output.t),
        "t_next": _scalar(step_output.t_next),
        "phase": step_output.phase,
        "loss_state": step_output.loss_state,
        "wall_time": wall_time,
        "L_obs_a": _scalar(eval_losses.L_obs_a),
        "L_obs_u": _scalar(eval_losses.L_obs_u),
        "L_pde": _scalar(eval_losses.L_pde),
        "clean_L_obs_a": _scalar(eval_losses.clean_L_obs_a),
        "clean_L_obs_u": _scalar(eval_losses.clean_L_obs_u),
        "eval_L_obs_a": _scalar(eval_losses.L_obs_a),
        "eval_L_obs_u": _scalar(eval_losses.L_obs_u),
        "eval_L_pde": _scalar(eval_losses.L_pde),
        "eval_clean_L_obs_a": _scalar(eval_losses.clean_L_obs_a),
        "eval_clean_L_obs_u": _scalar(eval_losses.clean_L_obs_u),
        "guidance_L_obs_a": _scalar(guidance_losses.L_obs_a),
        "guidance_L_obs_u": _scalar(guidance_losses.L_obs_u),
        "guidance_L_pde": _scalar(guidance_losses.L_pde),
        "guidance_clean_L_obs_a": _scalar(guidance_losses.clean_L_obs_a),
        "guidance_clean_L_obs_u": _scalar(guidance_losses.clean_L_obs_u),
        "rel_l2_a": relative_l2(phys_state.coef, ground_truth.coef),
        "rel_l2_u": relative_l2(phys_state.sol, ground_truth.sol),
        "obs_rel_l2_a": obs_relative_l2(phys_state.coef, ground_truth.coef, masks.coef),
        "obs_rel_l2_u": obs_relative_l2(phys_state.sol, ground_truth.sol, masks.sol),
        "pde_residual_norm": pde_residual_norm(eval_losses.pde_residual),
        "eval_pde_residual_norm": pde_residual_norm(eval_losses.pde_residual),
        "guidance_pde_residual_norm": pde_residual_norm(guidance_losses.pde_residual),
        "pde_residual_status": eval_losses.pde_residual_status,
        "guidance_pde_residual_status": guidance_losses.pde_residual_status,
        "pde_residual_equation": pde_meta.get("equation", ""),
        "interior_residual_norm": pde_meta.get("component_norms", {}).get("interior", 0.0),
        "boundary_residual_norm": pde_meta.get("component_norms", {}).get("boundary", 0.0),
        "initial_residual_norm": pde_meta.get("component_norms", {}).get("initial", 0.0),
        "endpoint_residual_norm": pde_meta.get("component_norms", {}).get("endpoint", 0.0),
        "bc_residual_status": "enabled" if pde_meta.get("bc_residual_enabled") else "disabled",
        "ic_residual_status": "enabled" if pde_meta.get("ic_residual_enabled") else "disabled",
        "boundary_condition_mode": pde_meta.get("boundary_condition_type", ""),
        "initial_condition_mode": pde_meta.get("initial_condition_type", ""),
        "pde_residual_mode": pde_meta.get("requested_residual_mode", pde_meta.get("mode", "")),
        "resolved_residual_mode": pde_meta.get("resolved_residual_mode", ""),
        "pde_residual_rhs": pde_meta.get("rhs", pde_meta.get("rhs_equation", "")),
        "pde_residual_channels_total": channels.get("total_channels", 0),
        "pde_residual_channels_interior": channels.get("interior_channels", 0),
        "pde_residual_channels_bc": channels.get("bc_channels", 0),
        "pde_residual_channels_ic": channels.get("ic_channels", 0),
        "pde_residual_channels_endpoint": channels.get("endpoint_channels", 0),
        "boundary_enforced": pde_meta.get("boundary_enforced", False),
        "boundary_enforced_by_operator": pde_meta.get("boundary_enforced_by_operator", False),
        "boundary_value_residual_applicable": pde_meta.get("boundary_value_residual_applicable", True),
        "pde_residual_region_applied_to": pde_meta.get("pde_residual_region_applied_to", ""),
        "pde_residual_region_skipped": pde_meta.get("pde_residual_region_skipped", False),
        "obs_loss_reduction_a": loss_reduction.get("obs_a", ""),
        "obs_loss_reduction_u": loss_reduction.get("obs_u", ""),
        "pde_loss_reduction": loss_reduction.get("pde", pde_meta.get("loss_reduction", "")),
        "obs_count_coef": obs_counts.get("coef", 0.0),
        "obs_count_sol": obs_counts.get("sol", 0.0),
        "pde_loss_interior": component_losses.get("interior", 0.0),
        "pde_loss_boundary": component_losses.get("boundary", 0.0),
        "pde_loss_initial": component_losses.get("initial", 0.0),
        "pde_loss_endpoint": component_losses.get("endpoint", 0.0),
        "pde_loss_component_total": component_losses.get("total", 0.0),
        "pde_loss_bc_weight": component_losses.get("bc_weight", 0.0),
        "pde_loss_ic_weight": component_losses.get("ic_weight", 0.0),
        "pde_loss_endpoint_weight": component_losses.get("endpoint_weight", 0.0),
        "guidance_pde_loss_interior": guidance_component_losses.get("interior", 0.0),
        "guidance_pde_loss_boundary": guidance_component_losses.get("boundary", 0.0),
        "guidance_pde_loss_initial": guidance_component_losses.get("initial", 0.0),
        "guidance_pde_loss_endpoint": guidance_component_losses.get("endpoint", 0.0),
        "guidance_pde_loss_component_total": guidance_component_losses.get("total", 0.0),
    }
    if gradient is not None:
        row.update(
            {
                "clip_scale": gradient.clip_scale,
                "grad_norm_obs_a": gradient.grad_norm_obs_a,
                "grad_norm_obs_u": gradient.grad_norm_obs_u,
                "grad_norm_pde": gradient.grad_norm_pde,
                "grad_norm_total": gradient.grad_norm_total,
                "gradient_target": gradient.metadata.get("gradient_target", ""),
                "guidance_update_scale": gradient.metadata.get("guidance_update_scale", 0.0),
                "stochastic_guidance_time": gradient.metadata.get("stochastic_guidance_time", ""),
            }
        )
    else:
        row.update(
            {
                "clip_scale": 1.0,
                "grad_norm_obs_a": 0.0,
                "grad_norm_obs_u": 0.0,
                "grad_norm_pde": 0.0,
                "grad_norm_total": 0.0,
                "gradient_target": "",
                "guidance_update_scale": 0.0,
                "stochastic_guidance_time": "",
            }
        )
    return row


def final_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    last = rows[-1].copy()
    last["num_recorded_steps"] = len(rows)
    return last


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _scalar(value: Any) -> float:
    try:
        return float(value.detach().cpu())
    except Exception:
        return float(value)

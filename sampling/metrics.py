from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def relative_l2(pred: Any, target: Any, eps: float = 1e-12) -> float:
    values = relative_l2_per_sample(pred, target, eps=eps)
    return sum(values) / len(values)


def relative_l2_per_sample(pred: Any, target: Any, eps: float = 1e-12) -> list[float]:
    import torch

    diff_flat = _batch_flatten(pred - target)
    target_flat = _batch_flatten(target)
    values = torch.linalg.vector_norm(diff_flat, dim=1) / torch.linalg.vector_norm(target_flat, dim=1).clamp_min(eps)
    return [float(value) for value in values.detach().cpu().tolist()]


def obs_relative_l2(pred: Any, target: Any, mask: Any, eps: float = 1e-12) -> float:
    values = obs_relative_l2_per_sample(pred, target, mask, eps=eps)
    return sum(values) / len(values)


def obs_relative_l2_per_sample(pred: Any, target: Any, mask: Any, eps: float = 1e-12) -> list[float]:
    import torch

    diff = (pred - target) * mask
    diff_flat = _batch_flatten(diff)
    target_flat = _batch_flatten(target * mask)
    values = torch.linalg.vector_norm(diff_flat, dim=1) / torch.linalg.vector_norm(target_flat, dim=1).clamp_min(eps)
    return [float(value) for value in values.detach().cpu().tolist()]


def pde_residual_norm(residual: Any | None, mask: Any | None = None, eps: float = 1e-12) -> float:
    values = pde_residual_norm_per_sample(residual, mask=mask, eps=eps)
    return float("nan") if not values else sum(values) / len(values)


def pde_residual_norm_per_sample(
    residual: Any | None,
    mask: Any | None = None,
    eps: float = 1e-12,
) -> list[float]:
    import torch

    if residual is None:
        return []
    residual_flat = _batch_flatten(residual)
    if mask is None:
        values = residual_flat.square().mean(dim=1).sqrt()
    else:
        expanded = torch.as_tensor(mask, dtype=residual.dtype, device=residual.device).expand_as(residual)
        mask_flat = _batch_flatten(expanded)
        numerator = (residual_flat.square() * mask_flat).sum(dim=1)
        denominator = mask_flat.sum(dim=1).clamp_min(eps)
        values = (numerator / denominator).sqrt()
    return [float(value) for value in values.detach().cpu().tolist()]


def _batch_flatten(value: Any) -> Any:
    if value.ndim <= 1:
        return value.reshape(1, -1)
    return value.reshape(value.shape[0], -1)


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
    guidance_loss_reduction = guidance_losses.metadata.get("guidance_loss_reduction", {})
    obs_counts = eval_losses.metadata.get("obs_counts", {})
    component_losses = pde_meta.get("component_losses", {}) or {}
    guidance_component_losses = guidance_pde_meta.get("guidance_component_losses", {}) or {}
    rel_l2_a_per_sample = relative_l2_per_sample(phys_state.coef, ground_truth.coef)
    rel_l2_u_per_sample = relative_l2_per_sample(phys_state.sol, ground_truth.sol)
    obs_rel_l2_a_per_sample = obs_relative_l2_per_sample(phys_state.coef, ground_truth.coef, masks.coef)
    obs_rel_l2_u_per_sample = obs_relative_l2_per_sample(phys_state.sol, ground_truth.sol, masks.sol)
    eval_pde_per_sample = pde_residual_norm_per_sample(
        eval_losses.pde_residual, mask=getattr(eval_losses, "pde_residual_mask", None)
    )
    guidance_pde_per_sample = pde_residual_norm_per_sample(
        guidance_losses.pde_residual, mask=getattr(guidance_losses, "pde_residual_mask", None)
    )
    component_norms = pde_meta.get("component_norms", {}) or {}
    missing_component_value = None if not eval_pde_per_sample else 0.0
    row = {
        "step": step,
        "t": _scalar(step_output.t),
        "t_next": _scalar(step_output.t_next),
        "phase": step_output.phase,
        "loss_state": step_output.loss_state,
        "endpoint_prediction_mode": getattr(step_output, "endpoint_prediction_mode", "single_step"),
        "endpoint_model_evaluations": getattr(step_output, "endpoint_model_evaluations", 1),
        "wall_time": wall_time,
        "L_obs_a": _scalar(eval_losses.L_obs_a),
        "L_obs_u": _scalar(eval_losses.L_obs_u),
        "L_pde": _optional_scalar(eval_losses.L_pde),
        "clean_L_obs_a": _scalar(eval_losses.clean_L_obs_a),
        "clean_L_obs_u": _scalar(eval_losses.clean_L_obs_u),
        "eval_L_obs_a": _scalar(eval_losses.L_obs_a),
        "eval_L_obs_u": _scalar(eval_losses.L_obs_u),
        "eval_L_pde": _optional_scalar(eval_losses.L_pde),
        "eval_clean_L_obs_a": _scalar(eval_losses.clean_L_obs_a),
        "eval_clean_L_obs_u": _scalar(eval_losses.clean_L_obs_u),
        "guidance_L_obs_a": _optional_scalar(_guidance_value(guidance_losses, "obs_a")),
        "guidance_L_obs_u": _optional_scalar(_guidance_value(guidance_losses, "obs_u")),
        "guidance_L_pde": _optional_scalar(_guidance_value(guidance_losses, "pde")),
        "guidance_clean_L_obs_a": _scalar(guidance_losses.clean_L_obs_a),
        "guidance_clean_L_obs_u": _scalar(guidance_losses.clean_L_obs_u),
        "rel_l2_a": sum(rel_l2_a_per_sample) / len(rel_l2_a_per_sample),
        "rel_l2_u": sum(rel_l2_u_per_sample) / len(rel_l2_u_per_sample),
        "obs_rel_l2_a": sum(obs_rel_l2_a_per_sample) / len(obs_rel_l2_a_per_sample),
        "obs_rel_l2_u": sum(obs_rel_l2_u_per_sample) / len(obs_rel_l2_u_per_sample),
        "rel_l2_a_per_sample": rel_l2_a_per_sample,
        "rel_l2_u_per_sample": rel_l2_u_per_sample,
        "obs_rel_l2_a_per_sample": obs_rel_l2_a_per_sample,
        "obs_rel_l2_u_per_sample": obs_rel_l2_u_per_sample,
        "relative_l2_reduction": "mean_of_per_sample_relative_l2",
        "pde_residual_norm": None if not eval_pde_per_sample else sum(eval_pde_per_sample) / len(eval_pde_per_sample),
        "eval_pde_residual_norm": (
            None if not eval_pde_per_sample else sum(eval_pde_per_sample) / len(eval_pde_per_sample)
        ),
        "guidance_pde_residual_norm": (
            None
            if not guidance_pde_per_sample
            else sum(guidance_pde_per_sample) / len(guidance_pde_per_sample)
        ),
        "pde_residual_norm_per_sample": eval_pde_per_sample,
        "guidance_pde_residual_norm_per_sample": guidance_pde_per_sample,
        "pde_residual_norm_reduction": "mean_of_per_sample_rms",
        "pde_residual_status": eval_losses.pde_residual_status,
        "guidance_pde_residual_status": guidance_losses.pde_residual_status,
        "pde_residual_error_type": pde_meta.get("error_type", ""),
        "pde_residual_error_message": pde_meta.get("error_message", ""),
        "pde_uses_ground_truth_fields": pde_meta.get("uses_ground_truth_fields", False),
        "pde_uses_ground_truth_endpoint_fields": pde_meta.get("uses_ground_truth_endpoint_fields", False),
        "pde_ground_truth_field_exception": pde_meta.get("ground_truth_field_exception"),
        "pde_coef_input_source": pde_meta.get("field_input_sources", {}).get("coef", ""),
        "pde_sol_input_source": pde_meta.get("field_input_sources", {}).get("sol", ""),
        "pde_auxiliary_field_input_sources": pde_meta.get("auxiliary_field_input_sources", {}),
        "pde_excluded_ground_truth_field_params": pde_meta.get("excluded_ground_truth_field_params", []),
        "pde_residual_equation": pde_meta.get("equation", ""),
        "interior_residual_norm": component_norms.get("interior", missing_component_value),
        "boundary_residual_norm": component_norms.get("boundary", missing_component_value),
        "endpoint_residual_norm": component_norms.get("endpoint", missing_component_value),
        "bc_residual_status": "enabled" if pde_meta.get("bc_residual_enabled") else "disabled",
        "boundary_condition_mode": pde_meta.get("boundary_condition_type", ""),
        "pde_residual_mode": pde_meta.get("requested_residual_mode", pde_meta.get("mode", "")),
        "resolved_residual_mode": pde_meta.get("resolved_residual_mode", ""),
        "pde_residual_rhs": pde_meta.get("rhs", pde_meta.get("rhs_equation", "")),
        "pde_residual_channels_total": channels.get("total_channels", 0),
        "pde_residual_channels_interior": channels.get("interior_channels", 0),
        "pde_residual_channels_bc": channels.get("bc_channels", 0),
        "pde_residual_channels_endpoint": channels.get("endpoint_channels", 0),
        "boundary_enforced": pde_meta.get("boundary_enforced", False),
        "boundary_enforced_by_operator": pde_meta.get("boundary_enforced_by_operator", False),
        "boundary_value_residual_applicable": pde_meta.get("boundary_value_residual_applicable", True),
        "pde_residual_region_applied_to": pde_meta.get("pde_residual_region_applied_to", ""),
        "pde_residual_region_skipped": pde_meta.get("pde_residual_region_skipped", False),
        "obs_loss_reduction_a": loss_reduction.get("obs_a", ""),
        "obs_loss_reduction_u": loss_reduction.get("obs_u", ""),
        "pde_loss_reduction": loss_reduction.get("pde", pde_meta.get("loss_reduction", "")),
        "guidance_obs_loss_reduction_a": guidance_loss_reduction.get("obs_a", ""),
        "guidance_obs_loss_reduction_u": guidance_loss_reduction.get("obs_u", ""),
        "guidance_pde_loss_reduction": guidance_loss_reduction.get("pde", ""),
        "obs_count_coef": obs_counts.get("coef", 0.0),
        "obs_count_sol": obs_counts.get("sol", 0.0),
        "pde_loss_interior": component_losses.get("interior", 0.0),
        "pde_loss_boundary": component_losses.get("boundary", 0.0),
        "pde_loss_endpoint": component_losses.get("endpoint", 0.0),
        "pde_loss_component_total": component_losses.get("total", 0.0),
        "pde_loss_bc_weight": component_losses.get("bc_weight", 0.0),
        "pde_loss_endpoint_weight": component_losses.get("endpoint_weight", 0.0),
        "guidance_pde_loss_interior": guidance_component_losses.get("interior", 0.0),
        "guidance_pde_loss_boundary": guidance_component_losses.get("boundary", 0.0),
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
                "clip_scope": gradient.metadata.get("clip_scope", ""),
                "guidance_update_scale": gradient.metadata.get("guidance_update_scale", 0.0),
                "stochastic_guidance_time": gradient.metadata.get("stochastic_guidance_time", ""),
                "deterministic_bt_mode": gradient.metadata.get("deterministic_bt_mode", ""),
                "deterministic_guidance_factor": gradient.metadata.get(
                    "deterministic_guidance_factor", 1.0
                ),
                "guidance_correction_raw_norm": gradient.metadata.get(
                    "guidance_correction_raw_norm", 0.0
                ),
                "guidance_correction_norm": gradient.metadata.get("guidance_correction_norm", 0.0),
                "guidance_correction_rms": gradient.metadata.get("guidance_correction_rms", 0.0),
                "correction_clip_scale": gradient.metadata.get("correction_clip_scale", 1.0),
                "nonfinite_correction_samples": gradient.metadata.get(
                    "nonfinite_correction_samples", 0
                ),
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
                "clip_scope": "",
                "guidance_update_scale": 0.0,
                "stochastic_guidance_time": "",
                "deterministic_bt_mode": "",
                "deterministic_guidance_factor": 1.0,
                "guidance_correction_raw_norm": 0.0,
                "guidance_correction_norm": 0.0,
                "guidance_correction_rms": 0.0,
                "correction_clip_scale": 1.0,
                "nonfinite_correction_samples": 0,
            }
        )
    return row


def final_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    last = rows[-1].copy()
    last["num_recorded_steps"] = len(rows)
    return last


def per_sample_metrics(
    phys_state: Any,
    ground_truth: Any,
    masks: Any,
    eval_losses: Any,
    *,
    step: int | None = None,
) -> list[dict[str, Any]]:
    """Return one independent, compact evaluation row per batch sample."""
    rel_a = relative_l2_per_sample(phys_state.coef, ground_truth.coef)
    rel_u = relative_l2_per_sample(phys_state.sol, ground_truth.sol)
    obs_a = obs_relative_l2_per_sample(phys_state.coef, ground_truth.coef, masks.coef)
    obs_u = obs_relative_l2_per_sample(phys_state.sol, ground_truth.sol, masks.sol)
    pde = pde_residual_norm_per_sample(
        eval_losses.pde_residual,
        mask=getattr(eval_losses, "pde_residual_mask", None),
    )
    sample_ids = list(ground_truth.metadata.get("sample_ids", []))
    rows = []
    for index in range(len(rel_a)):
        row = {
            "sample_index": index,
            "sample_id": sample_ids[index] if index < len(sample_ids) else str(index),
            "rel_l2_a": rel_a[index],
            "rel_l2_u": rel_u[index],
            "obs_rel_l2_a": obs_a[index],
            "obs_rel_l2_u": obs_u[index],
            "pde_residual_norm": pde[index] if index < len(pde) else None,
        }
        if step is not None:
            row = {"step": int(step), **row}
        rows.append(row)
    return rows


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


def _optional_scalar(value: Any) -> float | None:
    return None if value is None else _scalar(value)


def _guidance_value(losses: Any, component: str) -> Any:
    value = getattr(losses, f"guidance_L_{component}", None)
    return getattr(losses, f"L_{component}") if value is None else value

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data.specs import TEMPORAL_ENDPOINT_PDES
from sampling.config import normalize_residual_mode
from sampling.masks import PairMasks, residual_region_mask
from sampling.pde_residuals import compute_pde_residual
from sampling.state import SplitState


GROUND_TRUTH_FIELD_PDE_PARAM_KEYS = frozenset(
    {
        "trajectory",
        "full_trajectory",
        "trajectory_is_observed_ground_truth",
        "near_endpoint_temporal",
        "observed_initial",
        "true_initial",
        "initial_mask",
        "allow_true_initial_condition",
        "initial_condition_source",
    }
)


@dataclass
class ObservationTargets:
    coef_clean: Any
    sol_clean: Any
    coef_noisy: Any
    sol_noisy: Any


@dataclass
class GuidanceLossOutput:
    L_obs_a: Any
    L_obs_u: Any
    L_pde: Any
    obs_a_residual: Any
    obs_u_residual: Any
    pde_residual: Any | None
    clean_L_obs_a: Any
    clean_L_obs_u: Any
    pde_residual_status: str
    metadata: dict[str, Any]
    pde_residual_mask: Any | None = None
    guidance_L_obs_a: Any | None = None
    guidance_L_obs_u: Any | None = None
    guidance_L_pde: Any | None = None


def compute_guidance_losses(
    phys_state: SplitState,
    ground_truth: Any,
    masks: PairMasks,
    config: Any,
    observations: ObservationTargets | None = None,
) -> GuidanceLossOutput:
    import torch

    enabled = guidance_component_flags(config.guidance_components, getattr(config, "task", "both"))
    if float(getattr(config, "zeta_pde", 1.0)) == 0.0:
        enabled["pde"] = False
    target_coef = observations.coef_noisy if observations is not None else ground_truth.coef * masks.coef
    target_sol = observations.sol_noisy if observations is not None else ground_truth.sol * masks.sol
    clean_coef = observations.coef_clean if observations is not None else ground_truth.coef * masks.coef
    clean_sol = observations.sol_clean if observations is not None else ground_truth.sol * masks.sol

    obs_a_residual = (phys_state.coef - target_coef) * masks.coef
    obs_u_residual = (phys_state.sol - target_sol) * masks.sol
    clean_obs_a_residual = (phys_state.coef - clean_coef) * masks.coef
    clean_obs_u_residual = (phys_state.sol - clean_sol) * masks.sol

    zero = phys_state.coef.sum() * 0.0
    # Evaluation losses are properties of the prediction and must not change
    # when a guidance component or its zeta coefficient is disabled.
    L_obs_a = _masked_mse(phys_state.coef, target_coef, masks.coef)
    L_obs_u = _masked_mse(phys_state.sol, target_sol, masks.sol)
    obs_guidance_reduction = str(getattr(config, "obs_guidance_reduction", "mse"))
    if obs_guidance_reduction == "mse":
        raw_guidance_L_obs_a = L_obs_a
        raw_guidance_L_obs_u = L_obs_u
        obs_guidance_reduction_label = "masked_mse_over_observed_entries"
    elif obs_guidance_reduction == "l2_norm":
        raw_guidance_L_obs_a = _masked_l2_norm(phys_state.coef, target_coef, masks.coef)
        raw_guidance_L_obs_u = _masked_l2_norm(phys_state.sol, target_sol, masks.sol)
        obs_guidance_reduction_label = "mean_of_per_sample_masked_l2_norm"
    else:
        raise ValueError(f"Unknown obs_guidance_reduction={obs_guidance_reduction!r}")
    guidance_L_obs_a = raw_guidance_L_obs_a if enabled["obs_a"] else zero
    guidance_L_obs_u = raw_guidance_L_obs_u if enabled["obs_u"] else zero
    clean_L_obs_a = _masked_mse(phys_state.coef, clean_coef, masks.coef)
    clean_L_obs_u = _masked_mse(phys_state.sol, clean_sol, masks.sol)
    obs_counts = {
        "coef": _masked_count(phys_state.coef, masks.coef),
        "sol": _masked_count(phys_state.sol, masks.sol),
    }

    pde_field = None
    pde_field_mask = None
    status = "disabled"
    pde_meta: dict[str, Any] = {}
    # Compute a generated-state PDE residual for evaluation even when guidance
    # does not use it. An observed reference trajectory is deliberately skipped
    # because its residual is not a metric of the generated endpoint.
    raw_pde_params = _pde_params_with_residual_options(getattr(ground_truth, "pde_params", None), config)
    residual_mode = normalize_residual_mode(getattr(config, "residual_mode", "auto"))
    sparse_near_endpoint_exception_allowed = (
        residual_mode == "near_endpoint_temporal" and config.pde in TEMPORAL_ENDPOINT_PDES
    )
    if residual_mode == "near_endpoint_temporal" and not sparse_near_endpoint_exception_allowed:
        raise ValueError(
            f"near_endpoint_temporal is only supported for temporal endpoint PDEs; got {config.pde!r}"
        )
    field_data_dependent_mode = (
        residual_mode in {"full_trajectory_fd", "full_time_space"} and config.pde != "burger"
    )
    if field_data_dependent_mode:
        raise ValueError(
            f"{residual_mode} cannot be used for PDE guidance or generated-sample evaluation: "
            "the current model does not output the required intermediate time states, and observed ground-truth "
            "frames must not be mixed into PDE loss. Only Burgers supplies a full predicted time-space field."
        )
    pde_params, excluded_field_params = prediction_only_pde_params(
        raw_pde_params,
        allow_sparse_near_endpoint=sparse_near_endpoint_exception_allowed,
    )
    uses_sparse_near_endpoint_exception = (
        sparse_near_endpoint_exception_allowed and "near_endpoint_temporal" in pde_params
    )
    loss_components: dict[str, Any | None] = {
        "interior": None,
        "boundary": None,
        "endpoint": None,
        "interior_mask": None,
    }
    try:
        residual = compute_pde_residual(
            config.pde,
            phys_state.coef,
            phys_state.sol,
            pde_params=pde_params,
            k=getattr(config, "k", 1),
            residual_mode=getattr(config, "residual_mode", "auto"),
        )
        status = residual.status
        pde_meta = residual.metadata
        pde_field, pde_field_mask, compose_meta, loss_components = _compose_region_aware_pde_field(
            residual, config, masks
        )
        _require_finite(pde_field, f"{config.pde} PDE residual")
        pde_meta.update(compose_meta)
    except Exception as exc:
        raise RuntimeError(
            f"PDE residual computation failed for generated {config.pde!r} prediction: {exc}"
        ) from exc
    pde_meta["field_input_sources"] = {"coef": "model_output", "sol": "model_output"}
    pde_meta["uses_ground_truth_fields"] = uses_sparse_near_endpoint_exception
    pde_meta["uses_ground_truth_endpoint_fields"] = False
    pde_meta["ground_truth_field_exception"] = (
        "near_endpoint_temporal_sparse_observations"
        if uses_sparse_near_endpoint_exception
        else None
    )
    pde_meta["auxiliary_field_input_sources"] = (
        {
            "q_dt": "sparse_ground_truth_observations",
            "q_T_minus_dt": "sparse_ground_truth_observations",
        }
        if uses_sparse_near_endpoint_exception
        else {}
    )
    pde_meta["excluded_ground_truth_field_params"] = excluded_field_params
    if pde_field is not None and status != "disabled":
        L_pde, pde_component_losses = _componentwise_pde_mse_loss(
            loss_components,
            bc_weight=float(getattr(config, "bc_weight", 1.0)),
            endpoint_weight=float(getattr(config, "endpoint_bc_weight", 1.0)),
            fallback_field=pde_field,
        )
        _require_finite(L_pde, f"{config.pde} PDE loss")
        pde_meta["loss_reduction"] = "componentwise_mse_sum"
        pde_meta["component_losses"] = pde_component_losses
    else:
        L_pde = None
        pde_component_losses = {}

    if enabled["pde"]:
        if L_pde is None:
            raise RuntimeError(
                f"PDE guidance is enabled for {config.pde!r}, but residual_mode={residual_mode!r} "
                "does not produce an evaluable PDE loss"
            )
        guidance_L_pde = L_pde
        guidance_component_losses = pde_component_losses
    else:
        guidance_L_pde = zero
        guidance_component_losses = {
            "interior": 0.0,
            "boundary": 0.0,
            "endpoint": 0.0,
            "total": 0.0,
            "bc_weight": float(getattr(config, "bc_weight", 1.0)),
            "endpoint_weight": float(getattr(config, "endpoint_bc_weight", 1.0)),
        }
    pde_meta["guidance_component_losses"] = guidance_component_losses

    metadata = {
        "enabled": enabled,
        "pde": pde_meta,
        "pde_params_used": sorted(pde_params),
        "pde_field_input_sources": {"coef": "model_output", "sol": "model_output"},
        "pde_uses_ground_truth_fields": uses_sparse_near_endpoint_exception,
        "pde_uses_ground_truth_endpoint_fields": False,
        "pde_ground_truth_field_exception": pde_meta["ground_truth_field_exception"],
        "excluded_ground_truth_field_params": excluded_field_params,
        "residual_status": status,
        "loss_reduction": {
            "obs_a": "masked_mse_over_observed_entries",
            "obs_u": "masked_mse_over_observed_entries",
            "pde": "componentwise_mse_sum",
        },
        "guidance_loss_reduction": {
            "obs_a": obs_guidance_reduction_label,
            "obs_u": obs_guidance_reduction_label,
            "pde": "componentwise_mse_sum",
        },
        "loss_batch_reduction": "mean_of_per_sample",
        "obs_counts": obs_counts,
    }
    return GuidanceLossOutput(
        L_obs_a=L_obs_a,
        L_obs_u=L_obs_u,
        L_pde=L_pde,
        obs_a_residual=obs_a_residual,
        obs_u_residual=obs_u_residual,
        pde_residual=pde_field,
        clean_L_obs_a=clean_L_obs_a,
        clean_L_obs_u=clean_L_obs_u,
        pde_residual_status=status,
        metadata=metadata,
        pde_residual_mask=pde_field_mask,
        guidance_L_obs_a=guidance_L_obs_a,
        guidance_L_obs_u=guidance_L_obs_u,
        guidance_L_pde=guidance_L_pde,
    )


def guidance_component_flags(name: str, task: str = "both") -> dict[str, bool]:
    if name == "noguide":
        flags = {"obs_a": False, "obs_u": False, "pde": False}
        return _apply_task_gate(flags, task)
    if name == "obs_only":
        return _apply_task_gate({"obs_a": True, "obs_u": True, "pde": False}, task)
    if name == "pde_only":
        return _apply_task_gate({"obs_a": False, "obs_u": False, "pde": True}, task)
    if name == "obs_pde":
        return _apply_task_gate({"obs_a": True, "obs_u": True, "pde": True}, task)
    if name == "coef_obs_only":
        return _apply_task_gate({"obs_a": True, "obs_u": False, "pde": False}, task)
    if name == "sol_obs_only":
        return _apply_task_gate({"obs_a": False, "obs_u": True, "pde": False}, task)
    if name == "both_obs":
        if task != "both":
            raise ValueError("guidance_components='both_obs' is only valid for task='both'")
        return {"obs_a": True, "obs_u": True, "pde": False}
    raise ValueError(f"Unknown guidance_components={name!r}")


def _apply_task_gate(flags: dict[str, bool], task: str) -> dict[str, bool]:
    flags = flags.copy()
    if task == "forward":
        flags["obs_u"] = False
    elif task == "inverse":
        flags["obs_a"] = False
    elif task == "unconditional":
        flags["obs_a"] = False
        flags["obs_u"] = False
        flags["pde"] = False
    elif task != "both":
        raise ValueError(f"Unknown task={task!r}")
    return flags


def _mse(residual: Any) -> Any:
    return (residual**2).mean()


def _require_finite(value: Any, name: str) -> None:
    import torch

    if value is None:
        return
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _zero_like_reference(reference: Any) -> Any:
    return reference.sum() * 0.0


def _component_mse_or_zero(component: Any | None, reference: Any, mask: Any | None = None) -> Any:
    if component is None:
        return _zero_like_reference(reference)
    if component.numel() == 0:
        return _zero_like_reference(reference)
    return _masked_residual_mse(component, mask) if mask is not None else _mse(component)


def _componentwise_pde_mse_loss(
    components: dict[str, Any | None],
    *,
    bc_weight: float,
    endpoint_weight: float,
    fallback_field: Any,
) -> tuple[Any, dict[str, float]]:
    interior = components.get("interior")
    boundary = components.get("boundary")
    endpoint = components.get("endpoint")
    interior_mask = components.get("interior_mask")

    if interior is None and boundary is None and endpoint is None:
        loss = (
            _masked_residual_mse(fallback_field, interior_mask)
            if interior_mask is not None
            else _mse(fallback_field)
        )
        return loss, {
            "interior": float(loss.detach().cpu()),
            "boundary": 0.0,
            "endpoint": 0.0,
            "total": float(loss.detach().cpu()),
            "bc_weight": float(bc_weight),
            "endpoint_weight": float(endpoint_weight),
        }

    reference = interior
    if reference is None:
        for item in (boundary, endpoint, fallback_field):
            if item is not None:
                reference = item
                break
    if reference is None:
        reference = fallback_field

    L_int = _component_mse_or_zero(interior, reference, interior_mask)
    L_bc = _component_mse_or_zero(boundary, reference)
    L_ep = _component_mse_or_zero(endpoint, reference)

    total = L_int + float(bc_weight) * L_bc + float(endpoint_weight) * L_ep

    detached = {
        "interior": float(L_int.detach().cpu()),
        "boundary": float(L_bc.detach().cpu()),
        "endpoint": float(L_ep.detach().cpu()),
        "total": float(total.detach().cpu()),
        "bc_weight": float(bc_weight),
        "endpoint_weight": float(endpoint_weight),
    }
    return total, detached


def _masked_mse(pred: Any, target: Any, mask: Any, *, eps: float = 1e-12) -> Any:
    residual2, expanded_mask = _masked_squared_residual(pred, target, mask)
    batch = int(pred.shape[0]) if pred.ndim > 1 else 1
    numerator = residual2.reshape(batch, -1).sum(dim=1)
    denominator = expanded_mask.reshape(batch, -1).sum(dim=1)
    per_sample = numerator / denominator.clamp_min(eps)
    return per_sample.mean()


def _masked_l2_norm(pred: Any, target: Any, mask: Any) -> Any:
    """Mean of per-sample L2 norms over observed entries, matching DiffusionPDE at B=1."""
    import torch

    mask = torch.as_tensor(mask, dtype=pred.dtype, device=pred.device).expand_as(pred)
    target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
    residual = (pred - target) * mask
    batch = int(pred.shape[0]) if pred.ndim > 1 else 1
    per_sample = torch.linalg.vector_norm(residual.reshape(batch, -1), dim=1)
    return per_sample.mean()


def _masked_residual_mse(residual: Any, mask: Any, *, eps: float = 1e-12) -> Any:
    import torch

    mask = torch.as_tensor(mask, dtype=residual.dtype, device=residual.device).expand_as(residual)
    batch = int(residual.shape[0])
    numerator = ((residual**2) * mask).reshape(batch, -1).sum(dim=1)
    denominator = mask.reshape(batch, -1).sum(dim=1)
    return (numerator / denominator.clamp_min(eps)).mean()


def _masked_count(pred: Any, mask: Any) -> float:
    _, expanded_mask = _masked_squared_residual(pred, pred, mask)
    return float(expanded_mask.sum().detach().cpu())


def _masked_squared_residual(pred: Any, target: Any, mask: Any) -> tuple[Any, Any]:
    import torch

    mask = torch.as_tensor(mask, dtype=pred.dtype, device=pred.device)
    target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
    expanded_mask = mask.expand_as(pred)
    return ((pred - target) ** 2) * expanded_mask, expanded_mask


def _pde_params_with_residual_options(pde_params: dict[str, Any] | None, config: Any) -> dict[str, Any]:
    params = dict(pde_params or {})
    option_names = (
        "hermite_collocation_times",
        "hermite_num_collocation",
        "hermite_include_integral_residual",
        "hermite_integral_weight",
        "enforce_boundary_conditions",
        "boundary_condition_mode",
        "bc_weight",
        "endpoint_bc_weight",
        "boundary_residual_normalization",
        "allow_unknown_boundary_conditions",
        "ns_operator_mode",
        "coef_positive_mode",
        "coef_positive_floor",
    )
    for name in option_names:
        if hasattr(config, name):
            params[name] = getattr(config, name)
    return params


def prediction_only_pde_params(
    params: dict[str, Any],
    *,
    allow_sparse_near_endpoint: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Remove true fields, except explicitly authorized sparse near-endpoint observations."""
    excluded_keys = set(GROUND_TRUTH_FIELD_PDE_PARAM_KEYS)
    if allow_sparse_near_endpoint:
        excluded_keys.remove("near_endpoint_temporal")
    excluded = sorted(key for key in params if key in excluded_keys)
    filtered = {key: value for key, value in params.items() if key not in excluded_keys}
    if allow_sparse_near_endpoint and "near_endpoint_temporal" in filtered:
        filtered["near_endpoint_temporal"] = _sparsify_near_endpoint_temporal(
            filtered["near_endpoint_temporal"]
        )
    return filtered, excluded


def _sparsify_near_endpoint_temporal(value: Any) -> dict[str, Any]:
    import torch

    if not isinstance(value, dict):
        raise ValueError("near_endpoint_temporal must be a mapping of sparse observations and masks")
    required = {"q_dt", "q_T_minus_dt", "dt", "mask_0", "mask_T"}
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(
            "near_endpoint_temporal sparse-observation exception is missing required fields: "
            + ", ".join(missing)
        )
    sanitized = dict(value)
    for field_name, mask_name in (("q_dt", "mask_0"), ("q_T_minus_dt", "mask_T")):
        field = torch.as_tensor(value[field_name]).detach()
        if field.ndim == 3:
            field = field.unsqueeze(0)
        if field.ndim != 4:
            raise ValueError(f"{field_name} must be BCHW-compatible, got shape={tuple(field.shape)}")
        mask = torch.as_tensor(value[mask_name], dtype=field.dtype, device=field.device).detach()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"{mask_name} must be BCHW-compatible, got shape={tuple(mask.shape)}")
        if mask.shape[0] == 1 and field.shape[0] > 1:
            mask = mask.repeat(field.shape[0], 1, 1, 1)
        if mask.shape[0] != field.shape[0] or mask.shape[-2:] != field.shape[-2:]:
            raise ValueError(
                f"{mask_name} must match {field_name} batch/spatial shape; "
                f"got mask={tuple(mask.shape)}, field={tuple(field.shape)}"
            )
        if mask.shape[1] not in {1, field.shape[1]}:
            raise ValueError(
                f"{mask_name} must have one or {field.shape[1]} channels, got {mask.shape[1]}"
            )
        if not bool(torch.isfinite(mask).all().detach().cpu()) or bool(
            torch.any((mask != 0) & (mask != 1)).detach().cpu()
        ):
            raise ValueError(f"{mask_name} must be a finite binary sparse-observation mask")
        counts = mask.reshape(mask.shape[0], -1).sum(dim=1)
        if bool(torch.any(counts <= 0).detach().cpu()):
            raise ValueError(f"{mask_name} must observe at least one point in every batch sample")
        sanitized[field_name] = field * mask.expand_as(field)
        sanitized[mask_name] = mask
    metadata = dict(value.get("metadata", {}) if isinstance(value.get("metadata"), dict) else {})
    metadata.update(
        {
            "ground_truth_field_exception": "near_endpoint_temporal_sparse_observations",
            "observed_values_only": True,
            "unobserved_values_zeroed": True,
            "full_near_endpoint_frames_retained": False,
        }
    )
    sanitized["metadata"] = metadata
    return sanitized


def _component_norms(components: dict[str, Any] | None) -> dict[str, float]:
    import torch

    norms: dict[str, float] = {}
    for name in ("interior", "boundary", "endpoint"):
        value = None if components is None else components.get(name)
        if value is None:
            norms[name] = 0.0
        else:
            mask = None if components is None else components.get(f"{name}_mask")
            if mask is None:
                per_sample = value.square().reshape(value.shape[0], -1).mean(dim=1).sqrt()
            else:
                expanded = torch.as_tensor(mask, dtype=value.dtype, device=value.device).expand_as(value)
                numerator = (value.square() * expanded).reshape(value.shape[0], -1).sum(dim=1)
                denominator = expanded.reshape(value.shape[0], -1).sum(dim=1).clamp_min(1.0)
                per_sample = (numerator / denominator).sqrt()
            norms[name] = float(per_sample.mean().detach().cpu())
    return norms


def _compose_region_aware_pde_field(
    residual_output: Any, config: Any, masks: PairMasks
) -> tuple[Any, Any | None, dict[str, Any], dict[str, Any | None]]:
    import torch

    pde_field = residual_output.residual
    components = residual_output.components
    if components is None:
        raise ValueError("PDE residual output must provide named residual components")

    interior = components.get("interior")
    if interior is None:
        interior = pde_field
    boundary = components.get("boundary")
    endpoint = components.get("endpoint")

    metadata: dict[str, Any] = {
        "pde_residual_region": config.pde_residual_region,
        "boundary_region_masked": False,
        "endpoint_region_masked": False,
    }
    interior_masked = interior
    interior_mask = None
    if residual_output.metadata.get("resolved_residual_mode") == "near_endpoint_temporal" or residual_output.metadata.get("mode") == "near_endpoint_temporal":
        metadata.update(
            {
                "pde_residual_region_applied_to": "interior_only",
                "pde_residual_region_skipped": True,
                "reason": "near_endpoint_temporal interior is already sparse-temporal masked",
            }
        )
    else:
        region = residual_region_mask(
            config.pde_residual_region,
            masks.coef,
            masks.sol,
            tuple(interior.shape),
            task=getattr(config, "task", "both"),
        )
        if region is not None:
            interior_masked = interior * region
            interior_mask = region
        metadata.update(
            {
                "pde_residual_region_applied_to": "interior_only",
                "pde_residual_region_skipped": False,
            }
        )

    field = _append_components_for_logging(interior_masked, boundary, endpoint)
    loss_components = {
        "interior": interior_masked,
        "boundary": boundary,
        "endpoint": endpoint,
        "interior_mask": interior_mask,
    }
    metadata["component_norms"] = _component_norms(
        loss_components
    )
    channels = {
        "interior_channels": int(interior_masked.shape[1]),
        "bc_channels": int(boundary.shape[1]) if boundary is not None else 0,
        "endpoint_channels": int(endpoint.shape[1]) if endpoint is not None else 0,
        "total_channels": int(field.shape[1]),
    }
    metadata["residual_channels"] = channels
    field_mask = _append_component_masks(interior_masked, interior_mask, boundary, endpoint)
    return field, field_mask, metadata, loss_components


def _append_component_masks(
    interior: Any,
    interior_mask: Any | None,
    boundary: Any | None,
    endpoint: Any | None,
) -> Any | None:
    import torch

    if interior_mask is None:
        return None
    parts = [
        torch.as_tensor(interior_mask, dtype=interior.dtype, device=interior.device).expand_as(interior)
    ]
    for component in (boundary, endpoint):
        if component is not None:
            parts.append(torch.ones_like(component))
    return torch.cat(parts, dim=1)


def _append_components_for_logging(
    interior: Any,
    boundary: Any | None,
    endpoint: Any | None,
) -> Any:
    import torch

    parts = [interior]
    if boundary is not None:
        parts.append(boundary)
    if endpoint is not None:
        parts.append(endpoint)
    return torch.cat(parts, dim=1)

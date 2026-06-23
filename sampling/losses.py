from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sampling.masks import PairMasks, residual_region_mask
from sampling.pde_residuals import compute_pde_residual
from sampling.state import SplitState


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
    obs_loss_type = _resolve_obs_loss_type(config)
    pde_loss_type = _resolve_pde_loss_type(config)
    L_obs_a = _observation_loss(phys_state.coef, target_coef, masks.coef, obs_loss_type) if enabled["obs_a"] else zero
    L_obs_u = _observation_loss(phys_state.sol, target_sol, masks.sol, obs_loss_type) if enabled["obs_u"] else zero
    clean_L_obs_a = _observation_loss(phys_state.coef, clean_coef, masks.coef, obs_loss_type)
    clean_L_obs_u = _observation_loss(phys_state.sol, clean_sol, masks.sol, obs_loss_type)
    obs_counts = {
        "coef": _masked_count(phys_state.coef, masks.coef),
        "sol": _masked_count(phys_state.sol, masks.sol),
    }

    pde_field = None
    status = "disabled"
    pde_meta: dict[str, Any] = {}
    if enabled["pde"]:
        pde_params = _pde_params_with_residual_options(getattr(ground_truth, "pde_params", None), config)
        ic_mode = str(getattr(config, "initial_condition_mode", "auto"))
        if getattr(config, "enforce_initial_conditions", True) and enabled["obs_a"]:
            pde_params.setdefault("observed_initial", target_coef * masks.coef)
            pde_params.setdefault("initial_mask", masks.coef)
            pde_params.setdefault("initial_condition_source", "observation_loss_masked_coef")
        elif ic_mode == "observed_initial":
            raise ValueError(
                "initial_condition_mode='observed_initial' requires coefficient/initial observations; "
                "use task='forward' or 'both', or set initial_condition_mode='none/auto'."
            )
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
        pde_field, compose_meta = _compose_region_aware_pde_field(residual, config, masks)
        pde_meta.update(compose_meta)
        L_pde = _pde_loss(pde_field, pde_loss_type)
    else:
        L_pde = zero

    metadata = {
        "enabled": enabled,
        "pde": pde_meta,
        "pde_params_used": sorted(getattr(ground_truth, "pde_params", {}) or {}),
        "residual_status": status,
        "loss_reduction": {
            "obs_a": _loss_reduction_metadata("obs_a", obs_loss_type),
            "obs_u": _loss_reduction_metadata("obs_u", obs_loss_type),
            "pde": _loss_reduction_metadata("pde", pde_loss_type),
            "legacy_loss_type": getattr(config, "loss_type", None),
        },
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


def _reduce_loss(residual: Any, loss_type: str) -> Any:
    import torch

    if loss_type == "l1":
        return residual.abs().mean()
    if loss_type == "l2":
        return torch.linalg.vector_norm(residual) / max(residual.numel(), 1)
    if loss_type == "mse":
        return (residual**2).mean()
    raise ValueError(f"Unknown loss_type={loss_type!r}")


def _mse(residual: Any) -> Any:
    return (residual**2).mean()


def _masked_mse(pred: Any, target: Any, mask: Any, *, eps: float = 1e-12) -> Any:
    residual2, expanded_mask = _masked_squared_residual(pred, target, mask)
    denom = expanded_mask.sum().clamp_min(eps)
    return residual2.sum() / denom


def _masked_count(pred: Any, mask: Any) -> float:
    _, expanded_mask = _masked_squared_residual(pred, pred, mask)
    return float(expanded_mask.sum().detach().cpu())


def _masked_squared_residual(pred: Any, target: Any, mask: Any) -> tuple[Any, Any]:
    import torch

    mask = torch.as_tensor(mask, dtype=pred.dtype, device=pred.device)
    target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
    expanded_mask = mask.expand_as(pred)
    return ((pred - target) ** 2) * expanded_mask, expanded_mask


def _observation_loss(pred: Any, target: Any, mask: Any, loss_type: str) -> Any:
    if loss_type == "masked_mse":
        return _masked_mse(pred, target, mask)
    residual = (pred - target) * mask
    return _reduce_loss(residual, loss_type)


def _pde_loss(residual: Any, loss_type: str) -> Any:
    if loss_type == "mse":
        return _mse(residual)
    return _reduce_loss(residual, loss_type)


def _resolve_obs_loss_type(config: Any) -> str:
    return str(getattr(config, "obs_loss_type", getattr(config, "loss_type", "masked_mse")))


def _resolve_pde_loss_type(config: Any) -> str:
    return str(getattr(config, "pde_loss_type", getattr(config, "loss_type", "mse")))


def _loss_reduction_metadata(component: str, loss_type: str) -> str:
    if component.startswith("obs") and loss_type == "masked_mse":
        return "masked_mse_over_observed_entries"
    if component == "pde" and loss_type == "mse":
        return "mse_over_residual_field"
    return f"legacy_{loss_type}"


def _pde_params_with_residual_options(pde_params: dict[str, Any] | None, config: Any) -> dict[str, Any]:
    params = dict(pde_params or {})
    option_names = (
        "hermite_collocation_times",
        "hermite_num_collocation",
        "hermite_include_integral_residual",
        "hermite_integral_weight",
        "enforce_boundary_conditions",
        "enforce_initial_conditions",
        "boundary_condition_mode",
        "initial_condition_mode",
        "bc_weight",
        "ic_weight",
        "endpoint_bc_weight",
        "boundary_residual_normalization",
        "allow_unknown_boundary_conditions",
        "legacy_ignore_boundary",
    )
    for name in option_names:
        if hasattr(config, name):
            params[name] = getattr(config, name)
    return params


def _component_norms(components: dict[str, Any] | None) -> dict[str, float]:
    import torch

    norms: dict[str, float] = {}
    for name in ("interior", "boundary", "initial", "endpoint"):
        value = None if components is None else components.get(name)
        if value is None:
            norms[name] = 0.0
        else:
            norms[name] = float((torch.linalg.vector_norm(value) / max(value.numel(), 1)).detach().cpu())
    return norms


def _compose_region_aware_pde_field(residual_output: Any, config: Any, masks: PairMasks) -> tuple[Any, dict[str, Any]]:
    import torch

    pde_field = residual_output.residual
    components = residual_output.components
    if components is None:
        region = residual_region_mask(config.pde_residual_region, masks.coef, masks.sol, tuple(pde_field.shape))
        if region is not None:
            pde_field = pde_field * region
        return pde_field, {
            "component_norms": _component_norms(None),
            "pde_residual_region_applied_to": "total_legacy",
            "pde_residual_region": config.pde_residual_region,
        }

    interior = components.get("interior")
    if interior is None:
        interior = pde_field
    boundary = components.get("boundary")
    initial = components.get("initial")
    endpoint = components.get("endpoint")

    metadata: dict[str, Any] = {
        "pde_residual_region": config.pde_residual_region,
        "boundary_region_masked": False,
        "initial_region_masked": False,
        "endpoint_region_masked": False,
    }
    interior_masked = interior
    if residual_output.metadata.get("resolved_residual_mode") == "near_endpoint_temporal" or residual_output.metadata.get("mode") == "near_endpoint_temporal":
        metadata.update(
            {
                "pde_residual_region_applied_to": "interior_only",
                "pde_residual_region_skipped": True,
                "reason": "near_endpoint_temporal interior is already sparse-temporal masked",
            }
        )
    else:
        region = residual_region_mask(config.pde_residual_region, masks.coef, masks.sol, tuple(interior.shape))
        if region is not None:
            interior_masked = interior * region
        metadata.update(
            {
                "pde_residual_region_applied_to": "interior_only",
                "pde_residual_region_skipped": False,
            }
        )

    field = _append_weighted_components_for_loss(
        interior_masked,
        boundary,
        initial,
        endpoint,
        bc_weight=float(getattr(config, "bc_weight", 1.0)),
        ic_weight=float(getattr(config, "ic_weight", 1.0)),
        endpoint_weight=float(getattr(config, "endpoint_bc_weight", 1.0)),
    )
    metadata["component_norms"] = _component_norms(
        {"interior": interior_masked, "boundary": boundary, "initial": initial, "endpoint": endpoint}
    )
    channels = {
        "interior_channels": int(interior_masked.shape[1]),
        "bc_channels": int(boundary.shape[1]) if boundary is not None else 0,
        "ic_channels": int(initial.shape[1]) if initial is not None else 0,
        "endpoint_channels": int(endpoint.shape[1]) if endpoint is not None else 0,
        "total_channels": int(field.shape[1]),
    }
    metadata["residual_channels"] = channels
    return field, metadata


def _append_weighted_components_for_loss(
    interior: Any,
    boundary: Any | None,
    initial: Any | None,
    endpoint: Any | None,
    *,
    bc_weight: float,
    ic_weight: float,
    endpoint_weight: float,
) -> Any:
    import torch

    parts = [interior]
    if boundary is not None and bc_weight > 0.0:
        parts.append(torch.as_tensor(bc_weight, dtype=interior.dtype, device=interior.device).sqrt() * boundary)
    if initial is not None and ic_weight > 0.0:
        parts.append(torch.as_tensor(ic_weight, dtype=interior.dtype, device=interior.device).sqrt() * initial)
    if endpoint is not None and endpoint_weight > 0.0:
        parts.append(torch.as_tensor(endpoint_weight, dtype=interior.dtype, device=interior.device).sqrt() * endpoint)
    return torch.cat(parts, dim=1)

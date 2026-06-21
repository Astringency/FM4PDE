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

    obs_a_residual = (phys_state.coef * masks.coef) - target_coef
    obs_u_residual = (phys_state.sol * masks.sol) - target_sol
    clean_obs_a_residual = (phys_state.coef * masks.coef) - clean_coef
    clean_obs_u_residual = (phys_state.sol * masks.sol) - clean_sol

    zero = phys_state.coef.sum() * 0.0
    L_obs_a = _reduce_loss(obs_a_residual, config.loss_type) if enabled["obs_a"] else zero
    L_obs_u = _reduce_loss(obs_u_residual, config.loss_type) if enabled["obs_u"] else zero
    clean_L_obs_a = _reduce_loss(clean_obs_a_residual, config.loss_type)
    clean_L_obs_u = _reduce_loss(clean_obs_u_residual, config.loss_type)

    pde_field = None
    status = "disabled"
    pde_meta: dict[str, Any] = {}
    if enabled["pde"]:
        pde_params = _pde_params_with_residual_options(getattr(ground_truth, "pde_params", None), config)
        if getattr(config, "enforce_initial_conditions", True):
            observed_initial = target_coef * masks.coef
            pde_params.setdefault("observed_initial", observed_initial)
            pde_params.setdefault("initial_mask", masks.coef)
            pde_params.setdefault("initial_condition_source", "observation_loss_masked_coef")
        residual = compute_pde_residual(
            config.pde,
            phys_state.coef,
            phys_state.sol,
            pde_params=pde_params,
            k=getattr(config, "k", 1),
            residual_mode=getattr(config, "residual_mode", "auto"),
        )
        pde_field = residual.residual
        status = residual.status
        pde_meta = residual.metadata
        pde_meta["component_norms"] = _component_norms(residual.components)
        region = residual_region_mask(config.pde_residual_region, masks.coef, masks.sol, tuple(pde_field.shape))
        if region is not None:
            pde_field = pde_field * region
        L_pde = _reduce_loss(pde_field, config.loss_type)
    else:
        L_pde = zero

    metadata = {
        "enabled": enabled,
        "pde": pde_meta,
        "pde_params_used": sorted(getattr(ground_truth, "pde_params", {}) or {}),
        "residual_status": status,
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

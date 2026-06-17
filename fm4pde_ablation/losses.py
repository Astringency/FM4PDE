from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fm4pde_ablation.masks import PairMasks, residual_region_mask
from fm4pde_ablation.pde_residuals import compute_pde_residual
from fm4pde_ablation.state import SplitState


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
        residual = compute_pde_residual(config.pde, phys_state.coef, phys_state.sol, k=getattr(config, "k", 1))
        pde_field = residual.residual
        status = residual.status
        pde_meta = residual.metadata
        region = residual_region_mask(config.pde_residual_region, masks.coef, masks.sol, tuple(pde_field.shape))
        if region is not None:
            pde_field = pde_field * region
        L_pde = _reduce_loss(pde_field, config.loss_type)
    else:
        L_pde = zero

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
        metadata={"enabled": enabled, "pde": pde_meta},
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
        return _apply_task_gate({"obs_a": True, "obs_u": True, "pde": False}, task)
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sampling.losses import GuidanceLossOutput, guidance_component_flags


@dataclass
class GuidanceSchedule:
    zeta_obs_a_t: Any
    zeta_obs_u_t: Any
    zeta_pde_t: Any
    bt: Any
    metadata: dict[str, Any]


@dataclass
class GuidanceGradient:
    grad_obs_a: Any
    grad_obs_u: Any
    grad_pde: Any
    grad_total: Any
    grad_norm_obs_a: float
    grad_norm_obs_u: float
    grad_norm_pde: float
    grad_norm_total: float
    clip_scale: float
    component_clip_scales: dict[str, float]


def make_zeta_schedule(config: Any, t: Any, t_next: Any, bt: Any) -> GuidanceSchedule:
    import torch

    flags = guidance_component_flags(config.guidance_components, getattr(config, "task", "both"))
    ones = torch.ones_like(t_next)
    zeta_a = ones * float(config.zeta_obs_a) if flags["obs_a"] else ones * 0.0
    zeta_u = ones * float(config.zeta_obs_u) if flags["obs_u"] else ones * 0.0
    zeta_pde = ones * float(config.zeta_pde) if flags["pde"] else ones * 0.0

    schedule = config.guidance_schedule
    if schedule == "constant":
        factor = ones
    elif schedule == "delta":
        factor = (1.0 - t_next).clamp_min(0.0)
    elif schedule == "bt":
        factor = bt.abs() if hasattr(bt, "abs") else ones * abs(float(bt))
    elif schedule == "cosine":
        progress = t_next.clamp(0.0, 1.0)
        if getattr(config, "cosine_mode", "decay") == "ramp":
            factor = 0.5 - 0.5 * torch.cos(torch.pi * progress)
        else:
            factor = 0.5 + 0.5 * torch.cos(torch.pi * progress)
    elif schedule == "polynomial":
        factor = (1.0 - t_next).clamp_min(0.0) ** float(config.polynomial_power)
    elif schedule == "obs_decay":
        factor = ones
        if float(t_next.detach().cpu()) > float(config.obs_decay_start_ratio):
            factor = factor * float(config.obs_decay)
    else:
        raise ValueError(f"Unknown guidance_schedule={schedule!r}")
    return GuidanceSchedule(
        zeta_obs_a_t=zeta_a * factor,
        zeta_obs_u_t=zeta_u * factor,
        zeta_pde_t=zeta_pde * factor,
        bt=bt,
        metadata={"guidance_schedule": schedule, "factor": _scalar(factor)},
    )


def compute_guidance_gradient(losses: GuidanceLossOutput, x_loss: Any, schedule: GuidanceSchedule, config: Any) -> GuidanceGradient:
    import torch

    zero = torch.zeros_like(x_loss)
    grad_a = _grad_or_zero(losses.L_obs_a, x_loss, zero, retain_graph=True)
    grad_u = _grad_or_zero(losses.L_obs_u, x_loss, zero, retain_graph=True)
    grad_pde = _grad_or_zero(losses.L_pde, x_loss, zero, retain_graph=False)

    grad_a, scale_a = _clip_single(grad_a, config.clip_threshold, config.clip_mode == "per_component_norm")
    grad_u, scale_u = _clip_single(grad_u, config.clip_threshold, config.clip_mode == "per_component_norm")
    grad_pde, scale_pde = _clip_single(grad_pde, config.clip_threshold, config.clip_mode == "per_component_norm")

    total = schedule.zeta_obs_a_t * grad_a + schedule.zeta_obs_u_t * grad_u + schedule.zeta_pde_t * grad_pde
    clip_scale = 1.0
    if config.clip_mode == "global_norm":
        total, clip_scale = _clip_single(total, config.clip_threshold, True)
    elif config.clip_mode == "none":
        clip_scale = 1.0
    elif config.clip_mode != "per_component_norm":
        raise ValueError(f"Unknown clip_mode={config.clip_mode!r}")

    return GuidanceGradient(
        grad_obs_a=grad_a,
        grad_obs_u=grad_u,
        grad_pde=grad_pde,
        grad_total=total,
        grad_norm_obs_a=_norm(grad_a),
        grad_norm_obs_u=_norm(grad_u),
        grad_norm_pde=_norm(grad_pde),
        grad_norm_total=_norm(total),
        clip_scale=clip_scale,
        component_clip_scales={"obs_a": scale_a, "obs_u": scale_u, "pde": scale_pde},
    )


def apply_guidance_update(x_next: Any, gradient: GuidanceGradient, step_output: Any, schedule: GuidanceSchedule, config: Any) -> Any:
    scale = _update_scale(step_output.phase, step_output.t_next, step_output.step_size, schedule.bt, config)
    return x_next - scale * gradient.grad_total


def _update_scale(phase: str, t_next: Any, step_size: Any, bt: Any, config: Any) -> Any:
    if config.guidance_schedule == "bt" or phase == "deterministic":
        return bt * step_size
    return float(config.stochastic_guidance_coeff) * (1.0 - t_next).clamp_min(0.0)


def _grad_or_zero(loss: Any, x: Any, zero: Any, retain_graph: bool) -> Any:
    import torch

    if not getattr(loss, "requires_grad", False):
        return zero
    grad = torch.autograd.grad(loss, x, retain_graph=retain_graph, allow_unused=True)[0]
    return zero if grad is None else grad


def _clip_single(grad: Any, threshold: float, active: bool) -> tuple[Any, float]:
    import torch

    if not active:
        return grad, 1.0
    norm = torch.linalg.vector_norm(grad)
    scale = torch.minimum(torch.ones((), dtype=grad.dtype, device=grad.device), torch.as_tensor(threshold, dtype=grad.dtype, device=grad.device) / (norm + 1e-12))
    return grad * scale, float(scale.detach().cpu())


def _norm(grad: Any) -> float:
    import torch

    return float(torch.linalg.vector_norm(grad).detach().cpu())


def _scalar(x: Any) -> float:
    try:
        return float(x.detach().cpu())
    except Exception:
        return float(x)

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
    metadata: dict[str, Any]


def make_zeta_schedule(config: Any, t: Any, bt: Any, *, step: int) -> GuidanceSchedule:
    """Fixed component weights, with PDE guidance enabled at ceil(ratio*N)."""
    import math
    import torch

    flags = guidance_component_flags(config.guidance_components, getattr(config, "task", "both"))
    ones = torch.ones_like(t)
    start_step = math.ceil(float(config.pde_guidance_start_ratio) * config.num_steps)
    pde_factor = float(step >= start_step)
    return GuidanceSchedule(
        zeta_obs_a_t=ones * float(config.zeta_obs_a) * flags["obs_a"],
        zeta_obs_u_t=ones * float(config.zeta_obs_u) * flags["obs_u"],
        zeta_pde_t=ones * float(config.zeta_pde) * flags["pde"] * pde_factor,
        bt=bt,
        metadata={"pde_guidance_factor": pde_factor, "pde_guidance_start_step": start_step},
    )


def compute_guidance_gradient(
    losses: GuidanceLossOutput,
    grad_target: Any,
    schedule: GuidanceSchedule,
    config: Any,
) -> GuidanceGradient:
    import torch

    zero = torch.zeros_like(grad_target)
    enabled = losses.metadata.get("enabled")
    if not isinstance(enabled, dict):
        enabled = guidance_component_flags(config.guidance_components, getattr(config, "task", "both"))
        if float(getattr(config, "zeta_pde", 1.0)) == 0.0:
            enabled["pde"] = False
    # Losses are reported as the mean of per-sample losses. Differentiate
    # their sum so each sample receives the same gradient whether sampled
    # alone or as part of a larger batch.
    batch_scale = (
        int(grad_target.shape[0])
        if losses.metadata.get("loss_batch_reduction") == "mean_of_per_sample"
        else 1
    )
    guidance_L_obs_a = losses.guidance_L_obs_a if losses.guidance_L_obs_a is not None else losses.L_obs_a
    guidance_L_obs_u = losses.guidance_L_obs_u if losses.guidance_L_obs_u is not None else losses.L_obs_u
    guidance_L_pde = losses.guidance_L_pde if losses.guidance_L_pde is not None else losses.L_pde
    active_obs_a = bool(enabled["obs_a"] and _schedule_component_active(schedule.zeta_obs_a_t))
    active_obs_u = bool(enabled["obs_u"] and _schedule_component_active(schedule.zeta_obs_u_t))
    active_pde = bool(enabled["pde"] and _schedule_component_active(schedule.zeta_pde_t))
    grad_a = _grad_or_zero(
        guidance_L_obs_a,
        grad_target,
        zero,
        retain_graph=bool(active_obs_u or active_pde),
        enabled=active_obs_a,
        component="obs_a",
        batch_scale=batch_scale,
    )
    grad_u = _grad_or_zero(
        guidance_L_obs_u,
        grad_target,
        zero,
        retain_graph=active_pde,
        enabled=active_obs_u,
        component="obs_u",
        batch_scale=batch_scale,
    )
    grad_pde = _grad_or_zero(
        guidance_L_pde,
        grad_target,
        zero,
        retain_graph=False,
        enabled=active_pde,
        component="pde",
        batch_scale=batch_scale,
    )

    total = schedule.zeta_obs_a_t * grad_a + schedule.zeta_obs_u_t * grad_u + schedule.zeta_pde_t * grad_pde
    # Sum weighted gradients first, then clip each independent sample once.
    if not bool(torch.isfinite(total).all().detach().cpu()):
        raise FloatingPointError("Guidance gradient contains NaN or Inf")
    total, clip_scale = _clip_per_sample(total, config.clip_threshold)

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
        metadata={
            "grad_target_shape": list(grad_target.shape),
            "loss_gradient_batch_reduction": "sum_of_per_sample",
            "clip_scope": "per_sample",
            "scheduled_components_active": {
                "obs_a": active_obs_a,
                "obs_u": active_obs_u,
                "pde": active_pde,
            },
        },
    )


def _schedule_component_active(weight: Any) -> bool:
    import torch

    return bool(torch.any(torch.as_tensor(weight).detach() != 0).cpu())


def apply_guidance_update(
    x_next: Any,
    gradient: GuidanceGradient,
    step_output: Any,
    schedule: GuidanceSchedule,
    config: Any,
) -> Any:
    """Apply exactly the manuscript multiplier after global gradient clipping."""
    import torch

    if step_output.phase == "deterministic":
        # Preserve the original arithmetic wherever its intermediate is finite.
        # This is a numerical evaluation choice, not a time floor or scale cap.
        t = step_output.t
        denominator = torch.where(t == 0, torch.ones_like(t), t)
        raw_scale = step_output.step_size * ((1.0 - t) / denominator)
        # At subnormal t, 1/t can overflow even though dt/t is moderate.
        t64 = t.to(torch.float64)
        stable_scale = (
            step_output.step_size.to(torch.float64) / denominator.to(torch.float64)
            * (1.0 - t64)
        ).to(t.dtype)
        scale = torch.where(torch.isfinite(raw_scale), raw_scale, stable_scale)
        scale = torch.where(t == 0, torch.zeros_like(scale), scale)
    elif step_output.phase == "stochastic":
        scale = float(config.stochastic_guidance_coeff) * (1.0 - step_output.t)
    else:
        raise ValueError(f"Unknown phase={step_output.phase!r}")
    correction = scale * gradient.grad_total
    updated = x_next - correction
    if not bool(torch.isfinite(updated).all().detach().cpu()):
        raise FloatingPointError("Guided sampling update contains NaN or Inf")
    gradient.metadata.update({
        "guidance_update_scale": _scalar(scale),
        "guidance_correction_norm": _norm(correction),
        "guidance_correction_rms": _rms(correction),
    })
    return updated


def _grad_or_zero(
    loss: Any,
    x: Any,
    zero: Any,
    retain_graph: bool,
    *,
    enabled: bool,
    component: str,
    batch_scale: int,
) -> Any:
    import torch

    if not enabled:
        return zero
    if loss is None:
        raise RuntimeError(f"Enabled guidance loss {component!r} is unavailable")
    scaled_loss = loss * batch_scale
    if not getattr(scaled_loss, "requires_grad", False):
        raise RuntimeError(f"Enabled guidance loss {component!r} is detached from the autograd graph")
    grad = torch.autograd.grad(scaled_loss, x, retain_graph=retain_graph, allow_unused=True)[0]
    if grad is None:
        raise RuntimeError(
            f"Enabled guidance loss {component!r} is not connected to the current sampling state; "
            "check the loss-state computation graph"
        )
    return grad


def _clip_per_sample(grad: Any, threshold: float) -> tuple[Any, float]:
    """Global norm clip per sample: rho=Gc/max(Gc,||g||_2)."""
    import torch

    norms = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), dim=1)
    if grad.dtype != torch.float64 and not bool(torch.isfinite(norms).all().detach().cpu()):
        # Preserve the same norm formula when finite float32 entries overflow
        # its sum of squares. This does not change or discard the gradient.
        norms = torch.linalg.vector_norm(grad.double().reshape(grad.shape[0], -1), dim=1)
    bound = torch.as_tensor(threshold, dtype=grad.dtype, device=grad.device)
    scales = (bound / torch.maximum(bound, norms)).to(grad.dtype)
    clipped = grad * scales.reshape(grad.shape[0], *([1] * (grad.ndim - 1)))
    return clipped, float(scales.mean().detach().cpu())


def _norm(grad: Any) -> float:
    import torch

    return float(torch.linalg.vector_norm(grad).detach().cpu())


def _rms(value: Any) -> float:
    import torch

    return float(value.square().mean().sqrt().detach().cpu())


def _scalar(x: Any) -> float:
    try:
        return float(x.detach().cpu())
    except Exception:
        return float(x)

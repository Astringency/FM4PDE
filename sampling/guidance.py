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
    metadata: dict[str, Any]


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
    pde_factor = _pde_guidance_factor(config, t)
    return GuidanceSchedule(
        zeta_obs_a_t=zeta_a * factor,
        zeta_obs_u_t=zeta_u * factor,
        zeta_pde_t=zeta_pde * factor * pde_factor,
        bt=bt,
        metadata={
            "guidance_schedule": schedule,
            "factor": _scalar(factor),
            "pde_guidance_factor": _scalar(pde_factor),
            "pde_guidance_start_ratio": float(config.pde_guidance_start_ratio),
            "pde_guidance_ramp_ratio": float(config.pde_guidance_ramp_ratio),
        },
    )


def _pde_guidance_factor(config: Any, t: Any) -> Any:
    """Gate PDE guidance by normalized flow time without changing observation guidance."""
    import torch

    progress = t.clamp(0.0, 1.0)
    start = float(config.pde_guidance_start_ratio)
    ramp = float(config.pde_guidance_ramp_ratio)
    if ramp == 0.0:
        return (progress >= start).to(dtype=progress.dtype)
    return ((progress - start) / ramp).clamp(0.0, 1.0)


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

    grad_a, scale_a = _clip_per_sample(grad_a, config.clip_threshold, config.clip_mode == "per_component_norm")
    grad_u, scale_u = _clip_per_sample(grad_u, config.clip_threshold, config.clip_mode == "per_component_norm")
    grad_pde, scale_pde = _clip_per_sample(grad_pde, config.clip_threshold, config.clip_mode == "per_component_norm")

    total = schedule.zeta_obs_a_t * grad_a + schedule.zeta_obs_u_t * grad_u + schedule.zeta_pde_t * grad_pde
    clip_scale = 1.0
    if config.clip_mode == "global_norm":
        # A batch is a collection of independent inverse problems. Clipping
        # over the whole BCHW tensor would couple their sampling trajectories.
        total, clip_scale = _clip_per_sample(total, config.clip_threshold, True)
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
        metadata={
            "gradient_target": getattr(config, "gradient_target", "loss_state_direct"),
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
    scale = _update_scale(
        step_output.phase,
        step_output.t,
        step_output.t_next,
        step_output.step_size,
        schedule.bt,
        config,
    )
    deterministic_factor = 1.0
    if step_output.phase == "deterministic":
        deterministic_factor = _deterministic_guidance_factor(config, step_output.t)
        scale = scale * deterministic_factor
    correction = scale * gradient.grad_total
    nonfinite_samples = 0
    if step_output.phase == "deterministic" and bool(
        getattr(config, "deterministic_numerical_guard", True)
    ):
        correction, nonfinite_samples = _zero_nonfinite_samples(correction)
    raw_correction_norm = _norm(correction)
    correction_clip_scale = 1.0
    maximum_rms = float(getattr(config, "deterministic_correction_max_rms", 0.0))
    if step_output.phase == "deterministic" and maximum_rms > 0.0:
        correction, correction_clip_scale = _clip_correction_rms_per_sample(
            correction, maximum_rms
        )
    gradient.metadata["guidance_update_scale"] = _scalar(scale)
    gradient.metadata["stochastic_guidance_time"] = getattr(config, "stochastic_guidance_time", "t")
    gradient.metadata["deterministic_bt_mode"] = getattr(config, "deterministic_bt_mode", "legacy")
    gradient.metadata["deterministic_guidance_factor"] = _scalar(deterministic_factor)
    gradient.metadata["guidance_correction_raw_norm"] = raw_correction_norm
    gradient.metadata["guidance_correction_norm"] = _norm(correction)
    gradient.metadata["guidance_correction_rms"] = _rms(correction)
    gradient.metadata["correction_clip_scale"] = correction_clip_scale
    gradient.metadata["nonfinite_correction_samples"] = nonfinite_samples
    return x_next - correction


def _update_scale(phase: str, t: Any, t_next: Any, step_size: Any, bt: Any, config: Any) -> Any:
    if phase == "deterministic":
        return _deterministic_update_scale(t, t_next, step_size, bt, config)
    if config.guidance_schedule == "bt":
        return bt * step_size
    time_name = getattr(config, "stochastic_guidance_time", "t")
    if time_name == "t":
        time_value = t
    elif time_name == "t_next":
        time_value = t_next
    else:
        raise ValueError("stochastic_guidance_time must be 't' or 't_next'")
    return float(config.stochastic_guidance_coeff) * (1.0 - time_value).clamp_min(0.0)


def _deterministic_update_scale(t: Any, t_next: Any, step_size: Any, bt: Any, config: Any) -> Any:
    """Return a finite, explicitly configured deterministic guidance scale.

    ``legacy`` preserves historical behavior.  The other modes are ablation
    choices for the CondOT ``b_t=(1-t)/t`` boundary singularity.
    """
    import torch

    mode = getattr(config, "deterministic_bt_mode", "legacy")
    coeff = float(getattr(config, "deterministic_guidance_coeff", 1.0))
    raw = bt * step_size
    if mode == "legacy":
        scale = raw
    elif mode == "zero_at_t0":
        scale = torch.where(t <= 0.0, torch.zeros_like(raw), raw)
    elif mode == "t_next":
        safe_t_next = t_next.clamp_min(1e-6)
        scale = ((1.0 - safe_t_next).clamp_min(0.0) / safe_t_next) * step_size
    elif mode in {"clipped", "clipped_zero_at_t0"}:
        maximum = torch.as_tensor(
            float(getattr(config, "deterministic_bt_max_scale", 0.1)),
            dtype=raw.dtype,
            device=raw.device,
        )
        scale = torch.minimum(raw, maximum)
        if mode == "clipped_zero_at_t0":
            scale = torch.where(t <= 0.0, torch.zeros_like(scale), scale)
    elif mode == "stochastic_like":
        scale = (1.0 - t).clamp_min(0.0)
    elif mode == "capped_stochastic_like":
        maximum = torch.as_tensor(
            float(getattr(config, "deterministic_bt_max_scale", 0.1)),
            dtype=raw.dtype,
            device=raw.device,
        )
        # Here max_scale is the final trust-region cap, while coeff controls
        # the stochastic-shaped tail. Return directly to avoid multiplying
        # coeff twice below.
        return torch.minimum(coeff * (1.0 - t).clamp_min(0.0), maximum)
    else:
        raise ValueError(f"Unknown deterministic_bt_mode={mode!r}")
    return coeff * scale


def _deterministic_guidance_factor(config: Any, t: Any) -> Any:
    """Gate all deterministic guidance near the singular CondOT boundary."""
    import torch

    progress = t.clamp(0.0, 1.0)
    start = float(getattr(config, "deterministic_guidance_start_ratio", 0.0))
    ramp = float(getattr(config, "deterministic_guidance_ramp_ratio", 0.0))
    if ramp == 0.0:
        return (progress >= start).to(dtype=progress.dtype)
    return ((progress - start) / ramp).clamp(0.0, 1.0)


def _zero_nonfinite_samples(value: Any) -> tuple[Any, int]:
    """Reject an entire sample update when any element is NaN or infinite."""
    import torch

    batch_size = int(value.shape[0])
    finite = torch.isfinite(value.reshape(batch_size, -1)).all(dim=1)
    mask = finite.reshape(batch_size, *([1] * (value.ndim - 1)))
    guarded = torch.where(mask, value, torch.zeros_like(value))
    return guarded, int((~finite).sum().detach().cpu())


def _clip_correction_rms_per_sample(value: Any, maximum_rms: float) -> tuple[Any, float]:
    """Clip the applied state correction by per-sample RMS, independent of grid size."""
    import math

    elements_per_sample = int(value[0].numel())
    return _clip_per_sample(value, maximum_rms * math.sqrt(elements_per_sample), True)


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
            f"Enabled guidance loss {component!r} is not connected to gradient_target; "
            "check loss_state and gradient_target"
        )
    return grad


def _clip_single(grad: Any, threshold: float) -> tuple[Any, float]:
    import torch

    norm = torch.linalg.vector_norm(grad)
    scale = torch.minimum(torch.ones((), dtype=grad.dtype, device=grad.device), torch.as_tensor(threshold, dtype=grad.dtype, device=grad.device) / (norm + 1e-12))
    return grad * scale, float(scale.detach().cpu())


def _clip_per_sample(grad: Any, threshold: float, active: bool = True) -> tuple[Any, float]:
    """Clip each sample in the batch independently.  grad shape: [B, C, H, W]."""
    import torch

    if not active:
        return grad, 1.0
    B = int(grad.shape[0])
    if B <= 1:
        return _clip_single(grad, threshold)
    flat = grad.reshape(B, -1)
    norms = torch.linalg.vector_norm(flat, dim=1)  # [B]
    scales = torch.clamp(threshold / (norms + 1e-12), max=1.0)  # [B]
    clipped = grad * scales.reshape(B, *([1] * (grad.ndim - 1)))
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

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sampling.time_grid import endpoint_from_velocity


@dataclass
class SamplerStepOutput:
    x_raw_current: Any
    x_raw_next: Any
    x_endpoint: Any
    x_loss_state: Any
    t: Any
    t_next: Any
    step_size: Any
    phase: str
    loss_state: str
    wall_time: float


def phase_for_step(sampler_phase: str, switch_ratio: float, step_index: int, num_steps: int) -> str:
    if sampler_phase in {"deterministic", "stochastic"}:
        return sampler_phase
    switch_step = int(round(float(switch_ratio) * num_steps))
    if sampler_phase == "hybrid_d2s":
        return "deterministic" if step_index < switch_step else "stochastic"
    if sampler_phase == "hybrid_s2d":
        return "stochastic" if step_index < switch_step else "deterministic"
    raise ValueError(f"Unknown sampler_phase={sampler_phase!r}")


def sampler_step(
    net: Any,
    x_cur: Any,
    t: Any,
    t_next: Any,
    phase: str,
    step_method: str,
    loss_state: str,
    device: str | Any = "cpu",
    model_extra: dict[str, Any] | None = None,
) -> SamplerStepOutput:
    import torch

    start = time.time()
    step_size = t_next - t
    if phase == "deterministic":
        x_endpoint, x_next = _deterministic_step(net, x_cur, t, step_size, step_method, model_extra)
    elif phase == "stochastic":
        x_endpoint, x_next = _stochastic_step(net, x_cur, t, t_next, step_size, step_method, device, model_extra)
    else:
        raise ValueError(f"Unknown phase={phase!r}")

    normalized_loss_state = _normalize_loss_state(loss_state)
    selected = choose_loss_state(normalized_loss_state, x_cur, x_next, x_endpoint)
    return SamplerStepOutput(
        x_raw_current=x_cur,
        x_raw_next=x_next,
        x_endpoint=x_endpoint,
        x_loss_state=selected,
        t=t,
        t_next=t_next,
        step_size=step_size,
        phase=phase,
        loss_state=normalized_loss_state,
        wall_time=time.time() - start,
    )


def choose_loss_state(loss_state: str, x_cur: Any, x_next: Any, x_endpoint: Any) -> Any:
    if loss_state == "xt":
        return x_cur
    if loss_state == "x_next":
        return x_next
    if loss_state == "endpoint":
        return x_endpoint
    raise ValueError(f"Unknown loss_state={loss_state!r}")


def _normalize_loss_state(loss_state: str) -> str:
    return loss_state


def _deterministic_step(
    net: Any,
    x_cur: Any,
    t: Any,
    step_size: Any,
    method: str,
    model_extra: dict[str, Any] | None,
) -> tuple[Any, Any]:
    if method == "euler":
        v = _call_velocity_model(net, x_cur, t, model_extra)
        x_endpoint = endpoint_from_velocity(x_cur, v, t)
        return x_endpoint, x_cur + v * step_size
    if method == "midpoint":
        v = _call_velocity_model(net, x_cur, t, model_extra)
        t_mid = t + 0.5 * step_size
        x_mid = x_cur + 0.5 * step_size * v
        v_mid = _call_velocity_model(net, x_mid, t_mid, model_extra)
        x_endpoint = endpoint_from_velocity(x_mid, v_mid, t_mid)
        return x_endpoint, x_cur + step_size * v_mid
    raise ValueError(f"Unsupported step_method={method!r}")


def _stochastic_step(
    net: Any,
    x_cur: Any,
    t: Any,
    t_next: Any,
    step_size: Any,
    method: str,
    device: str | Any,
    model_extra: dict[str, Any] | None,
) -> tuple[Any, Any]:
    import torch

    if method == "euler":
        v = _call_velocity_model(net, x_cur, t, model_extra)
        x_endpoint = endpoint_from_velocity(x_cur, v, t)
    elif method == "midpoint":
        v = _call_velocity_model(net, x_cur, t, model_extra)
        # Midpoint is local to the current integration interval, just as in
        # the deterministic solver. The stochasticity belongs to the bridge
        # resampling below, not to a t-to-1 midpoint extrapolation.
        t_mid = t + 0.5 * step_size
        x_mid = x_cur + 0.5 * step_size * v
        x_endpoint = endpoint_from_velocity(
            x_mid,
            _call_velocity_model(net, x_mid, t_mid, model_extra),
            t_mid,
        )
    else:
        raise ValueError(f"Unsupported step_method={method!r}")
    target = torch.device(device if isinstance(device, str) else device)
    if target.type == "cuda" and not torch.cuda.is_available():
        target = torch.device("cpu")
    x0 = torch.randn_like(x_cur, device=target)
    x_next = (1.0 - t_next) * x0 + t_next * x_endpoint
    return x_endpoint, x_next


def _call_velocity_model(net: Any, x: Any, t: Any, model_extra: dict[str, Any] | None) -> Any:
    if model_extra is None:
        return net(x, t)
    if "_cfg_scale" not in model_extra:
        return net(x, t, extra=model_extra)
    extra = dict(model_extra)
    cfg_scale = extra.pop("_cfg_scale", None)
    null_label = extra.pop("_cfg_null_label", None)
    if cfg_scale is None:
        return net(x, t, extra=extra)
    if "label" not in extra or null_label is None:
        raise ValueError("CFG requires a conditional label and explicit null label")
    scale = float(cfg_scale)
    conditional = net(x, t, extra=extra) if scale != 0.0 else None
    if scale == 1.0:
        return conditional
    unconditional_extra = dict(extra)
    unconditional_extra["label"] = extra["label"].new_full(
        extra["label"].shape, int(null_label)
    )
    # Only the PDE category is dropped. Scalar conditioning remains present in
    # unconditional_extra by construction.
    unconditional = net(x, t, extra=unconditional_extra)
    if scale == 0.0:
        return unconditional
    return unconditional + scale * (conditional - unconditional)

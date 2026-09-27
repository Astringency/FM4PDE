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
    endpoint_prediction_mode: str
    endpoint_model_evaluations: int
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
    loss_state: str,
    device: str | Any = "cpu",
    model_extra: dict[str, Any] | None = None,
    stochastic_noise_source_batch_size: int | None = None,
    stochastic_noise_source_indices: list[int] | None = None,
) -> SamplerStepOutput:
    """One manuscript Euler or stochastic bridge proposal, with one velocity call."""
    import torch

    start = time.time()
    step_size = t_next - t
    velocity = _call_velocity_model(net, x_cur, t, model_extra)
    x_endpoint = endpoint_from_velocity(x_cur, velocity, t)
    if phase == "deterministic":
        x_next = x_cur + step_size * velocity
    elif phase == "stochastic":
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            target = torch.device("cpu")
        noise = _stochastic_bridge_noise_like(
            x_cur, device=target,
            source_batch_size=stochastic_noise_source_batch_size,
            source_indices=stochastic_noise_source_indices,
        )
        x_next = (1.0 - t_next) * noise + t_next * x_endpoint
    else:
        raise ValueError(f"Unknown phase={phase!r}")
    return SamplerStepOutput(
        x_raw_current=x_cur, x_raw_next=x_next, x_endpoint=x_endpoint,
        x_loss_state=choose_loss_state(loss_state, x_cur, x_next, x_endpoint),
        t=t, t_next=t_next, step_size=step_size, phase=phase,
        loss_state=loss_state, endpoint_prediction_mode="single_step",
        endpoint_model_evaluations=1, wall_time=time.time() - start,
    )


def choose_loss_state(loss_state: str, x_cur: Any, x_next: Any, x_endpoint: Any) -> Any:
    if loss_state == "xt":
        return x_cur
    if loss_state == "x_next":
        return x_next
    if loss_state == "endpoint":
        return x_endpoint
    raise ValueError(f"Unknown loss_state={loss_state!r}")


def _stochastic_bridge_noise_like(
    x_cur: Any,
    *,
    device: Any,
    source_batch_size: int | None = None,
    source_indices: list[int] | None = None,
) -> Any:
    """Draw bridge noise, optionally replaying selected rows from a larger batch draw."""
    import torch

    if source_batch_size is None and source_indices is None:
        return torch.randn_like(x_cur, device=device)
    source_batch_size = int(source_batch_size if source_batch_size is not None else x_cur.shape[0])
    source_indices = (
        list(range(int(x_cur.shape[0])))
        if source_indices is None
        else [int(value) for value in source_indices]
    )
    if source_batch_size < int(x_cur.shape[0]):
        raise ValueError("stochastic noise source_batch_size must be at least the current batch size")
    if len(source_indices) != int(x_cur.shape[0]):
        raise ValueError("stochastic noise source_indices must contain exactly one entry per current sample")
    if any(index < 0 or index >= source_batch_size for index in source_indices):
        raise ValueError("stochastic noise source_indices values must lie inside the source batch")
    template = x_cur.new_empty((source_batch_size, *x_cur.shape[1:]), device=device)
    source = torch.randn_like(template, device=device)
    index = torch.as_tensor(source_indices, dtype=torch.long, device=device)
    return source.index_select(0, index)


def _call_velocity_model(net: Any, x: Any, t: Any, model_extra: dict[str, Any] | None) -> Any:
    return net(x, t) if model_extra is None else net(x, t, extra=model_extra)

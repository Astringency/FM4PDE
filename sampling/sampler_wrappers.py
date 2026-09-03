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
    step_method: str,
    loss_state: str,
    device: str | Any = "cpu",
    model_extra: dict[str, Any] | None = None,
    stochastic_noise_source_batch_size: int | None = None,
    stochastic_noise_source_indices: list[int] | None = None,
    deterministic_endpoint_mode: str = "single_step",
    deterministic_endpoint_time_grid: Any | None = None,
    deterministic_rollout_checkpoint: bool = False,
) -> SamplerStepOutput:
    import torch

    start = time.time()
    step_size = t_next - t
    if phase == "deterministic":
        x_endpoint, x_next, endpoint_model_evaluations = _deterministic_step(
            net,
            x_cur,
            t,
            step_size,
            step_method,
            model_extra,
            endpoint_mode=deterministic_endpoint_mode,
            endpoint_time_grid=deterministic_endpoint_time_grid,
            rollout_checkpoint=deterministic_rollout_checkpoint,
        )
        endpoint_prediction_mode = deterministic_endpoint_mode
    elif phase == "stochastic":
        x_endpoint, x_next = _stochastic_step(
            net,
            x_cur,
            t,
            t_next,
            step_size,
            step_method,
            device,
            model_extra,
            stochastic_noise_source_batch_size=stochastic_noise_source_batch_size,
            stochastic_noise_source_indices=stochastic_noise_source_indices,
        )
        endpoint_prediction_mode = "single_step"
        endpoint_model_evaluations = 1 if step_method == "euler" else 2
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
        endpoint_prediction_mode=endpoint_prediction_mode,
        endpoint_model_evaluations=endpoint_model_evaluations,
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
    *,
    endpoint_mode: str,
    endpoint_time_grid: Any | None,
    rollout_checkpoint: bool,
) -> tuple[Any, Any, int]:
    if endpoint_mode == "rollout":
        return _deterministic_rollout_endpoint(
            net,
            x_cur,
            endpoint_time_grid,
            method,
            model_extra,
            use_checkpoint=rollout_checkpoint,
        )
    if endpoint_mode != "single_step":
        raise ValueError(f"Unsupported deterministic_endpoint_mode={endpoint_mode!r}")
    if method == "euler":
        v = _call_velocity_model(net, x_cur, t, model_extra)
        x_endpoint = endpoint_from_velocity(x_cur, v, t)
        return x_endpoint, x_cur + v * step_size, 1
    if method == "midpoint":
        v = _call_velocity_model(net, x_cur, t, model_extra)
        t_mid = t + 0.5 * step_size
        x_mid = x_cur + 0.5 * step_size * v
        v_mid = _call_velocity_model(net, x_mid, t_mid, model_extra)
        x_endpoint = endpoint_from_velocity(x_mid, v_mid, t_mid)
        return x_endpoint, x_cur + step_size * v_mid, 2
    raise ValueError(f"Unsupported step_method={method!r}")


def _deterministic_rollout_endpoint(
    net: Any,
    x_cur: Any,
    time_grid: Any | None,
    method: str,
    model_extra: dict[str, Any] | None,
    *,
    use_checkpoint: bool,
) -> tuple[Any, Any, int]:
    """Integrate the unguided ODE from the current grid point to ``t=1``.

    The first interval produces the raw sampler update.  Continuing through
    the remaining intervals produces the endpoint used by observation and PDE
    losses.  The entire rollout remains connected to ``x_cur`` so endpoint
    guidance can use the current-state chain rule.
    """
    if time_grid is None or int(time_grid.numel()) < 2:
        raise ValueError(
            "deterministic_endpoint_mode='rollout' requires the remaining sampling time grid"
        )
    state = x_cur
    x_next = None
    evaluations = 0
    for index in range(int(time_grid.numel()) - 1):
        t_cur = time_grid[index]
        t_next = time_grid[index + 1]
        state, count = _unguided_deterministic_interval(
            net,
            state,
            t_cur,
            t_next - t_cur,
            method,
            model_extra,
            use_checkpoint=use_checkpoint,
        )
        evaluations += count
        if x_next is None:
            x_next = state
    if x_next is None:
        raise RuntimeError("deterministic endpoint rollout did not advance any interval")
    return state, x_next, evaluations


def _unguided_deterministic_interval(
    net: Any,
    x_cur: Any,
    t: Any,
    step_size: Any,
    method: str,
    model_extra: dict[str, Any] | None,
    *,
    use_checkpoint: bool,
) -> tuple[Any, int]:
    if method == "euler":
        v = _rollout_velocity(net, x_cur, t, model_extra, use_checkpoint)
        return x_cur + step_size * v, 1
    if method == "midpoint":
        v = _rollout_velocity(net, x_cur, t, model_extra, use_checkpoint)
        t_mid = t + 0.5 * step_size
        x_mid = x_cur + 0.5 * step_size * v
        v_mid = _rollout_velocity(net, x_mid, t_mid, model_extra, use_checkpoint)
        return x_cur + step_size * v_mid, 2
    raise ValueError(f"Unsupported step_method={method!r}")


def _rollout_velocity(
    net: Any,
    x: Any,
    t: Any,
    model_extra: dict[str, Any] | None,
    use_checkpoint: bool,
) -> Any:
    if not use_checkpoint or not getattr(x, "requires_grad", False):
        return _call_velocity_model(net, x, t, model_extra)
    from torch.utils.checkpoint import checkpoint

    return checkpoint(
        lambda state: _call_velocity_model(net, state, t, model_extra),
        x,
        use_reentrant=False,
    )


def _stochastic_step(
    net: Any,
    x_cur: Any,
    t: Any,
    t_next: Any,
    step_size: Any,
    method: str,
    device: str | Any,
    model_extra: dict[str, Any] | None,
    *,
    stochastic_noise_source_batch_size: int | None = None,
    stochastic_noise_source_indices: list[int] | None = None,
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
    x0 = _stochastic_bridge_noise_like(
        x_cur,
        device=target,
        source_batch_size=stochastic_noise_source_batch_size,
        source_indices=stochastic_noise_source_indices,
    )
    x_next = (1.0 - t_next) * x0 + t_next * x_endpoint
    return x_endpoint, x_next


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

from __future__ import annotations

from typing import Any


def make_time_grid(kind: str, num_steps: int, device: str | Any = "cpu", eta: float = 0.4,
                   *, sampler_phase: str = "deterministic", switch_ratio: float = 0.5) -> Any:
    """Uniform grid, or the manuscript geometric grid during deterministic steps.

    Hybrids retain the full-budget geometric nodes strictly inside their D
    interval and insert the exact switching time. The remaining steps divide
    the S interval uniformly. Thus the switching time is a time, not a step
    fraction; no separate stage budget or growth parameter is introduced.
    """
    import torch

    if kind == "geometric" and eta <= 0:
        raise ValueError("geometric growth eta must be positive")
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    target = torch.device(device if isinstance(device, str) else device)
    if target.type == "cuda" and not torch.cuda.is_available():
        target = torch.device("cpu")
    if kind == "uniform":
        grid = torch.linspace(0.0, 1.0, num_steps + 1, device=target)
    elif kind == "geometric":
        if num_steps == 1:
            grid = torch.tensor([0.0, 1.0], device=target)
        else:
            t0 = 1.0 / ((1.0 + eta) ** (num_steps - 1))
            values = [0.0, t0]
            for _ in range(num_steps - 1):
                values.append(values[-1] * (1.0 + eta))
            values[-1] = 1.0
            grid = torch.tensor(values, device=target, dtype=torch.float32)
        if sampler_phase in {"hybrid_d2s", "hybrid_s2d"}:
            if not 0.0 <= switch_ratio <= 1.0:
                raise ValueError("switch_ratio must be in [0, 1]")
            switch = grid.new_tensor(switch_ratio)
            if switch_ratio in {0.0, 1.0}:
                pure_d = ((sampler_phase == "hybrid_d2s" and switch_ratio == 1.0)
                          or (sampler_phase == "hybrid_s2d" and switch_ratio == 0.0))
                if not pure_d:
                    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=target)
            elif sampler_phase == "hybrid_d2s":
                prefix = grid[grid < switch]
                remaining = num_steps - len(prefix)
                if remaining < 1:
                    raise ValueError("The geometric D prefix leaves no step for S; increase num_steps or change the switch")
                tail = torch.linspace(switch_ratio, 1.0, remaining + 1, device=target)
                grid = torch.cat([prefix, tail])
            else:
                tail = grid[grid > switch]
                remaining = num_steps - len(tail)
                if remaining < 1:
                    raise ValueError("The geometric D suffix leaves no step for S; increase num_steps or change the switch")
                prefix = torch.linspace(0.0, switch_ratio, remaining + 1, device=target)
                grid = torch.cat([prefix, tail])
        elif sampler_phase != "deterministic":
            raise ValueError("A geometric grid requires a deterministic phase")
    else:
        raise ValueError(f"Unknown time_grid={kind!r}")
    if not bool(torch.all(grid[1:] >= grid[:-1])):
        raise ValueError(f"{kind} time grid is not monotone")
    return grid.clamp(0.0, 1.0)


def condot_guidance_coefficient(t: Any) -> Any:
    """Unfloored (1-t)/t for t>0; zero denotes the skipped t=0 correction."""
    import torch

    # Diagnostics need float64 when positive geometric nodes are subnormal.
    t = torch.as_tensor(t).to(torch.float64)
    safe_t = torch.where(t == 0, torch.ones_like(t), t)
    return (1.0 - safe_t) / safe_t


def endpoint_from_velocity(x_t: Any, velocity: Any, t: Any) -> Any:
    """CondOT endpoint, including the exact identity at t=1."""
    return x_t + (1.0 - t) * velocity

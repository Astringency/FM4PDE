from __future__ import annotations

from typing import Any


def make_time_grid(kind: str, num_steps: int, device: str | Any = "cpu", eta: float = 0.4) -> Any:
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
    else:
        raise ValueError(f"Unknown time_grid={kind!r}")
    if not bool(torch.all(grid[1:] >= grid[:-1])):
        raise ValueError(f"{kind} time grid is not monotone")
    return grid.clamp(0.0, 1.0)


def condot_guidance_coefficient(t: Any) -> Any:
    """Manuscript b_t=(1-tilde_t)/tilde_t, with tilde_t=max(t,1e-6)."""
    import torch

    t = torch.as_tensor(t)
    safe_t = t.clamp_min(1e-6)
    return (1.0 - safe_t) / safe_t


def endpoint_from_velocity(x_t: Any, velocity: Any, t: Any) -> Any:
    """CondOT endpoint, including the exact identity at t=1."""
    return x_t + (1.0 - t) * velocity

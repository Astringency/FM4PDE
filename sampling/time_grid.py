from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SchedulerCoefficients:
    alpha_t: Any
    sigma_t: Any
    d_alpha_t: Any
    d_sigma_t: Any


@dataclass
class AffineCoefficients:
    a_t: Any
    b_t: Any


def make_time_grid(kind: str, num_steps: int, device: str | Any = "cpu", eta: float = 0.4) -> Any:
    import torch

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
    elif kind == "cosine":
        s = torch.linspace(0.0, 1.0, num_steps + 1, device=target)
        grid = 1.0 - torch.cos(s * torch.pi / 2.0)
        grid = grid / grid[-1].clamp_min(1e-12)
    else:
        raise ValueError(f"Unknown time_grid={kind!r}")
    if not bool(torch.all(grid[1:] >= grid[:-1])):
        raise ValueError(f"{kind} time grid is not monotone")
    return grid.clamp(0.0, 1.0)


def scheduler_coefficients(
    t: Any,
    scheduler: str = "CondOT",
    n: float = 2.0,
    beta_min: float = 1.0,
    beta_max: float = 2.0,
    eps: float = 1e-6,
) -> SchedulerCoefficients:
    import torch

    if not hasattr(t, "shape"):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = t.clamp(eps, 1.0 - eps)
    if scheduler == "CondOT":
        return SchedulerCoefficients(t, 1.0 - t, torch.ones_like(t), -torch.ones_like(t))
    if scheduler == "PolynomialConvex":
        alpha = t**n
        sigma = 1.0 - alpha
        deriv = n * (t ** (n - 1.0))
        return SchedulerCoefficients(alpha, sigma, deriv, -deriv)
    if scheduler == "VP":
        b = beta_min
        B = beta_max
        T = 0.5 * (1.0 - t) ** 2 * (B - b) + (1.0 - t) * b
        dT = -(1.0 - t) * (B - b) - b
        exp_neg_T = torch.exp(-T)
        sigma = torch.sqrt((1.0 - exp_neg_T).clamp_min(eps))
        return SchedulerCoefficients(
            alpha_t=torch.exp(-0.5 * T),
            sigma_t=sigma,
            d_alpha_t=-0.5 * dT * torch.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * exp_neg_T / sigma,
        )
    if scheduler == "LVP":
        sigma = torch.sqrt((1.0 - t**2).clamp_min(eps))
        return SchedulerCoefficients(t, sigma, torch.ones_like(t), -t / sigma)
    if scheduler == "Cosine":
        pi = torch.pi
        return SchedulerCoefficients(
            alpha_t=torch.sin(pi / 2.0 * t),
            sigma_t=torch.cos(pi / 2.0 * t).clamp_min(eps),
            d_alpha_t=pi / 2.0 * torch.cos(pi / 2.0 * t),
            d_sigma_t=-pi / 2.0 * torch.sin(pi / 2.0 * t),
        )
    raise ValueError(f"Unknown scheduler={scheduler!r}")


def affine_coefficients(coeffs: SchedulerCoefficients, training: str = "velocity", eps: float = 1e-6) -> AffineCoefficients:
    alpha = coeffs.alpha_t.clamp_min(eps)
    sigma = coeffs.sigma_t
    if training == "velocity":
        a_t = coeffs.d_alpha_t / alpha
        b_t = -(
            coeffs.d_sigma_t * sigma * alpha - coeffs.d_alpha_t * (sigma**2)
        ) / alpha
    elif training == "x1":
        a_t = 1.0 / alpha
        b_t = (sigma**2) / alpha
    elif training == "x0":
        a_t = alpha.new_zeros(alpha.shape)
        b_t = -sigma
    else:
        raise ValueError(f"Unknown training target: {training}")
    return AffineCoefficients(a_t=a_t, b_t=b_t)


def endpoint_from_velocity(x_t: Any, velocity: Any, t: Any, eps: float = 1e-6) -> Any:
    """CondOT velocity parameterization: x_1 = x_t + (1 - t) v_theta(x_t,t)."""
    return x_t + (1.0 - t).clamp_min(eps) * velocity

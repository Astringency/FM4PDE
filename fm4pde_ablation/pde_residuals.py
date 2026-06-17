from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ResidualOutput:
    residual: Any
    status: str
    metadata: dict[str, Any]


def compute_pde_residual(pde: str, coef: Any, sol: Any, *, k: int = 1) -> ResidualOutput:
    table: dict[str, Callable[..., ResidualOutput]] = {
        "darcy": _darcy,
        "poisson": _poisson,
        "helmholtz": lambda a, u: _helmholtz(a, u, k=k),
        "nsnonbounded": _nsnonbounded,
        "burger": _burger,
        "reaction_diffusion": _reaction_diffusion,
        "shallow_water": _shallow_water,
    }
    if pde not in table:
        raise ValueError(f"Unsupported PDE residual: {pde}")
    return table[pde](coef, sol)


def residual_status(pde: str) -> str:
    if pde in {"darcy", "poisson", "helmholtz"}:
        return "reliable"
    if pde in {"reaction_diffusion", "shallow_water", "burger"}:
        return "approximate"
    if pde == "nsnonbounded":
        return "placeholder"
    return "disabled"


def _darcy(a: Any, u: Any) -> ResidualOutput:
    import torch

    deriv_x, deriv_y = _central_kernels(u)
    ux = torch.nn.functional.conv2d(u, deriv_x, padding=(0, 1))
    uy = torch.nn.functional.conv2d(u, deriv_y, padding=(1, 0))
    div = torch.nn.functional.conv2d(a * ux, deriv_x, padding=(0, 1)) + torch.nn.functional.conv2d(
        a * uy, deriv_y, padding=(1, 0)
    )
    residual = div + 1.0
    return ResidualOutput(_zero_boundary(residual), "reliable", {"equation": "div(a grad u) + 1"})


def _poisson(a: Any, u: Any) -> ResidualOutput:
    residual = _laplacian(u) - a
    return ResidualOutput(_zero_boundary(residual), "reliable", {"equation": "laplace(u) - f"})


def _helmholtz(a: Any, u: Any, k: int = 1) -> ResidualOutput:
    residual = _laplacian(u) + float(k**2) * u - a
    return ResidualOutput(_zero_boundary(residual), "reliable", {"equation": "laplace(u) + k^2 u - f", "k": k})


def _nsnonbounded(a: Any, u: Any) -> ResidualOutput:
    import torch

    deriv_x, deriv_y = _central_kernels(u)
    residual = torch.nn.functional.conv2d(u, deriv_x, padding=(0, 1)) + torch.nn.functional.conv2d(
        u, deriv_y, padding=(1, 0)
    )
    return ResidualOutput(
        _zero_boundary(residual),
        "placeholder",
        {"warning": "Current FM4PDE legacy NS residual is grad_x(wT)+grad_y(wT), not full Navier-Stokes."},
    )


def _burger(a: Any, u: Any) -> ResidualOutput:
    import torch

    deriv_t = torch.tensor([[-1.0], [0.0], [1.0]], dtype=u.dtype, device=u.device).view(1, 1, 3, 1) / 2.0
    deriv_x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=u.dtype, device=u.device).view(1, 1, 1, 3) / 2.0
    ut = torch.nn.functional.conv2d(u, deriv_t, padding=(1, 0))
    ux = torch.nn.functional.conv2d(u, deriv_x, padding=(0, 1))
    uxx = torch.nn.functional.conv2d(ux, deriv_x, padding=(0, 1))
    residual = ut + u * ux - 0.01 * uxx
    return ResidualOutput(_zero_boundary(residual), "approximate", {"nu": 0.01, "axis": "BCHW-as-time-space"})


def _reaction_diffusion(a: Any, u: Any) -> ResidualOutput:
    import torch

    if a.shape[1] != 2 or u.shape[1] != 2:
        raise ValueError(f"reaction_diffusion expects 2+2 channels, got a={a.shape}, u={u.shape}")
    total_time = 5.0
    k = 5e-3
    d_u = 1e-3
    d_v = 5e-3
    a_u, a_v = a[:, 0:1], a[:, 1:2]
    u_u, u_v = u[:, 0:1], u[:, 1:2]
    u_t = (u_u - a_u) / total_time
    v_t = (u_v - a_v) / total_time
    lap_u = _periodic_laplacian(u_u)
    lap_v = _periodic_laplacian(u_v)
    res_u = u_t - (d_u * lap_u + u_u - u_u**3 - k - u_v)
    res_v = v_t - (d_v * lap_v + u_u - u_v)
    return ResidualOutput(
        torch.cat([res_u, res_v], dim=1),
        "approximate",
        {"two_time_level_approx": True, "T": total_time, "D_u": d_u, "D_v": d_v},
    )


def _shallow_water(a: Any, u: Any) -> ResidualOutput:
    import torch

    if a.shape[1] != 3 or u.shape[1] != 3:
        raise ValueError(f"shallow_water expects conservative variables [h,hu,hv], got a={a.shape}, u={u.shape}")
    h0, hu0, hv0 = a[:, 0:1], a[:, 1:2], a[:, 2:3]
    h, hu, hv = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    h_mid = 0.5 * (h0 + h)
    hu_mid = 0.5 * (hu0 + hu)
    hv_mid = 0.5 * (hv0 + hv)
    eps = 1e-6
    mass = (h - h0) + _dx(hu_mid) + _dy(hv_mid)
    mom_x = (hu - hu0) + _dx((hu_mid**2) / h_mid.clamp_min(eps) + 0.5 * h_mid**2) + _dy(
        hu_mid * hv_mid / h_mid.clamp_min(eps)
    )
    mom_y = (hv - hv0) + _dx(hu_mid * hv_mid / h_mid.clamp_min(eps)) + _dy(
        (hv_mid**2) / h_mid.clamp_min(eps) + 0.5 * h_mid**2
    )
    return ResidualOutput(
        torch.cat([mass, mom_x, mom_y], dim=1),
        "approximate",
        {"variables": ["h", "hu", "hv"], "two_time_level_approx": True},
    )


def _central_kernels(reference: Any) -> tuple[Any, Any]:
    import torch

    deriv_x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=reference.dtype, device=reference.device).view(1, 1, 1, 3) / 2.0
    deriv_y = torch.tensor([[-1.0], [0.0], [1.0]], dtype=reference.dtype, device=reference.device).view(1, 1, 3, 1) / 2.0
    return deriv_x, deriv_y


def _laplacian(u: Any) -> Any:
    import torch

    h = 1.0 / max(int(u.shape[-1]) - 1, 1)
    padded = torch.nn.functional.pad(u, (1, 1, 1, 1), "constant", 0)
    return (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
        - 4.0 * u
    ) / (h**2)


def _periodic_laplacian(u: Any) -> Any:
    h = 1.0 / max(int(u.shape[-1]) - 1, 1)
    return (
        u.roll(1, dims=2)
        + u.roll(-1, dims=2)
        + u.roll(1, dims=3)
        + u.roll(-1, dims=3)
        - 4.0 * u
    ) / (h**2)


def _zero_boundary(x: Any) -> Any:
    y = x.clone()
    if y.shape[-1] > 1 and y.shape[-2] > 1:
        y[..., 0, :] = 0
        y[..., -1, :] = 0
        y[..., :, 0] = 0
        y[..., :, -1] = 0
    return y


def _dx(f: Any) -> Any:
    import torch

    return (torch.nn.functional.pad(f, (1, 1, 0, 0), mode="replicate")[:, :, :, 2:] -
            torch.nn.functional.pad(f, (1, 1, 0, 0), mode="replicate")[:, :, :, :-2]) / 2.0


def _dy(f: Any) -> Any:
    import torch

    return (torch.nn.functional.pad(f, (0, 0, 1, 1), mode="replicate")[:, :, 2:, :] -
            torch.nn.functional.pad(f, (0, 0, 1, 1), mode="replicate")[:, :, :-2, :]) / 2.0

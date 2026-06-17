from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ResidualOutput:
    residual: Any
    status: str
    metadata: dict[str, Any]


def compute_pde_residual(
    pde: str,
    coef: Any,
    sol: Any,
    *,
    pde_params: dict[str, Any] | None = None,
    k: int = 1,
    residual_mode: str = "auto",
) -> ResidualOutput:
    pde_params = pde_params or {}
    table: dict[str, Callable[..., ResidualOutput]] = {
        "darcy": _darcy,
        "poisson": _poisson,
        "helmholtz": lambda a, u: _helmholtz(a, u, k=k),
        "nsnonbounded": lambda a, u: _nsnonbounded(a, u, residual_mode=residual_mode),
        "burger": _burger,
        "reaction_diffusion": _reaction_diffusion,
        "shallow_water": _shallow_water,
        "heat": lambda a, u: _heat(a, u, pde_params=pde_params),
        "wave": lambda a, u: _wave(a, u, pde_params=pde_params),
        "advection_diffusion": lambda a, u: _advection_diffusion(a, u, pde_params=pde_params),
        "steady_heat_conduction": lambda a, u: _steady_heat_conduction(a, u, pde_params=pde_params),
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
        return "disabled"
    if pde in {"heat", "wave", "advection_diffusion", "steady_heat_conduction"}:
        return "approximate"
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
    return _out(_zero_boundary(residual), "reliable", {"equation": "div(a grad u) + 1"})


def _poisson(a: Any, u: Any) -> ResidualOutput:
    residual = _laplacian(u) - a
    return _out(_zero_boundary(residual), "reliable", {"equation": "laplace(u) - f"})


def _helmholtz(a: Any, u: Any, k: int = 1) -> ResidualOutput:
    residual = _laplacian(u) + float(k**2) * u - a
    return _out(_zero_boundary(residual), "reliable", {"equation": "laplace(u) + k^2 u - f", "k": k})


def _nsnonbounded(a: Any, u: Any, residual_mode: str = "auto") -> ResidualOutput:
    import torch

    residual = torch.zeros_like(u[:, :1])
    return _out(
        residual,
        "disabled",
        {
            "equation": "Navier-Stokes vorticity residual",
            "warning": "NS PDE guidance is disabled because no reliable two-time-level vorticity residual is implemented.",
            "residual_mode": residual_mode,
        },
    )


def _burger(a: Any, u: Any) -> ResidualOutput:
    import torch

    deriv_t = torch.tensor([[-1.0], [0.0], [1.0]], dtype=u.dtype, device=u.device).view(1, 1, 3, 1) / 2.0
    deriv_x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=u.dtype, device=u.device).view(1, 1, 1, 3) / 2.0
    ut = torch.nn.functional.conv2d(u, deriv_t, padding=(1, 0))
    ux = torch.nn.functional.conv2d(u, deriv_x, padding=(0, 1))
    uxx = torch.nn.functional.conv2d(ux, deriv_x, padding=(0, 1))
    residual = ut + u * ux - 0.01 * uxx
    return _out(
        _zero_boundary(residual),
        "approximate",
        {"equation": "u_t + u u_x - nu u_xx", "nu": 0.01, "axis": "BCHW-as-time-space"},
    )


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
    return _out(
        torch.cat([res_u, res_v], dim=1),
        "approximate",
        {
            "equation": "two-time-level FitzHugh-Nagumo reaction-diffusion residual",
            "two_time_level_approx": True,
            "T": total_time,
            "D_u": d_u,
            "D_v": d_v,
        },
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
    return _out(
        torch.cat([mass, mom_x, mom_y], dim=1),
        "approximate",
        {
            "equation": "two-time-level shallow-water conservative residual",
            "variables": ["h", "hu", "hv"],
            "two_time_level_approx": True,
        },
    )


def _heat(a: Any, u: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    if a.shape[1] != 1 or u.shape[1] != 1:
        raise ValueError(f"heat expects 1+1 channels, got a={a.shape}, u={u.shape}")
    alpha = _param_field(pde_params, "alpha", u, default=1.0)
    u_mid = 0.5 * (a + u)
    residual = (u - a) - alpha * _laplacian(u_mid)
    return _out(
        _zero_boundary(residual),
        "approximate",
        {
            "equation": "u_t - alpha * laplace(u_mid)",
            "two_time_level_approx": True,
            "pde_params_used": _used_params(pde_params, ("alpha",)),
        },
    )


def _wave(a: Any, u: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    import torch

    if a.shape[1] != 2 or u.shape[1] != 2:
        raise ValueError(f"wave expects 2+2 channels [u,v], got a={a.shape}, u={u.shape}")
    c = _param_field(pde_params, "c", u[:, 0:1], default=1.0)
    u0, v0 = a[:, 0:1], a[:, 1:2]
    u_t, v_t = u[:, 0:1], u[:, 1:2]
    u_mid = 0.5 * (u0 + u_t)
    v_mid = 0.5 * (v0 + v_t)
    res_u = (u_t - u0) - v_mid
    res_v = (v_t - v0) - (c**2) * _laplacian(u_mid)
    return _out(
        torch.cat([_zero_boundary(res_u), _zero_boundary(res_v)], dim=1),
        "approximate",
        {
            "equation": "u_t - v, v_t - c^2 laplace(u_mid)",
            "two_time_level_approx": True,
            "pde_params_used": _used_params(pde_params, ("c",)),
        },
    )


def _advection_diffusion(a: Any, u: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    if a.shape[1] != 1 or u.shape[1] != 1:
        raise ValueError(f"advection_diffusion expects 1+1 channels, got a={a.shape}, u={u.shape}")
    bx = _param_field(pde_params, "b_x", u, default=0.0)
    by = _param_field(pde_params, "b_y", u, default=0.0)
    kappa = _param_field(pde_params, "kappa", u, default=1.0)
    u_mid = 0.5 * (a + u)
    residual = (u - a) + bx * _dx(u_mid) + by * _dy(u_mid) - kappa * _laplacian(u_mid)
    return _out(
        _zero_boundary(residual),
        "approximate",
        {
            "equation": "u_t + b_x u_x + b_y u_y - kappa laplace(u_mid)",
            "two_time_level_approx": True,
            "pde_params_used": _used_params(pde_params, ("b_x", "b_y", "kappa")),
        },
    )


def _steady_heat_conduction(a: Any, u: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    if a.shape[1] != 1 or u.shape[1] != 1:
        raise ValueError(f"steady_heat_conduction expects 1+1 channels, got a={a.shape}, u={u.shape}")
    conductivity = (1.0 + 0.05 * (u - 298.0)).clamp_min(0.1)
    residual = _nonlinear_heat_conduction_residual(u, conductivity, a[:, :1])
    return _out(
        _zero_boundary(residual),
        "approximate",
        {
            "equation": "-div(lambda(u) grad u) - f, lambda(u)=max(1+0.05*(u-298),0.1)",
            "boundary_condition": "bottom Dirichlet u=u_D; top/left/right zero Neumann",
            "pde_params_used": _used_params(pde_params, ("u_D",)),
        },
    )


def _central_kernels(reference: Any) -> tuple[Any, Any]:
    import torch

    hx = 1.0 / max(int(reference.shape[-1]) - 1, 1)
    hy = 1.0 / max(int(reference.shape[-2]) - 1, 1)
    deriv_x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=reference.dtype, device=reference.device).view(1, 1, 1, 3) / (2.0 * hx)
    deriv_y = torch.tensor([[-1.0], [0.0], [1.0]], dtype=reference.dtype, device=reference.device).view(1, 1, 3, 1) / (2.0 * hy)
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
    hx = 1.0 / max(int(u.shape[-1]) - 1, 1)
    hy = 1.0 / max(int(u.shape[-2]) - 1, 1)
    return (
        (u.roll(1, dims=2) + u.roll(-1, dims=2) - 2.0 * u) / (hy**2)
        + (u.roll(1, dims=3) + u.roll(-1, dims=3) - 2.0 * u) / (hx**2)
    )


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

    h = 1.0 / max(int(f.shape[-1]) - 1, 1)
    return (torch.nn.functional.pad(f, (1, 1, 0, 0), mode="replicate")[:, :, :, 2:] -
            torch.nn.functional.pad(f, (1, 1, 0, 0), mode="replicate")[:, :, :, :-2]) / (2.0 * h)


def _dy(f: Any) -> Any:
    import torch

    h = 1.0 / max(int(f.shape[-2]) - 1, 1)
    return (torch.nn.functional.pad(f, (0, 0, 1, 1), mode="replicate")[:, :, 2:, :] -
            torch.nn.functional.pad(f, (0, 0, 1, 1), mode="replicate")[:, :, :-2, :]) / (2.0 * h)


def _param_field(params: dict[str, Any], name: str, reference: Any, default: float) -> Any:
    import torch

    if name not in params:
        return torch.full((reference.shape[0], 1, 1, 1), float(default), dtype=reference.dtype, device=reference.device)
    value = torch.as_tensor(params[name], dtype=reference.dtype, device=reference.device).reshape(-1)
    if value.numel() == 1:
        value = value.repeat(reference.shape[0])
    if value.numel() != reference.shape[0]:
        raise ValueError(f"PDE parameter {name!r} must have batch length {reference.shape[0]}, got {tuple(value.shape)}")
    return value.view(reference.shape[0], 1, 1, 1)


def _nonlinear_heat_conduction_residual(u: Any, conductivity: Any, source: Any) -> Any:
    import torch

    h = 1.0 / max(int(u.shape[-1]) - 1, 1)
    inv_h2 = 1.0 / (h**2)
    residual = torch.zeros_like(u[:, :1])
    if u.shape[-2] <= 2 or u.shape[-1] <= 2:
        return residual
    center_u = u[:, :, 1:-1, 1:-1]
    center_l = conductivity[:, :, 1:-1, 1:-1]
    accum = torch.zeros_like(center_u)
    for neighbor_u, neighbor_l in (
        (u[:, :, :-2, 1:-1], conductivity[:, :, :-2, 1:-1]),
        (u[:, :, 2:, 1:-1], conductivity[:, :, 2:, 1:-1]),
        (u[:, :, 1:-1, :-2], conductivity[:, :, 1:-1, :-2]),
        (u[:, :, 1:-1, 2:], conductivity[:, :, 1:-1, 2:]),
    ):
        face = 0.5 * (center_l + neighbor_l) * inv_h2
        accum = accum + face * (center_u - neighbor_u)
    residual[:, :, 1:-1, 1:-1] = accum - source[:, :, 1:-1, 1:-1]
    return residual


def _used_params(params: dict[str, Any], names: tuple[str, ...]) -> dict[str, bool]:
    return {name: name in params for name in names}


def _out(residual: Any, status: str, metadata: dict[str, Any]) -> ResidualOutput:
    metadata = dict(metadata)
    metadata.setdefault("residual_status", status)
    metadata.setdefault("status", status)
    return ResidualOutput(residual, status, metadata)

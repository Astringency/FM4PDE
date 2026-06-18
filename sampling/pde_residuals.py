from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data.specs import FULL_TIME_SPACE_PDES, STATIC_PDES, TEMPORAL_ENDPOINT_PDES, get_pde_spec

ENDPOINT_SECANT_WARNING = "coarse two-time-level endpoint residual; not a full spatiotemporal PDE residual"
TEMPORAL_ONLY_MODES = {"hermite_bridge", "near_endpoint_temporal", "endpoint_secant", "full_trajectory_fd"}


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
    normalized_mode = _normalize_residual_mode(residual_mode)
    if pde in TEMPORAL_ENDPOINT_PDES:
        return _endpoint_time_dependent_residual(
            pde,
            coef,
            sol,
            pde_params=pde_params,
            requested_mode=residual_mode,
            normalized_mode=normalized_mode,
        )
    if normalized_mode == "disabled":
        return _disabled_residual(coef, sol, pde, residual_mode)
    if pde in STATIC_PDES and normalized_mode in TEMPORAL_ONLY_MODES:
        raise ValueError(f"{pde} is in the static residual family; temporal residual_mode={normalized_mode!r} is invalid")
    if pde in FULL_TIME_SPACE_PDES and normalized_mode in {"hermite_bridge", "near_endpoint_temporal", "endpoint_secant"}:
        raise ValueError(f"{pde} already uses a full time-space residual; temporal endpoint mode {normalized_mode!r} is invalid")
    table: dict[str, Callable[..., ResidualOutput]] = {
        "darcy": _darcy,
        "poisson": _poisson,
        "helmholtz": lambda a, u: _helmholtz(a, u, k=k),
        "burger": lambda a, u: _burger(a, u, residual_mode=residual_mode),
        "steady_heat_conduction": lambda a, u: _steady_heat_conduction(a, u, pde_params=pde_params),
    }
    if pde not in table:
        raise ValueError(f"Unsupported PDE residual: {pde}")
    out = table[pde](coef, sol)
    out.metadata.setdefault("requested_residual_mode", residual_mode)
    out.metadata.setdefault(
        "resolved_residual_mode",
        "full_time_space" if pde == "burger" else ("static_nonlinear_boundary" if pde == "steady_heat_conduction" else "static"),
    )
    _apply_family_metadata(out, pde, normalized_mode)
    return out


def residual_status(pde: str) -> str:
    if pde in STATIC_PDES or pde in FULL_TIME_SPACE_PDES:
        return "reliable"
    if pde in TEMPORAL_ENDPOINT_PDES:
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
    return _out(_zero_boundary(residual), "reliable", {"equation": "div(a grad u) + 1", "mode": "static"})


def _poisson(a: Any, u: Any) -> ResidualOutput:
    residual = _laplacian(u) - a
    return _out(_zero_boundary(residual), "reliable", {"equation": "laplace(u) - f", "mode": "static"})


def _helmholtz(a: Any, u: Any, k: int = 1) -> ResidualOutput:
    residual = _laplacian(u) + float(k**2) * u - a
    return _out(_zero_boundary(residual), "reliable", {"equation": "laplace(u) + k^2 u - f", "k": k, "mode": "static"})


def _burger(a: Any, u: Any, residual_mode: str = "auto") -> ResidualOutput:
    import torch

    deriv_t = torch.tensor([[-1.0], [0.0], [1.0]], dtype=u.dtype, device=u.device).view(1, 1, 3, 1) / 2.0
    deriv_x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=u.dtype, device=u.device).view(1, 1, 1, 3) / 2.0
    ut = torch.nn.functional.conv2d(u, deriv_t, padding=(1, 0))
    ux = torch.nn.functional.conv2d(u, deriv_x, padding=(0, 1))
    uxx = torch.nn.functional.conv2d(ux, deriv_x, padding=(0, 1))
    residual = ut + u * ux - 0.01 * uxx
    return _out(
        _zero_boundary(residual),
        "reliable",
        {
            "equation": "u_t + u u_x - nu u_xx",
            "nu": 0.01,
            "axis": "BCHW-as-time-space",
            "mode": "full_time_space",
            "requested_residual_mode": residual_mode,
            "resolved_residual_mode": "full_time_space",
            "full_trajectory_required": True,
            "residual_family": "full_time_space",
            "temporal_derivative_mode": "full_fd",
            "endpoint_only": False,
            "uses_generated_trajectory": True,
            "uses_extra_temporal_observations": False,
            "two_time_level_approx": False,
        },
    )


def _endpoint_time_dependent_residual(
    pde: str,
    a: Any,
    u: Any,
    *,
    pde_params: dict[str, Any],
    requested_mode: str,
    normalized_mode: str,
) -> ResidualOutput:
    resolved_mode = _resolve_endpoint_mode(normalized_mode)
    if resolved_mode == "disabled":
        return _disabled_residual(a, u, pde, requested_mode)
    if resolved_mode == "hermite_bridge":
        out = _hermite_bridge_residual(pde, a, u, pde_params)
    elif resolved_mode == "near_endpoint_temporal":
        out = _near_endpoint_temporal_residual(pde, a, u, pde_params)
    elif resolved_mode == "endpoint_secant":
        out = _endpoint_secant_residual(pde, a, u, pde_params)
    elif resolved_mode == "full_trajectory_fd":
        out = _full_trajectory_fd_residual(pde, a, u, pde_params)
    else:
        raise ValueError(f"residual_mode={requested_mode!r} is invalid for temporal endpoint PDE {pde!r}")
    out.metadata.setdefault("requested_residual_mode", requested_mode)
    out.metadata.setdefault("resolved_residual_mode", resolved_mode)
    _apply_family_metadata(out, pde, resolved_mode, requested_mode=requested_mode)
    return out


def _resolve_endpoint_mode(mode: str) -> str:
    if mode == "auto":
        return "hermite_bridge"
    return mode


def _endpoint_secant_residual(
    pde: str,
    a: Any,
    u: Any,
    pde_params: dict[str, Any],
) -> ResidualOutput:
    if pde == "heat":
        return _heat_endpoint_secant(a, u, pde_params)
    if pde == "wave":
        return _wave_endpoint_secant(a, u, pde_params)
    if pde == "advection_diffusion":
        return _advection_diffusion_endpoint_secant(a, u, pde_params)
    if pde == "reaction_diffusion":
        return _reaction_diffusion_endpoint_secant(a, u, pde_params)
    if pde == "shallow_water":
        return _shallow_water_endpoint_secant(a, u, pde_params)
    if pde == "nsnonbounded":
        return _generic_endpoint_secant(pde, a, u, pde_params)
    raise ValueError(f"endpoint_secant residual is not implemented for {pde!r}")


def _heat_endpoint_secant(a: Any, u: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    if a.shape[1] != 1 or u.shape[1] != 1:
        raise ValueError(f"heat expects 1+1 channels, got a={a.shape}, u={u.shape}")
    alpha = _param_field(pde_params, "alpha", u, default=1.0)
    time_scale, time_meta = _time_scale_field(pde_params, u)
    u_mid = 0.5 * (a + u)
    residual = (u - a) / time_scale - alpha * _laplacian(u_mid)
    return _out(
        _zero_boundary(residual),
        "approximate",
        {
            "equation": "(uT - u0) / T - alpha * laplace(u_mid)",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _used_params(pde_params, ("alpha", "T", "total_time", "dt")),
        },
    )


def _wave_endpoint_secant(a: Any, u: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    import torch

    if a.shape[1] != 2 or u.shape[1] != 2:
        raise ValueError(f"wave expects 2+2 channels [u,v], got a={a.shape}, u={u.shape}")
    c = _param_field(pde_params, "c", u[:, 0:1], default=1.0)
    u0, v0 = a[:, 0:1], a[:, 1:2]
    u_t, v_t = u[:, 0:1], u[:, 1:2]
    time_scale, time_meta = _time_scale_field(pde_params, u[:, 0:1])
    u_mid = 0.5 * (u0 + u_t)
    v_mid = 0.5 * (v0 + v_t)
    res_u = (u_t - u0) / time_scale - v_mid
    res_v = (v_t - v0) / time_scale - (c**2) * _laplacian(u_mid)
    return _out(
        torch.cat([_zero_boundary(res_u), _zero_boundary(res_v)], dim=1),
        "approximate",
        {
            "equation": "(uT - u0) / T - v_mid, (vT - v0) / T - c^2 laplace(u_mid)",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _used_params(pde_params, ("c", "T", "total_time", "dt")),
        },
    )


def _advection_diffusion_endpoint_secant(
    a: Any,
    u: Any,
    pde_params: dict[str, Any],
) -> ResidualOutput:
    if a.shape[1] != 1 or u.shape[1] != 1:
        raise ValueError(f"advection_diffusion expects 1+1 channels, got a={a.shape}, u={u.shape}")
    bx = _param_field(pde_params, "b_x", u, default=0.0)
    by = _param_field(pde_params, "b_y", u, default=0.0)
    kappa = _param_field(pde_params, "kappa", u, default=1.0)
    time_scale, time_meta = _time_scale_field(pde_params, u)
    u_mid = 0.5 * (a + u)
    residual = (u - a) / time_scale + bx * _dx(u_mid) + by * _dy(u_mid) - kappa * _laplacian(u_mid)
    return _out(
        _zero_boundary(residual),
        "approximate",
        {
            "equation": "(uT - u0) / T + b_x u_x + b_y u_y - kappa laplace(u_mid)",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _used_params(pde_params, ("b_x", "b_y", "kappa", "T", "total_time", "dt")),
        },
    )


def _reaction_diffusion_endpoint_secant(
    a: Any,
    u: Any,
    pde_params: dict[str, Any],
) -> ResidualOutput:
    import torch

    if a.shape[1] != 2 or u.shape[1] != 2:
        raise ValueError(f"reaction_diffusion expects 2+2 channels, got a={a.shape}, u={u.shape}")
    a_u, a_v = a[:, 0:1], a[:, 1:2]
    u_u, u_v = u[:, 0:1], u[:, 1:2]
    time_scale, time_meta = _time_scale_field(pde_params, u_u, default=_default_total_time("reaction_diffusion"))
    d_u = _param_field_any(pde_params, ("D_u", "Du"), u_u, default=2e-3)
    d_v = _param_field_any(pde_params, ("D_v", "Dv"), u_v, default=4e-3)
    k = _param_field(pde_params, "k", u_u, default=3e-3)
    u_t = (u_u - a_u) / time_scale
    v_t = (u_v - a_v) / time_scale
    lap_u = _neumann_laplacian(u_u, pde_params)
    lap_v = _neumann_laplacian(u_v, pde_params)
    res_u = u_t - (d_u * lap_u + u_u - u_u**3 - k - u_v)
    res_v = v_t - (d_v * lap_v + u_u - u_v)
    return _out(
        torch.cat([res_u, res_v], dim=1),
        "approximate",
        {
            "equation": "two-time-level FitzHugh-Nagumo reaction-diffusion residual",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _used_params(
                pde_params,
                (
                    "D_u",
                    "Du",
                    "D_v",
                    "Dv",
                    "k",
                    "T",
                    "total_time",
                    "dt",
                    "x_range",
                    "y_range",
                    "x_left",
                    "x_right",
                    "y_bottom",
                    "y_top",
                    "dx",
                    "dy",
                ),
            ),
            **_reaction_diffusion_spatial_metadata(pde_params, u_u),
        },
    )


def _shallow_water_endpoint_secant(
    a: Any,
    u: Any,
    pde_params: dict[str, Any],
) -> ResidualOutput:
    import torch

    if a.shape[1] != 3 or u.shape[1] != 3:
        raise ValueError(f"shallow_water expects conservative variables [h,hu,hv], got a={a.shape}, u={u.shape}")
    h0, hu0, hv0 = a[:, 0:1], a[:, 1:2], a[:, 2:3]
    h, hu, hv = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    h_mid = 0.5 * (h0 + h)
    hu_mid = 0.5 * (hu0 + hu)
    hv_mid = 0.5 * (hv0 + hv)
    eps = _float_param(pde_params, "eps", 1e-6)
    g = _param_field(pde_params, "g", h, default=1.0)
    time_scale, time_meta = _time_scale_field(pde_params, h)
    h_safe = h_mid.clamp_min(eps)
    mass = (h - h0) / time_scale + _dx(hu_mid) + _dy(hv_mid)
    mom_x = (hu - hu0) / time_scale + _dx((hu_mid**2) / h_safe + 0.5 * g * h_mid**2) + _dy(
        hu_mid * hv_mid / h_safe
    )
    mom_y = (hv - hv0) / time_scale + _dx(hu_mid * hv_mid / h_safe) + _dy(
        (hv_mid**2) / h_safe + 0.5 * g * h_mid**2
    )
    return _out(
        torch.cat([mass, mom_x, mom_y], dim=1),
        "approximate",
        {
            "equation": "two-time-level shallow-water conservative residual",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "variables": ["h", "hu", "hv"],
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _used_params(pde_params, ("g", "eps", "T", "total_time", "dt")),
        },
    )


def _generic_endpoint_secant(pde: str, a: Any, u: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    _validate_time_dependent_state(pde, a, u)
    time_scale, time_meta = _time_scale_field(pde_params, u, default=_default_total_time(pde))
    q_mid = 0.5 * (a + u)
    residual = (u - a) / time_scale - _rhs_time_dependent(pde, q_mid, pde_params)
    residual = _apply_endpoint_residual_boundary(pde, residual)
    return _out(
        residual,
        "approximate",
        {
            "equation": f"{pde} endpoint secant residual",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _rhs_param_usage(pde, pde_params),
            **_rhs_metadata(pde, pde_params, q_mid),
        },
    )


def _full_trajectory_fd_residual(pde: str, q0: Any, qT: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    import torch

    trajectory = _extract_full_trajectory(pde_params)
    trajectory = torch.as_tensor(trajectory, dtype=q0.dtype, device=q0.device)
    if trajectory.ndim == 4:
        trajectory = trajectory.unsqueeze(0)
    if trajectory.ndim != 5:
        raise ValueError(
            "full_trajectory_fd mode requires trajectory with layout [B,T,C,H,W] or [B,C,T,H,W], "
            f"got shape {tuple(trajectory.shape)}"
        )
    expected_channels = get_pde_spec(pde).coef_channels
    if trajectory.shape[2] == expected_channels:
        btchw = trajectory
    elif trajectory.shape[1] == expected_channels:
        btchw = trajectory.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(
            "full_trajectory_fd mode requires trajectory with layout [B,T,C,H,W] or [B,C,T,H,W], "
            f"got shape {tuple(trajectory.shape)} for {expected_channels} state channels"
        )
    if btchw.shape[1] < 3:
        raise ValueError("full_trajectory_fd mode requires at least three trajectory time points")
    if btchw.shape[0] == 1 and q0.shape[0] > 1:
        btchw = btchw.repeat(q0.shape[0], 1, 1, 1, 1)
    if btchw.shape[0] != q0.shape[0] or btchw.shape[2:] != q0.shape[1:]:
        raise ValueError(
            "full_trajectory_fd trajectory batch/channel/spatial shape must match endpoints; "
            f"trajectory={tuple(btchw.shape)}, q0={tuple(q0.shape)}"
        )
    dt, dt_meta = _trajectory_dt_field(pde_params, q0, int(btchw.shape[1]))
    residuals = []
    for idx in range(int(btchw.shape[1])):
        q = btchw[:, idx]
        if idx == 0:
            q_t = (btchw[:, 1] - btchw[:, 0]) / dt
        elif idx == int(btchw.shape[1]) - 1:
            q_t = (btchw[:, -1] - btchw[:, -2]) / dt
        else:
            q_t = (btchw[:, idx + 1] - btchw[:, idx - 1]) / (2.0 * dt)
        residuals.append(_apply_endpoint_residual_boundary(pde, q_t - _rhs_time_dependent(pde, q, pde_params)))
    residual = torch.cat(residuals, dim=1)
    return _out(
        residual,
        "reliable",
        {
            "equation": f"{pde} full trajectory finite-difference residual",
            "mode": "full_trajectory_fd",
            "resolved_residual_mode": "full_trajectory_fd",
            "trajectory_layout": "BTCHW",
            "trajectory_time_points": int(btchw.shape[1]),
            "time_delta": dt_meta,
            "rhs": _rhs_name(pde),
            "pde_params_used": _rhs_param_usage(pde, pde_params),
            **_rhs_metadata(pde, pde_params, btchw[:, 0]),
        },
    )


def _extract_full_trajectory(params: dict[str, Any]) -> Any:
    for name in ("full_trajectory", "trajectory"):
        if name in params:
            return params[name]
    raise ValueError("full_trajectory_fd mode requires explicit full trajectory state in pde_params['full_trajectory'] or pde_params['trajectory']")


def _trajectory_dt_field(params: dict[str, Any], reference: Any, n_time: int) -> tuple[Any, dict[str, Any]]:
    import torch

    if "trajectory_dt" in params:
        dt = _param_field(params, "trajectory_dt", reference, default=1.0)
        return dt, {"source": "trajectory_dt", "defaulted": False, "values": _metadata_values(dt)}
    if "dt" in params:
        dt = _param_field(params, "dt", reference, default=1.0)
        return dt, {"source": "dt", "defaulted": False, "values": _metadata_values(dt)}
    for name in ("T", "total_time"):
        if name in params:
            total = _param_field(params, name, reference, default=1.0)
            dt = total / float(max(n_time - 1, 1))
            return dt, {"source": f"{name}/(n_time-1)", "defaulted": False, "values": _metadata_values(dt)}
    dt = torch.full((reference.shape[0], 1, 1, 1), 1.0 / float(max(n_time - 1, 1)), dtype=reference.dtype, device=reference.device)
    return dt, {"source": "default_unit_interval/(n_time-1)", "defaulted": True, "value": _metadata_values(dt)}


def _hermite_bridge_residual(pde: str, q0: Any, qT: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    import torch

    _validate_time_dependent_state(pde, q0, qT)
    time_scale, time_meta = _time_scale_field(pde_params, q0, default=_default_total_time(pde))
    f0 = _rhs_time_dependent(pde, q0, pde_params)
    fT = _rhs_time_dependent(pde, qT, pde_params)
    collocation_times = _hermite_collocation_times(pde_params)
    residuals = []
    for s_value in collocation_times:
        s = torch.as_tensor(s_value, dtype=q0.dtype, device=q0.device)
        s2 = s * s
        s3 = s2 * s
        h = (
            (2.0 * s3 - 3.0 * s2 + 1.0) * q0
            + (s3 - 2.0 * s2 + s) * time_scale * f0
            + (-2.0 * s3 + 3.0 * s2) * qT
            + (s3 - s2) * time_scale * fT
        )
        dh_ds = (
            (6.0 * s2 - 6.0 * s) * q0
            + (3.0 * s2 - 4.0 * s + 1.0) * time_scale * f0
            + (-6.0 * s2 + 6.0 * s) * qT
            + (3.0 * s2 - 2.0 * s) * time_scale * fT
        )
        residuals.append(_apply_endpoint_residual_boundary(pde, dh_ds / time_scale - _rhs_time_dependent(pde, h, pde_params)))

    include_integral = _hermite_include_integral(pde_params)
    integral_weight = _hermite_integral_weight(pde_params)
    if include_integral:
        weight = torch.as_tensor(integral_weight, dtype=q0.dtype, device=q0.device).sqrt()
        integral_residual = qT - q0 - 0.5 * time_scale * (f0 + fT)
        residuals.append(_apply_endpoint_residual_boundary(pde, weight * integral_residual))
    residual = torch.cat(residuals, dim=1)
    return _out(
        residual,
        "approximate",
        {
            "equation": f"{pde} endpoint-induced cubic Hermite bridge residual",
            "mode": "hermite_bridge",
            "bridge_type": "cubic_hermite_endpoint_pde_derivatives",
            "collocation_times": collocation_times,
            "num_collocation": len(collocation_times),
            "include_integral_residual": include_integral,
            "integral_weight": integral_weight,
            "time_scale": time_meta,
            "rhs": _rhs_name(pde),
            "two_time_level_approx": False,
            "endpoint_only": True,
            "full_trajectory_required": False,
            "residual_channels": int(residual.shape[1]),
            "state_channels": int(q0.shape[1]),
            "pde_params_used": _rhs_param_usage(pde, pde_params),
            **_rhs_metadata(pde, pde_params, q0),
        },
    )


def _near_endpoint_temporal_residual(pde: str, q0: Any, qT: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    import torch

    _validate_time_dependent_state(pde, q0, qT)
    near = _extract_near_endpoint_temporal_params(pde_params)
    q_dt = _as_bchw_like(near["q_dt"], q0, "q_dt")
    q_T_minus_dt = _as_bchw_like(near["q_T_minus_dt"], qT, "q_T_minus_dt")
    dt = _temporal_delta_field(near["dt"], q0)
    mask_0 = _as_mask_like(near["mask_0"], q0, "mask_0")
    mask_T = _as_mask_like(near["mask_T"], qT, "mask_T")

    td0 = (q_dt - q0) / dt
    tdT = (qT - q_T_minus_dt) / dt
    r0 = _apply_endpoint_residual_boundary(pde, td0 - _rhs_time_dependent(pde, q0, pde_params))
    rT = _apply_endpoint_residual_boundary(pde, tdT - _rhs_time_dependent(pde, qT, pde_params))
    scale0, count0 = _mask_normalization(mask_0, r0)
    scaleT, countT = _mask_normalization(mask_T, rT)
    r0_masked = r0 * mask_0 * scale0
    rT_masked = rT * mask_T * scaleT
    residual = torch.cat([r0_masked, rT_masked], dim=1)
    source_meta = near.get("metadata", {})
    return _out(
        residual,
        "approximate",
        {
            "equation": f"{pde} near-endpoint sparse temporal residual",
            "mode": "near_endpoint_temporal",
            "rhs": _rhs_name(pde),
            "two_time_level_approx": False,
            "endpoint_only": False,
            "full_trajectory_required": False,
            "extra_temporal_observations_required": True,
            "requires_extra_temporal_observations": True,
            "uses_sparse_temporal_observations": True,
            "uses_extra_temporal_observations": True,
            "uses_generated_trajectory": False,
            "time_delta": _metadata_values(dt),
            "mask_observed_points_0": _metadata_values(count0),
            "mask_observed_points_T": _metadata_values(countT),
            "mask_normalization_0": _metadata_values(scale0),
            "mask_normalization_T": _metadata_values(scaleT),
            "num_observed_points": {
                "near_0": _metadata_values(count0),
                "near_T": _metadata_values(countT),
            },
            "near_endpoint_metadata": source_meta,
            "pde_params_used": _rhs_param_usage(pde, pde_params),
            **_rhs_metadata(pde, pde_params, q0),
        },
    )


def _rhs_time_dependent(pde: str, q: Any, pde_params: dict[str, Any]) -> Any:
    if pde == "heat":
        return _rhs_heat(q, pde_params)
    if pde == "wave":
        return _rhs_wave(q, pde_params)
    if pde == "advection_diffusion":
        return _rhs_advection_diffusion(q, pde_params)
    if pde == "reaction_diffusion":
        return _rhs_reaction_diffusion(q, pde_params)
    if pde == "shallow_water":
        return _rhs_shallow_water(q, pde_params)
    if pde == "nsnonbounded":
        return _rhs_nsnonbounded(q, pde_params)
    raise ValueError(f"No time-dependent RHS is implemented for {pde!r}")


def _rhs_heat(q: Any, pde_params: dict[str, Any]) -> Any:
    alpha = _param_field(pde_params, "alpha", q, default=1.0)
    return alpha * _laplacian(q)


def _rhs_wave(q: Any, pde_params: dict[str, Any]) -> Any:
    import torch

    if q.shape[1] != 2:
        raise ValueError(f"wave RHS expects [u,v] channels, got {q.shape}")
    c = _param_field(pde_params, "c", q[:, 0:1], default=1.0)
    u, v = q[:, 0:1], q[:, 1:2]
    return torch.cat([v, (c**2) * _laplacian(u)], dim=1)


def _rhs_advection_diffusion(q: Any, pde_params: dict[str, Any]) -> Any:
    bx = _param_field(pde_params, "b_x", q, default=0.0)
    by = _param_field(pde_params, "b_y", q, default=0.0)
    kappa = _param_field(pde_params, "kappa", q, default=1.0)
    return -bx * _dx(q) - by * _dy(q) + kappa * _laplacian(q)


def _rhs_reaction_diffusion(q: Any, pde_params: dict[str, Any]) -> Any:
    import torch

    if q.shape[1] != 2:
        raise ValueError(f"reaction_diffusion RHS expects [u,v] channels, got {q.shape}")
    u, v = q[:, 0:1], q[:, 1:2]
    d_u = _param_field_any(pde_params, ("D_u", "Du"), u, default=2e-3)
    d_v = _param_field_any(pde_params, ("D_v", "Dv"), v, default=4e-3)
    k = _param_field(pde_params, "k", u, default=3e-3)
    f_u = d_u * _neumann_laplacian(u, pde_params) + u - u**3 - k - v
    f_v = d_v * _neumann_laplacian(v, pde_params) + u - v
    return torch.cat([f_u, f_v], dim=1)


def _rhs_shallow_water(q: Any, pde_params: dict[str, Any]) -> Any:
    import torch

    if q.shape[1] != 3:
        raise ValueError(f"shallow_water RHS expects [h,hu,hv] channels, got {q.shape}")
    h, hu, hv = q[:, 0:1], q[:, 1:2], q[:, 2:3]
    eps = _float_param(pde_params, "eps", 1e-6)
    g = _param_field(pde_params, "g", h, default=1.0)
    h_safe = h.clamp_min(eps)
    flux_x = torch.cat(
        [
            hu,
            (hu**2) / h_safe + 0.5 * g * h**2,
            hu * hv / h_safe,
        ],
        dim=1,
    )
    flux_y = torch.cat(
        [
            hv,
            hu * hv / h_safe,
            (hv**2) / h_safe + 0.5 * g * h**2,
        ],
        dim=1,
    )
    return -_dx(flux_x) - _dy(flux_y)


def _rhs_nsnonbounded(q: Any, pde_params: dict[str, Any]) -> Any:
    import torch

    if q.shape[1] != 1:
        raise ValueError(f"nsnonbounded RHS expects scalar vorticity channel, got {q.shape}")
    w = q[:, :1]
    w_zero_mean = w - w.mean(dim=(-2, -1), keepdim=True)
    psi_hat, kx, ky, k2 = _periodic_stream_function_fft(w_zero_mean)
    u_vel = torch.fft.ifft2(1j * ky * psi_hat, dim=(-2, -1)).real
    v_vel = torch.fft.ifft2(-1j * kx * psi_hat, dim=(-2, -1)).real
    w_hat = torch.fft.fft2(w_zero_mean, dim=(-2, -1))
    w_x = torch.fft.ifft2(1j * kx * w_hat, dim=(-2, -1)).real
    w_y = torch.fft.ifft2(1j * ky * w_hat, dim=(-2, -1)).real
    lap_w = torch.fft.ifft2(-k2 * w_hat, dim=(-2, -1)).real
    nu = _param_field_any(pde_params, ("nu", "viscosity"), w, default=1e-3)
    forcing = _forcing_field(pde_params, w)
    return -u_vel * w_x - v_vel * w_y + nu * lap_w + forcing


def _periodic_stream_function_fft(w: Any) -> tuple[Any, Any, Any, Any]:
    import torch

    h, width = int(w.shape[-2]), int(w.shape[-1])
    ky_1d = 2.0 * torch.pi * torch.fft.fftfreq(h, d=1.0 / max(h, 1), device=w.device, dtype=w.dtype)
    kx_1d = 2.0 * torch.pi * torch.fft.fftfreq(width, d=1.0 / max(width, 1), device=w.device, dtype=w.dtype)
    ky = ky_1d.view(1, 1, h, 1)
    kx = kx_1d.view(1, 1, 1, width)
    k2 = kx**2 + ky**2
    w_hat = torch.fft.fft2(w, dim=(-2, -1))
    psi_hat = torch.zeros_like(w_hat)
    mask = k2 > 0
    psi_hat = torch.where(mask, w_hat / torch.where(mask, k2, torch.ones_like(k2)), psi_hat)
    return psi_hat, kx, ky, k2


def _forcing_field(params: dict[str, Any], reference: Any) -> Any:
    import torch

    if "forcing" not in params:
        return torch.zeros_like(reference[:, :1])
    forcing = torch.as_tensor(params["forcing"], dtype=reference.dtype, device=reference.device)
    if forcing.ndim == 0:
        return torch.full_like(reference[:, :1], float(forcing))
    if forcing.ndim == 2:
        forcing = forcing.view(1, 1, *forcing.shape)
    elif forcing.ndim == 3:
        forcing = forcing.unsqueeze(1)
    if forcing.ndim != 4:
        raise ValueError(f"forcing must be scalar or BCHW-compatible, got shape={tuple(forcing.shape)}")
    if forcing.shape[0] == 1 and reference.shape[0] > 1:
        forcing = forcing.repeat(reference.shape[0], 1, 1, 1)
    if forcing.shape[1] != 1 or forcing.shape[0] != reference.shape[0] or forcing.shape[-2:] != reference.shape[-2:]:
        raise ValueError(f"forcing must match vorticity batch/spatial shape, got {tuple(forcing.shape)} vs {tuple(reference.shape)}")
    return forcing


def _validate_time_dependent_state(pde: str, q0: Any, qT: Any) -> None:
    expected_channels = {
        "heat": 1,
        "wave": 2,
        "advection_diffusion": 1,
        "reaction_diffusion": 2,
        "shallow_water": 3,
        "nsnonbounded": 1,
    }[pde]
    if q0.shape != qT.shape:
        raise ValueError(f"{pde} endpoint residual expects matching endpoint shapes, got {q0.shape} and {qT.shape}")
    if q0.ndim != 4 or q0.shape[1] != expected_channels:
        raise ValueError(f"{pde} expects BCHW endpoint states with {expected_channels} channels, got {q0.shape}")


def _apply_endpoint_residual_boundary(pde: str, residual: Any) -> Any:
    if pde in {"heat", "wave", "advection_diffusion"}:
        return _zero_boundary(residual)
    return residual


def _rhs_name(pde: str) -> str:
    return {
        "heat": "alpha_laplacian",
        "wave": "first_order_wave_u_t_v_v_t_c2_laplacian_u",
        "advection_diffusion": "advection_diffusion_rhs",
        "reaction_diffusion": "fitzhugh_nagumo_reaction_diffusion_rhs",
        "shallow_water": "standard_2d_conservative_shallow_water_flux_rhs",
        "nsnonbounded": "periodic_fft_vorticity_transport_rhs",
    }[pde]


def _rhs_param_usage(pde: str, params: dict[str, Any]) -> dict[str, bool]:
    names = {
        "heat": ("alpha", "T", "total_time", "dt", "hermite_collocation_times", "hermite_num_collocation"),
        "wave": ("c", "T", "total_time", "dt", "hermite_collocation_times", "hermite_num_collocation"),
        "advection_diffusion": (
            "b_x",
            "b_y",
            "kappa",
            "T",
            "total_time",
            "dt",
            "hermite_collocation_times",
            "hermite_num_collocation",
        ),
        "reaction_diffusion": (
            "D_u",
            "Du",
            "D_v",
            "Dv",
            "k",
            "T",
            "total_time",
            "dt",
            "x_range",
            "y_range",
            "x_left",
            "x_right",
            "y_bottom",
            "y_top",
            "dx",
            "dy",
            "hermite_collocation_times",
            "hermite_num_collocation",
        ),
        "shallow_water": ("g", "eps", "T", "total_time", "dt", "hermite_collocation_times", "hermite_num_collocation"),
        "nsnonbounded": ("nu", "viscosity", "forcing", "T", "total_time", "dt", "hermite_collocation_times", "hermite_num_collocation"),
    }[pde]
    return _used_params(params, names)


def _default_total_time(pde: str) -> float:
    return 1.0


def _hermite_settings(params: dict[str, Any]) -> dict[str, Any]:
    nested = params.get("hermite_bridge", {})
    return nested if isinstance(nested, dict) else {}


def _hermite_collocation_times(params: dict[str, Any]) -> list[float]:
    settings = _hermite_settings(params)
    raw_times = settings.get("collocation_times", params.get("hermite_collocation_times"))
    if raw_times is not None:
        times = _as_float_list(raw_times, "hermite_collocation_times")
    else:
        raw_num = settings.get("num_collocation", params.get("hermite_num_collocation"))
        if raw_num is None or int(raw_num) <= 0:
            times = [0.25, 0.5, 0.75]
        else:
            n = int(raw_num)
            times = [(idx + 1.0) / (n + 1.0) for idx in range(n)]
    if not times:
        raise ValueError("hermite_bridge requires at least one collocation time")
    bad = [value for value in times if not 0.0 < value < 1.0]
    if bad:
        raise ValueError(f"hermite_collocation_times must be inside (0, 1), got {bad}")
    return times


def _hermite_include_integral(params: dict[str, Any]) -> bool:
    settings = _hermite_settings(params)
    return _as_bool(settings.get("include_integral_residual", params.get("hermite_include_integral_residual", True)))


def _hermite_integral_weight(params: dict[str, Any]) -> float:
    settings = _hermite_settings(params)
    weight = float(settings.get("integral_weight", params.get("hermite_integral_weight", 1.0)))
    if weight < 0.0:
        raise ValueError("hermite_integral_weight must be non-negative")
    return weight


def _extract_near_endpoint_temporal_params(params: dict[str, Any]) -> dict[str, Any]:
    nested = params.get("near_endpoint_temporal", {})
    if isinstance(nested, dict):
        source = nested
        required = ("q_dt", "q_T_minus_dt", "dt", "mask_0", "mask_T")
        missing = [key for key in required if key not in source]
        if not missing:
            return dict(source)
    else:
        missing = []
    flat_map = {
        "q_dt": "near_q_dt",
        "q_T_minus_dt": "near_q_T_minus_dt",
        "dt": "near_dt",
        "mask_0": "near_mask_0",
        "mask_T": "near_mask_T",
    }
    flat_missing = [out_key for out_key, in_key in flat_map.items() if in_key not in params]
    if flat_missing:
        combined_missing = sorted(set(missing + flat_missing))
        raise ValueError(
            "near_endpoint_temporal mode requires extra near-endpoint sparse temporal observations and masks; "
            f"missing: {combined_missing}"
        )
    return {out_key: params[in_key] for out_key, in_key in flat_map.items()}


def _as_bchw_like(value: Any, reference: Any, name: str) -> Any:
    import torch

    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be BCHW-compatible, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] == 1 and reference.shape[0] > 1:
        tensor = tensor.repeat(reference.shape[0], 1, 1, 1)
    if tensor.shape != reference.shape:
        raise ValueError(f"{name} must match endpoint shape {tuple(reference.shape)}, got {tuple(tensor.shape)}")
    return tensor


def _as_mask_like(value: Any, reference: Any, name: str) -> Any:
    import torch

    mask = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"{name} must be mask BCHW-compatible, got shape {tuple(mask.shape)}")
    if mask.shape[0] == 1 and reference.shape[0] > 1:
        mask = mask.repeat(reference.shape[0], 1, 1, 1)
    if mask.shape[0] != reference.shape[0] or mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError(f"{name} must match endpoint batch/spatial shape {tuple(reference.shape)}, got {tuple(mask.shape)}")
    if mask.shape[1] not in {1, reference.shape[1]}:
        raise ValueError(f"{name} must have either one channel or {reference.shape[1]} channels, got {mask.shape[1]}")
    return mask


def _temporal_delta_field(value: Any, reference: Any) -> Any:
    import torch

    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.repeat(reference.shape[0])
    if tensor.numel() != reference.shape[0]:
        raise ValueError(f"near_endpoint_temporal dt must be scalar or batch length {reference.shape[0]}, got {tuple(tensor.shape)}")
    if bool(torch.any(tensor <= 0).detach().cpu()):
        raise ValueError("near_endpoint_temporal dt must be positive")
    return tensor.view(reference.shape[0], 1, 1, 1)


def _mask_normalization(mask: Any, residual: Any) -> tuple[Any, Any]:
    import torch

    b, c, h, w = residual.shape
    count = mask.reshape(b, -1).sum(dim=1).clamp_min(1.0)
    total = float(h * w if mask.shape[1] == 1 else c * h * w)
    scale = torch.sqrt(torch.full_like(count, total) / count).view(b, 1, 1, 1)
    return scale, count


def _disabled_residual(a: Any, u: Any, pde: str, residual_mode: str) -> ResidualOutput:
    import torch

    reference = u if hasattr(u, "shape") else a
    residual = torch.zeros_like(reference[:, :1])
    return _out(
        residual,
        "disabled",
        {
            "equation": f"{pde} residual disabled",
            "requested_residual_mode": residual_mode,
            "resolved_residual_mode": "disabled",
            "mode": "disabled",
        },
    )


def _steady_heat_conduction(a: Any, u: Any, pde_params: dict[str, Any]) -> ResidualOutput:
    if a.shape[1] != 1 or u.shape[1] != 1:
        raise ValueError(f"steady_heat_conduction expects 1+1 channels, got a={a.shape}, u={u.shape}")
    conductivity = (1.0 + 0.05 * (u - 298.0)).clamp_min(0.1)
    u_d = _param_field(pde_params, "u_D", u, default=298.0)
    residual = _steady_heat_residual_with_boundary(u, conductivity, a[:, :1], u_d)
    return _out(
        residual,
        "reliable",
        {
            "equation": "-div(lambda(u) grad u) - f, lambda(u)=max(1+0.05*(u-298),0.1)",
            "mode": "static_nonlinear_boundary",
            "resolved_residual_mode": "static_nonlinear_boundary",
            "residual_family": "static",
            "temporal_derivative_mode": "none",
            "endpoint_only": False,
            "boundary_condition": "bottom Dirichlet u=u_D; top/left/right zero Neumann",
            "boundary_enforced": True,
            "boundary_residual": {
                "included_in_field": True,
                "bottom": "u[..., 0, :] - u_D",
                "top": "(u[..., -1, :] - u[..., -2, :]) / dy",
                "left": "(u[..., 1:-1, 0] - u[..., 1:-1, 1]) / dx",
                "right": "(u[..., 1:-1, -1] - u[..., 1:-1, -2]) / dx",
            },
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


def _neumann_laplacian(u: Any, pde_params: dict[str, Any] | None = None) -> Any:
    import torch

    pde_params = pde_params or {}
    hx, hy = _rd_grid_spacing_fields(pde_params, u)
    padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode="replicate")
    lap_y = (padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1] - 2.0 * u) / (hy**2)
    lap_x = (padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:] - 2.0 * u) / (hx**2)
    return lap_x + lap_y


def _rd_grid_spacing_fields(pde_params: dict[str, Any], reference: Any) -> tuple[Any, Any]:
    hx = _param_field(pde_params, "dx", reference, default=0.0) if "dx" in pde_params else None
    hy = _param_field(pde_params, "dy", reference, default=0.0) if "dy" in pde_params else None
    x_left, x_right, _ = _rd_axis_bounds(pde_params, "x", reference)
    y_bottom, y_top, _ = _rd_axis_bounds(pde_params, "y", reference)
    if hx is None:
        hx = (x_right - x_left) / max(int(reference.shape[-1]), 1)
    if hy is None:
        hy = (y_top - y_bottom) / max(int(reference.shape[-2]), 1)
    return hx.abs().clamp_min(1e-12), hy.abs().clamp_min(1e-12)


def _rd_axis_bounds(pde_params: dict[str, Any], axis: str, reference: Any) -> tuple[Any, Any, str]:
    import torch

    if axis == "x":
        range_name, left_name, right_name = "x_range", "x_left", "x_right"
        default_left, default_right = -1.0, 1.0
    elif axis == "y":
        range_name, left_name, right_name = "y_range", "y_bottom", "y_top"
        default_left, default_right = -1.0, 1.0
    else:
        raise ValueError(f"Unknown axis={axis!r}")

    if left_name in pde_params and right_name in pde_params:
        return (
            _param_field(pde_params, left_name, reference, default=default_left),
            _param_field(pde_params, right_name, reference, default=default_right),
            f"{left_name}/{right_name}",
        )
    if range_name in pde_params:
        values = torch.as_tensor(pde_params[range_name], dtype=reference.dtype, device=reference.device)
        if values.ndim == 1 and values.numel() == 2:
            left = values[0].repeat(reference.shape[0])
            right = values[1].repeat(reference.shape[0])
        elif values.ndim >= 2 and values.shape[-1] == 2:
            flat = values.reshape(-1, 2)
            if flat.shape[0] == 1:
                flat = flat.repeat(reference.shape[0], 1)
            if flat.shape[0] != reference.shape[0]:
                raise ValueError(f"{range_name} must be length 2 or batch x 2, got shape={tuple(values.shape)}")
            left, right = flat[:, 0], flat[:, 1]
        else:
            raise ValueError(f"{range_name} must be length 2 or batch x 2, got shape={tuple(values.shape)}")
        return left.view(reference.shape[0], 1, 1, 1), right.view(reference.shape[0], 1, 1, 1), range_name
    left = torch.full((reference.shape[0], 1, 1, 1), default_left, dtype=reference.dtype, device=reference.device)
    right = torch.full((reference.shape[0], 1, 1, 1), default_right, dtype=reference.dtype, device=reference.device)
    return left, right, "default"


def _rhs_metadata(pde: str, pde_params: dict[str, Any], reference: Any) -> dict[str, Any]:
    if pde == "reaction_diffusion":
        return _reaction_diffusion_spatial_metadata(pde_params, reference)
    if pde == "shallow_water":
        return {"rhs": "standard_2d_conservative_shallow_water_flux_rhs"}
    if pde == "nsnonbounded":
        nu_source = "nu" if "nu" in pde_params else ("viscosity" if "viscosity" in pde_params else "default")
        forcing_present = "forcing" in pde_params
        return {
            "equation": "2D vorticity Navier-Stokes",
            "velocity_reconstruction": "periodic_fft_streamfunction",
            "boundary_assumption": "periodic",
            "mean_vorticity_handling": "zero_mean_projection",
            "assumptions": ["periodic_boundary", "zero_mean_vorticity_for_poisson_solve"],
            "nu_source": nu_source,
            "nu_defaulted": nu_source == "default",
            "forcing": "provided" if forcing_present else "zero",
            "forcing_source": "forcing" if forcing_present else None,
            "forcing_defaulted": not forcing_present,
        }
    return {}


def _reaction_diffusion_spatial_metadata(pde_params: dict[str, Any], reference: Any) -> dict[str, Any]:
    x_left, x_right, x_source = _rd_axis_bounds(pde_params, "x", reference)
    y_bottom, y_top, y_source = _rd_axis_bounds(pde_params, "y", reference)
    hx, hy = _rd_grid_spacing_fields(pde_params, reference)
    return {
        "boundary_condition": "homogeneous_neumann",
        "laplacian": "neumann",
        "domain": {
            "x": [_metadata_values(x_left), _metadata_values(x_right)],
            "y": [_metadata_values(y_bottom), _metadata_values(y_top)],
            "source": {"x": x_source, "y": y_source},
            "default": x_source == "default" and y_source == "default",
            "defaulted": x_source == "default" and y_source == "default",
        },
        "grid_spacing": {
            "dx": _metadata_values(hx),
            "dy": _metadata_values(hy),
            "source": {
                "dx": "dx" if "dx" in pde_params else "domain_width/num_cells",
                "dy": "dy" if "dy" in pde_params else "domain_height/num_cells",
            },
        },
    }


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


def _param_field_any(params: dict[str, Any], names: tuple[str, ...], reference: Any, default: float) -> Any:
    for name in names:
        if name in params:
            return _param_field(params, name, reference, default=default)
    return _param_field({}, names[0], reference, default=default)


def _time_scale_field(params: dict[str, Any], reference: Any, default: float = 1.0) -> tuple[Any, dict[str, Any]]:
    for name in ("T", "total_time", "dt"):
        if name in params:
            field = _param_field(params, name, reference, default=1.0)
            return field, {
                "source": name,
                "defaulted": False,
                "values": _metadata_values(field),
                "candidate_order": ["T", "total_time", "dt"],
            }
    field = _param_field({}, "T", reference, default=default)
    return field, {
        "source": "default",
        "defaulted": True,
        "value": default,
        "reason": "pde_params did not contain T, total_time, or dt",
        "candidate_order": ["T", "total_time", "dt"],
    }


def _float_param(params: dict[str, Any], name: str, default: float) -> float:
    import torch

    if name not in params:
        return float(default)
    value = torch.as_tensor(params[name]).detach().reshape(-1)
    if value.numel() < 1:
        return float(default)
    return float(value[0].cpu().item())


def _normalize_residual_mode(mode: str) -> str:
    aliases = {
        "": "auto",
        "default": "auto",
    }
    normalized = aliases.get(str(mode), str(mode))
    valid = {
        "auto",
        "hermite_bridge",
        "near_endpoint_temporal",
        "endpoint_secant",
        "full_trajectory_fd",
        "full_time_space",
        "disabled",
    }
    if normalized not in valid:
        raise ValueError(f"residual_mode={mode!r} is invalid; expected one of {sorted(valid)}")
    return normalized


def _as_float_list(value: Any, name: str) -> list[float]:
    import torch

    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"1", "true", "yes", "y", "on"}:
            return True
        if value.lower() in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _metadata_values(field: Any) -> Any:
    flat = field.detach().reshape(-1).cpu().tolist()
    if len(flat) == 1:
        return flat[0]
    return flat


def _steady_heat_residual_with_boundary(u: Any, conductivity: Any, source: Any, u_d: Any) -> Any:
    residual = _nonlinear_heat_conduction_residual(u, conductivity, source)
    h_size = int(u.shape[-2])
    w_size = int(u.shape[-1])
    if h_size <= 0 or w_size <= 0:
        return residual
    dx = 1.0 / max(w_size - 1, 1)
    dy = 1.0 / max(h_size - 1, 1)
    u_d_row = u_d[..., 0, 0].unsqueeze(-1)
    residual[..., 0, :] = u[..., 0, :] - u_d_row
    if h_size > 1:
        residual[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / dy
    if w_size > 1 and h_size > 2:
        residual[..., 1:-1, 0] = (u[..., 1:-1, 0] - u[..., 1:-1, 1]) / dx
        residual[..., 1:-1, -1] = (u[..., 1:-1, -1] - u[..., 1:-1, -2]) / dx
    return residual


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


def _apply_family_metadata(
    out: ResidualOutput,
    pde: str,
    resolved_mode: str,
    *,
    requested_mode: str | None = None,
) -> None:
    spec = get_pde_spec(pde)
    family = "full_trajectory" if resolved_mode == "full_trajectory_fd" else spec.residual_family
    out.metadata.setdefault("pde", pde)
    out.metadata.setdefault("residual_family", family)
    out.metadata.setdefault("requested_residual_mode", requested_mode if requested_mode is not None else resolved_mode)
    out.metadata.setdefault("resolved_residual_mode", resolved_mode)
    if family == "static":
        out.metadata.setdefault("temporal_derivative_mode", "none")
        out.metadata.setdefault("endpoint_only", False)
        out.metadata.setdefault("uses_generated_trajectory", False)
        out.metadata.setdefault("uses_extra_temporal_observations", False)
        out.metadata.setdefault("two_time_level_approx", False)
    elif family == "full_time_space":
        out.metadata.setdefault("temporal_derivative_mode", "full_fd")
        out.metadata.setdefault("endpoint_only", False)
        out.metadata.setdefault("uses_generated_trajectory", True)
        out.metadata.setdefault("uses_extra_temporal_observations", False)
        out.metadata.setdefault("two_time_level_approx", False)
    elif resolved_mode == "hermite_bridge":
        out.metadata.setdefault("temporal_derivative_mode", "hermite_bridge")
        out.metadata.setdefault("endpoint_only", True)
        out.metadata.setdefault("uses_generated_trajectory", False)
        out.metadata.setdefault("uses_extra_temporal_observations", False)
        out.metadata.setdefault("two_time_level_approx", False)
    elif resolved_mode == "near_endpoint_temporal":
        out.metadata.setdefault("temporal_derivative_mode", "near_endpoint_sparse_fd")
        out.metadata.setdefault("endpoint_only", False)
        out.metadata.setdefault("uses_generated_trajectory", False)
        out.metadata.setdefault("uses_extra_temporal_observations", True)
        out.metadata.setdefault("requires_extra_temporal_observations", True)
        out.metadata.setdefault("two_time_level_approx", False)
    elif resolved_mode == "endpoint_secant":
        out.metadata.setdefault("temporal_derivative_mode", "endpoint_secant")
        out.metadata.setdefault("endpoint_only", True)
        out.metadata.setdefault("uses_generated_trajectory", False)
        out.metadata.setdefault("uses_extra_temporal_observations", False)
        out.metadata.setdefault("two_time_level_approx", True)
    elif resolved_mode == "full_trajectory_fd":
        out.metadata.setdefault("temporal_derivative_mode", "full_fd")
        out.metadata.setdefault("endpoint_only", False)
        out.metadata.setdefault("uses_generated_trajectory", True)
        out.metadata.setdefault("uses_extra_temporal_observations", False)
        out.metadata.setdefault("two_time_level_approx", False)

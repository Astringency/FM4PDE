from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable

from data.specs import FULL_TIME_SPACE_PDES, STATIC_PDES, TEMPORAL_ENDPOINT_PDES, get_pde_spec

ENDPOINT_SECANT_WARNING = "coarse two-time-level endpoint residual; not a full spatiotemporal PDE residual"
TEMPORAL_ONLY_MODES = {"hermite_bridge", "near_endpoint_temporal", "endpoint_secant", "full_trajectory_fd"}


@dataclass
class ResidualOutput:
    residual: Any
    status: str
    metadata: dict[str, Any]
    components: dict[str, Any] | None = None


@dataclass
class BoundaryConditionSpec:
    kind: str
    value: Any | None = None
    sides: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    strict: bool = True


@dataclass
class InitialConditionSpec:
    kind: str
    value: Any | None = None
    source: str = "unknown"
    strict: bool = True


@dataclass
class PDEConstraintSpec:
    pde: str
    bc: BoundaryConditionSpec
    ic: InitialConditionSpec | None
    bc_weight: float
    ic_weight: float
    endpoint_weight: float
    normalize_by_mask: bool
    boundary_residual_normalization: str
    allow_unknown_bc: bool
    legacy_ignore_boundary: bool
    enforce_boundary_conditions: bool
    enforce_initial_conditions: bool


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
    constraint_spec = _constraint_spec_from_params(pde, pde_params)
    normalized_mode = _normalize_residual_mode(residual_mode)
    if pde in TEMPORAL_ENDPOINT_PDES:
        return _endpoint_time_dependent_residual(
            pde,
            coef,
            sol,
            pde_params=pde_params,
            constraint_spec=constraint_spec,
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
        "darcy": lambda a, u: _darcy(a, u, constraint_spec),
        "poisson": lambda a, u: _poisson(a, u, constraint_spec),
        "helmholtz": lambda a, u: _helmholtz(a, u, constraint_spec, k=k),
        "burger": lambda a, u: _burger(a, u, constraint_spec, pde_params=pde_params, residual_mode=residual_mode),
        "steady_heat_conduction": lambda a, u: _steady_heat_conduction(a, u, pde_params=pde_params, spec=constraint_spec),
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


def _darcy(a: Any, u: Any, spec: PDEConstraintSpec) -> ResidualOutput:
    import torch

    if a.shape[-2:] != u.shape[-2:] or a.shape[-2] != a.shape[-1]:
        raise ValueError(f"Darcy generator-aligned residual expects matching square fields, got a={a.shape}, u={u.shape}")
    cell_to_nodal, nodal_to_cell_inverse = _darcy_spline_matrices(int(u.shape[-1]), u.dtype, u.device)
    coefficient = cell_to_nodal @ a @ cell_to_nodal.T
    solution = nodal_to_cell_inverse @ u @ nodal_to_cell_inverse.T
    h = 1.0 / max(int(u.shape[-1]) - 1, 1)
    center_u = solution[..., 1:-1, 1:-1]
    center_a = coefficient[..., 1:-1, 1:-1]
    operator = torch.zeros_like(center_u)
    for neighbor_u, neighbor_a in (
        (solution[..., :-2, 1:-1], coefficient[..., :-2, 1:-1]),
        (solution[..., 2:, 1:-1], coefficient[..., 2:, 1:-1]),
        (solution[..., 1:-1, :-2], coefficient[..., 1:-1, :-2]),
        (solution[..., 1:-1, 2:], coefficient[..., 1:-1, 2:]),
    ):
        operator = operator + 0.5 * (center_a + neighbor_a) * (center_u - neighbor_u) / (h**2)
    interior = torch.zeros_like(solution)
    interior[..., 1:-1, 1:-1] = operator - 1.0
    out = _with_constraints(
        pde="darcy",
        state=solution,
        interior=interior,
        status="reliable",
        metadata={
            "equation": "-div(a grad u) - 1",
            "mode": "static",
            "discretization": "matlab_spline_nodal_conservative",
            "stored_grid": "cell_centered",
            "operator_grid": "nodal_endpoint_included",
        },
        spec=spec,
    )
    return out


def _poisson(a: Any, u: Any, spec: PDEConstraintSpec) -> ResidualOutput:
    interior = _interior_only(_laplacian(u) - a)
    return _with_constraints(
        pde="poisson",
        state=u,
        interior=interior,
        status="reliable",
        metadata={"equation": "laplace(u) - f", "mode": "static"},
        spec=spec,
    )


def _helmholtz(a: Any, u: Any, spec: PDEConstraintSpec, k: int = 1) -> ResidualOutput:
    import torch

    n = int(u.shape[-1])
    if u.shape[-2] != n:
        raise ValueError(f"Helmholtz generator-aligned residual expects square fields, got {u.shape}")
    h = 1.0 / max(n - 1, 1)
    one_d = torch.diag(torch.full((n,), -2.0, dtype=u.dtype, device=u.device))
    if n > 1:
        off = torch.ones(n - 1, dtype=u.dtype, device=u.device)
        one_d = one_d + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    one_d = one_d / (h**2)
    one_d[0] = 0.0
    one_d[0, 0] = 1.0
    one_d[-1] = 0.0
    one_d[-1, -1] = 1.0
    rhs = a.clone()
    rhs[..., 0, :] = 0.0
    rhs[..., -1, :] = 0.0
    rhs[..., :, 0] = 0.0
    rhs[..., :, -1] = 0.0
    interior = one_d @ u + u @ one_d.T + float(k**2) * u - rhs
    # The current MATLAB generator does not replace the complete 2-D boundary
    # rows. Its Kronecker system above is the authoritative boundary equation.
    generator_spec = replace(spec, enforce_boundary_conditions=False)
    return _with_constraints(
        pde="helmholtz",
        state=u,
        interior=interior,
        status="reliable",
        metadata={
            "equation": "(I kron L + L kron I + k^2 I)u - f_zero_boundary",
            "k": k,
            "mode": "static",
            "discretization": "matlab_kronecker_generator",
            "generator_boundary_note": "1-D boundary rows are modified before the Kronecker sum; this is not strict 2-D zero Dirichlet",
        },
        spec=generator_spec,
    )


def _burger(
    a: Any,
    u: Any,
    spec: PDEConstraintSpec,
    pde_params: dict[str, Any],
    residual_mode: str = "auto",
) -> ResidualOutput:
    import torch

    if u.ndim != 4 or u.shape[1] != 1 or u.shape[-2] < 3:
        raise ValueError(f"Burgers full trajectory must be [B,1,T,X] with T>=3, got {u.shape}")
    final_time = _float_param(pde_params, "T", _float_param(pde_params, "total_time", 1.0))
    dt = _float_param(pde_params, "trajectory_dt", final_time / max(int(u.shape[-2]) - 1, 1))
    domain_length = _float_param(pde_params, "domain_length", 1.0)
    dx = domain_length / max(int(u.shape[-1]), 1)
    nu = _float_param(pde_params, "nu", _float_param(pde_params, "viscosity", 0.01))
    state = u[:, :, 1:-1]
    ut = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dt)
    ux, uxx = _periodic_trajectory_x_derivatives(state, domain_length=domain_length)
    interior = ut + state * ux - nu * uxx
    return _with_constraints(
        pde="burger",
        state=u,
        interior=interior,
        status="reliable",
        metadata={
            "equation": "u_t + u u_x - nu u_xx",
            "nu": nu,
            "dt": dt,
            "dx": dx,
            "axis": "BCHW-as-time-space",
            "mode": "full_trajectory_fd",
            "requested_residual_mode": residual_mode,
            "resolved_residual_mode": "full_trajectory_fd",
            "full_trajectory_required": True,
            "residual_family": "full_time_space",
            "temporal_derivative_mode": "full_fd",
            "endpoint_only": False,
            "uses_generated_trajectory": True,
            "uses_extra_temporal_observations": False,
            "two_time_level_approx": False,
        },
        spec=spec,
        pde_params=pde_params,
    )


def _endpoint_time_dependent_residual(
    pde: str,
    a: Any,
    u: Any,
    *,
    pde_params: dict[str, Any],
    constraint_spec: PDEConstraintSpec,
    requested_mode: str,
    normalized_mode: str,
) -> ResidualOutput:
    resolved_mode = _resolve_endpoint_mode(normalized_mode)
    if resolved_mode == "disabled":
        return _disabled_residual(a, u, pde, requested_mode)
    if resolved_mode == "hermite_bridge":
        out = _hermite_bridge_residual(pde, a, u, pde_params, constraint_spec)
    elif resolved_mode == "near_endpoint_temporal":
        out = _near_endpoint_temporal_residual(pde, a, u, pde_params, constraint_spec)
    elif resolved_mode == "endpoint_secant":
        out = _endpoint_secant_residual(pde, a, u, pde_params, constraint_spec)
    elif resolved_mode == "full_trajectory_fd":
        out = _full_trajectory_fd_residual(pde, a, u, pde_params, constraint_spec)
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
    spec: PDEConstraintSpec,
) -> ResidualOutput:
    if pde == "heat":
        return _heat_endpoint_secant(a, u, pde_params, spec)
    if pde == "wave":
        return _wave_endpoint_secant(a, u, pde_params, spec)
    if pde == "advection_diffusion":
        return _advection_diffusion_endpoint_secant(a, u, pde_params, spec)
    if pde == "reaction_diffusion":
        return _reaction_diffusion_endpoint_secant(a, u, pde_params, spec)
    if pde == "shallow_water":
        return _shallow_water_endpoint_secant(a, u, pde_params, spec)
    if pde == "nsnonbounded":
        return _generic_endpoint_secant(pde, a, u, pde_params, spec)
    raise ValueError(f"endpoint_secant residual is not implemented for {pde!r}")


def _heat_endpoint_secant(a: Any, u: Any, pde_params: dict[str, Any], spec: PDEConstraintSpec) -> ResidualOutput:
    if a.shape[1] != 1 or u.shape[1] != 1:
        raise ValueError(f"heat expects 1+1 channels, got a={a.shape}, u={u.shape}")
    _required_param_field(pde_params, "alpha", u)
    time_scale, time_meta = _time_scale_field(pde_params, u)
    u_mid = 0.5 * (a + u)
    interior = (u - a) / time_scale - _rhs_heat(u_mid, pde_params)
    return _with_constraints(
        pde="heat",
        state=u_mid,
        interior=interior,
        status="approximate",
        metadata={
            "equation": "(uT - u0) / T - alpha * laplace(u_mid)",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _used_params(pde_params, ("alpha", "T", "total_time", "dt")),
            **_rhs_metadata("heat", pde_params, u_mid),
        },
        spec=spec,
        endpoint_states=[a, u, u_mid],
        initial_state=a,
        pde_params=pde_params,
    )


def _wave_endpoint_secant(a: Any, u: Any, pde_params: dict[str, Any], spec: PDEConstraintSpec) -> ResidualOutput:
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
    res_v = (v_t - v0) / time_scale - (c**2) * _laplacian_for_bc(u_mid, pde_params, _operator_boundary_kind("wave", pde_params))
    interior = torch.cat([res_u, res_v], dim=1)
    return _with_constraints(
        pde="wave",
        state=torch.cat([u_mid, v_mid], dim=1),
        interior=interior,
        status="approximate",
        metadata={
            "equation": "(uT - u0) / T - v_mid, (vT - v0) / T - c^2 laplace(u_mid)",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _used_params(pde_params, ("c", "T", "total_time", "dt")),
            **_rhs_metadata("wave", pde_params, u_mid),
        },
        spec=spec,
        endpoint_states=[a, u, torch.cat([u_mid, v_mid], dim=1)],
        initial_state=a,
        pde_params=pde_params,
    )


def _advection_diffusion_endpoint_secant(
    a: Any,
    u: Any,
    pde_params: dict[str, Any],
    spec: PDEConstraintSpec,
) -> ResidualOutput:
    if a.shape[1] != 1 or u.shape[1] != 1:
        raise ValueError(f"advection_diffusion expects 1+1 channels, got a={a.shape}, u={u.shape}")
    time_scale, time_meta = _time_scale_field(pde_params, u)
    u_mid = 0.5 * (a + u)
    interior = (u - a) / time_scale - _rhs_advection_diffusion(u_mid, pde_params)
    return _with_constraints(
        pde="advection_diffusion",
        state=u_mid,
        interior=interior,
        status="approximate",
        metadata={
            "equation": "(uT - u0) / T + b_x u_x + b_y u_y - kappa laplace(u_mid)",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _used_params(pde_params, ("b_x", "b_y", "kappa", "T", "total_time", "dt")),
            **_rhs_metadata("advection_diffusion", pde_params, u_mid),
        },
        spec=spec,
        endpoint_states=[a, u, u_mid],
        initial_state=a,
        pde_params=pde_params,
    )


def _reaction_diffusion_endpoint_secant(
    a: Any,
    u: Any,
    pde_params: dict[str, Any],
    spec: PDEConstraintSpec,
) -> ResidualOutput:
    import torch

    if a.shape[1] != 2 or u.shape[1] != 2:
        raise ValueError(f"reaction_diffusion expects 2+2 channels, got a={a.shape}, u={u.shape}")
    a_u, a_v = a[:, 0:1], a[:, 1:2]
    u_u, u_v = u[:, 0:1], u[:, 1:2]
    time_scale, time_meta = _time_scale_field(pde_params, u_u, default=_default_total_time("reaction_diffusion"))
    u_mid = 0.5 * (a_u + u_u)
    v_mid = 0.5 * (a_v + u_v)
    defaults = _reaction_diffusion_defaults(pde_params)
    d_u = _param_field_any(pde_params, ("D_u", "Du"), u_mid, default=defaults["D_u"])
    d_v = _param_field_any(pde_params, ("D_v", "Dv"), v_mid, default=defaults["D_v"])
    k = _param_field(pde_params, "k", u_mid, default=defaults["k"])
    u_t = (u_u - a_u) / time_scale
    v_t = (u_v - a_v) / time_scale
    lap_u = _reaction_diffusion_laplacian(u_mid, pde_params)
    lap_v = _reaction_diffusion_laplacian(v_mid, pde_params)
    res_u = u_t - (d_u * lap_u + u_mid - u_mid**3 - k - v_mid)
    res_v = v_t - (d_v * lap_v + u_mid - v_mid)
    interior = torch.cat([res_u, res_v], dim=1)
    return _with_constraints(
        pde="reaction_diffusion",
        state=torch.cat([u_mid, v_mid], dim=1),
        interior=interior,
        status="approximate",
        metadata={
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
            **_reaction_diffusion_spatial_metadata(pde_params, u_mid),
        },
        spec=spec,
        endpoint_states=[a, u, torch.cat([u_mid, v_mid], dim=1)],
        initial_state=a,
        pde_params=pde_params,
    )


def _shallow_water_endpoint_secant(
    a: Any,
    u: Any,
    pde_params: dict[str, Any],
    spec: PDEConstraintSpec,
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
    rhs_mid = _rhs_shallow_water(torch.cat([h_mid, hu_mid, hv_mid], dim=1), pde_params)
    mass = (h - h0) / time_scale - rhs_mid[:, 0:1]
    mom_x = (hu - hu0) / time_scale - rhs_mid[:, 1:2]
    mom_y = (hv - hv0) / time_scale - rhs_mid[:, 2:3]
    interior = torch.cat([mass, mom_x, mom_y], dim=1)
    return _with_constraints(
        pde="shallow_water",
        state=torch.cat([h_mid, hu_mid, hv_mid], dim=1),
        interior=interior,
        status="approximate",
        metadata={
            "equation": "two-time-level shallow-water conservative residual",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "variables": ["h", "hu", "hv"],
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _used_params(pde_params, ("g", "eps", "T", "total_time", "dt")),
            **_rhs_metadata("shallow_water", pde_params, h_mid),
        },
        spec=spec,
        endpoint_states=[a, u, torch.cat([h_mid, hu_mid, hv_mid], dim=1)],
        initial_state=a,
        pde_params=pde_params,
    )


def _generic_endpoint_secant(
    pde: str,
    a: Any,
    u: Any,
    pde_params: dict[str, Any],
    spec: PDEConstraintSpec,
) -> ResidualOutput:
    _validate_time_dependent_state(pde, a, u)
    time_scale, time_meta = _time_scale_field(pde_params, u, default=_default_total_time(pde))
    q_mid = 0.5 * (a + u)
    residual = (u - a) / time_scale - _rhs_time_dependent(pde, q_mid, pde_params)
    residual = _apply_endpoint_residual_boundary(pde, residual)
    out = _with_constraints(
        pde=pde,
        state=q_mid,
        interior=residual,
        status="approximate",
        metadata={
            "equation": f"{pde} endpoint secant residual",
            "mode": "endpoint_secant",
            "warning": ENDPOINT_SECANT_WARNING,
            "two_time_level_approx": True,
            "time_scale": time_meta,
            "pde_params_used": _rhs_param_usage(pde, pde_params),
            **_rhs_metadata(pde, pde_params, q_mid),
        },
        spec=spec,
        endpoint_states=[a, u, q_mid],
        initial_state=a,
        pde_params=pde_params,
    )
    return out


def _full_trajectory_fd_residual(pde: str, q0: Any, qT: Any, pde_params: dict[str, Any], spec: PDEConstraintSpec) -> ResidualOutput:
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
    expected_channels = 1 if pde == "wave" and (trajectory.shape[1] == 1 or trajectory.shape[2] == 1) else get_pde_spec(pde).coef_channels
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
    endpoint_shape = q0.shape[1:]
    expected_endpoint_shape = endpoint_shape if not (pde == "wave" and expected_channels == 1) else (1, *endpoint_shape[1:])
    if btchw.shape[0] != q0.shape[0] or btchw.shape[2:] != expected_endpoint_shape:
        raise ValueError(
            "full_trajectory_fd trajectory batch/channel/spatial shape must match endpoints; "
            f"trajectory={tuple(btchw.shape)}, q0={tuple(q0.shape)}"
        )
    dt, dt_meta = _trajectory_dt_field(pde_params, q0, int(btchw.shape[1]))
    if pde == "wave" and expected_channels == 1:
        state = btchw[:, 1:-1, 0:1]
        u_tt = (btchw[:, 2:, 0:1] - 2.0 * state + btchw[:, :-2, 0:1]) / (dt[:, None] ** 2)
        c = _param_field(pde_params, "c", q0[:, :1], default=1.0)
        flat = state.reshape(-1, 1, *state.shape[-2:])
        lap = _laplacian_for_bc(flat, pde_params, _operator_boundary_kind("wave", pde_params)).reshape_as(state)
        interior_btchw = u_tt - c[:, None] ** 2 * lap
        interior = interior_btchw.permute(0, 2, 1, 3, 4).reshape(q0.shape[0], -1, *q0.shape[-2:])
        state_for_constraints = state.mean(dim=1)
        endpoint_states = [btchw[:, 0], btchw[:, -1]]
        wave_state_form = "displacement_only_second_order"
    else:
        state = btchw[:, 1:-1]
        q_t = (btchw[:, 2:] - btchw[:, :-2]) / (2.0 * dt[:, None])
        flat_state = state.reshape(-1, *state.shape[2:])
        flat_rhs = _rhs_time_dependent(pde, flat_state, _repeat_batch_params(pde_params, int(state.shape[1])))
        rhs = flat_rhs.reshape_as(state)
        interior_btchw = q_t - rhs
        interior = interior_btchw.permute(0, 2, 1, 3, 4).reshape(q0.shape[0], -1, *q0.shape[-2:])
        state_for_constraints = state.mean(dim=1)
        endpoint_states = [btchw[:, idx] for idx in range(int(btchw.shape[1]))]
        wave_state_form = "first_order_state" if pde == "wave" else None
    out = _with_constraints(
        pde=pde,
        state=state_for_constraints,
        interior=interior,
        status="reliable",
        metadata={
            "equation": f"{pde} full trajectory finite-difference residual",
            "mode": "full_trajectory_fd",
            "resolved_residual_mode": "full_trajectory_fd",
            "trajectory_layout": "BTCHW",
            "trajectory_time_points": int(btchw.shape[1]),
            "trajectory_residual_time_points": int(btchw.shape[1]) - 2,
            "time_delta": dt_meta,
            "wave_state_form": wave_state_form,
            "rhs": _rhs_name(pde),
            "pde_params_used": _rhs_param_usage(pde, pde_params),
            **_rhs_metadata(pde, pde_params, btchw[:, 0]),
        },
        spec=spec,
        endpoint_states=endpoint_states,
        initial_state=btchw[:, 0],
        pde_params=pde_params,
    )
    observed = bool(pde_params.get("trajectory_is_observed_ground_truth", False))
    out.metadata["trajectory_role"] = "observed_ground_truth" if observed else "predicted_or_explicit_input"
    out.metadata["guidance_compatible"] = not observed
    return out


def _extract_full_trajectory(params: dict[str, Any]) -> Any:
    for name in ("full_trajectory", "trajectory"):
        if name in params:
            return params[name]
    raise ValueError("full_trajectory_fd mode requires explicit full trajectory state in pde_params['full_trajectory'] or pde_params['trajectory']")


def _constraint_spec_from_params(pde: str, params: dict[str, Any]) -> PDEConstraintSpec:
    bc_mode = str(params.get("boundary_condition_mode", params.get("boundary_condition", "auto")))
    ic_mode = str(params.get("initial_condition_mode", "auto"))
    legacy = _as_bool(params.get("legacy_ignore_boundary", False)) or bc_mode == "legacy_ignore"
    allow_unknown = _as_bool(params.get("allow_unknown_boundary_conditions", False))
    enforce_bc = _as_bool(params.get("enforce_boundary_conditions", True))
    enforce_ic = _as_bool(params.get("enforce_initial_conditions", True))
    bc = _resolve_boundary_spec(pde, bc_mode, params, legacy, allow_unknown)
    ic = _resolve_initial_spec(ic_mode, params)
    return PDEConstraintSpec(
        pde=pde,
        bc=bc,
        ic=ic,
        bc_weight=float(params.get("bc_weight", 1.0)),
        ic_weight=float(params.get("ic_weight", 1.0)),
        endpoint_weight=float(params.get("endpoint_bc_weight", params.get("lambda_endpoint", 1.0))),
        normalize_by_mask=str(params.get("boundary_residual_normalization", "sqrt_grid_over_mask")) != "mean",
        boundary_residual_normalization=str(params.get("boundary_residual_normalization", "sqrt_grid_over_mask")),
        allow_unknown_bc=allow_unknown,
        legacy_ignore_boundary=legacy,
        enforce_boundary_conditions=enforce_bc,
        enforce_initial_conditions=enforce_ic,
    )


def _resolve_boundary_spec(
    pde: str,
    mode: str,
    params: dict[str, Any],
    legacy: bool,
    allow_unknown: bool,
) -> BoundaryConditionSpec:
    if legacy:
        return BoundaryConditionSpec(kind="none", source="legacy_ignore", strict=False)
    aliases = {
        "dirichlet_zero": ("dirichlet", 0.0, "config"),
        "neumann_zero": ("neumann", 0.0, "config"),
        "periodic": ("periodic", None, "config"),
        "mixed": ("mixed", None, "config"),
        "none": ("none", None, "config"),
        "open": ("open", 0.0, "config"),
        "wall": ("wall", 0.0, "config"),
    }
    if mode in aliases:
        kind, value, source = aliases[mode]
        if pde == "burger" and kind == "periodic":
            kind = "periodic_x"
        sides = params.get("boundary_sides", {})
        return BoundaryConditionSpec(kind=kind, value=value, sides=dict(sides) if isinstance(sides, dict) else {}, source=source)
    if mode != "auto":
        raise ValueError(f"boundary_condition_mode={mode!r} is invalid")
    if "boundary_condition_kind" in params:
        kind = _normalize_boundary_kind(str(params["boundary_condition_kind"]), pde)
        return BoundaryConditionSpec(
            kind=kind,
            value=params.get("boundary_condition_value"),
            sides=dict(params.get("boundary_sides", {})) if isinstance(params.get("boundary_sides", {}), dict) else {},
            source="metadata",
        )
    defaults: dict[str, tuple[str, Any, str]] = {
        "darcy": ("dirichlet", 0.0, "default_confirmed_static_dataset"),
        "poisson": ("dirichlet", 0.0, "default_confirmed_static_dataset"),
        "helmholtz": ("dirichlet", 0.0, "default_confirmed_static_dataset"),
        "heat": ("periodic", None, "pair_h5_generator_default"),
        "wave": ("periodic", None, "pair_h5_generator"),
        "advection_diffusion": ("periodic", None, "pair_h5_generator"),
        "reaction_diffusion": ("neumann", 0.0, "reaction_diffusion_generator"),
        "shallow_water": ("open", 0.0, "clawpack_extrap_boundary"),
        "burger": ("periodic_x", None, "burgers_dataset_convention"),
        "steady_heat_conduction": ("mixed", None, "steady_heat_conduction_generator"),
        "nsnonbounded": ("periodic", None, "default_periodic_torus"),
    }
    if pde in defaults:
        kind, value, source = defaults[pde]
        sides = {}
        if pde == "steady_heat_conduction":
            sides = {
                "bottom": {"kind": "dirichlet", "value_param": "u_D"},
                "top": {"kind": "neumann", "value": 0.0},
                "left": {"kind": "neumann", "value": 0.0},
                "right": {"kind": "neumann", "value": 0.0},
            }
        return BoundaryConditionSpec(kind=kind, value=value, sides=sides, source=source, strict=True)
    if allow_unknown:
        return BoundaryConditionSpec(kind="unknown", source="unknown", strict=False)
    raise ValueError(f"Could not resolve boundary condition for PDE {pde!r}; set boundary_condition_mode or allow_unknown_boundary_conditions=true")


def _normalize_boundary_kind(kind: str, pde: str) -> str:
    text = kind.lower()
    if "periodic" in text:
        return "periodic_x" if pde == "burger" else "periodic"
    if "neumann" in text:
        return "neumann"
    if "dirichlet" in text:
        return "dirichlet"
    if "extrap" in text or "open" in text:
        return "open"
    if "wall" in text or "reflect" in text:
        return "wall"
    return text


def _resolve_initial_spec(mode: str, params: dict[str, Any]) -> InitialConditionSpec:
    if mode == "legacy_ignore":
        return InitialConditionSpec(kind="none", source="legacy_ignore", strict=False)
    if mode in {"none", "auto"}:
        if "observed_initial" in params:
            return InitialConditionSpec(kind="observed_initial", value=params["observed_initial"], source="pde_params.observed_initial")
        if "true_initial" in params and _as_bool(params.get("allow_true_initial_condition", False)):
            return InitialConditionSpec(kind="trajectory_initial", value=params["true_initial"], source="pde_params.true_initial")
        if mode == "none":
            return InitialConditionSpec(kind="none", source="config", strict=False)
        return InitialConditionSpec(kind="unknown", source="not_available", strict=False)
    if mode == "observed_initial":
        if "observed_initial" in params:
            return InitialConditionSpec(kind=mode, value=params["observed_initial"], source="pde_params.observed_initial")
        return InitialConditionSpec(kind="unknown", source=f"missing_{mode}", strict=True)
    if mode == "trajectory_initial":
        if "true_initial" in params:
            return InitialConditionSpec(kind=mode, value=params["true_initial"], source="pde_params.true_initial_explicit_extra_condition")
        return InitialConditionSpec(kind="unknown", source=f"missing_{mode}", strict=True)
    if mode == "endpoint_initial":
        if "observed_initial" in params:
            return InitialConditionSpec(kind=mode, value=params["observed_initial"], source="pde_params.observed_initial")
        if "true_initial" in params:
            return InitialConditionSpec(kind=mode, value=params["true_initial"], source="pde_params.true_initial_explicit_extra_condition")
        return InitialConditionSpec(kind="unknown", source=f"missing_{mode}", strict=True)
    raise ValueError(f"initial_condition_mode={mode!r} is invalid")


def _with_constraints(
    pde: str,
    state: Any,
    interior: Any,
    status: str,
    metadata: dict[str, Any],
    spec: PDEConstraintSpec,
    *,
    bc_states: list[Any] | None = None,
    endpoint_states: list[Any] | None = None,
    initial_state: Any | None = None,
    endpoint_component: Any | None = None,
    pde_params: dict[str, Any] | None = None,
) -> ResidualOutput:
    import torch

    pde_params = pde_params or {}
    if spec.legacy_ignore_boundary:
        residual = _append_constraint_residuals(
            interior,
            None,
            None,
            endpoint_component,
            bc_weight=0.0,
            ic_weight=0.0,
            endpoint_weight=spec.endpoint_weight,
        )
        components = {"interior": interior, "boundary": None, "initial": None, "endpoint": endpoint_component}
        out = _out(residual, status, metadata, components=components)
        _constraint_metadata(
            out,
            spec,
            interior,
            None,
            None,
            endpoint_component,
            pde=pde,
            pde_params=pde_params,
            legacy_used=True,
            unresolved=[],
        )
        return out
    bc = None
    unresolved = []
    if spec.enforce_boundary_conditions:
        sources = bc_states if bc_states is not None else (endpoint_states if endpoint_states is not None else [state])
        bc_parts = [
            part
            for part in (_compute_boundary_residual(pde, q, spec.bc, spec.boundary_residual_normalization, pde_params) for q in sources)
            if part is not None
        ]
        if bc_parts:
            bc = torch.cat(bc_parts, dim=1)
    ic = None
    if spec.enforce_initial_conditions and spec.ic is not None:
        ic_source_state = initial_state
        if ic_source_state is None and endpoint_states:
            ic_source_state = endpoint_states[0]
        if ic_source_state is None:
            ic_source_state = state
        ic = _compute_initial_residual(ic_source_state, spec.ic, pde_params)
        if ic is None and spec.ic.kind == "unknown":
            unresolved.append("initial_condition")
    residual = _append_constraint_residuals(
        interior,
        bc,
        ic,
        endpoint_component,
        bc_weight=spec.bc_weight,
        ic_weight=spec.ic_weight,
        endpoint_weight=spec.endpoint_weight,
    )
    components = {"interior": interior, "boundary": bc, "initial": ic, "endpoint": endpoint_component}
    out = _out(residual, status, metadata, components=components)
    _constraint_metadata(
        out,
        spec,
        interior,
        bc,
        ic,
        endpoint_component,
        pde=pde,
        pde_params=pde_params,
        legacy_used=False,
        unresolved=unresolved,
    )
    return out


def _append_constraint_residuals(
    interior: Any,
    bc: Any | None,
    ic: Any | None,
    endpoint: Any | None,
    *,
    bc_weight: float,
    ic_weight: float,
    endpoint_weight: float,
) -> Any:
    import torch

    parts = [interior]
    if bc is not None and bc_weight > 0.0:
        parts.append(torch.as_tensor(bc_weight, dtype=interior.dtype, device=interior.device).sqrt() * bc)
    if ic is not None and ic_weight > 0.0:
        parts.append(torch.as_tensor(ic_weight, dtype=interior.dtype, device=interior.device).sqrt() * ic)
    if endpoint is not None and endpoint_weight > 0.0:
        parts.append(torch.as_tensor(endpoint_weight, dtype=interior.dtype, device=interior.device).sqrt() * endpoint)
    return torch.cat(parts, dim=1)


def _constraint_metadata(
    out: ResidualOutput,
    spec: PDEConstraintSpec,
    interior: Any,
    bc: Any | None,
    ic: Any | None,
    endpoint: Any | None,
    *,
    pde: str,
    pde_params: dict[str, Any],
    legacy_used: bool,
    unresolved: list[str],
) -> None:
    periodic_operator = spec.bc.kind in {"periodic", "periodic_x"}
    ghost_cell_operator = pde in {"reaction_diffusion", "shallow_water"} and spec.bc.kind in {"neumann", "open"}
    operator_encoded_boundary = periodic_operator or ghost_cell_operator
    boundary_enforced = bc is not None or (spec.enforce_boundary_conditions and operator_encoded_boundary)
    out.metadata.update(
        {
            "interior_residual_enabled": True,
            "bc_residual_enabled": bc is not None,
            "ic_residual_enabled": ic is not None,
            "endpoint_residual_enabled": endpoint is not None,
            "boundary_condition_type": spec.bc.kind,
            "boundary_condition_source": spec.bc.source,
            "initial_condition_type": spec.ic.kind if spec.ic is not None else "none",
            "initial_condition_source": spec.ic.source if spec.ic is not None else "none",
            "hard_coded_defaults": [spec.bc.source] if "default" in spec.bc.source else [],
            "unresolved_conditions": unresolved + (["boundary_condition"] if spec.bc.kind == "unknown" else []),
            "legacy_ignore_boundary": spec.legacy_ignore_boundary,
            "legacy_boundary_ignored": legacy_used,
            "boundary_enforced": boundary_enforced,
            "boundary_enforced_by_operator": bool(spec.enforce_boundary_conditions and operator_encoded_boundary and bc is None),
            "boundary_value_residual_applicable": not operator_encoded_boundary,
            "grid_convention": "endpoint_false_periodic" if periodic_operator else ("cell_centered_ghost_cells" if ghost_cell_operator else "closed_interval_or_metadata"),
            "initial_enforced": ic is not None,
            "boundary_residual_normalization": spec.boundary_residual_normalization,
            "residual_channels": {
                "interior_channels": int(interior.shape[1]),
                "bc_channels": int(bc.shape[1]) if bc is not None else 0,
                "ic_channels": int(ic.shape[1]) if ic is not None else 0,
                "endpoint_channels": int(endpoint.shape[1]) if endpoint is not None else 0,
                "total_channels": int(out.residual.shape[1]),
            },
        }
    )
    if bc is not None and spec.bc.kind in {"neumann", "open", "mixed", "wall"}:
        dx, dy, spacing_source = _boundary_grid_spacing_info(interior, pde_params, pde=pde, bc_kind=spec.bc.kind)
        out.metadata.update(
            {
                "boundary_residual_spacing_source": spacing_source,
                "boundary_residual_dx": _metadata_values(dx),
                "boundary_residual_dy": _metadata_values(dy),
            }
        )


def _compute_boundary_residual(
    pde: str,
    q: Any,
    bc: BoundaryConditionSpec,
    normalization: str,
    pde_params: dict[str, Any],
) -> Any | None:
    # RD and SWE store finite-volume cell centres. Their Neumann/extrapolation
    # conditions define ghost cells used by the spatial operator; equality of
    # adjacent physical cells is not a boundary condition.
    if pde in {"reaction_diffusion", "shallow_water"} and bc.kind in {"neumann", "open"}:
        return None
    if bc.kind in {"none", "unknown"}:
        return None
    if bc.kind == "dirichlet":
        return _dirichlet_residual(q, bc.value if bc.value is not None else 0.0, bc.sides, normalization)
    if bc.kind == "neumann":
        return _neumann_residual(q, bc.value if bc.value is not None else 0.0, bc.sides, normalization, pde_params=pde_params, pde=pde)
    if bc.kind == "periodic":
        return _periodic_residual(q, axes=("x", "y"), include_derivative_continuity=True, normalization=normalization, pde_params=pde_params)
    if bc.kind == "periodic_x":
        return _periodic_residual(q, axes=("x",), include_derivative_continuity=True, normalization=normalization, pde_params=pde_params)
    if bc.kind == "mixed":
        return _mixed_boundary_residual(pde, q, bc, normalization, pde_params)
    if bc.kind == "wall":
        return _wall_residual(q, normalization, pde_params=pde_params, pde=pde)
    if bc.kind == "open":
        return _neumann_residual(q, 0.0, {}, normalization, pde_params=pde_params, pde=pde)
    raise ValueError(f"Unsupported boundary condition kind={bc.kind!r}")


def _compute_initial_residual(q0: Any, ic: InitialConditionSpec, pde_params: dict[str, Any]) -> Any | None:
    if ic.kind in {"none", "unknown"} or ic.value is None:
        return None
    target = _as_bchw_like(ic.value, q0, ic.source)
    residual = q0 - target
    if "initial_mask" in pde_params:
        mask = _as_mask_like(pde_params["initial_mask"], q0, "initial_mask")
        residual = residual * mask
        return _normalize_masked_residual(residual, mask, "sqrt_grid_over_mask")
    return residual


def _trajectory_dt_field(params: dict[str, Any], reference: Any, n_time: int) -> tuple[Any, dict[str, Any]]:
    import torch

    if "trajectory_dt" in params:
        dt = _param_field(params, "trajectory_dt", reference, default=1.0)
        return dt, {"source": "trajectory_dt", "defaulted": False, "values": _metadata_values(dt)}
    if "trajectory_time_values" in params:
        values = torch.as_tensor(params["trajectory_time_values"], dtype=reference.dtype, device=reference.device)
        if values.ndim != 1 or values.numel() != n_time:
            raise ValueError(f"trajectory_time_values must contain {n_time} entries, got shape {tuple(values.shape)}")
        deltas = values[1:] - values[:-1]
        if not torch.allclose(deltas, deltas[:1].expand_as(deltas), rtol=1e-5, atol=1e-8):
            raise ValueError("full_trajectory_fd currently requires uniformly spaced trajectory_time_values")
        dt = torch.full((reference.shape[0], 1, 1, 1), float(deltas[0]), dtype=reference.dtype, device=reference.device)
        return dt, {"source": "trajectory_time_values", "defaulted": False, "values": _metadata_values(dt)}
    for name in ("T", "total_time"):
        if name in params:
            total = _param_field(params, name, reference, default=1.0)
            dt = total / float(max(n_time - 1, 1))
            return dt, {"source": f"{name}/(n_time-1)", "defaulted": False, "values": _metadata_values(dt)}
    for name in ("saved_dt", "dt"):
        if name in params:
            dt = _param_field(params, name, reference, default=1.0)
            return dt, {"source": name, "defaulted": False, "values": _metadata_values(dt)}
    dt = torch.full((reference.shape[0], 1, 1, 1), 1.0 / float(max(n_time - 1, 1)), dtype=reference.dtype, device=reference.device)
    return dt, {"source": "default_unit_interval/(n_time-1)", "defaulted": True, "value": _metadata_values(dt)}


def _hermite_bridge_residual(pde: str, q0: Any, qT: Any, pde_params: dict[str, Any], spec: PDEConstraintSpec) -> ResidualOutput:
    import torch

    _validate_time_dependent_state(pde, q0, qT)
    time_scale, time_meta = _time_scale_field(pde_params, q0, default=_default_total_time(pde))
    f0 = _rhs_time_dependent(pde, q0, pde_params)
    fT = _rhs_time_dependent(pde, qT, pde_params)
    collocation_times = _hermite_collocation_times(pde_params)
    residuals = []
    collocation_states = []
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
        collocation_states.append(h)
        residuals.append(dh_ds / time_scale - _rhs_time_dependent(pde, h, pde_params))

    include_integral = _hermite_include_integral(pde_params)
    integral_weight = _hermite_integral_weight(pde_params)
    endpoint_component = None
    if include_integral:
        weight = torch.as_tensor(integral_weight, dtype=q0.dtype, device=q0.device).sqrt()
        integral_residual = qT - q0 - 0.5 * time_scale * (f0 + fT)
        endpoint_component = weight * integral_residual
    interior = torch.cat(residuals, dim=1)
    equation = (
        "2D vorticity Navier-Stokes endpoint Hermite bridge residual"
        if pde == "nsnonbounded"
        else f"{pde} endpoint-induced cubic Hermite bridge residual"
    )
    return _with_constraints(
        pde=pde,
        state=qT,
        interior=interior,
        status="approximate",
        metadata={
            "equation": equation,
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
            "residual_channels": int(interior.shape[1]),
            "state_channels": int(q0.shape[1]),
            "pde_params_used": _rhs_param_usage(pde, pde_params),
            **_rhs_metadata(pde, pde_params, q0),
        },
        spec=spec,
        bc_states=[q0, *collocation_states, qT],
        endpoint_states=[q0, qT],
        initial_state=q0,
        endpoint_component=endpoint_component,
        pde_params=pde_params,
    )


def _near_endpoint_temporal_residual(pde: str, q0: Any, qT: Any, pde_params: dict[str, Any], spec: PDEConstraintSpec) -> ResidualOutput:
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
    r0 = td0 - _rhs_time_dependent(pde, q0, pde_params)
    rT = tdT - _rhs_time_dependent(pde, qT, pde_params)
    scale0, count0 = _mask_normalization(mask_0, r0)
    scaleT, countT = _mask_normalization(mask_T, rT)
    r0_masked = r0 * mask_0 * scale0
    rT_masked = rT * mask_T * scaleT
    interior = torch.cat([r0_masked, rT_masked], dim=1)
    source_meta = near.get("metadata", {})
    return _with_constraints(
        pde=pde,
        state=qT,
        interior=interior,
        status="approximate",
        metadata={
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
        spec=spec,
        bc_states=[q0, qT],
        endpoint_states=[q0, qT],
        initial_state=q0,
        pde_params=pde_params,
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


def _repeat_batch_params(params: dict[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat sample-wise parameters to match flattened trajectory states."""
    import torch

    repeated: dict[str, Any] = {}
    for name, value in params.items():
        if name in {"trajectory", "full_trajectory", "trajectory_time_values"}:
            continue
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] > 1:
            repeated[name] = value.repeat_interleave(repeats, dim=0)
        else:
            repeated[name] = value
    return repeated


def _rhs_heat(q: Any, pde_params: dict[str, Any]) -> Any:
    alpha = _required_param_field(pde_params, "alpha", q)
    return alpha * _laplacian_for_bc(q, pde_params, _operator_boundary_kind("heat", pde_params))


def _rhs_wave(q: Any, pde_params: dict[str, Any]) -> Any:
    import torch

    if q.shape[1] != 2:
        raise ValueError(f"wave RHS expects [u,v] channels, got {q.shape}")
    c = _param_field(pde_params, "c", q[:, 0:1], default=1.0)
    u, v = q[:, 0:1], q[:, 1:2]
    return torch.cat([v, (c**2) * _laplacian_for_bc(u, pde_params, _operator_boundary_kind("wave", pde_params))], dim=1)


def _rhs_advection_diffusion(q: Any, pde_params: dict[str, Any]) -> Any:
    bx = _required_param_field(pde_params, "b_x", q)
    by = _required_param_field(pde_params, "b_y", q)
    kappa = _required_param_field(pde_params, "kappa", q)
    bc_kind = _operator_boundary_kind("advection_diffusion", pde_params)
    return -bx * _dx_for_bc(q, pde_params, bc_kind) - by * _dy_for_bc(q, pde_params, bc_kind) + kappa * _laplacian_for_bc(q, pde_params, bc_kind)


def _rhs_reaction_diffusion(q: Any, pde_params: dict[str, Any]) -> Any:
    import torch

    if q.shape[1] != 2:
        raise ValueError(f"reaction_diffusion RHS expects [u,v] channels, got {q.shape}")
    u, v = q[:, 0:1], q[:, 1:2]
    defaults = _reaction_diffusion_defaults(pde_params)
    d_u = _param_field_any(pde_params, ("D_u", "Du"), u, default=defaults["D_u"])
    d_v = _param_field_any(pde_params, ("D_v", "Dv"), v, default=defaults["D_v"])
    k = _param_field(pde_params, "k", u, default=defaults["k"])
    f_u = d_u * _reaction_diffusion_laplacian(u, pde_params) + u - u**3 - k - v
    f_v = d_v * _reaction_diffusion_laplacian(v, pde_params) + u - v
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
    bc_kind = _operator_boundary_kind("shallow_water", pde_params)
    return -_dx_for_bc(flux_x, pde_params, bc_kind) - _dy_for_bc(flux_y, pde_params, bc_kind)


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
    forcing = _forcing_field(pde_params, w) if "forcing" in pde_params else _ns_default_forcing(w)
    result = -u_vel * w_x - v_vel * w_y + nu * lap_w + forcing
    # Numerical protection: clamp to prevent NaN/Inf propagation in Hermite bridge
    result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    result = torch.clamp(result, -1e6, 1e6)
    return result


def _periodic_stream_function_fft(w: Any) -> tuple[Any, Any, Any, Any]:
    import torch

    h, width = int(w.shape[-2]), int(w.shape[-1])
    # Generator arrays are [x,y]: the first stored spatial axis is x.
    kx_1d = 2.0 * torch.pi * torch.fft.fftfreq(h, d=1.0 / max(h, 1), device=w.device, dtype=w.dtype)
    ky_1d = 2.0 * torch.pi * torch.fft.fftfreq(width, d=1.0 / max(width, 1), device=w.device, dtype=w.dtype)
    kx = kx_1d.view(1, 1, h, 1)
    ky = ky_1d.view(1, 1, 1, width)
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


def _ns_default_forcing(reference: Any) -> Any:
    import torch

    h, width = int(reference.shape[-2]), int(reference.shape[-1])
    x = torch.arange(h, dtype=reference.dtype, device=reference.device).view(1, 1, h, 1) / max(h, 1)
    y = torch.arange(width, dtype=reference.dtype, device=reference.device).view(1, 1, 1, width) / max(width, 1)
    forcing = 0.1 * (torch.sin(2.0 * torch.pi * (x + y)) + torch.cos(2.0 * torch.pi * (x + y)))
    return forcing.expand(reference.shape[0], 1, h, width)


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
    del pde
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


def _steady_heat_conduction(a: Any, u: Any, pde_params: dict[str, Any], spec: PDEConstraintSpec) -> ResidualOutput:
    if a.shape[1] != 1 or u.shape[1] != 1:
        raise ValueError(f"steady_heat_conduction expects 1+1 channels, got a={a.shape}, u={u.shape}")
    conductivity = (1.0 + 0.05 * (u - 298.0)).clamp_min(0.1)
    u_d = _required_param_field(pde_params, "u_D", u)
    interior = _interior_only(_nonlinear_heat_conduction_residual(u, conductivity, a[:, :1]))
    return _with_constraints(
        pde="steady_heat_conduction",
        state=u,
        interior=interior,
        status="reliable",
        metadata={
            "equation": "-div(lambda(u) grad u) - f, lambda(u)=max(1+0.05*(u-298),0.1)",
            "mode": "static_nonlinear_boundary",
            "resolved_residual_mode": "static_nonlinear_boundary",
            "residual_family": "static",
            "temporal_derivative_mode": "none",
            "endpoint_only": False,
            "boundary_condition": "bottom Dirichlet u=u_D; top/left/right zero Neumann",
            "boundary_enforced": True,
            "boundary_condition_type": "mixed",
            "boundary_residual": {
                "included_in_field": True,
                "bottom": "u[..., 0, :] - u_D",
                "top_interior": "u[..., -1, 1:-1] - u[..., -2, 1:-1]",
                "left_after_bottom": "u[..., 1:, 0] - u[..., 1:, 1]",
                "right_after_bottom": "u[..., 1:, -1] - u[..., 1:, -2]",
                "row_precedence": "bottom, sides, top interior",
            },
            "pde_params_used": _used_params(pde_params, ("u_D",)),
        },
        spec=spec,
        pde_params=pde_params,
    )


@lru_cache(maxsize=16)
def _darcy_spline_matrices_numpy(resolution: int) -> tuple[Any, Any]:
    import numpy as np
    from scipy.interpolate import CubicSpline

    cell = (np.arange(resolution, dtype=np.float64) + 0.5) / resolution
    nodal = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    identity = np.eye(resolution, dtype=np.float64)
    cell_to_nodal = CubicSpline(cell, identity, axis=0)(nodal)
    nodal_to_cell = CubicSpline(nodal, identity, axis=0)(cell)
    return cell_to_nodal, np.linalg.inv(nodal_to_cell)


def _darcy_spline_matrices(resolution: int, dtype: Any, device: Any) -> tuple[Any, Any]:
    import torch

    cell_to_nodal, nodal_to_cell_inverse = _darcy_spline_matrices_numpy(resolution)
    return (
        torch.as_tensor(cell_to_nodal, dtype=dtype, device=device),
        torch.as_tensor(nodal_to_cell_inverse, dtype=dtype, device=device),
    )


def _periodic_trajectory_x_derivatives(state: Any, domain_length: float) -> tuple[Any, Any]:
    import torch

    n = int(state.shape[-1])
    modes = torch.fft.fftfreq(n, d=domain_length / max(n, 1), device=state.device, dtype=state.dtype)
    wave_number = 2.0 * torch.pi * modes.view(1, 1, 1, n)
    state_hat = torch.fft.fft(state, dim=-1)
    first = torch.fft.ifft(1j * wave_number * state_hat, dim=-1).real
    second = torch.fft.ifft(-(wave_number**2) * state_hat, dim=-1).real
    return first, second


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


def _periodic_grid_spacing(reference: Any, pde_params: dict[str, Any] | None = None) -> tuple[Any, Any]:
    import torch

    pde_params = pde_params or {}
    if "dx" in pde_params:
        dx = _param_field(pde_params, "dx", reference, default=1.0)
    else:
        domain_x = _float_param(pde_params, "domain_length_x", _float_param(pde_params, "domain_length", 1.0))
        dx = torch.full((reference.shape[0], 1, 1, 1), domain_x / max(int(reference.shape[-2]), 1), dtype=reference.dtype, device=reference.device)
    if "dy" in pde_params:
        dy = _param_field(pde_params, "dy", reference, default=1.0)
    else:
        domain_y = _float_param(pde_params, "domain_length_y", _float_param(pde_params, "domain_length", 1.0))
        dy = torch.full((reference.shape[0], 1, 1, 1), domain_y / max(int(reference.shape[-1]), 1), dtype=reference.dtype, device=reference.device)
    return dx.abs().clamp_min(1e-12), dy.abs().clamp_min(1e-12)


def _closed_interval_grid_spacing(reference: Any, pde_params: dict[str, Any] | None = None) -> tuple[Any, Any]:
    import torch

    pde_params = pde_params or {}
    if "dx" in pde_params:
        dx = _param_field(pde_params, "dx", reference, default=1.0)
    else:
        dx = torch.full((reference.shape[0], 1, 1, 1), 1.0 / max(int(reference.shape[-2]) - 1, 1), dtype=reference.dtype, device=reference.device)
    if "dy" in pde_params:
        dy = _param_field(pde_params, "dy", reference, default=1.0)
    else:
        dy = torch.full((reference.shape[0], 1, 1, 1), 1.0 / max(int(reference.shape[-1]) - 1, 1), dtype=reference.dtype, device=reference.device)
    return dx.abs().clamp_min(1e-12), dy.abs().clamp_min(1e-12)


def _grid_spacing_for_bc(reference: Any, pde_params: dict[str, Any] | None, bc_kind: str) -> tuple[Any, Any]:
    if bc_kind in {"periodic", "periodic_x"}:
        return _periodic_grid_spacing(reference, pde_params)
    return _closed_interval_grid_spacing(reference, pde_params)


def _boundary_grid_spacing(
    reference: Any,
    pde_params: dict[str, Any] | None = None,
    pde: str | None = None,
    bc_kind: str = "neumann",
) -> tuple[Any, Any]:
    dx, dy, _ = _boundary_grid_spacing_info(reference, pde_params, pde=pde, bc_kind=bc_kind)
    return dx, dy


def _boundary_grid_spacing_info(
    reference: Any,
    pde_params: dict[str, Any] | None = None,
    pde: str | None = None,
    bc_kind: str = "neumann",
) -> tuple[Any, Any, str]:
    pde_params = pde_params or {}
    if bc_kind in {"periodic", "periodic_x"}:
        raise ValueError("periodic boundary residuals must use periodic spacing helpers")
    if "dx" in pde_params or "dy" in pde_params:
        dx, dy = _closed_interval_grid_spacing(reference, pde_params)
        return dx, dy, "pde_params_dx_dy"
    if pde == "reaction_diffusion":
        dx, dy = _rd_grid_spacing_fields(pde_params, reference)
        return dx, dy, "reaction_diffusion_domain_metadata"
    dx, dy = _closed_interval_grid_spacing(reference, pde_params)
    return dx, dy, "closed_interval_default"


def _periodic_dx(f: Any, dx: Any | None = None) -> Any:
    import torch

    if dx is None:
        dx, _ = _periodic_grid_spacing(f, None)
    n = int(f.shape[-2])
    modes = torch.fft.fftfreq(n, d=1.0 / max(n, 1), device=f.device, dtype=f.dtype).view(1, 1, n, 1)
    wave_number = 2.0 * torch.pi * modes / (dx * n)
    return torch.fft.ifft2(1j * wave_number * torch.fft.fft2(f, dim=(-2, -1)), dim=(-2, -1)).real


def _periodic_dy(f: Any, dy: Any | None = None) -> Any:
    import torch

    if dy is None:
        _, dy = _periodic_grid_spacing(f, None)
    n = int(f.shape[-1])
    modes = torch.fft.fftfreq(n, d=1.0 / max(n, 1), device=f.device, dtype=f.dtype).view(1, 1, 1, n)
    wave_number = 2.0 * torch.pi * modes / (dy * n)
    return torch.fft.ifft2(1j * wave_number * torch.fft.fft2(f, dim=(-2, -1)), dim=(-2, -1)).real


def _periodic_laplacian_2d(f: Any, dx: Any | None = None, dy: Any | None = None) -> Any:
    import torch

    if dx is None or dy is None:
        dx_default, dy_default = _periodic_grid_spacing(f, None)
        dx = dx_default if dx is None else dx
        dy = dy_default if dy is None else dy
    nx, ny = int(f.shape[-2]), int(f.shape[-1])
    mode_x = torch.fft.fftfreq(nx, d=1.0 / max(nx, 1), device=f.device, dtype=f.dtype).view(1, 1, nx, 1)
    mode_y = torch.fft.fftfreq(ny, d=1.0 / max(ny, 1), device=f.device, dtype=f.dtype).view(1, 1, 1, ny)
    kx = 2.0 * torch.pi * mode_x / (dx * nx)
    ky = 2.0 * torch.pi * mode_y / (dy * ny)
    return torch.fft.ifft2(-(kx**2 + ky**2) * torch.fft.fft2(f, dim=(-2, -1)), dim=(-2, -1)).real


def _periodic_laplacian(u: Any) -> Any:
    return _periodic_laplacian_2d(u)


def _dirichlet_laplacian(u: Any, pde_params: dict[str, Any] | None = None, boundary_value: float = 0.0) -> Any:
    import torch

    dx, dy = _closed_interval_grid_spacing(u, pde_params)
    padded = torch.nn.functional.pad(u, (1, 1, 1, 1), "constant", float(boundary_value))
    lap_x = (padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1] - 2.0 * u) / (dx**2)
    lap_y = (padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:] - 2.0 * u) / (dy**2)
    return lap_x + lap_y


def _neumann_laplacian(u: Any, pde_params: dict[str, Any] | None = None) -> Any:
    import torch

    pde_params = pde_params or {}
    if any(name in pde_params for name in ("x_range", "y_range", "x_left", "x_right", "y_bottom", "y_top")):
        hx, hy = _rd_grid_spacing_fields(pde_params, u)
    else:
        hx, hy = _closed_interval_grid_spacing(u, pde_params)
    padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode="replicate")
    lap_x = (padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1] - 2.0 * u) / (hx**2)
    lap_y = (padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:] - 2.0 * u) / (hy**2)
    return lap_x + lap_y


def _reaction_diffusion_laplacian(u: Any, pde_params: dict[str, Any] | None = None) -> Any:
    import torch

    pde_params = pde_params or {}
    hx, hy = _rd_grid_spacing_fields(pde_params, u)
    padded = torch.nn.functional.pad(u, (1, 1, 1, 1), mode="replicate")
    lap_x = (padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1] - 2.0 * u) / (hx**2)
    lap_y = (padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:] - 2.0 * u) / (hy**2)
    return lap_x + lap_y


def _laplacian_for_bc(q: Any, pde_params: dict[str, Any], bc_kind: str) -> Any:
    if bc_kind in {"periodic", "periodic_x"}:
        dx, dy = _periodic_grid_spacing(q, pde_params)
        return _periodic_laplacian_2d(q, dx=dx, dy=dy)
    if bc_kind in {"neumann", "open", "wall"}:
        return _neumann_laplacian(q, pde_params)
    if bc_kind == "dirichlet":
        return _dirichlet_laplacian(q, pde_params, boundary_value=0.0)
    return _laplacian(q)


def _rd_grid_spacing_fields(pde_params: dict[str, Any], reference: Any) -> tuple[Any, Any]:
    hx = _param_field(pde_params, "dx", reference, default=0.0) if "dx" in pde_params else None
    hy = _param_field(pde_params, "dy", reference, default=0.0) if "dy" in pde_params else None
    x_left, x_right, _ = _rd_axis_bounds(pde_params, "x", reference)
    y_bottom, y_top, _ = _rd_axis_bounds(pde_params, "y", reference)
    if hx is None:
        hx = (x_right - x_left) / max(int(reference.shape[-2]), 1)
    if hy is None:
        hy = (y_top - y_bottom) / max(int(reference.shape[-1]), 1)
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
    if pde in {"heat", "wave", "advection_diffusion"}:
        bc_kind = _operator_boundary_kind(pde, pde_params)
        dx, dy = _grid_spacing_for_bc(reference, pde_params, bc_kind)
        op = "periodic_roll_stencil" if bc_kind in {"periodic", "periodic_x"} else (
            "neumann_replicate_stencil" if bc_kind in {"neumann", "open", "wall"} else "dirichlet_zero_stencil"
        )
        return {
            "rhs_boundary_condition_type": bc_kind,
            "rhs_spatial_operator": op,
            "grid_spacing": {"dx": _metadata_values(dx), "dy": _metadata_values(dy)},
        }
    if pde == "reaction_diffusion":
        return _reaction_diffusion_spatial_metadata(pde_params, reference)
    if pde == "shallow_water":
        bc_kind = _operator_boundary_kind(pde, pde_params)
        return {
            "rhs": "standard_2d_conservative_shallow_water_flux_rhs",
            "rhs_boundary_condition_type": bc_kind,
            "rhs_spatial_operator": "periodic_roll_flux_divergence" if bc_kind == "periodic" else "open_replicate_flux_divergence",
        }
    if pde == "nsnonbounded":
        nu_source = "nu" if "nu" in pde_params else ("viscosity" if "viscosity" in pde_params else "default")
        forcing_present = "forcing" in pde_params
        return {
            "rhs_equation": "2D vorticity Navier-Stokes",
            "velocity_reconstruction": "periodic_fft_streamfunction",
            "boundary_assumption": "periodic",
            "domain": "periodic_torus_[0,1)^2",
            "grid_convention": "endpoint_false_periodic",
            "mean_vorticity_handling": "zero_mean_projection",
            "assumptions": ["periodic_boundary", "zero_mean_vorticity_for_poisson_solve"],
            "nu_source": nu_source,
            "nu_defaulted": nu_source == "default",
            "forcing": "provided" if forcing_present else "fixed_ns_forcing",
            "forcing_source": "provided" if forcing_present else "default_fixed_ns_forcing",
            "forcing_defaulted": not forcing_present,
            "forcing_grid": "endpoint_false",
            "forcing_formula": "0.1 * (sin(2*pi*(x+y)) + cos(2*pi*(x+y)))",
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


def _interior_only(x: Any) -> Any:
    y = x.clone()
    if y.shape[-1] > 1 and y.shape[-2] > 1:
        y[..., 0, :] = 0
        y[..., -1, :] = 0
        y[..., :, 0] = 0
        y[..., :, -1] = 0
    return y


def _interior_time_space(x: Any) -> Any:
    y = _interior_only(x)
    if y.shape[-2] > 1:
        y[..., 0, :] = 0
        y[..., -1, :] = 0
    return y


def _boundary_mask_like(x: Any) -> dict[str, Any]:
    import torch

    base = torch.zeros_like(x[:, :1])
    masks = {}
    masks["bottom"] = base.clone()
    masks["bottom"][..., 0, :] = 1.0
    masks["top"] = base.clone()
    masks["top"][..., -1, :] = 1.0
    masks["left"] = base.clone()
    masks["left"][..., :, 0] = 1.0
    masks["right"] = base.clone()
    masks["right"][..., :, -1] = 1.0
    masks["corners"] = base.clone()
    masks["corners"][..., 0, 0] = 1.0
    masks["corners"][..., 0, -1] = 1.0
    masks["corners"][..., -1, 0] = 1.0
    masks["corners"][..., -1, -1] = 1.0
    boundary = base.clone()
    boundary[..., 0, :] = 1.0
    boundary[..., -1, :] = 1.0
    boundary[..., :, 0] = 1.0
    boundary[..., :, -1] = 1.0
    masks["boundary"] = boundary
    return masks


def _normalize_masked_residual(residual: Any, mask: Any, mode: str) -> Any:
    import torch

    if mode == "mean":
        return residual
    count = mask.expand_as(residual).reshape(residual.shape[0], -1).sum(dim=1).clamp_min(1.0)
    total = float(residual[0].numel())
    scale = torch.sqrt(torch.full_like(count, total) / count).view(residual.shape[0], 1, 1, 1)
    if mode in {"sqrt_grid_over_mask", "mask_mean"}:
        return residual * scale
    raise ValueError(f"boundary_residual_normalization={mode!r} is invalid")


def _dirichlet_residual(u: Any, value: Any, sides: dict[str, Any] | None = None, normalization: str = "sqrt_grid_over_mask") -> Any:
    import torch

    sides = sides or {}
    target = _boundary_value(value, u)
    residual = torch.zeros_like(u)
    mask = torch.zeros_like(u[:, :1])
    active = _active_sides(sides)
    if "left" in active:
        residual[..., :, 0] = u[..., :, 0] - target[..., 0, 0].unsqueeze(-1)
        mask[..., :, 0] = 1.0
    if "right" in active:
        residual[..., :, -1] = u[..., :, -1] - target[..., 0, 0].unsqueeze(-1)
        mask[..., :, -1] = 1.0
    if "bottom" in active:
        residual[..., 0, :] = u[..., 0, :] - target[..., 0, 0].unsqueeze(-1)
        mask[..., 0, :] = 1.0
    if "top" in active:
        residual[..., -1, :] = u[..., -1, :] - target[..., 0, 0].unsqueeze(-1)
        mask[..., -1, :] = 1.0
    return _normalize_masked_residual(residual, mask, normalization)


def _neumann_residual(
    u: Any,
    normal_derivative_value: Any,
    sides: dict[str, Any] | None = None,
    normalization: str = "sqrt_grid_over_mask",
    pde_params: dict[str, Any] | None = None,
    pde: str | None = None,
) -> Any:
    import torch

    value = _boundary_value(normal_derivative_value, u)
    residual = torch.zeros_like(u)
    mask = torch.zeros_like(u[:, :1])
    h, w = int(u.shape[-2]), int(u.shape[-1])
    dx, dy = _boundary_grid_spacing(u, pde_params, pde=pde, bc_kind="neumann")
    dx_line = dx[..., 0, 0].unsqueeze(-1)
    dy_line = dy[..., 0, 0].unsqueeze(-1)
    active = _active_sides(sides or {})
    val = value[..., 0, 0].unsqueeze(-1)
    if w > 1 and "left" in active:
        residual[..., :, 0] = (u[..., :, 1] - u[..., :, 0]) / dx_line - val
        mask[..., :, 0] = 1.0
    if w > 1 and "right" in active:
        residual[..., :, -1] = (u[..., :, -1] - u[..., :, -2]) / dx_line - val
        mask[..., :, -1] = 1.0
    if h > 1 and "bottom" in active:
        residual[..., 0, :] = (u[..., 1, :] - u[..., 0, :]) / dy_line - val
        mask[..., 0, :] = 1.0
    if h > 1 and "top" in active:
        residual[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / dy_line - val
        mask[..., -1, :] = 1.0
    return _normalize_masked_residual(residual, mask, normalization)


def _periodic_residual(
    u: Any,
    axes: tuple[str, ...] = ("x", "y"),
    include_derivative_continuity: bool = True,
    normalization: str = "sqrt_grid_over_mask",
    pde_params: dict[str, Any] | None = None,
) -> Any:
    import torch

    pde_params = pde_params or {}
    if not _as_bool(pde_params.get("periodic_duplicate_endpoint", False)):
        return None
    parts = []
    if "x" in axes:
        r = torch.zeros_like(u)
        mask = torch.zeros_like(u[:, :1])
        jump = u[..., :, 0] - u[..., :, -1]
        r[..., :, 0] = jump
        r[..., :, -1] = -jump
        mask[..., :, 0] = 1.0
        mask[..., :, -1] = 1.0
        parts.append(_normalize_masked_residual(r, mask, normalization))
        if include_derivative_continuity and u.shape[-1] > 2:
            d = torch.zeros_like(u)
            djump = (u[..., :, 1] - u[..., :, 0]) - (u[..., :, -1] - u[..., :, -2])
            d[..., :, 0] = djump
            d[..., :, -1] = -djump
            parts.append(_normalize_masked_residual(d, mask, normalization))
    if "y" in axes:
        r = torch.zeros_like(u)
        mask = torch.zeros_like(u[:, :1])
        jump = u[..., 0, :] - u[..., -1, :]
        r[..., 0, :] = jump
        r[..., -1, :] = -jump
        mask[..., 0, :] = 1.0
        mask[..., -1, :] = 1.0
        parts.append(_normalize_masked_residual(r, mask, normalization))
        if include_derivative_continuity and u.shape[-2] > 2:
            d = torch.zeros_like(u)
            djump = (u[..., 1, :] - u[..., 0, :]) - (u[..., -1, :] - u[..., -2, :])
            d[..., 0, :] = djump
            d[..., -1, :] = -djump
            parts.append(_normalize_masked_residual(d, mask, normalization))
    return torch.cat(parts, dim=1) if parts else None


def _mixed_boundary_residual(
    pde: str,
    q: Any,
    spec: BoundaryConditionSpec,
    normalization: str,
    pde_params: dict[str, Any],
) -> Any:
    import torch

    if pde == "steady_heat_conduction":
        u_d = _required_param_field(pde_params, "u_D", q)
        # Match build_linear_system(): bottom rows take precedence over
        # sides, and side rows take precedence over top at the corners.
        residual = torch.zeros_like(q)
        residual[..., 0, :] = q[..., 0, :] - u_d[..., 0, :]
        if q.shape[-2] > 1 and q.shape[-1] > 1:
            residual[..., 1:, 0] = q[..., 1:, 0] - q[..., 1:, 1]
            residual[..., 1:, -1] = q[..., 1:, -1] - q[..., 1:, -2]
        if q.shape[-2] > 1 and q.shape[-1] > 2:
            residual[..., -1, 1:-1] = q[..., -1, 1:-1] - q[..., -2, 1:-1]
        return residual
    parts = []
    for side, side_spec in (spec.sides or {}).items():
        kind = side_spec.get("kind") if isinstance(side_spec, dict) else None
        value = side_spec.get("value", spec.value) if isinstance(side_spec, dict) else spec.value
        if kind == "dirichlet":
            parts.append(_dirichlet_residual(q, value if value is not None else 0.0, {side: {}}, normalization))
        elif kind == "neumann":
            parts.append(
                _neumann_residual(
                    q,
                    value if value is not None else 0.0,
                    {side: {}},
                    normalization,
                    pde_params=pde_params,
                    pde=pde,
                )
            )
    return torch.cat(parts, dim=1) if parts else None


def _wall_residual(q: Any, normalization: str, pde_params: dict[str, Any] | None = None, pde: str | None = None) -> Any:
    import torch

    if q.shape[1] < 3:
        return _neumann_residual(q, 0.0, {}, normalization, pde_params=pde_params, pde=pde)
    residual = torch.zeros_like(q)
    mask = torch.zeros_like(q[:, :1])
    residual[:, 1:2, :, 0] = q[:, 1:2, :, 0]
    residual[:, 1:2, :, -1] = q[:, 1:2, :, -1]
    residual[:, 2:3, 0, :] = q[:, 2:3, 0, :]
    residual[:, 2:3, -1, :] = q[:, 2:3, -1, :]
    mask[..., :, 0] = 1.0
    mask[..., :, -1] = 1.0
    mask[..., 0, :] = 1.0
    mask[..., -1, :] = 1.0
    return _normalize_masked_residual(residual, mask, normalization)


def _boundary_value(value: Any, reference: Any) -> Any:
    import torch

    if hasattr(value, "shape") or isinstance(value, (list, tuple)):
        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if tensor.ndim == 0:
            return torch.full((reference.shape[0], reference.shape[1], 1, 1), float(tensor), dtype=reference.dtype, device=reference.device)
        if tensor.ndim == 1:
            if tensor.numel() == 1:
                tensor = tensor.repeat(reference.shape[0])
            if tensor.numel() == reference.shape[0]:
                return tensor.view(reference.shape[0], 1, 1, 1).expand(-1, reference.shape[1], -1, -1)
        if tensor.ndim == 4:
            if tensor.shape[0] == 1 and reference.shape[0] > 1:
                tensor = tensor.repeat(reference.shape[0], 1, 1, 1)
            return tensor
    return torch.full((reference.shape[0], reference.shape[1], 1, 1), float(value), dtype=reference.dtype, device=reference.device)


def _active_sides(sides: dict[str, Any]) -> set[str]:
    if not sides:
        return {"left", "right", "bottom", "top"}
    return {side for side in ("left", "right", "bottom", "top") if side in sides}


def _dx(f: Any) -> Any:
    import torch

    h = 1.0 / max(int(f.shape[-2]) - 1, 1)
    padded = torch.nn.functional.pad(f, (0, 0, 1, 1), mode="replicate")
    return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2.0 * h)


def _dy(f: Any) -> Any:
    import torch

    h = 1.0 / max(int(f.shape[-1]) - 1, 1)
    padded = torch.nn.functional.pad(f, (1, 1, 0, 0), mode="replicate")
    return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2.0 * h)


def _dirichlet_dx(f: Any, pde_params: dict[str, Any] | None = None, boundary_value: float = 0.0) -> Any:
    import torch

    dx, _ = _closed_interval_grid_spacing(f, pde_params)
    padded = torch.nn.functional.pad(f, (0, 0, 1, 1), mode="constant", value=float(boundary_value))
    return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2.0 * dx)


def _dirichlet_dy(f: Any, pde_params: dict[str, Any] | None = None, boundary_value: float = 0.0) -> Any:
    import torch

    _, dy = _closed_interval_grid_spacing(f, pde_params)
    padded = torch.nn.functional.pad(f, (1, 1, 0, 0), mode="constant", value=float(boundary_value))
    return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2.0 * dy)


def _dx_for_bc(f: Any, pde_params: dict[str, Any], bc_kind: str) -> Any:
    if bc_kind in {"periodic", "periodic_x"}:
        dx, _ = _periodic_grid_spacing(f, pde_params)
        return _periodic_dx(f, dx=dx)
    if bc_kind == "dirichlet":
        return _dirichlet_dx(f, pde_params, boundary_value=0.0)
    dx, _ = _nonperiodic_operator_spacing(f, pde_params)
    return _replicate_dx(f, dx)


def _dy_for_bc(f: Any, pde_params: dict[str, Any], bc_kind: str) -> Any:
    if bc_kind == "periodic":
        _, dy = _periodic_grid_spacing(f, pde_params)
        return _periodic_dy(f, dy=dy)
    if bc_kind == "periodic_x":
        _, dy = _nonperiodic_operator_spacing(f, pde_params)
        return _replicate_dy(f, dy)
    if bc_kind == "dirichlet":
        return _dirichlet_dy(f, pde_params, boundary_value=0.0)
    _, dy = _nonperiodic_operator_spacing(f, pde_params)
    return _replicate_dy(f, dy)


def _nonperiodic_operator_spacing(f: Any, pde_params: dict[str, Any]) -> tuple[Any, Any]:
    if any(name in pde_params for name in ("x_range", "y_range", "x_left", "x_right", "y_bottom", "y_top")):
        return _rd_grid_spacing_fields(pde_params, f)
    return _closed_interval_grid_spacing(f, pde_params)


def _replicate_dx(f: Any, dx: Any) -> Any:
    import torch

    padded = torch.nn.functional.pad(f, (0, 0, 1, 1), mode="replicate")
    return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2.0 * dx)


def _replicate_dy(f: Any, dy: Any) -> Any:
    import torch

    padded = torch.nn.functional.pad(f, (1, 1, 0, 0), mode="replicate")
    return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2.0 * dy)


def _operator_boundary_kind(pde: str, pde_params: dict[str, Any]) -> str:
    if "boundary_condition_kind" in pde_params:
        return _normalize_boundary_kind(str(pde_params["boundary_condition_kind"]), pde)
    mode = str(pde_params.get("boundary_condition_mode", pde_params.get("boundary_condition", "auto")))
    aliases = {
        "dirichlet_zero": "dirichlet",
        "neumann_zero": "neumann",
        "periodic": "periodic_x" if pde == "burger" else "periodic",
        "open": "open",
        "wall": "wall",
        "mixed": "mixed",
        "none": "none",
        "legacy_ignore": "none",
    }
    if mode != "auto":
        return aliases.get(mode, _normalize_boundary_kind(mode, pde))
    defaults = {
        "heat": "periodic",
        "wave": "periodic",
        "advection_diffusion": "periodic",
        "reaction_diffusion": "neumann",
        "shallow_water": "open",
        "burger": "periodic_x",
        "darcy": "dirichlet",
        "poisson": "dirichlet",
        "helmholtz": "dirichlet",
        "steady_heat_conduction": "mixed",
    }
    return defaults.get(pde, "none")


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


def _required_param_field(params: dict[str, Any], name: str, reference: Any) -> Any:
    if name not in params or params[name] is None:
        raise ValueError(f"Missing required sample-level PDE parameter {name!r}; load it from the data file")
    return _param_field(params, name, reference, default=0.0)


def _reaction_diffusion_defaults(params: dict[str, Any]) -> dict[str, float]:
    profile = str(params.get("generator_profile", "current")).lower()
    if profile in {"legacy", "pdebench_legacy", "reaction_diffusion_old"}:
        return {"D_u": 1e-3, "D_v": 5e-3, "k": 5e-3}
    return {"D_u": 2e-3, "D_v": 4e-3, "k": 3e-3}


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


def _out(residual: Any, status: str, metadata: dict[str, Any], components: dict[str, Any] | None = None) -> ResidualOutput:
    metadata = dict(metadata)
    metadata.setdefault("residual_status", status)
    metadata.setdefault("status", status)
    return ResidualOutput(residual, status, metadata, components=components)


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

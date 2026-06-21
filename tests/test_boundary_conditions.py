import pytest

torch = pytest.importorskip("torch")

from sampling.pde_residuals import (
    _dirichlet_residual,
    _neumann_residual,
    _periodic_residual,
    _steady_heat_residual_with_boundary,
    compute_pde_residual,
)


def test_zero_boundary_no_longer_used_as_bc():
    u = torch.randn(1, 1, 8, 8)
    for pde in ("poisson", "heat", "advection_diffusion"):
        if pde == "poisson":
            out = compute_pde_residual(pde, torch.zeros_like(u), u, pde_params={"boundary_condition_mode": "dirichlet_zero"})
        else:
            out = compute_pde_residual(pde, u, u, pde_params={"boundary_condition_mode": "periodic"}, residual_mode="hermite_bridge")
        assert out.metadata["interior_residual_enabled"] is True
        assert out.metadata["bc_residual_enabled"] is True
        assert out.components["boundary"] is not None
        assert out.metadata["legacy_boundary_ignored"] is False


def test_poisson_dirichlet_zero_bc():
    u = torch.zeros(1, 1, 8, 8)
    u[..., 0, :] = 2.0
    bc = _dirichlet_residual(u, 0.0, {}, "mean")
    assert bc.abs().sum() > 0
    u[..., 0, :] = 0.0
    bc = _dirichlet_residual(u, 0.0, {}, "mean")
    assert torch.allclose(bc, torch.zeros_like(bc))


def test_neumann_zero_bc():
    const = torch.ones(1, 1, 8, 8)
    assert torch.allclose(_neumann_residual(const, 0.0, {}, "mean"), torch.zeros_like(const))
    sloped = const.clone()
    sloped[..., :, 0] = 0.0
    assert _neumann_residual(sloped, 0.0, {}, "mean").abs().sum() > 0


def test_periodic_bc():
    u = torch.randn(1, 1, 8, 8)
    u[..., :, -1] = u[..., :, 0]
    u[..., -1, :] = u[..., 0, :]
    assert _periodic_residual(u, include_derivative_continuity=False, normalization="mean").abs().sum() == pytest.approx(0.0)
    u[..., 0, 3] += 1.0
    assert _periodic_residual(u, include_derivative_continuity=False, normalization="mean").abs().sum() > 0

    q0 = torch.randn(1, 2, 8, 8)
    qT = torch.randn(1, 2, 8, 8)
    out = compute_pde_residual(
        "reaction_diffusion",
        q0,
        qT,
        pde_params={"boundary_condition_mode": "neumann_zero", "T": torch.tensor([1.0])},
        residual_mode="endpoint_secant",
    )
    assert out.metadata["boundary_condition_type"] == "neumann"
    assert out.metadata["bc_residual_enabled"] is True


def test_hermite_bridge_boundary_conditions():
    for pde, channels, mode in (
        ("heat", 1, "periodic"),
        ("wave", 2, "periodic"),
        ("advection_diffusion", 1, "periodic"),
        ("reaction_diffusion", 2, "neumann_zero"),
        ("shallow_water", 3, "open"),
    ):
        q0 = torch.randn(1, channels, 8, 8)
        qT = torch.randn(1, channels, 8, 8)
        if pde == "shallow_water":
            q0[:, :1] = q0[:, :1].abs() + 1.0
            qT[:, :1] = qT[:, :1].abs() + 1.0
        out = compute_pde_residual(pde, q0, qT, pde_params={"boundary_condition_mode": mode}, residual_mode="hermite_bridge")
        assert out.components["interior"] is not None
        assert out.components["boundary"] is not None
        assert out.metadata["residual_channels"]["bc_channels"] >= channels


def test_initial_condition_masked():
    q0 = torch.zeros(1, 1, 4, 4)
    qT = torch.zeros_like(q0)
    target = torch.ones_like(q0)
    mask = torch.zeros_like(q0)
    mask[..., 1, 1] = 1.0
    out = compute_pde_residual(
        "heat",
        q0,
        qT,
        pde_params={
            "boundary_condition_mode": "periodic",
            "observed_initial": target * mask,
            "initial_mask": mask,
        },
        residual_mode="endpoint_secant",
    )
    ic = out.components["initial"]
    assert ic[..., 1, 1].abs().sum() > 0
    assert ic[..., 0, 0].abs().sum() == pytest.approx(0.0)


def test_unknown_bc_raises():
    with pytest.raises(ValueError):
        compute_pde_residual(
            "poisson",
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 1, 4, 4),
            pde_params={"boundary_condition_mode": "not_a_mode", "allow_unknown_boundary_conditions": False},
        )


def test_steady_heat_conduction_migrated_bc():
    u = torch.ones(1, 1, 6, 6) * 298.0
    a = torch.zeros_like(u)
    params = {"u_D": torch.tensor([300.0]), "boundary_condition_mode": "mixed"}
    out = compute_pde_residual("steady_heat_conduction", a, u, pde_params=params)
    old = _steady_heat_residual_with_boundary(u, (1.0 + 0.05 * (u - 298.0)).clamp_min(0.1), a, torch.ones_like(u) * 300.0)
    assert out.components["boundary"] is not None
    assert out.residual[:, :1, 1:-1, 1:-1].shape == old[:, :, 1:-1, 1:-1].shape
    assert out.metadata["boundary_condition_type"] == "mixed"


def test_ns_still_disabled():
    out = compute_pde_residual("nsnonbounded", torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), residual_mode="hermite_bridge")
    assert out.status == "disabled"
    assert out.metadata["bc_residual_enabled"] is False

import pytest

torch = pytest.importorskip("torch")

from sampling.pde_residuals import compute_pde_residual, residual_status


TEMPORAL_ENDPOINT_CASES = [
    ("heat", 1),
    ("wave", 2),
    ("advection_diffusion", 1),
    ("reaction_diffusion", 2),
    ("shallow_water", 3),
    ("nsnonbounded", 1),
]


def _state(pde, batch=2, h=8, w=8):
    channels = dict(TEMPORAL_ENDPOINT_CASES)[pde]
    q0 = torch.randn(batch, channels, h, w)
    qT = torch.randn(batch, channels, h, w)
    if pde == "shallow_water":
        q0[:, 0:1] = q0[:, 0:1].abs() + 0.5
        qT[:, 0:1] = qT[:, 0:1].abs() + 0.5
    return q0, qT


def test_temporal_residual_modes_all_endpoint_pdes():
    for pde, channels in TEMPORAL_ENDPOINT_CASES:
        q0, qT = _state(pde)
        out = compute_pde_residual(pde, q0, qT, residual_mode="hermite_bridge")
        assert out.residual.shape[0] == 2
        assert out.residual.shape[-2:] == (8, 8)
        assert out.metadata["residual_channels"]["interior_channels"] == 3 * channels
        assert out.metadata["residual_channels"]["endpoint_channels"] == channels
        if pde in {"heat", "wave", "advection_diffusion", "nsnonbounded"}:
            assert out.metadata["residual_channels"]["bc_channels"] == 0
            assert out.metadata["boundary_enforced"] is True
            assert out.metadata["boundary_enforced_by_operator"] is True
        else:
            assert out.metadata["residual_channels"]["bc_channels"] > 0
        assert out.status == "approximate"
        assert out.metadata["status"] == "approximate"
        assert out.metadata["endpoint_only"] is True
        assert out.metadata["temporal_derivative_mode"] == "hermite_bridge"
        assert out.metadata["residual_family"] == "temporal_endpoint"
        assert out.metadata["resolved_residual_mode"] == "hermite_bridge"
        assert torch.isfinite(out.residual).all()


def test_auto_uses_hermite_for_endpoint_time_dependent():
    for pde, _channels in TEMPORAL_ENDPOINT_CASES:
        q0, qT = _state(pde)
        out = compute_pde_residual(pde, q0, qT, residual_mode="auto")
        assert out.metadata["resolved_residual_mode"] == "hermite_bridge"
        assert out.metadata["mode"] == "hermite_bridge"


def test_temporal_endpoint_secant_all_endpoint_pdes():
    for pde, channels in TEMPORAL_ENDPOINT_CASES:
        q0, qT = _state(pde)
        out = compute_pde_residual(pde, q0, qT, residual_mode="endpoint_secant")
        assert out.residual.shape[0] == 2
        assert out.residual.shape[-2:] == (8, 8)
        assert out.metadata["residual_channels"]["interior_channels"] == channels
        if pde in {"heat", "wave", "advection_diffusion", "nsnonbounded"}:
            assert out.metadata["residual_channels"]["bc_channels"] == 0
            assert out.metadata["boundary_enforced_by_operator"] is True
        else:
            assert out.metadata["residual_channels"]["bc_channels"] > 0
        assert out.metadata["two_time_level_approx"] is True
        assert out.metadata["temporal_derivative_mode"] == "endpoint_secant"
        assert out.metadata["resolved_residual_mode"] == "endpoint_secant"
        assert out.metadata["warning"].startswith("coarse two-time-level")
        assert torch.isfinite(out.residual).all()


def test_endpoint_secant_uses_time_scale_from_params():
    h = w = 7
    heat = compute_pde_residual(
        "heat",
        torch.zeros(1, 1, h, w),
        torch.ones(1, 1, h, w),
        pde_params={"alpha": torch.tensor([0.0]), "T": torch.tensor([2.0])},
        residual_mode="endpoint_secant",
    )
    assert heat.metadata["time_scale"]["source"] == "T"
    assert heat.residual[0, 0, 3, 3].item() == pytest.approx(0.5)

    adv = compute_pde_residual(
        "advection_diffusion",
        torch.zeros(1, 1, h, w),
        torch.ones(1, 1, h, w),
        pde_params={
            "b_x": torch.tensor([0.0]),
            "b_y": torch.tensor([0.0]),
            "kappa": torch.tensor([0.0]),
            "dt": torch.tensor([5.0]),
        },
        residual_mode="endpoint_secant",
    )
    assert adv.metadata["time_scale"]["source"] == "dt"
    assert adv.residual[0, 0, 3, 3].item() == pytest.approx(0.2)


def test_near_endpoint_temporal_all_endpoint_pdes_requires_aux():
    for pde, channels in TEMPORAL_ENDPOINT_CASES:
        with pytest.raises(ValueError, match="near_endpoint_temporal mode requires extra near-endpoint"):
            compute_pde_residual(
                pde,
                torch.zeros(1, channels, 5, 5),
                torch.zeros(1, channels, 5, 5),
                residual_mode="near_endpoint_temporal",
            )


def test_near_endpoint_temporal_all_endpoint_pdes_with_aux():
    for pde, channels in TEMPORAL_ENDPOINT_CASES:
        q0, qT = _state(pde, batch=1, h=5, w=5)
        mask_0 = torch.zeros(1, 1, 5, 5)
        mask_T = torch.zeros(1, 1, 5, 5)
        mask_0[..., 2, 2] = 1.0
        mask_T[..., 1, 3] = 1.0
        out = compute_pde_residual(
            pde,
            q0,
            qT,
            pde_params={
                "near_endpoint_temporal": {
                    "q_dt": q0 + 0.1,
                    "q_T_minus_dt": qT - 0.1,
                    "dt": torch.tensor([0.1]),
                    "mask_0": mask_0,
                    "mask_T": mask_T,
                },
            },
            residual_mode="near_endpoint_temporal",
        )
        assert out.residual.shape[0] == 1
        assert out.residual.shape[-2:] == (5, 5)
        assert out.metadata["temporal_derivative_mode"] == "near_endpoint_sparse_fd"
        assert out.metadata["endpoint_only"] is False
        assert out.metadata["uses_extra_temporal_observations"] is True
        assert torch.isfinite(out.residual).all()


def test_full_trajectory_fd_requires_trajectory_for_endpoint_pdes():
    for pde, _channels in TEMPORAL_ENDPOINT_CASES:
        q0, qT = _state(pde, batch=1, h=5, w=5)
        with pytest.raises(ValueError, match="full_trajectory_fd mode requires explicit full trajectory"):
            compute_pde_residual(pde, q0, qT, residual_mode="full_trajectory_fd")


def test_burger_is_full_time_space():
    u = torch.randn(2, 1, 8, 8)
    out = compute_pde_residual("burger", u, u)
    assert out.status == "reliable"
    assert out.metadata["residual_family"] == "full_time_space"
    assert out.metadata["temporal_derivative_mode"] == "full_fd"
    assert out.metadata["endpoint_only"] is False
    assert out.metadata["mode"] == "full_time_space"
    assert out.metadata["resolved_residual_mode"] == "full_time_space"


def test_steady_heat_is_static_not_temporal():
    h = w = 7
    out = compute_pde_residual(
        "steady_heat_conduction",
        torch.ones(2, 1, h, w),
        torch.ones(2, 1, h, w) * 298.0,
        pde_params={"u_D": torch.tensor([298.0, 300.0])},
    )
    assert tuple(out.components["interior"].shape) == (2, 1, h, w)
    assert out.components["boundary"] is not None
    assert out.status == "reliable"
    assert out.metadata["residual_family"] == "static"
    assert out.metadata["temporal_derivative_mode"] == "none"
    assert out.metadata["mode"] == "static_nonlinear_boundary"
    assert out.metadata["resolved_residual_mode"] == "static_nonlinear_boundary"
    assert out.components["boundary"][1, 0, 0, 0].item() == pytest.approx(-2.0 * (h * w / w) ** 0.5)


def test_nsnonbounded_residual_enabled_and_backward():
    q0 = torch.randn(1, 1, 8, 8, requires_grad=True)
    qT = torch.randn(1, 1, 8, 8, requires_grad=True)
    out = compute_pde_residual("nsnonbounded", q0, qT, residual_mode="hermite_bridge")
    assert out.status == "approximate"
    assert out.metadata["rhs_equation"] == "2D vorticity Navier-Stokes"
    assert out.metadata["resolved_residual_mode"] == "hermite_bridge"
    assert out.metadata["forcing_defaulted"] is True
    out.residual.square().mean().backward()
    assert q0.grad is not None
    assert qT.grad is not None


def test_nsnonbounded_residual_status_is_approximate():
    assert residual_status("nsnonbounded") == "approximate"


def test_nsnonbounded_endpoint_secant_uses_default_fixed_forcing():
    q0 = torch.zeros(1, 1, 8, 8)
    qT = torch.zeros_like(q0)
    out = compute_pde_residual(
        "nsnonbounded",
        q0,
        qT,
        pde_params={"T": torch.tensor([1.0]), "nu": torch.tensor([1e-3])},
        residual_mode="endpoint_secant",
    )
    assert out.status == "approximate"
    assert out.metadata["resolved_residual_mode"] == "endpoint_secant"
    assert out.metadata["rhs_equation"] == "2D vorticity Navier-Stokes"
    assert out.metadata["forcing_defaulted"] is True
    assert out.metadata["forcing_source"] == "default_fixed_ns_forcing"
    assert out.residual.abs().sum() > 0
    assert out.residual[0, 0, 0, 0].item() == pytest.approx(-0.1)


def test_legacy_ignore_boundary_keeps_hermite_integral_endpoint_residual():
    q0 = torch.zeros(1, 1, 6, 6)
    qT = torch.ones_like(q0)
    out = compute_pde_residual(
        "heat",
        q0,
        qT,
        pde_params={
            "legacy_ignore_boundary": True,
            "hermite_include_integral_residual": True,
            "hermite_integral_weight": 1.0,
        },
        residual_mode="hermite_bridge",
    )
    channels = out.metadata["residual_channels"]
    assert out.metadata["legacy_boundary_ignored"] is True
    assert out.metadata["endpoint_residual_enabled"] is True
    assert channels["total_channels"] == channels["interior_channels"] + channels["endpoint_channels"]

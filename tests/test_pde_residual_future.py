import pytest

torch = pytest.importorskip("torch")

from sampling.pde_residuals import compute_pde_residual


TIME_DEPENDENT_CASES = [
    ("heat", 1),
    ("wave", 2),
    ("advection_diffusion", 1),
    ("reaction_diffusion", 2),
    ("shallow_water", 3),
]


def _state(pde, batch=2, h=7, w=7):
    channels = dict(TIME_DEPENDENT_CASES)[pde]
    q0 = torch.randn(batch, channels, h, w)
    qT = torch.randn(batch, channels, h, w)
    if pde == "shallow_water":
        q0[:, 0:1] = q0[:, 0:1].abs() + 0.1
        qT[:, 0:1] = qT[:, 0:1].abs() + 0.1
    return q0, qT


def test_hermite_bridge_residual_shapes_and_metadata():
    for pde, channels in TIME_DEPENDENT_CASES:
        q0, qT = _state(pde)
        out = compute_pde_residual(pde, q0, qT, residual_mode="hermite_bridge")
        assert tuple(out.residual.shape) == (2, 4 * channels, 7, 7)
        assert out.status == "approximate"
        assert out.metadata["mode"] == "hermite_bridge"
        assert out.metadata["resolved_residual_mode"] == "hermite_bridge"
        assert out.metadata["collocation_times"] == [0.25, 0.5, 0.75]
        assert out.metadata["two_time_level_approx"] is False
        assert out.metadata["endpoint_only"] is True
        assert out.metadata["full_trajectory_required"] is False


def test_auto_uses_hermite_for_endpoint_time_dependent():
    for pde, _channels in TIME_DEPENDENT_CASES:
        q0, qT = _state(pde)
        out = compute_pde_residual(pde, q0, qT, residual_mode="auto")
        assert out.metadata["resolved_residual_mode"] == "hermite_bridge"
        assert out.metadata["mode"] == "hermite_bridge"


def test_endpoint_secant_legacy_available():
    h = w = 7
    heat = compute_pde_residual(
        "heat",
        torch.zeros(1, 1, h, w),
        torch.ones(1, 1, h, w),
        pde_params={"alpha": torch.tensor([0.0]), "T": torch.tensor([2.0])},
        residual_mode="endpoint_secant",
    )
    assert heat.metadata["mode"] == "endpoint_secant"
    assert heat.metadata["warning"].startswith("coarse two-time-level")
    assert heat.metadata["two_time_level_approx"] is True
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
        residual_mode="legacy_endpoint_secant",
    )
    assert adv.metadata["mode"] == "legacy_endpoint_secant"
    assert adv.metadata["warning"].startswith("coarse two-time-level")
    assert adv.residual[0, 0, 3, 3].item() == pytest.approx(0.2)


def test_future_pde_residual_uses_time_scale_from_params_in_endpoint_secant():
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

    wave = compute_pde_residual(
        "wave",
        torch.zeros(1, 2, h, w),
        torch.cat([torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)], dim=1),
        pde_params={"c": torch.tensor([0.0]), "total_time": torch.tensor([4.0])},
        residual_mode="endpoint_secant",
    )
    assert wave.metadata["time_scale"]["source"] == "total_time"
    assert wave.residual[0, 0, 3, 3].item() == pytest.approx(0.25)

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


def test_near_endpoint_temporal_requires_observations():
    with pytest.raises(ValueError, match="near_endpoint_temporal mode requires extra near-endpoint"):
        compute_pde_residual(
            "heat",
            torch.zeros(1, 1, 5, 5),
            torch.zeros(1, 1, 5, 5),
            pde_params={"alpha": torch.tensor([0.0])},
            residual_mode="near_endpoint_temporal",
        )


def test_near_endpoint_temporal_masked_residual():
    h = w = 5
    q0 = torch.zeros(1, 1, h, w)
    qT = torch.zeros(1, 1, h, w)
    mask_0 = torch.zeros(1, 1, h, w)
    mask_T = torch.zeros(1, 1, h, w)
    mask_0[..., 2, 2] = 1.0
    mask_T[..., 1, 3] = 1.0
    out = compute_pde_residual(
        "heat",
        q0,
        qT,
        pde_params={
            "alpha": torch.tensor([0.0]),
            "near_endpoint_temporal": {
                "q_dt": torch.ones_like(q0),
                "q_T_minus_dt": -torch.ones_like(qT),
                "dt": torch.tensor([1.0]),
                "mask_0": mask_0,
                "mask_T": mask_T,
            },
        },
        residual_mode="near_endpoint_temporal",
    )
    assert tuple(out.residual.shape) == (1, 2, h, w)
    assert out.metadata["mode"] == "near_endpoint_temporal"
    assert out.metadata["mask_observed_points_0"] == pytest.approx(1.0)
    assert out.metadata["mask_normalization_0"] == pytest.approx(5.0)
    assert out.residual[0, 0, 2, 2].item() == pytest.approx(5.0)
    assert out.residual[0, 0].count_nonzero().item() == 1
    assert out.residual[0, 1, 1, 3].item() == pytest.approx(5.0)
    assert out.residual[0, 1].count_nonzero().item() == 1


def test_shallow_water_flux_standard_form_shape():
    q0, qT = _state("shallow_water", batch=1, h=6, w=6)
    q0[:, 0:1] = 0.0
    qT[:, 0:1] = 0.0
    out = compute_pde_residual("shallow_water", q0, qT, residual_mode="hermite_bridge")
    assert tuple(out.residual.shape) == (1, 12, 6, 6)
    assert torch.isfinite(out.residual).all()
    assert out.metadata["rhs"] == "standard_2d_conservative_shallow_water_flux_rhs"


def test_static_future_pde_residual_shape_still_unchanged():
    h = w = 7
    steady = compute_pde_residual(
        "steady_heat_conduction",
        torch.ones(2, 1, h, w),
        torch.ones(2, 1, h, w) * 298.0,
        pde_params={"u_D": torch.tensor([298.0, 300.0])},
    )
    assert tuple(steady.residual.shape) == (2, 1, h, w)
    assert "lambda(u)" in steady.metadata["equation"]
    assert steady.metadata["boundary_enforced"] is True
    assert steady.residual[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert steady.residual[1, 0, 0, 0].item() == pytest.approx(-2.0)


def test_ns_residual_still_disabled():
    out = compute_pde_residual("nsnonbounded", torch.zeros(1, 1, 6, 6), torch.randn(1, 1, 6, 6))
    assert out.status == "disabled"
    assert out.metadata["resolved_residual_mode"] == "disabled"
    assert "vorticity transport" in out.metadata["disabled_reason"]
    assert torch.count_nonzero(out.residual) == 0

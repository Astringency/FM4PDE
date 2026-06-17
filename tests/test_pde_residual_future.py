import pytest

torch = pytest.importorskip("torch")

from fm4pde_ablation.pde_residuals import compute_pde_residual


def test_future_pde_residual_shapes_and_metadata():
    h = w = 7
    heat = compute_pde_residual(
        "heat",
        torch.zeros(2, 1, h, w),
        torch.ones(2, 1, h, w),
        pde_params={"alpha": torch.tensor([0.1, 0.2])},
    )
    assert tuple(heat.residual.shape) == (2, 1, h, w)
    assert heat.metadata["pde_params_used"]["alpha"] is True

    wave = compute_pde_residual(
        "wave",
        torch.zeros(2, 2, h, w),
        torch.ones(2, 2, h, w),
        pde_params={"c": torch.tensor([1.0, 1.5])},
    )
    assert tuple(wave.residual.shape) == (2, 2, h, w)
    assert wave.status == "approximate"

    adv = compute_pde_residual(
        "advection_diffusion",
        torch.zeros(2, 1, h, w),
        torch.ones(2, 1, h, w),
        pde_params={"b_x": torch.zeros(2), "b_y": torch.ones(2), "kappa": torch.ones(2) * 0.1},
    )
    assert tuple(adv.residual.shape) == (2, 1, h, w)
    assert adv.metadata["pde_params_used"]["kappa"] is True

    steady = compute_pde_residual(
        "steady_heat_conduction",
        torch.ones(2, 1, h, w),
        torch.ones(2, 1, h, w) * 298.0,
        pde_params={"u_D": torch.tensor([298.0, 300.0])},
    )
    assert tuple(steady.residual.shape) == (2, 1, h, w)
    assert "lambda(u)" in steady.metadata["equation"]


def test_ns_residual_is_disabled_placeholder_not_legacy_gradient():
    out = compute_pde_residual("nsnonbounded", torch.zeros(1, 1, 6, 6), torch.randn(1, 1, 6, 6))
    assert out.status == "disabled"
    assert torch.count_nonzero(out.residual) == 0

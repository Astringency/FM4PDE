import pytest

torch = pytest.importorskip("torch")

from sampling.pde_residuals import _neumann_laplacian, compute_pde_residual


def test_neumann_laplacian_constant_field_is_zero():
    field = torch.ones(2, 1, 16, 16)
    lap = _neumann_laplacian(field, {"x_left": torch.tensor([-1.0, -1.0]), "x_right": torch.tensor([1.0, 1.0])})

    assert torch.allclose(lap, torch.zeros_like(field))


def test_reaction_diffusion_residual_uses_neumann_metadata_on_arbitrary_shape():
    q0 = torch.randn(1, 2, 16, 16)
    qT = torch.randn(1, 2, 16, 16)

    out = compute_pde_residual(
        "reaction_diffusion",
        q0,
        qT,
        pde_params={
            "T": torch.tensor([1.0]),
            "D_u": torch.tensor([2e-3]),
            "D_v": torch.tensor([4e-3]),
            "k": torch.tensor([3e-3]),
            "x_left": torch.tensor([-1.0]),
            "x_right": torch.tensor([1.0]),
            "y_bottom": torch.tensor([-1.0]),
            "y_top": torch.tensor([1.0]),
        },
        residual_mode="hermite_bridge",
    )

    assert tuple(out.residual.shape) == (1, 8, 16, 16)
    assert torch.isfinite(out.residual).all()
    assert out.metadata["boundary_condition"] == "homogeneous_neumann"
    assert out.metadata["laplacian"] == "neumann"
    assert out.metadata["domain"]["x"] == [-1.0, 1.0]
    assert out.metadata["grid_spacing"]["dx"] == pytest.approx(0.125)
    assert out.metadata["pde_params_used"]["D_u"] is True
    assert out.metadata["pde_params_used"]["D_v"] is True
    assert out.metadata["pde_params_used"]["k"] is True
    assert out.metadata["pde_params_used"]["T"] is True


def test_reaction_diffusion_residual_default_domain_works_on_32x32():
    q0 = torch.randn(1, 2, 32, 32)
    qT = torch.randn(1, 2, 32, 32)

    out = compute_pde_residual(
        "reaction_diffusion",
        q0,
        qT,
        pde_params={"T": torch.tensor([1.0]), "D_u": torch.tensor([2e-3]), "D_v": torch.tensor([4e-3]), "k": torch.tensor([3e-3])},
        residual_mode="endpoint_secant",
    )

    assert tuple(out.residual.shape) == (1, 2, 32, 32)
    assert torch.isfinite(out.residual).all()
    assert out.metadata["boundary_condition"] == "homogeneous_neumann"
    assert out.metadata["domain"]["defaulted"] is True
    assert out.metadata["grid_spacing"]["dx"] == pytest.approx(2.0 / 32.0)

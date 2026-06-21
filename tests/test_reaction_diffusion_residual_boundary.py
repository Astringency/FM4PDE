import pytest

torch = pytest.importorskip("torch")

from sampling.pde_residuals import _neumann_laplacian, _neumann_residual, compute_pde_residual


def test_reaction_diffusion_neumann_bc_uses_domain_spacing():
    n = 16
    x = torch.linspace(-1.0, 1.0, n)
    field = x.view(1, 1, 1, n).repeat(1, 1, n, 1)
    pde_params = {
        "x_left": torch.tensor([-1.0]),
        "x_right": torch.tensor([1.0]),
        "y_bottom": torch.tensor([-1.0]),
        "y_top": torch.tensor([1.0]),
    }

    residual = _neumann_residual(
        field,
        0.0,
        {"left": {}, "right": {}},
        "mean",
        pde_params=pde_params,
        pde="reaction_diffusion",
    )

    expected = (2.0 / (n - 1)) / (2.0 / n)
    assert residual[..., :, 0].mean().item() == pytest.approx(expected)
    assert residual[..., :, -1].mean().item() == pytest.approx(expected)
    assert residual[..., :, 0].mean().item() != pytest.approx(1.0)


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

    assert out.residual.shape[0] == 1
    assert out.residual.shape[-2:] == (16, 16)
    assert out.metadata["residual_channels"]["interior_channels"] == 6
    assert out.metadata["residual_channels"]["endpoint_channels"] == 2
    assert out.metadata["residual_channels"]["bc_channels"] == 10
    assert torch.isfinite(out.residual).all()
    assert out.metadata["boundary_condition"] == "homogeneous_neumann"
    assert out.metadata["laplacian"] == "neumann"
    assert out.metadata["domain"]["x"] == [-1.0, 1.0]
    assert out.metadata["grid_spacing"]["dx"] == pytest.approx(0.125)
    assert out.metadata["boundary_residual_spacing_source"] == "reaction_diffusion_domain_metadata"
    assert out.metadata["boundary_residual_dx"] == pytest.approx(0.125)
    assert out.metadata["boundary_residual_dy"] == pytest.approx(0.125)
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

    assert out.residual.shape[0] == 1
    assert out.residual.shape[-2:] == (32, 32)
    assert out.metadata["residual_channels"]["interior_channels"] == 2
    assert out.metadata["residual_channels"]["bc_channels"] == 4
    assert torch.isfinite(out.residual).all()
    assert out.metadata["boundary_condition"] == "homogeneous_neumann"
    assert out.metadata["domain"]["defaulted"] is True
    assert out.metadata["grid_spacing"]["dx"] == pytest.approx(2.0 / 32.0)

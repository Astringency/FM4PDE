import math

import pytest

torch = pytest.importorskip("torch")

from sampling.pde_residuals import (
    _neumann_residual,
    _reaction_diffusion_laplacian,
    _rhs_nsnonbounded,
)


def test_reaction_diffusion_laplacian_uses_bcyx_axes_on_non_square_grid():
    height, width = 9, 14
    dx = 2.0 / width
    dy = 4.0 / height
    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx - 1.0
    y = (torch.arange(height, dtype=torch.float64) + 0.5) * dy - 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    field = (xx.square() + 3.0 * yy.square()).view(1, 1, height, width)
    lap = _reaction_diffusion_laplacian(
        field,
        {"x_range": [-1.0, 1.0], "y_range": [-2.0, 2.0]},
    )
    assert torch.allclose(lap[..., 1:-1, 1:-1], torch.full_like(lap[..., 1:-1, 1:-1], 8.0))


def test_reaction_diffusion_neumann_residual_uses_dx_for_left_right_and_dy_for_top_bottom():
    height, width = 5, 8
    dx, dy = 0.25, 0.5
    x = torch.arange(width, dtype=torch.float64) * dx
    y = torch.arange(height, dtype=torch.float64) * dy
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    field = (xx + 2.0 * yy).view(1, 1, height, width)
    residual = _neumann_residual(
        field,
        0.0,
        normalization="mean",
        pde_params={"dx": dx, "dy": dy},
        pde="reaction_diffusion",
    )
    assert residual[..., 1:-1, 0].abs().mean().item() == pytest.approx(1.0)
    assert residual[..., 0, 1:-1].abs().mean().item() == pytest.approx(2.0)


def _manual_generator_aligned_ns_rhs(w, viscosity, forcing):
    n = w.shape[-1]
    kmax = math.floor(n / 2)
    modes = torch.cat((torch.arange(0, kmax), torch.arange(-kmax, 0))).to(w)
    ky = modes.view(1, n).expand(n, n)
    kx = ky.t()
    lap = 4.0 * math.pi**2 * (kx.square() + ky.square())
    lap_safe = lap.clone()
    lap_safe[0, 0] = 1.0
    w_hat = torch.fft.fft2(w[:, 0], norm="forward")
    psi_hat = w_hat / lap_safe
    u = torch.fft.ifft2(1j * 2.0 * math.pi * ky * psi_hat, norm="forward").real
    v = torch.fft.ifft2(-1j * 2.0 * math.pi * kx * psi_hat, norm="forward").real
    wx = torch.fft.ifft2(1j * 2.0 * math.pi * kx * w_hat, norm="forward").real
    wy = torch.fft.ifft2(1j * 2.0 * math.pi * ky * w_hat, norm="forward").real
    dealias = ((kx.abs() <= (2.0 / 3.0) * kmax) & (ky.abs() <= (2.0 / 3.0) * kmax))
    nonlinear = torch.fft.ifft2(
        torch.fft.fft2(u * wx + v * wy, norm="forward") * dealias,
        norm="forward",
    ).real
    force = torch.fft.ifft2(
        torch.fft.fft2(forcing[:, 0], norm="forward") * dealias,
        norm="forward",
    ).real
    lap_w = torch.fft.ifft2(-lap * w_hat, norm="forward").real
    return (-nonlinear + viscosity * lap_w + force).unsqueeze(1)


def test_ns_default_rhs_matches_generator_two_thirds_operator():
    torch.manual_seed(3)
    w = torch.randn(2, 1, 16, 16, dtype=torch.float64)
    w = w - w.mean(dim=(-2, -1), keepdim=True)
    forcing = torch.randn_like(w)
    expected = _manual_generator_aligned_ns_rhs(w, 1e-3, forcing)
    actual = _rhs_nsnonbounded(
        w,
        {"nu": 1e-3, "forcing": forcing, "ns_operator_mode": "generator_dealiased"},
    )
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_ns_nonfinite_rhs_fails_instead_of_being_clamped():
    w = torch.zeros(1, 1, 8, 8)
    w[..., 0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        _rhs_nsnonbounded(w, {"nu": 1e-3, "forcing": 0.0})

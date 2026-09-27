"""CPU regressions for endpoint-averaged secant dynamics and guidance gradients.

Run with: python -m sampling.validate_secant_residual -v
"""
import math
import unittest
from types import SimpleNamespace

import torch

from sampling.config import AblationConfig
from sampling.losses import compute_guidance_losses
from sampling.masks import PairMasks
from sampling.pde_residuals import compute_pde_residual
from sampling.state import SplitState


def secant(pde, a, u, params):
    return compute_pde_residual(
        pde, a, u, pde_params=params, residual_mode="endpoint_secant"
    )


class SecantResidualTests(unittest.TestCase):
    def test_reaction_cubic_and_batchwise_time(self):
        # Spatially constant states have zero diffusion. Averaging the cubic
        # endpoint reactions gives residuals 4 and 3.5, unlike a midpoint RHS.
        a = torch.tensor([1.0, 0.0], dtype=torch.float64).view(1, 2, 1, 1)
        u = torch.tensor([2.0, 0.0], dtype=torch.float64).view(1, 2, 1, 1)
        a = a.expand(2, 2, 4, 4).clone().requires_grad_()
        u = u.expand(2, 2, 4, 4).clone().requires_grad_()
        params = {"T": torch.tensor([1.0, 2.0]), "k": 0.0}
        expected = torch.tensor([[4.0, -1.5], [3.5, -1.5]], dtype=a.dtype)
        expected = expected[:, :, None, None].expand_as(a)
        actual = secant("reaction_diffusion", a, u, params)
        torch.testing.assert_close(actual.residual, expected)
        self.assertEqual(actual.metadata["secant_rhs_evaluation"], "endpoint_average")

        # Exercise the public guidance path and its 1/d_h normalization too.
        cfg = AblationConfig(
            pde="reaction_diffusion", task="both", img_channels=4,
            guidance_components="pde_only", residual_mode="endpoint_secant",
        )
        truth = SimpleNamespace(coef=torch.zeros_like(a), sol=torch.zeros_like(u), pde_params=params)
        masks = PairMasks(torch.zeros_like(a), torch.zeros_like(u), {})
        losses = compute_guidance_losses(SplitState(a, u), truth, masks, cfg)
        torch.testing.assert_close(losses.L_pde, expected.square().mean())
        torch.testing.assert_close(losses.guidance_L_pde, expected.square().mean())
        for gradient in torch.autograd.grad(losses.guidance_L_pde, (a, u)):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0.0)

    def test_ns_distinct_shear_endpoints_have_no_self_advection(self):
        # Each endpoint is a single shear mode, so its own advection vanishes.
        # Their midpoint has interacting modes with different Laplacian
        # eigenvalues; the old implementation produces a spurious cross term.
        n = 16
        x = torch.arange(n, dtype=torch.float64).view(1, 1, n, 1) / n
        y = torch.arange(n, dtype=torch.float64).view(1, 1, 1, n) / n
        a = torch.sin(2 * math.pi * x).expand(1, 1, n, n)
        u = torch.sin(4 * math.pi * y).expand_as(a)
        forcing = 0.05 * torch.cos(2 * math.pi * (x + y))
        nu, time = 0.003, 0.7
        expected = (u - a) / time + 0.5 * nu * (
            (2 * math.pi)**2 * a + (4 * math.pi)**2 * u
        ) - forcing
        actual = secant("nsnonbounded", a, u, {
            "T": time, "nu": nu, "forcing": forcing,
        })
        torch.testing.assert_close(actual.residual, expected, rtol=1e-10, atol=1e-10)

    def test_shallow_water_averages_endpoint_pressure_fluxes(self):
        # Zero momenta isolate the nonlinear gravitational pressure h^2.
        # Include the original open-boundary replicated ghost-cell stencil.
        n, g, time = 8, 1.3, 0.6
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n, dtype=torch.float64).view(1, 1, n, 1)
        h0 = (1 + x).expand(1, 1, n, n)
        hT = (3 + 2 * x).expand_as(h0)
        zero = torch.zeros_like(h0)
        a, u = torch.cat([h0, zero, zero], 1), torch.cat([hT, zero, zero], 1)
        xp, xm = (x + dx).clamp_max(1), (x - dx).clamp_min(0)
        p_plus = 0.25 * g * ((1 + xp)**2 + (3 + 2 * xp)**2)
        p_minus = 0.25 * g * ((1 + xm)**2 + (3 + 2 * xm)**2)
        momentum = ((p_plus - p_minus) / (2 * dx)).expand_as(h0)
        expected = torch.cat([(hT - h0) / time, momentum, zero], 1)
        actual = secant("shallow_water", a, u, {"T": time, "g": g})
        torch.testing.assert_close(actual.residual, expected, rtol=1e-10, atol=1e-10)

    def test_linear_equations_preserve_fourier_mode_residuals(self):
        n, time, k = 16, 0.75, 2 * math.pi
        x = torch.arange(n, dtype=torch.float64).view(1, 1, n, 1) / n
        sine = torch.sin(k * x).expand(1, 1, n, n)
        cosine = torch.cos(k * x).expand_as(sine)
        cases = [
            ("heat", sine, 2 * sine, {"T": time, "alpha": 0.03},
             sine / time + 1.5 * 0.03 * k**2 * sine),
            ("advection_diffusion", sine, 2 * sine,
             {"T": time, "b_x": 0.4, "b_y": -0.2, "kappa": 0.01},
             sine / time + 1.5 * 0.4 * k * cosine + 1.5 * 0.01 * k**2 * sine),
            ("wave", torch.cat([sine, 0.3 * sine], 1),
             torch.cat([2 * sine, -0.5 * sine], 1), {"T": time, "c": 1.2},
             torch.cat([sine / time + 0.1 * sine,
                        -0.8 * sine / time + 1.5 * 1.2**2 * k**2 * sine], 1)),
        ]
        for pde, a, u, params, expected in cases:
            with self.subTest(pde=pde):
                actual = secant(pde, a, u, params)
                torch.testing.assert_close(actual.residual, expected, rtol=1e-10, atol=1e-10)

    def test_both_endpoint_gradients_match_finite_differences(self):
        generator = torch.Generator().manual_seed(20260927)
        cases = [
            ("heat", 1, {"alpha": 0.02}),
            ("wave", 2, {"c": 0.7}),
            ("advection_diffusion", 1, {"b_x": 0.3, "b_y": -0.1, "kappa": 0.01}),
            ("reaction_diffusion", 2, {}),
            ("shallow_water", 3, {}),
            ("nsnonbounded", 1, {}),
        ]
        for pde, channels, params in cases:
            with self.subTest(pde=pde):
                a = 0.1 * torch.randn(1, channels, 8, 8, dtype=torch.float64, generator=generator)
                u = 0.1 * torch.randn(1, channels, 8, 8, dtype=torch.float64, generator=generator)
                if pde == "shallow_water":
                    a[:, 0] += 2.0
                    u[:, 0] += 2.0
                a.requires_grad_()
                u.requires_grad_()
                params = {**params, "T": 0.7}
                self.assertTrue(torch.autograd.gradcheck(
                    lambda q0, qT: secant(pde, q0, qT, params).residual,
                    (a, u), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True,
                ))


if __name__ == "__main__":
    unittest.main()

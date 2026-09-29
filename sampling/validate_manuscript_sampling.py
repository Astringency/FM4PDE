"""CPU regressions for the manuscript sampling equations and raw Darcy coefficients.

Run with: python -m sampling.validate_manuscript_sampling -v
"""
import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sampling.config import AblationConfig, load_config
from sampling.guidance import (
    _clip_per_sample, apply_guidance_update, compute_guidance_gradient, make_zeta_schedule,
)
from sampling.losses import compute_guidance_losses
from sampling.masks import PairMasks
from sampling.pde_residuals import compute_pde_residual
from sampling.sampler_wrappers import sampler_step
from sampling.state import SplitState
from sampling.time_grid import condot_guidance_coefficient, endpoint_from_velocity, make_time_grid


class AffineVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t, extra=None):
        self.calls += 1
        return 0.2 * x + t


class ManuscriptSamplingTests(unittest.TestCase):
    def test_proposals_endpoint_and_single_network_evaluation(self):
        x = torch.arange(8, dtype=torch.float64).reshape(2, 1, 2, 2) / 10
        t, tn = torch.tensor(0.3), torch.tensor(0.31)
        v = 0.2 * x + t
        endpoint = x + (1 - t) * v
        for phase in ['deterministic', 'stochastic']:
            for state in ['xt', 'x_next', 'endpoint']:
                with self.subTest(phase=phase, state=state):
                    net = AffineVelocity()
                    torch.manual_seed(52)
                    noise = torch.randn_like(x)
                    expected = x + (tn - t) * v if phase == 'deterministic' else (1 - tn) * noise + tn * endpoint
                    torch.manual_seed(52)
                    out = sampler_step(net, x, t, tn, phase, state)
                    torch.testing.assert_close(out.x_raw_next, expected)
                    torch.testing.assert_close(out.x_endpoint, endpoint)
                    torch.testing.assert_close(out.x_loss_state, {'xt': x, 'x_next': expected, 'endpoint': endpoint}[state])
                    self.assertEqual(net.calls, 1)
        torch.testing.assert_close(endpoint_from_velocity(x, v, torch.tensor(1.)), x, rtol=0, atol=0)

    def test_update_multiplier_including_first_step_and_tiny_positive_time(self):
        cfg = AblationConfig()
        self.assertEqual(cfg.stochastic_guidance_coeff, 0.1)
        proposal = torch.ones(2, 1, 2, 2, dtype=torch.float64)
        direction = torch.full_like(proposal, 0.7)
        for phase in ['deterministic', 'stochastic']:
            for time in [0., 1e-7, .01, .4, .99]:
                with self.subTest(phase=phase, time=time):
                    t = torch.tensor(time, dtype=proposal.dtype)
                    dt = torch.tensor(.01, dtype=proposal.dtype)
                    schedule = make_zeta_schedule(cfg, t, condot_guidance_coefficient(t), step=1)
                    step = SimpleNamespace(phase=phase, t=t, t_next=t+dt, step_size=dt)
                    gradient = SimpleNamespace(grad_total=direction, metadata={})
                    gamma = (0. if time == 0 else .01 * (1-time)/time) if phase == 'deterministic' else .1*(1-time)
                    actual = apply_guidance_update(proposal, gradient, step, schedule, cfg)
                    torch.testing.assert_close(actual, proposal-gamma*direction, rtol=1e-13, atol=1e-13)

    def test_unfloored_geometric_scale_is_finite_at_subnormal_times(self):
        cfg = AblationConfig(sampler_phase='deterministic', num_steps=2000)
        grid = make_time_grid('geometric', cfg.num_steps, eta=.05)
        times, increments = grid[:-1], torch.diff(grid)
        self.assertGreater(grid[1].item(), 0.)
        self.assertLess(grid[1].item(), torch.finfo(grid.dtype).tiny)
        # Test the full published grid against a float64 mathematical reference.
        actual = torch.empty_like(times)
        for k, (time, increment) in enumerate(zip(times, increments)):
            schedule = make_zeta_schedule(cfg, time, condot_guidance_coefficient(time), step=k)
            step = SimpleNamespace(phase='deterministic', t=time, step_size=increment)
            gradient = SimpleNamespace(grad_total=torch.ones(1), metadata={})
            actual[k] = apply_guidance_update(torch.zeros(1), gradient, step, schedule, cfg)[0]
        expected = torch.zeros_like(times, dtype=torch.float64)
        expected[1:] = -(increments[1:].double()/times[1:].double())*(1-times[1:].double())
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.isfinite(condot_guidance_coefficient(times)).all())
        self.assertEqual(actual[0].item(), 0.)
        torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-9)

    def test_weighted_chain_rule_global_clip_and_batch_independence(self):
        cfg = AblationConfig(task='both', clip_threshold=0.7, zeta_obs_a=2., zeta_obs_u=3., zeta_pde=4., pde_guidance_start_ratio=0.)
        values = torch.tensor([.1, .2, .3, .4, 2., -1., 4., 3.], dtype=torch.float64).reshape(2, 1, 2, 2)

        def gradient_for(x):
            x = x.clone().requires_grad_()
            endpoint = x + .6 * (.2*x + .4)
            losses = SimpleNamespace(
                L_obs_a=endpoint.square().mean(), L_obs_u=(endpoint-1).square().mean(), L_pde=(2*endpoint+1).square().mean(),
                guidance_L_obs_a=None, guidance_L_obs_u=None, guidance_L_pde=None,
                metadata={'loss_batch_reduction': 'mean_of_per_sample'},
            )
            t = torch.tensor(.4, dtype=x.dtype)
            schedule = make_zeta_schedule(cfg, t, condot_guidance_coefficient(t), step=40)
            return compute_guidance_gradient(losses, x, schedule, cfg).grad_total

        endpoint = 1.12 * values + .24
        # d/dx [2 E^2 + 3(E-1)^2 + 4(2E+1)^2] / 4 entries.
        expected = 1.12 * (4*endpoint + 6*(endpoint-1) + 16*(2*endpoint+1)) / 4
        norms = expected.flatten(1).norm(dim=1)
        expected = expected * (.7/torch.maximum(norms, torch.full_like(norms, .7)))[:, None, None, None]
        actual = gradient_for(values)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, torch.cat([gradient_for(row[None]) for row in values]))
        zero, scale = _clip_per_sample(torch.zeros_like(values), .7)
        self.assertEqual(scale, 1.)
        self.assertEqual(zero.count_nonzero(), 0)
        huge = torch.full((1, 1, 2, 2), 1e25, dtype=torch.float32)
        clipped, _ = _clip_per_sample(huge, 2.)
        torch.testing.assert_close(clipped, torch.ones_like(huge))

    def test_pde_gate_is_step_based_on_both_published_grids(self):
        cfg = AblationConfig(sampler_phase='deterministic', num_steps=100)
        for kind in ['uniform', 'geometric']:
            grid = make_time_grid(kind, 100, eta=.05)
            self.assertEqual(len(grid), 101)
            self.assertEqual(grid[0], 0.)
            self.assertEqual(grid[-1], 1.)
            if kind == 'geometric':
                expected = 1.05 ** (torch.arange(1, 101, dtype=torch.float64)-100)
                torch.testing.assert_close(grid[1:].double(), expected, rtol=1e-6, atol=1e-8)
            for step in [0, 79, 80, 99]:
                schedule = make_zeta_schedule(cfg, grid[step], condot_guidance_coefficient(grid[step]), step=step)
                self.assertEqual(float(schedule.zeta_pde_t), float(step >= 80))

    def test_darcy_keeps_continuous_zero_and_negative_coefficients(self):
        coord = torch.linspace(0, 1, 8, dtype=torch.float64)
        u = (coord*(1-coord))[:, None] * (coord*(1-coord))[None, :]
        u = u[None, None]

        def residual(a):
            return compute_pde_residual('darcy', a, u).components['interior'][..., 1:-1, 1:-1]

        reference = residual(torch.ones_like(u)) + 1
        for coefficient in [-2., 0., 6., 6.1, 10.3]:
            with self.subTest(coefficient=coefficient):
                # Darcy's interior operator is linear in a, with source term -1.
                torch.testing.assert_close(residual(torch.full_like(u, coefficient)), coefficient*reference-1)

        a = torch.full_like(u, 6., requires_grad=True)
        cfg = AblationConfig(pde='darcy', task='both', guidance_components='pde_only')
        truth = SimpleNamespace(coef=torch.zeros_like(a), sol=torch.zeros_like(u), pde_params={})
        masks = PairMasks(torch.zeros_like(a), torch.zeros_like(u), {})
        loss = compute_guidance_losses(SplitState(a, u), truth, masks, cfg).guidance_L_pde
        grad = torch.autograd.grad(loss, a)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.norm().item(), 0.)
        self.assertTrue(torch.autograd.gradcheck(residual, (a,), fast_mode=True))

    def test_nonfinite_update_is_reported_and_not_silently_discarded(self):
        cfg = AblationConfig()
        t = torch.tensor(.2)
        step = SimpleNamespace(phase='deterministic', t=t, t_next=t+.01, step_size=torch.tensor(.01))
        schedule = make_zeta_schedule(cfg, t, condot_guidance_coefficient(t), step=20)
        gradient = SimpleNamespace(grad_total=torch.full((1, 1, 2, 2), float('nan')), metadata={})
        with self.assertRaises(FloatingPointError):
            apply_guidance_update(torch.zeros_like(gradient.grad_total), gradient, step, schedule, cfg)

    def test_shallow_water_zero_depth_guard_is_preserved(self):
        state = torch.zeros(1, 3, 8, 8, dtype=torch.float64)
        state[:, 1] = torch.linspace(0, 1, 8)[None, :, None]
        state[:, 2] = .5
        result = compute_pde_residual('shallow_water', state, state, pde_params={'T': 1.}, residual_mode='endpoint_secant')
        self.assertTrue(torch.isfinite(result.residual).all())

    def test_retained_hermite_near_endpoint_and_burgers_residuals(self):
        a = torch.full((1, 1, 8, 8), 2., dtype=torch.float64, requires_grad=True)
        u = torch.full_like(a, 3., requires_grad=True)
        params = {'T': 1., 'alpha': .01}
        result = compute_pde_residual('heat', a, u, pde_params=params, residual_mode='hermite_bridge')
        # Constant spatial fields have G=0. Hermite H=2+3s^2-2s^3,
        # hence dH/ds=(1.125,1.5,1.125) at the three manuscript points.
        expected = torch.tensor([1.125, 1.5, 1.125], dtype=a.dtype)[None, :, None, None].expand(1, 3, 8, 8)
        torch.testing.assert_close(result.residual, expected)
        self.assertIsNone(result.components['endpoint'])

        mask = torch.zeros_like(a)
        mask[..., 2, 3] = 1
        near = {'q_dt': torch.full_like(a, 2.1), 'q_T_minus_dt': torch.full_like(u, 2.9),
                'dt': .1, 'mask_0': mask, 'mask_T': mask}
        truth = SimpleNamespace(coef=a.detach(), sol=u.detach(), pde_params={**params, 'near_endpoint_temporal': near})
        cfg = AblationConfig(pde='heat', task='both', guidance_components='pde_only', residual_mode='near_endpoint_temporal')
        masks = PairMasks(mask, mask, {})
        loss = compute_guidance_losses(SplitState(a, u), truth, masks, cfg).L_pde
        # Both observed time differences are exactly 1; unobserved entries
        # must contribute nothing and must not dilute the normalized loss.
        torch.testing.assert_close(loss, torch.ones_like(loss))
        for grad in torch.autograd.grad(loss, (a, u)):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm().item(), 0.)

        trajectory = torch.full_like(a, 2.)
        result = compute_pde_residual('burger', trajectory, trajectory, residual_mode='full_time_space')
        torch.testing.assert_close(result.residual, torch.zeros_like(result.residual))
        self.assertEqual(result.metadata['resolved_residual_mode'], 'full_time_space')

    def test_removed_config_options_are_rejected(self):
        for key in ['coef_positive_mode', 'deterministic_bt_max_scale', 'step_method', 'gradient_target', 'cfg_scale']:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'old.yaml'
                path.write_text(f'{key}: 1\n')
                with self.assertRaisesRegex(ValueError, 'Unknown config fields'):
                    load_config(path)
        with self.assertRaises(ValueError):
            AblationConfig(time_grid='cosine').validate()
        with self.assertRaises(ValueError):
            AblationConfig(time_grid='geometric', sampler_phase='stochastic').validate()

    def test_public_runner_completes_100_steps_for_all_phases(self):
        from data.transform import PDEStandardizer
        from sampling.data import PDEGroundTruth
        from sampling.runner import run_single_ablation

        a = torch.ones(1, 1, 8, 8)
        u = torch.ones_like(a)*.1
        truth = PDEGroundTruth(
            pde='poisson', coef=a, sol=u, pair=torch.cat([a, u], 1), pde_params={},
            channel_names_coef=['a'], channel_names_sol=['u'], metadata={'synthetic': True},
        )
        masks = PairMasks(torch.ones_like(a), torch.ones_like(u), {})
        for phase, grid in [('stochastic', 'uniform'), ('deterministic', 'uniform'), ('deterministic', 'geometric'), ('hybrid_d2s', 'uniform'), ('hybrid_s2d', 'uniform')]:
            with self.subTest(phase=phase, grid=grid), tempfile.TemporaryDirectory() as tmp:
                cfg = AblationConfig(
                    task='both', sampler_phase=phase, time_grid=grid, time_grid_eta=.05,
                    num_steps=100, img_resolution=8, num_obs=64, output_dir=tmp,
                    zeta_obs_a=1e-4, zeta_obs_u=1e-4, zeta_pde=1e-8,
                )
                net = AffineVelocity()
                normalizer = PDEStandardizer(torch.zeros(2), torch.ones(2))
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_single_ablation(cfg, (net, normalizer, {}), ground_truth=truth, observation_masks=masks)
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(net.calls, 100)
                self.assertTrue(math.isfinite(result['rel_l2_u']))


if __name__ == '__main__':
    unittest.main()

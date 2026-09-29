"""CPU checks for manuscript metrics, saved artifacts and experiment summaries.

Run with: python -m sampling.validate_reproduction -v
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.paper.residual_metrics import full_pde_loss, evaluate_fields
from experiments.paper.run import digest, write_json
from experiments.paper.summarize import summarize
from sampling.config import AblationConfig
from sampling.losses import compute_guidance_losses, ObservationTargets
from sampling.masks import PairMasks
from sampling.metrics import per_sample_metrics, pde_loss_per_sample
from sampling.state import SplitState


class ReproductionTests(unittest.TestCase):
    def test_noisy_observed_error_uses_measurements_and_field_error_uses_clean_truth(self):
        a = torch.ones(2, 1, 8, 8, dtype=torch.float64)
        clean, measured = 2*a, 3*a
        masks = PairMasks(a, a, {})
        truth = SimpleNamespace(coef=a, sol=clean, pde_params={}, metadata={})
        state = SplitState(a, measured)
        losses = compute_guidance_losses(state, truth, masks, AblationConfig(task='both'),
                                         ObservationTargets(a, clean, a, measured))
        rows = per_sample_metrics(state, truth, masks, losses)
        self.assertEqual([r['obs_rel_l2_u'] for r in rows], [0., 0.])
        self.assertEqual([r['rel_l2_u'] for r in rows], [.5, .5])

    def test_weighted_component_mses_are_not_concatenated_residual_mse(self):
        interior = torch.full((2, 1, 4, 4), 2.)
        boundary = torch.full_like(interior, 3.)
        endpoint = torch.full_like(interior, 4.)
        losses = SimpleNamespace(
            pde_components=dict(interior=interior, boundary=boundary, endpoint=endpoint),
            metadata={'pde': {'component_losses': {'bc_weight': 2., 'endpoint_weight': .5}}})
        self.assertEqual(pde_loss_per_sample(losses), [30., 30.])

    def test_poisson_full_loss_includes_boundary_and_survives_disabled_guidance(self):
        a = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
        u = torch.full_like(a, 2.)
        cfg = AblationConfig(task='both', boundary_condition_mode='dirichlet_zero',
                             guidance_components='noguide', zeta_pde=0.)
        data = dict(coef_final=a, sol_final=u, coef_ground_truth=a, sol_ground_truth=u,
                    config=cfg.asdict(), pde_params={})
        self.assertEqual(evaluate_fields('poisson', a, u, a, u)['residual_mse'], [0.])
        self.assertAlmostEqual(full_pde_loss(data)[0], 4., places=12)

    def test_helmholtz_full_loss_includes_modified_boundary_rows(self):
        a = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
        u = torch.ones_like(a)
        cfg = AblationConfig(pde='helmholtz', task='both')
        data = dict(coef_final=a, sol_final=u, coef_ground_truth=a, sol_ground_truth=u,
                    config=cfg.asdict(), pde_params={})
        # Constant u: 36 interior entries have residual 1, 24 edges 2, 4 corners 3.
        self.assertAlmostEqual(full_pde_loss(data)[0], (36 + 24*4 + 4*9)/64, places=12)
        self.assertEqual(evaluate_fields('helmholtz', a, u, a, u)['residual_mse'], [1.])

    def test_saved_near_endpoint_measurements_can_be_reevaluated(self):
        from sampling.runner import _sanitize_pde_params_for_artifact
        a = torch.full((1, 1, 8, 8), 2., dtype=torch.float64)
        u = torch.full_like(a, 3.)
        mask = torch.zeros_like(a)
        mask[..., 2, 3] = 1
        params = dict(T=1., alpha=.01, near_endpoint_temporal=dict(
            q_dt=a+.1, q_T_minus_dt=u-.1, dt=.1, mask_0=mask, mask_T=mask))
        cfg = AblationConfig(pde='heat', task='both', residual_mode='near_endpoint_temporal')
        data = dict(coef_final=a, sol_final=u, coef_ground_truth=a, sol_ground_truth=u,
                    config=cfg.asdict(), pde_params=_sanitize_pde_params_for_artifact(params, cfg))
        self.assertAlmostEqual(full_pde_loss(data)[0], 1., places=12)

    def make_summary_input(self, folder):
        records = []
        for offset, errors in [(0, [.1, .2]), (2, [.9])]:
            cfg = AblationConfig(task='both', offset=offset, batch_size=len(errors),
                                 output_dir=str(folder/str(offset)))
            truth = torch.ones(len(errors), 1, 8, 8, dtype=torch.float64)
            prediction = truth * (1 + torch.tensor(errors, dtype=truth.dtype)[:, None, None, None])
            path = folder/f'{offset}.pt'
            torch.save(dict(coef_final=prediction, sol_final=prediction, coef_ground_truth=truth,
                            sol_ground_truth=truth, config=cfg.asdict(), pde_params={}), path)
            records.append(dict(id=str(offset), result=str(path), result_sha256=digest(path),
                                identity=dict(config=cfg.asdict())))
        write_json(folder/'index.json', records)
        return records

    def test_summary_pools_samples_before_mean_std_and_full_pde_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.make_summary_input(folder)
            summarize(folder)
            rows = json.loads((folder/'summary.json').read_text())
            field = next(r for r in rows if r['field'] == 'u')
            self.assertEqual(field['count'], 3)
            self.assertAlmostEqual(field['mean'], .4, places=12)
            self.assertAlmostEqual(field['std'], ((.09+.04+.25)/2)**.5, places=12)
            self.assertTrue(any(r['metric'] == 'pde_loss' for r in rows))

    def test_summary_rejects_duplicate_samples_and_changed_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            records = self.make_summary_input(folder)
            duplicate = dict(records[0], id='duplicate')
            write_json(folder/'index.json', records+[duplicate])
            with self.assertRaisesRegex(ValueError, 'Duplicate sample'):
                summarize(folder)
            write_json(folder/'index.json', records)
            Path(records[0]['result']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                summarize(folder)

    def test_public_runner_saves_actual_measurements_and_sample_losses(self):
        from data.transform import PDEStandardizer
        from sampling.data import PDEGroundTruth
        from sampling.runner import run_single_ablation
        from sampling.validate_manuscript_sampling import AffineVelocity
        a = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)/64 + 1
        u = .1*a
        truth = PDEGroundTruth('poisson', a, u, torch.cat([a,u],1), {}, ['a'], ['u'], {})
        masks = PairMasks(torch.ones_like(a), torch.ones_like(u), {})
        with tempfile.TemporaryDirectory() as temporary:
            cfg = AblationConfig(task='both', num_steps=3, img_resolution=8, num_obs=64,
                output_dir=temporary, noise_level=.1, zeta_obs_a=1e-4, zeta_obs_u=1e-4, zeta_pde=1e-8)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_single_ablation(cfg, (AffineVelocity(), PDEStandardizer(torch.zeros(2),torch.ones(2)), {}),
                                             ground_truth=truth, observation_masks=masks)
            data = torch.load(Path(result['run_dir'])/'result.pt', weights_only=False)
            self.assertFalse(torch.equal(data['observations']['sol'], u))
            expected = (data['sol_final']-data['observations']['sol']).norm()/data['observations']['sol'].norm()
            self.assertAlmostEqual(data['metrics_per_sample'][0]['obs_rel_l2_u'], expected.item(), places=6)
            # Sampling evaluates float32 residuals; offline reporting recomputes
            # them in float64 from the saved fields.
            torch.testing.assert_close(torch.tensor(data['metrics_per_sample'][0]['L_pde']),
                                       torch.tensor(full_pde_loss(data)[0]), rtol=1e-6, atol=1e-8)

    def test_runtime_identity_records_numerical_settings(self):
        from experiments.paper.provenance import runtime_identity
        identity = runtime_identity('cpu')
        self.assertIn('torch', identity['packages'])
        self.assertEqual(identity['threads'], torch.get_num_threads())
        old = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.benchmark = not old
            self.assertNotEqual(identity, runtime_identity('cpu'))
        finally:
            torch.backends.cudnn.benchmark = old

    def test_burgers_solver_cache_checks_recorded_source_and_inputs(self):
        import numpy as np
        import scipy.io
        from experiments.paper import consistency_helpers as h
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            initial = np.ones((1,128))
            scipy.io.savemat(folder/'initial.mat', dict(initial=initial))
            scipy.io.savemat(folder/'solved.mat', dict(trajectory=np.ones((1,128,128))))
            receipt = dict(input_sha256=digest(folder/'initial.mat'),
                output_sha256=digest(folder/'solved.mat'),
                generator_sha256=digest(h.ROOT/'data/DataGen/static/burgers1.m'),
                wrapper_sha256=digest(Path(h.__file__).with_name('burger_solve.m')),
                parameters=dict(viscosity=.01,spatial_points=128,time_points=128,T=1,dt=1/127))
            write_json(folder/'solve.json', receipt)
            self.assertEqual(h.solve(initial,folder,SimpleNamespace()).shape, (1,1,128,128))
            with self.assertRaisesRegex(ValueError, 'different initial conditions'):
                h.solve(2*initial,folder,SimpleNamespace())
            receipt['generator_sha256'] = 'old source'
            write_json(folder/'solve.json', receipt)
            with self.assertRaisesRegex(ValueError, 'source or parameters changed'):
                h.solve(initial,folder,SimpleNamespace())

    def test_plot_uses_saved_measurements_and_selected_sample_metrics(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from plot.plot import plot_from_result_pt
        a = torch.ones(2, 1, 8, 8)
        u = 2*a
        prediction = u.clone()
        prediction[1] = 3.
        data = dict(coef_ground_truth=a, sol_ground_truth=u, coef_final=a, sol_final=prediction,
                    observations=dict(coef=a, sol=4*a), masks=dict(coef=a, sol=a),
                    config=dict(noise_level=.1), metrics=dict(rel_l2_u=.25, L_pde=10.),
                    metrics_per_sample=[dict(L_pde=2.), dict(L_pde=18.)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'result.pt'
            torch.save(data, path)
            fig = plot_from_result_pt(path, sample_index=1)
            try:
                np.testing.assert_array_equal(fig.axes[5].images[0].get_array(), np.full((8,8), 4.))
                self.assertIn('rel_l2=0.5000', fig.axes[7].get_title())
                self.assertIn('L_pde=1.80e+01', fig._suptitle.get_text())
            finally:
                plt.close(fig)

    def test_q_uses_three_cohort_means_and_requires_all_methods(self):
        from experiments.paper.score import score_means
        means = dict(
            fm4pde=dict(relative_l2_joint=2., pde_loss=1., solver_relative_l2=1.),
            recfno=dict(relative_l2_joint=1., pde_loss=8., solver_relative_l2=1.),
            senseiver=dict(relative_l2_joint=2., pde_loss=4., solver_relative_l2=8.),
            voronoicnn=dict(relative_l2_joint=4., pde_loss=2., solver_relative_l2=1.))
        scores = score_means(means)
        self.assertAlmostEqual(scores['fm4pde'], 2**(1/3))
        self.assertAlmostEqual(scores['recfno'], 2.)
        self.assertAlmostEqual(scores['senseiver'], 4.)
        with self.assertRaisesRegex(ValueError, 'four methods'):
            score_means({'fm4pde': means['fm4pde']})
        means['fm4pde']['pde_loss'] = 0.
        with self.assertRaisesRegex(ValueError, 'strictly positive'):
            score_means(means)

    def test_q_csv_pools_samples_and_rejects_missing_or_duplicate_ids(self):
        import csv
        from experiments.paper.score import collect, METHODS
        with tempfile.TemporaryDirectory() as temporary:
            inputs = []
            for method in METHODS:
                path = Path(temporary)/f'{method}.csv'
                rows = []
                for index in range(2):
                    error, loss = ((1., 9.) if index == 0 else (9., 1.)) if method == 'fm4pde' else (3., 3.)
                    rows.append(dict(pde='poisson', distribution='id', index=index,
                                     relative_l2_a=error, relative_l2_u=error,
                                     pde_loss=loss, solver_relative_l2=1.))
                with path.open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                inputs.append((method,path))
            report = collect(inputs, expected_count=2)
            self.assertAlmostEqual(report['rows'][0]['Q']['fm4pde'], (25/9)**(1/3))
            with self.assertRaisesRegex(ValueError, 'expected input IDs'):
                collect(inputs, expected_count=100)
            with self.assertRaisesRegex(ValueError, 'Duplicate input'):
                collect(inputs+[inputs[0]], expected_count=2)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

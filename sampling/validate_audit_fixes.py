"""CPU regression checks for manuscript cohorts, timing controls and NS means.

Run with: python -m sampling.validate_audit_fixes -v
"""
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from experiments.paper.run import ROOT, digest, write_json
from experiments.paper.summarize import summarize
from sampling.config import load_config


class AuditFixTests(unittest.TestCase):
    def make_ablation_records(self, folder, pde, *, split_batches, with_cohorts):
        manifest = ROOT/'configs/experiments/ablations/observation_pde_guidance.yaml'
        jobs = {j['id']: j for j in yaml.safe_load(manifest.read_text())['jobs']}
        ids = {'poisson': ('archive_0589', 'weight_poisson_0.1'),
               'burger': ('archive_0099', 'weight_burger_10')}[pde]
        records = []
        for experiment, job_id in enumerate(ids):
            job = jobs[job_id]
            batches = [(0, 10), (10, 10)] if split_batches and experiment == 0 else [(0, 20)]
            for shift, count in batches:
                cfg = load_config(ROOT/f'configs/main/both/{pde}.yaml',
                                  {**job['overrides'], 'batch_size': count,
                                   'offset': job['overrides']['offset'] + shift})
                truth = torch.ones(count, 1, 8, 8, dtype=torch.float64)
                prediction = truth * (2 if experiment == 0 else 4)
                identity = dict(config=cfg.asdict())
                if with_cohorts:
                    identity['cohort'] = job['cohort']
                name = job_id if shift == 0 else f'{job_id}_batch_{shift}'
                path = folder/f'{name}.pt'
                torch.save(dict(coef_final=prediction, sol_final=prediction,
                                coef_ground_truth=truth, sol_ground_truth=truth,
                                config=cfg.asdict(), pde_params={}), path)
                records.append(dict(id=name, identity=identity, result=str(path),
                                    result_sha256=digest(path)))
        write_json(folder/'index.json', records)
        return manifest

    def check_ablation_summary(self, folder):
        rows = json.loads((folder/'summary.json').read_text())
        means = {row['settings']['cohort']: (row['count'], row['mean'])
                 for row in rows if row['field'] == 'u'}
        self.assertEqual(means, {'guidance_components': (20, 1.), 'pde_weight': (20, 3.)})

    def test_ablation_cohorts_stay_separate_while_batches_pool(self):
        for pde in ('poisson', 'burger'):
            with self.subTest(pde=pde), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                self.make_ablation_records(folder, pde, split_batches=True, with_cohorts=True)
                summarize(folder)
                self.check_ablation_summary(folder)

    def test_old_receipts_can_be_regrouped_without_resampling(self):
        for pde in ('poisson', 'burger'):
            with self.subTest(pde=pde), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                manifest = self.make_ablation_records(folder, pde, split_batches=False, with_cohorts=False)
                before = digest(folder/'index.json')
                summarize(folder, manifest)
                self.check_ablation_summary(folder)
                self.assertEqual(digest(folder/'index.json'), before)

    def test_summary_rejects_conflicting_cohort_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = self.make_ablation_records(folder, 'poisson', split_batches=False, with_cohorts=True)
            records = json.loads((folder/'index.json').read_text())
            records[0]['identity']['cohort'] = 'wrong_cohort'
            write_json(folder/'index.json', records)
            with self.assertRaisesRegex(ValueError, 'Cohort differs'):
                summarize(folder, manifest)

    def test_timing_uses_manuscript_weights_and_stochastic_updates(self):
        from experiments.paper.timing_inputs import plan_overrides, scientific_controls
        expected = {
            'poisson': (5e4, 9e7, .1, 50.),
            'helmholtz': (2.5e4, 1e8, .1, 150.),
            'darcy': (10., 5e8, .1, 100.),
            'nsnonbounded': (6e5, 6e5, 100., 150.),
            'burger': (0., 52428800., 10., 50.),
        }
        protocol = json.loads((ROOT/'configs/experiments/comparison/timing_protocol.json').read_text())
        with patch.dict('os.environ', {}, clear=True):
            for pde, weights in expected.items():
                values = plan_overrides(pde)
                self.assertEqual(tuple(values[k] for k in ('zeta_obs_a', 'zeta_obs_u', 'zeta_pde', 'clip_threshold')), weights)
                self.assertEqual(values['sampler_phase'], 'stochastic')
                self.assertEqual(protocol['fm_configs'][pde], scientific_controls(pde))
            self.assertEqual(plan_overrides('burger')['sensor_mode'], 'time_slices')
            # Old input bundles supply assets, never obsolete scientific controls.
            with patch.dict('os.environ', {'TIMING_INPUT_ROOT': '/archive'}):
                for pde in expected:
                    values = plan_overrides(pde)
                    self.assertEqual(values['zeta_pde'], expected[pde][2])
                    self.assertEqual(values['checkpoint_path'], f'/archive/weights/fm_{pde}.pth')

    def timing_inputs_without_assets(self, pde):
        from experiments.paper.timing_inputs import TimingInputs
        inputs = TimingInputs.__new__(TimingInputs)
        inputs.pde = pde
        root = ROOT/'configs/experiments/comparison'
        inputs.protocol = json.loads((root/'timing_protocol.json').read_text())
        self.assertEqual(digest(root/'timing_masks.npz'), inputs.protocol['masks_sha256'])
        with np.load(root/'timing_masks.npz') as arrays:
            inputs.masks = {key: arrays[key].copy() for key in arrays.files}
        inputs.ids = inputs.protocol['evaluation_ids']
        inputs.pilot = inputs.protocol['pilot_id']
        inputs.order = [*inputs.ids, inputs.pilot]
        return inputs

    def test_timing_masks_observe_five_complete_time_slices(self):
        for pde in ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger'):
            inputs = self.timing_inputs_without_assets(pde)
            inputs._verify_random_selection()
            for index in inputs.order:
                for field in ('a', 'u'):
                    mask = inputs.observation_mask(index, field)
                    self.assertEqual(mask.sum(), 640 if pde == 'burger' else 500)
                    if pde == 'burger':
                        self.assertEqual(np.count_nonzero(mask.sum(axis=1)), 5)
                        self.assertTrue(np.all(mask.sum(axis=0) == 5))

    def test_legacy_timing_masks_are_adapted_before_both_methods_use_them(self):
        inputs = self.timing_inputs_without_assets('burger')
        inputs.protocol.pop('burger_observation_layout')
        expected = {k: v.copy() for k, v in inputs.masks.items()}
        for key in inputs.masks:
            if key.startswith('burger_'):
                inputs.masks[key] = inputs.masks[key].T.copy()
        inputs._verify_random_selection()
        inputs.public = False
        zeros = np.zeros((len(inputs.order), 1, 128, 128), dtype=np.float32)
        inputs.data = {'burger_a': zeros, 'burger_u': zeros}
        for index in inputs.order:
            _, masks = inputs.case(index, 'cpu')
            for field, mask in [('a', masks.coef), ('u', masks.sol)]:
                np.testing.assert_array_equal(mask[0, 0].numpy(), expected[f'burger_{index}_{field}'])

    def solve_ns(self, initial, force, *, visc=.1, T=.2, dt=.002):
        from data.DataGen.time_dependent.no_bound_ns.ns_2d import navier_stokes_2d
        with contextlib.redirect_stderr(io.StringIO()):
            return navier_stokes_2d(initial, force, visc, T, dt, 1)[2][..., -1]

    def test_ns_constant_vorticity_is_not_damped(self):
        initial = torch.ones(2, 8, 8)
        initial[1] *= -2
        result = self.solve_ns(initial, torch.zeros(8, 8))
        torch.testing.assert_close(result, initial, rtol=0, atol=1e-7)

    def test_ns_mean_responds_only_to_mean_forcing(self):
        initial = torch.ones(1, 8, 8)
        result = self.solve_ns(initial, torch.full((8, 8), .25))
        torch.testing.assert_close(result, initial + .25*.2, rtol=0, atol=5e-6)

    def test_ns_nonzero_mode_retains_crank_nicolson_diffusion(self):
        coordinate = torch.arange(8, dtype=torch.float32)/8
        mode = torch.sin(2*math.pi*coordinate)[None, :, None].expand(1, 8, 8)
        visc, dt, steps = .1, .002, 100
        amplification = ((1-.5*dt*visc*4*math.pi**2)/(1+.5*dt*visc*4*math.pi**2))**steps
        result = self.solve_ns(1+mode, torch.zeros(8, 8), visc=visc, T=dt*steps, dt=dt)
        torch.testing.assert_close(result, 1+amplification*mode, rtol=0, atol=4e-6)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

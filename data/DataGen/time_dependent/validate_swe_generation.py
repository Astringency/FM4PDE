"""Small PyClaw integration checks for shallow-water generation.

Run with: python -m data.DataGen.time_dependent.validate_swe_generation -v
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from data.DataGen.time_dependent.gen_swe import generate_dataset, parse_args


class ShallowWaterGenerationTests(unittest.TestCase):
    def check_sample(self, group, radius, height):
        self.assertEqual(float(group.attrs['dam_radius']), radius)
        self.assertEqual(float(group.attrs['inner_height']), height)
        initial = group['data/h'][0, ..., 0]
        self.assertTrue(np.isfinite(group['data/h'][:]).all())
        self.assertAlmostEqual(float(initial.min()), 1., places=6)
        if float(group.attrs.get('transition_width', 0)) == 0:
            # Inspect the field, not just the metadata: the historical bug
            # recorded a random height while initializing every field at 2.
            self.assertAlmostEqual(float(initial.max()), height, places=6)
            self.assertTrue(np.any(initial == 1.))
            self.assertTrue(np.any(initial == np.float32(height)))
        self.assertEqual(float(np.abs(group['data/hu'][0]).max()), 0.)
        self.assertEqual(float(np.abs(group['data/hv'][0]).max()), 0.)

    def test_training_shards_use_fixed_actual_depth_and_historical_radii(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--out-dir', tmp, '--split', 'train', '--total-samples', '3',
                               '--samples-per-file', '2', '--resolution', '16',
                               '--tsteps', '1', '--T', '0.01'])
            paths = generate_dataset(args)
            self.assertEqual(len(paths), 2)
            # The first radii in the released training data, seeds 0 and 1.
            expected_first_radii = [0.5547846749285816, 0.5047286498801027]
            seen = []
            for path in paths:
                with h5py.File(path, 'r') as file:
                    for key in sorted(k for k in file if k.isdigit()):
                        group = file[key]
                        seed = int(group.attrs['seed'])
                        radius = float(np.random.default_rng(seed).uniform(.3, .7))
                        self.check_sample(group, radius, 2.)
                        if seed < 2:
                            self.assertEqual(radius, expected_first_radii[seed])
                        seen.append(seed)
            self.assertEqual(seen, [0, 1, 2])

    def test_test_profiles_preserve_random_radius_height_and_phase_draws(self):
        for profile, base_seed in [('id', 10000000), ('smooth', 20000000), ('rough', 30000000)]:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                args = parse_args(['--out-dir', tmp, '--split', 'test', '--dataset-type', profile,
                                   '--total-samples', '2', '--resolution', '16',
                                   '--tsteps', '1', '--T', '0.01'])
                path, = generate_dataset(args)
                heights = []
                with h5py.File(path, 'r') as file:
                    for index in range(2):
                        rng = np.random.default_rng(base_seed + index)
                        radius, height = float(rng.uniform(.4, .8)), float(rng.uniform(2., 3.))
                        phase = float(rng.uniform(0., 2*np.pi))
                        group = file[f'{index:06d}']
                        self.check_sample(group, radius, height)
                        self.assertEqual(float(group.attrs['boundary_phase']), phase)
                        heights.append(height)
                self.assertNotEqual(heights[0], heights[1])

    def test_datacheck_entrypoint_uses_the_same_split_settings(self):
        from data.DataGen.python import datacheck_generate
        args = SimpleNamespace(resolution=16, n_time=2, n_train=2, n_test=2, T=.01)
        with tempfile.TemporaryDirectory() as tmp:
            # Preview rendering is unrelated to the generator's numerical data.
            with patch.object(datacheck_generate, 'write_shallow_water_preview',
                              side_effect=lambda path, preview, **kwargs: preview):
                result = datacheck_generate.generate_shallow_water(Path(tmp), args)
            self.assertEqual(result['status'], 'ok')
            for path, split, base_seed in zip(result['files'], ['train', 'test'], [0, 10000000]):
                with h5py.File(path, 'r') as file:
                    for index in range(2):
                        rng = np.random.default_rng(base_seed + index)
                        radius, height = ((float(rng.uniform(.3, .7)), 2.) if split == 'train'
                                          else (float(rng.uniform(.4, .8)), float(rng.uniform(2., 3.))))
                        self.check_sample(file[f'{index:06d}'], radius, height)


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()

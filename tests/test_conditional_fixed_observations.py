"""CPU checks for the fixed-observation and canonical-prefix contract."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('conditional_fixed', ROOT / 'plot/run_conditional_sample_scaling.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FixedObservationsTest(unittest.TestCase):
    def test_exact_prefix_partition(self):
        for batch_size in [3, 7, 32, 64]:
            ranges = list(runner.batch_ranges(batch_size))
            self.assertEqual([i for start, stop in ranges for i in range(start, stop)], list(range(1000)))
            self.assertTrue(all(0 < stop - start <= batch_size for start, stop in ranges))
            self.assertTrue(set(runner.KS) <= {stop for _, stop in ranges})

    def test_fixed_targets_repeat_across_draws_and_partitions(self):
        from sampling.masks import make_pair_masks
        for shared in [False, True]:
            cfg = SimpleNamespace(task='both', num_obs=500, sensor_mode='per_sample_random',
                                  shared_mask=shared, mask_seed=20262412)
            coef = torch.arange(128 * 128).float().reshape(1, 1, 128, 128)
            truth = SimpleNamespace(coef=coef, sol=coef * .01 + 2)
            fixed = runner.fixed_observations(cfg, truth)
            for count in [1, 3, 64]:
                gt = SimpleNamespace(coef=truth.coef.expand(count, -1, -1, -1),
                                     sol=truth.sol.expand(count, -1, -1, -1))
                masks, obs, proof = runner.expand_fixed(fixed, gt, count, 'cpu')
                self.assertEqual(proof['hashes'], fixed['hashes'])
                self.assertTrue(torch.equal(obs.coef_noisy, gt.coef * masks.coef))
                self.assertTrue(torch.equal(obs.sol_noisy, gt.sol * masks.sol))
                self.assertTrue(torch.equal(masks.coef, masks.coef[:1].expand_as(masks.coef)))
            old = make_pair_masks((64, 1, 128, 128), (64, 1, 128, 128), 500,
                                  'per_sample_random', shared, cfg.mask_seed)
            self.assertFalse(torch.equal(old.coef[0], old.coef[1]))
            self.assertTrue(torch.equal(old.coef[:1], fixed['masks'][:, :1]))

    def test_modified_observed_value_is_rejected(self):
        cfg = SimpleNamespace(task='forward', num_obs=500, sensor_mode='per_sample_random',
                              shared_mask=False, mask_seed=1)
        truth = SimpleNamespace(coef=torch.ones(1, 1, 128, 128), sol=torch.ones(1, 1, 128, 128))
        fixed = runner.fixed_observations(cfg, truth)
        fixed['observations'][0, 0] += fixed['masks'][0, 0]
        with self.assertRaises(AssertionError):
            runner.expand_fixed(fixed, truth, 1, 'cpu')


if __name__ == '__main__':
    unittest.main()

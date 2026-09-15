"""CPU checks for the fixed-observation and canonical-prefix contract."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import json
import tempfile
from unittest.mock import patch

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

    def test_checkpoint_resume_and_independent_audit(self):
        # A small deterministic fake sampler exercises the real 1000-row
        # checkpoint/export pipeline without running a network or using CUDA.
        import importlib
        exporter = importlib.import_module('export_conditional_sample_scaling')
        source = Path('/home/tat512/C01Python/audit/paper_revision_20260908/inputs/poisson')
        if not source.exists():
            self.skipTest('Local archived input fixture is unavailable')
        protocol = json.loads((source / 'protocol.json').read_text())
        selection_path = ROOT / 'plot/conditional_sample_scaling_selection.json'
        selection = json.loads(selection_path.read_text())
        truths = torch.load(source / 'truths.pt', map_location='cpu', weights_only=False)
        identity = dict(protocol_sha256=runner.digest(source / 'protocol.json'),
                        truth_sha256=protocol['truth_sha256'], weights_sha256=protocol['weights_sha256'],
                        selection_sha256=runner.digest(selection_path), runner_sha256=runner.digest(runner.__file__),
                        source_hashes=runner.source_identity(), historical_commit=runner.HISTORICAL_COMMIT,
                        torch=torch.__version__, tf32=True, fused_guidance=True)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'cases/forward/offset1500'; folder.mkdir(parents=True)
            args = SimpleNamespace(batch_size=64)
            cfg = runner.configured_case(protocol, selection, source, 'forward', 1500, folder)
            calls = []

            def fake_sample(config, truth, bundle, indices, fixed):
                calls.append(list(indices))
                factors = 1 + torch.tensor(list(indices)).float().view(-1, 1, 1, 1) * .0001
                prediction = fixed['truth'] * factors
                return prediction, prediction.double().mean(0).float(), dict(seconds=1e-9,
                    peak_bytes=1, batch_size=len(indices), seed_indices=list(indices), num_steps=100, nfe=100,
                    observations=dict(hashes=fixed['hashes'], all_rows_identical=True, checked_rows=len(indices),
                                      observed_fields=fixed['observed_fields']),
                    initial_noise_hashes=[str(i) for i in indices])

            with patch.object(runner, 'fast_sample', fake_sample), patch.object(torch.cuda, 'synchronize'):
                runner.run_case(args, cfg, truths[1500], None, identity, folder)
                self.assertEqual(len(calls), 20)
                # Completed data is verified and reused, never sampled again.
                runner.run_case(args, cfg, truths[1500], None, identity, folder)
                self.assertEqual(len(calls), 20)
                # Simulate interruption before committing the final pool.
                (folder / 'complete.json').unlink()
                runner.run_case(args, cfg, truths[1500], None, identity, folder)
                self.assertEqual(len(calls), 20)
            rows, _, proof, _ = exporter.audit_case(folder, source, protocol, selection, truths)
            self.assertEqual(len(rows), 5)
            self.assertTrue(proof['all_1000_draw_observations_identical'])
            # A corrupted completed prediction must not be silently accepted.
            with (folder / 'pool.pt').open('ab') as f:
                f.write(b'corruption')
            with self.assertRaises(AssertionError):
                exporter.audit_case(folder, source, protocol, selection, truths)


if __name__ == '__main__':
    unittest.main()

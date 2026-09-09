"""Unit fixtures are artificial arrays, not experimental results."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import torch

path = Path(__file__).resolve().parents[1] / 'reproducibility/revision_20260909/replay_baseline_frozen.py'
spec = importlib.util.spec_from_file_location('frozen_replay', path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class CacheBindingTests(unittest.TestCase):
    def fixture(self):
        name = 'poisson_test_example.mat'
        raw = {'split': 'test', 'full_tensor': torch.zeros(1000, 2, 4, 4),
               'sample_indices': torch.arange(1000),
               'global_sample_ids': [f'{name}:{i}' for i in range(1000)],
               'file_paths': ['/original/'+name]}
        payload = {'schema_version': 'baseline-frozen-raw-v1', 'pde': 'poisson',
                   'load_full_trajectory': False, 'raw': raw,
                   'provenance': {'source_mat_sha256': '1'*64,
                                  'source_mat_path': '/original/'+name,
                                  'source_indices': list(range(1000))}}
        entry = {'pde': 'poisson', 'load_full_trajectory': False,
                 'source_mat_sha256': '1'*64, 'tensors': replay.tensors(raw)}
        summary = {'pde': 'poisson', 'data_files_json': json.dumps({'test': [name]})}
        return payload, entry, summary, argparse.Namespace(load_full_trajectory=True)

    def test_static_flags_share_cache_without_rewriting_original_flag(self):
        payload, entry, summary, args = self.fixture()
        self.assertIs(replay.validate_cache(payload, entry, summary, args), payload['raw'])
        self.assertTrue(args.load_full_trajectory)

    def test_wrong_array_source_or_sample_identity_fails(self):
        for mutation in ('array', 'mat_hash', 'sample_id'):
            payload, entry, summary, args = self.fixture()
            if mutation == 'array':
                payload['raw']['full_tensor'][17, 0, 0, 0] = 1
            elif mutation == 'mat_hash':
                payload['provenance']['source_mat_sha256'] = '2'*64
            else:
                payload['raw']['global_sample_ids'][17] = 'renamed-cache.pt:17'
            with self.assertRaises(AssertionError):
                replay.validate_cache(payload, entry, summary, args)

    def test_evaluation_sensor_seed_overrides_training_seed(self):
        snapshot = {'args': {'seed': 1, 'sensor_seed': 1, 'baseline': 'recfno',
                             'load_full_trajectory': False, 'task': 'forward'}}
        summary = {'sensor_seed': 77, 'load_full_trajectory': True,
                   'synthetic_data': False, 'test_size': 1000, 'split': 'test',
                   'task': 'sparse_solution'}
        before = copy.deepcopy(snapshot)
        args = replay.resolve_args(snapshot, summary)
        self.assertEqual(args.sensor_seed, 77)
        self.assertEqual(args.task, 'sparse_solution')
        self.assertTrue(args.load_full_trajectory)
        self.assertEqual(snapshot, before)


if __name__ == '__main__':
    unittest.main()

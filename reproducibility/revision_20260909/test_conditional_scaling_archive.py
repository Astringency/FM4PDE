"""Check that changing frozen data/timing cannot reuse a scientific approval."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_conditional_scaling_archive as archive


class ScientificBindingTests(unittest.TestCase):
    def setUp(self):
        self.jobs = [(task, offset) for task in archive.TASKS for offset in range(1500, 1532)]
        jobs = [list(job) for index, job in enumerate(self.jobs) if index % 4 == 0]
        self.environment = dict(commit=archive.COMMIT, script_sha256=archive.PRODUCER_SHA,
                                K=archive.KS, tf32=True,
                                args=dict(batch_size=64, fused_guidance=True, num_shards=4, shard_index=0))
        self.approved = {}
        self.timings = {}
        receipts = []
        for task, offset in jobs:
            for k in archive.KS:
                path = f'production/{task}/offset{offset}/K{k}'
                self.approved[path + '.pt'] = 'a' * 64
                self.timings[task, offset, k] = dict(seconds='1.25', compute_seconds='1.0', peak_bytes='123')
                receipts.append(dict(path=path + '.json', sha256='b' * 64,
                                     data=dict(task=task, offset=offset, K=k, num_steps=100, nfe_per_draw=100,
                                               script_sha256=archive.PRODUCER_SHA, result_sha256='a' * 64,
                                               seconds=1.25, compute_seconds=1.0, peak_bytes=123)))
        self.metadata = dict(shards=[dict(shard=0, producer_live=False,
                                         environment=dict(path='production/environment_run_0.json', sha256='c' * 64,
                                                          data=self.environment),
                                         complete=dict(path='production/complete_0.json', sha256='d' * 64,
                                                       data=dict(jobs=jobs)))], receipts=receipts)

    def check(self, metadata):
        with patch.object(archive.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(metadata))):
            return archive.check_remote_gate('server197', dict(source_path='/unused/frozen/source'),
                                             self.approved, {0: self.environment}, self.timings)

    def test_complete_matching_shard_passes(self):
        _, evidence = self.check(self.metadata)
        self.assertEqual(len(evidence), 242)  # 120 tensors, 120 receipts, environment and completion.

    def test_changed_tensor_and_its_receipt_are_refused(self):
        changed = copy.deepcopy(self.metadata)
        changed['receipts'][0]['data']['result_sha256'] = 'e' * 64
        with self.assertRaisesRegex(RuntimeError, 'approved tensor SHA'):
            self.check(changed)

    def test_changed_timing_is_refused(self):
        changed = copy.deepcopy(self.metadata)
        changed['receipts'][0]['data']['seconds'] = 1.26
        with self.assertRaisesRegex(RuntimeError, 'approved CSV'):
            self.check(changed)

    def test_missing_receipt_is_refused(self):
        changed = copy.deepcopy(self.metadata)
        changed['receipts'].pop()
        with self.assertRaisesRegex(RuntimeError, 'static shard assignment'):
            self.check(changed)


if __name__ == '__main__':
    unittest.main()

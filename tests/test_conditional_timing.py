import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'plot/audit_conditional_timing.py'
SPEC = importlib.util.spec_from_file_location('conditional_timing', PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TimingExposureTests(unittest.TestCase):
    def test_boundary_and_partial_overlap(self):
        initial = {'external': [{'gpu': gpu, 'process_started_unix': 100.} for gpu in [6, 7]]}
        handoff = {'rows': [dict(gpu=6, task='forward', offset=1500,
            start=i, stop=i+1, seconds=10., receipt_mtime_unix=end)
            for i, end in enumerate([99., 100., 105., 111.])]}
        rows = list(AUDIT.exposure_index(initial, handoff).values())
        self.assertEqual([x['potential_overlap_seconds'] for x in rows], [0., 0., 5., 10.])
        self.assertEqual([x['potentially_exposed'] for x in rows], [False, False, True, True])

    def test_duplicate_receipt_rejected(self):
        initial = {'external': [{'gpu': gpu, 'process_started_unix': 100.} for gpu in [6, 7]]}
        row = dict(gpu=6, task='forward', offset=1500, start=0, stop=1,
                   seconds=10., receipt_mtime_unix=105.)
        with self.assertRaises(AssertionError):
            AUDIT.exposure_index(initial, {'rows': [row, row]})

    def test_statistics_keep_real_times(self):
        result = AUDIT.describe([1., 2., 3., 100.])
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['mean_seconds'], 26.5)
        self.assertEqual(result['median_seconds'], 2.5)
        self.assertAlmostEqual(result['p95_seconds'], 85.45)


if __name__ == '__main__':
    unittest.main()

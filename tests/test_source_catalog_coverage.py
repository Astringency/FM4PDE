"""A valid partial archive must not pass as the complete source catalog."""
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'reproducibility/revision_20260909/source_snapshots.py'
SPEC = importlib.util.spec_from_file_location('source_snapshots', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceCatalogCoverage(unittest.TestCase):
    def setUp(self):
        self.commit = 'a' * 40
        self.catalog = {'repositories': [dict(
            name='FM4PDE', retain_history=True,
            snapshots=[dict(revision=self.commit, purpose='test fixture')]) ]}
        self.tree = dict(repository='FM4PDE', commit=self.commit, tree='b' * 40)
        self.history = dict(repository='FM4PDE', format='self-contained Git bundle')

    def test_complete_catalog(self):
        self.assertEqual(MODULE.verify_catalog_coverage(
            self.catalog, [self.tree, self.history]), 1)

    def test_missing_required_version_even_with_valid_other_archives(self):
        unrelated = dict(self.tree, commit='c' * 40)
        with self.assertRaisesRegex(RuntimeError, 'required source'):
            MODULE.verify_catalog_coverage(self.catalog, [unrelated, self.history])

    def test_source_without_restorable_history_is_incomplete(self):
        with self.assertRaisesRegex(RuntimeError, 'required Git history'):
            MODULE.verify_catalog_coverage(self.catalog, [self.tree])


if __name__ == '__main__':
    unittest.main()

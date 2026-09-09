"""Exercise archive safety using disposable local trees, never production paths."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('archive_results', Path(__file__).with_name('archive_results.py'))
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class ArchiveSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.root = self.base / 'archive'
        self.root.mkdir()
        self.entry = dict(id='test_study', source_host='server197', source_path=str(self.source),
                          destination_relative='main/revision_20260909/test_study/raw', status='ready', role='primary')

    def tearDown(self):
        self.temp.cleanup()

    def test_copy_repeat_and_refuse_different_destination(self):
        (self.source / 'tensor.pt').write_bytes(b'original tensor')
        result = archive.copy_local(self.entry, self.root)
        final = Path(result['destination'])
        self.assertEqual(result['status'], 'archived')
        self.assertEqual(archive.copy_local(self.entry, self.root)['status'], 'already_verified')
        before = archive.tree(final)
        (self.source / 'tensor.pt').write_bytes(b'different tensor')
        with self.assertRaisesRegex(RuntimeError, 'Existing destination differs'):
            archive.copy_local(self.entry, self.root)
        self.assertEqual(archive.tree(final), before)
        receipt = json.loads((final / archive.RECEIPT).read_text())
        self.assertEqual(receipt['files']['tensor.pt']['sha256'], archive.sha(final / 'tensor.pt'))
        self.assertEqual(archive.verify_entry(self.entry, self.root)['status'], 'verified')
        (final / 'tensor.pt').write_bytes(b'destination damage')
        with self.assertRaisesRegex(RuntimeError, 'differs from the recorded'):
            archive.verify_entry(self.entry, self.root)

    def test_pending_refused_before_creating_target_files(self):
        self.entry['status'] = 'pending'
        with self.assertRaisesRegex(RuntimeError, 'Not an executable'):
            archive.copy_local(self.entry, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_frozen_hash_refused_before_copy(self):
        (self.source / 'protocol.json').write_text('{}')
        self.entry['evidence_sha256'] = {'protocol.json': '0' * 64}
        with self.assertRaisesRegex(RuntimeError, 'Frozen metadata mismatch'):
            archive.copy_local(self.entry, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_symlink_directory_materialized_and_selective_copy_bounded(self):
        target = self.base / 'linked_truth'
        target.mkdir()
        (target / 'truth.pt').write_bytes(b'truth')
        (self.source / 'inputs').symlink_to(target, target_is_directory=True)
        (self.source / 'unrelated.pt').write_bytes(b'must not copy')
        self.entry.update(dereference_symlinks=True, include_files=['inputs/truth.pt'])
        result = archive.copy_local(self.entry, self.root)
        final = Path(result['destination'])
        self.assertFalse((final / 'inputs').is_symlink())
        self.assertEqual((final / 'inputs/truth.pt').read_bytes(), b'truth')
        self.assertFalse((final / 'unrelated.pt').exists())

    def test_file_source_and_evidence(self):
        weight = self.source / 'weights.pth'
        weight.write_bytes(b'checkpoint bytes')
        self.entry.update(source_path=str(weight), source_kind='file', evidence_sha256={'weights.pth': archive.sha(weight)})
        result = archive.copy_local(self.entry, self.root)
        self.assertEqual((Path(result['destination']) / 'weights.pth').read_bytes(), weight.read_bytes())

    def test_corrupt_staging_not_published(self):
        (self.source / 'tensor.pt').write_bytes(b'original')
        expected = archive.tree(self.source)
        stage, final = archive.prepare(self.entry, self.root)
        (stage / 'payload/tensor.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'Transfer hash mismatch'):
            archive.promote(stage, expected, self.root)
        self.assertFalse(final.exists())

    def test_symlink_escape_and_reserved_receipt_refused(self):
        outside = self.base / 'outside'
        outside.mkdir()
        (self.root / 'main').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'outside the target'):
            archive.prepare(self.entry, self.root)
        (self.source / archive.RECEIPT).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'reserved root archive receipt'):
            archive.source_tree(self.entry, self.source)

    def test_changed_source_detected_while_hashing(self):
        target = self.source / 'mutable.pt'
        target.write_bytes(b'before')
        original = archive.sha
        def mutate(path):
            result = original(path)
            Path(path).write_bytes(b'after longer content')
            return result
        with mock.patch.object(archive, 'sha', side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, 'changed while hashing'):
                archive.tree(self.source)

    def test_empty_selection_and_parent_traversal_refused(self):
        self.entry['include_files'] = []
        with self.assertRaisesRegex(ValueError, 'empty include_files'):
            archive.validate_entry(self.entry)
        del self.entry['include_files']
        self.entry['destination_relative'] = 'main/revision_20260909/../unsafe'
        with self.assertRaisesRegex(ValueError, 'Invalid destination'):
            archive.validate_entry(self.entry)

    def test_persistent_submission_reconnects_without_duplicate_tmux(self):
        message = {'mode':'_verify','payload':self.entry,'entry':self.entry,'nonce':'fixed-test-request'}
        with mock.patch.object(archive, 'run') as launch:
            first = archive.submit_job(message, self.root)
            launch.assert_called_once()
        with mock.patch.object(archive, 'run') as launch, mock.patch.object(archive.subprocess, 'run') as status:
            status.return_value.returncode = 0
            second = archive.submit_job(message, self.root)
            launch.assert_not_called()
        self.assertEqual(first['job'], second['job'])
        (Path(first['job']) / 'result.json').write_text(json.dumps({'state':'complete','result':{'status':'verified'}}))
        with mock.patch.object(archive, 'run') as launch:
            done = archive.submit_job(message, self.root)
            launch.assert_not_called()
        self.assertEqual(done['state'], 'complete')

    def test_persistent_runner_saves_result_and_executes_once(self):
        message = {'mode':'_verify','payload':self.entry,'entry':self.entry,'nonce':'runner-test-request'}
        with mock.patch.object(archive, 'run'):
            directory = Path(archive.submit_job(message, self.root)['job'])
        def child(args, **kwargs):
            kwargs['stdout'].write(json.dumps({'status':'verified','bytes':123}))
            return mock.Mock(returncode=0)
        with mock.patch.object(archive.subprocess, 'run', side_effect=child) as child_run:
            archive.run_job(directory, self.root)
            archive.run_job(directory, self.root)
            child_run.assert_called_once()
        result = json.loads((directory / 'result.json').read_text())
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['result']['bytes'], 123)


if __name__ == '__main__':
    unittest.main()

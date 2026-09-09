"""Check that archive CSV comparisons permit only the declared relocation."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproducibility/revision_20260909'))
from verify_ns_archived_main import ORIGINAL, compare_csv, write_csv


class ArchiveComparisonTests(unittest.TestCase):
    def test_only_result_root_can_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = dict(dist='id', setting='full_forward', offset='2000', error_a='0.2',
                gpu='original GPU', result=str(ORIGINAL/'main_results/id/full_forward/offset2000.pt'))
            new = {**old, 'result':str(root/'archive/main_results/id/full_forward/offset2000.pt')}
            write_csv(root/'old.csv', [old]);write_csv(root/'new.csv', [new])
            result = compare_csv(root/'old.csv', root/'new.csv', ['dist','setting','offset'], ['error_a'], root/'archive')
            self.assertEqual(result['permitted_result_path_changes'], 1)
            self.assertTrue(result['numeric_differences']['error_a']['exact'])
            for field, value in [('gpu','another GPU'), ('error_a','0.3'),
                                 ('result',str(root/'archive/main_results/id/full_forward/offset2001.pt'))]:
                with self.subTest(field=field):
                    write_csv(root/'new.csv', [{**new, field:value}])
                    with self.assertRaises(AssertionError):
                        compare_csv(root/'old.csv', root/'new.csv', ['dist','setting','offset'], ['error_a'], root/'archive')

    def test_duplicate_physical_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            row=dict(dist='id',setting='full_forward',offset='2000',error_a='0.2')
            write_csv(root/'old.csv',[row,row]);write_csv(root/'new.csv',[row,row])
            with self.assertRaises(AssertionError):
                compare_csv(root/'old.csv',root/'new.csv',['dist','setting','offset'],['error_a'],root/'archive')


if __name__=='__main__':
    unittest.main()

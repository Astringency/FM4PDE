"""Check displayed tables after substituting independently recomputed old norms.

This creates a new diagnostic snapshot. Neither the publication snapshot nor
its original float32-reported values are changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--control-audit', type=Path, required=True)
    parser.add_argument('--published-tables', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a fresh diagnostic directory'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'plot'))
    from export_paper_ablation_tables import export
    audit = json.loads(args.control_audit.read_text())
    assert audit['status'] == 'pass' and audit['predictions'] == 316 and audit['fields_checked'] == 632
    records = json.loads((args.snapshot/'records.json').read_text())
    controls = {r['id']: r for r in records if r['source'] == 'unchanged'}
    assert set(controls) == {r['id'] for r in audit['rows']} and len(controls) == 316
    for raw in audit['rows']:
        record = controls[raw['id']]
        for field in ('a', 'u'):
            values = raw['fields'][field]
            assert record['rel_l2_'+field] == values['archived']
            record['rel_l2_'+field] = values['recomputed_float64']
    source = args.output/'diagnostic_snapshot'
    source.mkdir(parents=True)
    (source/'records.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    (source/'manifest.json').write_bytes((args.snapshot/'manifest.json').read_bytes())
    export(SimpleNamespace(source=source, output=args.output/'tables', development=False))
    comparisons = {p.name: sha(p) == sha(args.published_tables/p.name)
                   for p in (args.output/'tables').glob('*.tex')}
    assert len(comparisons) == 20
    result = dict(status='pass' if all(comparisons.values()) else 'differences',
        source_snapshot_sha256=sha(args.snapshot/'records.json'),
        control_audit_sha256=sha(args.control_audit),
        scope='Only 316 unchanged controls: substitute independently recomputed float64 relative norms; '
              'compare displayed values and ranking in all 20 tables. Diagnostics are retained.',
        original_data_and_publication_values_preserved=True, tables_byte_identical=comparisons)
    (args.output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

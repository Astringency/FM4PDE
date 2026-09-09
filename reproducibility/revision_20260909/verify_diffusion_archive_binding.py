#!/usr/bin/env python3
"""Bind an existing numerical audit to independently verified archived bytes.

No predictions, errors, or observation losses are recomputed here. The audit
and archive receipts remain read-only, and their exact SHA256 values are bound
in the resulting report.
"""
import argparse
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-manifest', type=Path, required=True)
    parser.add_argument('--archive-receipt', type=Path, required=True)
    parser.add_argument('--archive-receipt-sha256', required=True)
    parser.add_argument('--per-example-checks', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.dependency_manifest.read_text())
    known = {Path(row['path']).name: row for row in manifest['files']}
    for path in (args.per_example_checks, args.validation):
        expected = known[path.name]
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise RuntimeError('Audit artifact differs from frozen dependency manifest: ' + path.name)
    receipt_sha = sha(args.archive_receipt)
    if receipt_sha != args.archive_receipt_sha256:
        raise RuntimeError('Archive receipt differs from its independent verification outcome')
    receipt = json.loads(args.archive_receipt.read_text())
    source_prefix = manifest['per_example_structure']['join_source_prefix']
    if (receipt['source']['id'] != 'old_diffusion_main'
            or receipt['source']['source_host'] != 'server216'
            or receipt['source']['source_path'].rstrip('/') + '/' != source_prefix):
        raise RuntimeError('Numerical audit and archive identify different original sources')
    validation = json.loads(args.validation.read_text())
    if not (validation['status'] == 'pass' and validation['complete']
            and validation['examples'] == 26000 and validation['cells'] == 26
            and validation['n_per_cell'] == 1000
            and validation['per_example_sha256'] == sha(args.per_example_checks)
            and validation['manifest_sha256'] == manifest['manifest_sha256']):
        raise RuntimeError('The numerical audit is not complete or does not bind these exact artifacts')
    with gzip.open(args.per_example_checks, 'rt') as stream:
        rows = json.load(stream)
    if len(rows) != 26000:
        raise RuntimeError('Expected 26,000 original prediction records')
    cells = Counter()
    indices = {}
    checked = {'result': set(), 'metrics': set()}
    masks = {}
    absent = 0
    mismatches = []
    for row in rows:
        cell = (row['study'], row['pde'], row['task'])
        cells[cell] += 1
        indices.setdefault(cell, set()).add(row['index'])
        for field, mask in row['observation_masks'].items():
            key = (*cell, field)
            aggregate = masks.setdefault(key, dict(counts=Counter(), shapes=set(),
                                                   dtypes=set(), hashes=[]))
            aggregate['counts'][row['observation_counts'][field]] += 1
            aggregate['shapes'].add(tuple(mask['shape']))
            aggregate['dtypes'].add(mask['dtype'])
            aggregate['hashes'].append((row['index'], mask['sha256']))
        for group in ('result', 'metrics'):
            source, digest = row[group + '_path'], row[group + '_sha256']
            if source is None:
                if group != 'metrics' or digest is not None:
                    raise RuntimeError('A prediction or its SHA is missing')
                absent += 1
                continue
            if not source.startswith(source_prefix):
                raise RuntimeError('Unexpected original source prefix')
            relative = source[len(source_prefix):]
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise RuntimeError('Invalid source-relative artifact path')
            recorded = receipt['files'].get(relative)
            if not recorded or recorded.get('kind') != 'file' or recorded.get('sha256') != digest:
                mismatches.append(dict(path=relative, expected_sha256=digest,
                                       archived=recorded))
            checked[group].add(relative)
    if (len(cells) != 26 or set(cells.values()) != {1000}
            or any(values != set(range(1000)) for values in indices.values())
            or len(checked['result']) != 26000 or len(checked['metrics']) != 19000 or absent != 7000):
        raise RuntimeError('Audit coverage or unique source-file counts differ from the frozen contract')
    report = dict(
        status='pass' if not mismatches else 'fail',
        observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Source-relative file SHA association of a completed numerical audit with an independently verified archive; no new numerical computation.',
        archive_receipt_sha256=receipt_sha,
        original_source=receipt['source']['source_path'],
        archived_destination_relative=receipt['source']['destination_relative'],
        dependency_manifest_sha256=sha(args.dependency_manifest),
        numerical_validation_sha256=sha(args.validation),
        per_example_audit_sha256=sha(args.per_example_checks),
        cells=26, examples=26000, prediction_files=len(checked['result']),
        existing_metric_json_files=len(checked['metrics']),
        examples_without_separate_metric_json=absent,
        original_fallback='The audit uses original pickle loss[-1] for 2,000 Burgers predictions, and loss.global_a/global_u[-1] for the remaining 5,000 predictions without separate metric JSON.',
        fallback_metric_sources=dict(Counter(row['metric_source'] for row in rows
                                             if row['metrics_path'] is None)),
        mismatch_count=len(mismatches), mismatches=mismatches,
        per_cell_counts=[dict(study=key[0], pde=key[1], task=key[2], examples=value)
                         for key, value in sorted(cells.items())],
        observation_mask_summary=[dict(
            study=key[0], pde=key[1], task=key[2], field=key[3],
            records=sum(value['counts'].values()),
            observation_count_frequencies=dict(sorted(value['counts'].items())),
            shapes=[list(shape) for shape in sorted(value['shapes'])],
            dtypes=sorted(value['dtypes']),
            unique_mask_sha256=len({digest for _, digest in value['hashes']}),
            ordered_mask_sha256_sequence_digest=hashlib.sha256(
                ''.join(f'{index}:{digest}\n' for index, digest in sorted(value['hashes'])).encode()).hexdigest(),
        ) for key, value in sorted(masks.items())],
    )
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('mismatches', 'per_cell_counts', 'observation_mask_summary')}))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

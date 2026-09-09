#!/usr/bin/env python3
"""Freeze native DiffusionPDE inputs and the archive-bound numerical audit."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

EXPECTED_MANIFEST = '29b6548318a662d134ec1689f264701f84dc582c4a2cb59a64e8e43f026ab0ad'
TARGET = '/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs'
DESTINATION = Path('main/revision_20260909/diffusion_original_main/native_frozen_inputs')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-manifest', type=Path, required=True)
    parser.add_argument('--binding-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.dependency_manifest.read_bytes()
    if sha(raw) != EXPECTED_MANIFEST:
        raise RuntimeError('Owner dependency manifest differs from the reviewed version')
    owner = json.loads(raw)
    binding_path = args.binding_review / 'archive_binding_validation.json'
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    if not (binding['status'] == 'pass' and binding['mismatch_count'] == 0
            and binding['prediction_files'] == 26000
            and binding['existing_metric_json_files'] == 19000
            and binding['dependency_manifest_sha256'] == EXPECTED_MANIFEST
            and binding['numerical_validation_sha256'] == owner['validation_sha256']):
        raise RuntimeError('The completed numerical audit has not been bound to archived source bytes')
    if args.output.exists():
        raise RuntimeError('Refusing to replace an existing frozen addendum')
    groups = {}
    root = Path(owner['source_root'])
    for row in owner['files']:
        relative = Path(row['path']).relative_to(root)
        if relative.parts[0] not in ('frozen_inputs_v3', 'full_saved_prediction_audit'):
            raise RuntimeError('Unexpected source tree, including a possible incomplete cache version')
        groups.setdefault(relative.parts[0], []).append(row)
    if sorted(len(rows) for rows in groups.values()) != [3, 6]:
        raise RuntimeError('Expected exactly six frozen-input files and three numerical-audit files')
    entries = []
    for group, rows in sorted(groups.items()):
        source = root / group
        selected = {str(Path(row['path']).relative_to(source)): row for row in rows}
        entries.append(dict(
            id='diffusion_native_' + group, source_host=owner['source_host'],
            source_path=str(source), destination_relative=str(DESTINATION / group),
            status='ready', role='dependency', source_kind='directory',
            purpose='Exact native input slices or completed independent numerical audit; original files retained.',
            include_files=sorted(selected),
            evidence_sha256={name: row['sha256'] for name, row in sorted(selected.items())},
            observed=dict(exists=True, files=len(rows), logical_bytes=sum(row['bytes'] for row in rows)),
            owner_inventory_sha256=EXPECTED_MANIFEST,
            archive_binding_review_sha256=sha(binding_bytes),
        ))
    evidence = args.output / 'evidence'
    evidence.mkdir(parents=True)
    for row in owner['local_provenance']:
        path = Path(row['path']); data = path.read_bytes()
        if len(data) != row['bytes'] or sha(data) != row['sha256']:
            raise RuntimeError('Local provenance changed: ' + str(path))
        target = evidence / path.name
        if target.exists():
            raise RuntimeError('Provenance basename collision')
        target.write_bytes(data)
    (evidence / args.dependency_manifest.name).write_bytes(raw)
    (evidence / binding_path.name).write_bytes(binding_bytes)
    for name in ('receipt_metadata_read.json', 'source_audit_metadata_read.json'):
        (evidence / name).write_bytes((args.binding_review / name).read_bytes())
    paths = sorted(evidence.iterdir())
    entries.append(dict(
        id='diffusion_native_provenance', source_host='local', source_path=str(evidence),
        destination_relative=str(DESTINATION / 'provenance'), status='ready', role='dependency',
        purpose='Native input selection and producer identities; exact numerical-audit to archive SHA association.',
        evidence_sha256={path.name: sha(path.read_bytes()) for path in paths},
        observed=dict(exists=True, files=len(paths), logical_bytes=sum(path.stat().st_size for path in paths)),
    ))
    inventory = dict(schema_version=1, observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     target_host='server197', target_root=TARGET, entries=entries,
                     source_owner_inventory_sha256=EXPECTED_MANIFEST,
                     original_results_archive_receipt_sha256=binding['archive_receipt_sha256'],
                     archive_binding_review_sha256=sha(binding_bytes),
                     excluded=owner['not_included'],
                     derived_materialization='MAT/HDF5 files can be regenerated from these native NPZ arrays; they are not copied again.')
    path = args.output / 'archive_inventory.diffusion_native_addendum.json'
    path.write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), sha256=sha(path.read_bytes()), entries=len(entries),
                          total_bytes=sum(entry['observed']['logical_bytes'] for entry in entries))))


if __name__ == '__main__':
    main()

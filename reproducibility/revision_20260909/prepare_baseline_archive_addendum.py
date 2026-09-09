#!/usr/bin/env python3
"""Freeze the authorized selected-baseline weights/configuration addendum."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

DEPENDENCY_SHA = 'fb353c6e4fb27f1898505bdfbf32b0045c5ee2ff00634beb91f673d246740946'
SOURCE_ROOT = Path('/large_storage/zhangxf/outputs/FM4PDEbaseline/runs')
TARGET_ROOT = Path('/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs')
STUDY = Path('main/revision_20260909/legacy_baseline_extracts/dependencies')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--reviewed-inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = args.output / 'evidence'
    if args.output.exists():
        raise RuntimeError('Refusing to replace an existing frozen addendum directory')
    sources = {name: (args.coverage / name).read_bytes() for name in (
        'baseline_dependency_files.json', 'baseline_checkpoint_duplicate_references.json',
        'baseline_all_selected_runs.json')}
    assert digest(sources['baseline_dependency_files.json']) == DEPENDENCY_SHA
    dependencies = json.loads(sources['baseline_dependency_files.json'])
    references = json.loads(sources['baseline_checkpoint_duplicate_references.json'])
    runs = json.loads(sources['baseline_all_selected_runs.json'])
    reviewed_bytes = args.reviewed_inventory.read_bytes()
    reviewed = json.loads(reviewed_bytes)
    original_entries = {e['id']: e for e in reviewed['entries']}
    assert references['dependency_manifest_sha256'] == DEPENDENCY_SHA
    assert len(dependencies['checkpoints']) == 48 and len(dependencies['metadata']) == 761
    assert sum(row['bytes'] for row in dependencies['checkpoints']) == 573544957
    assert sum(row['bytes'] for row in dependencies['metadata']) == 33372614
    base = {r['path']: r for r in dependencies['checkpoints']}
    assert len(base) == len(references['checkpoints']) == 48
    binding = []
    new_weights = []
    required = {}
    for row in references['checkpoints']:
        assert all(row[k] == base[row['path']][k] for k in ('bytes', 'sha256', 'expected_sha256'))
        assert row['exists'] and row['matches_expected'] and row['sha256'] == row['expected_sha256']
        relative = Path(row['path']).relative_to(SOURCE_ROOT)
        reference = row.get('already_archived_reference')
        if reference:
            source = original_entries[reference['entry_id']]
            relative_reference = Path(reference['source_path']).relative_to(source['source_path'])
            archive_file = TARGET_ROOT / source['destination_relative'] / relative_reference
            required[reference['entry_id']] = source['destination_relative']
            destination = dict(existing_entry_id=reference['entry_id'],
                               existing_relative_file=str(relative_reference),
                               archive_file=str(archive_file))
        else:
            new_weights.append(row)
            destination = dict(new_entry_id='baseline_selected_checkpoints_addendum',
                               archive_file=str(TARGET_ROOT / STUDY / 'checkpoints/runs' / relative))
        binding.append(dict(source_host='server197', original_file=row['path'],
                            bytes=row['bytes'], sha256=row['sha256'], **destination))
    assert len(new_weights) == 33 and len(binding) - len(new_weights) == 15
    entries = []
    commits = sorted({r['summary']['commit_hash'] for r in runs if r['summary'].get('commit_hash')})
    for group, rows, source_root, suffix in [
        ('checkpoints', new_weights, SOURCE_ROOT, 'checkpoints/runs'),
        ('metadata', dependencies['metadata'], SOURCE_ROOT.parent, 'metadata'),
    ]:
        assert all(r['exists'] and len(r['sha256']) == 64 for r in rows)
        selected = {str(Path(r['path']).relative_to(source_root)): r for r in rows}
        assert len(selected) == len(rows)
        entries.append(dict(
            id=f'baseline_selected_{group}_addendum', source_host='server197',
            source_path=str(source_root), destination_relative=str(STUDY / suffix),
            status='ready', role='dependency', source_kind='directory',
            purpose='Exact selected baseline dependencies; retain original relative paths and source bytes.',
            include_files=sorted(selected),
            evidence_sha256={p: r['sha256'] for p, r in sorted(selected.items())},
            observed=dict(exists=True, files=len(rows), logical_bytes=sum(r['bytes'] for r in rows)),
            producer_commits=commits, owner_dependency_manifest_sha256=DEPENDENCY_SHA,
        ))
    evidence.mkdir(parents=True)
    for name, data in sources.items():
        (evidence / name).write_bytes(data)
    mapping = dict(
        observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='48 selected checkpoints; 15 references to identical archived files and 33 new copies; 761 metadata files.',
        source_dependency_inventory_sha256=DEPENDENCY_SHA,
        required_verified_archive_entries=required, checkpoint_bindings=binding,
        metadata_entry='baseline_selected_metadata_addendum',
        metadata_source_root=str(SOURCE_ROOT.parent),
        metadata_archive_root=str(TARGET_ROOT / STUDY / 'metadata'),
        excluded='Original MAT datasets, derived input subsets, and additional Burgers dependencies are outside this addendum.',
    )
    (evidence / 'archive_dependency_bindings.json').write_text(json.dumps(mapping, indent=2) + '\n')
    files = sorted(evidence.iterdir())
    entries.append(dict(
        id='baseline_selected_dependency_provenance', source_host='local', source_path=str(evidence),
        destination_relative=str(STUDY / 'provenance'), status='ready', role='dependency',
        purpose='Frozen selected-run summaries, dependency hashes and all 48 checkpoint archive bindings.',
        evidence_sha256={p.name: digest(p.read_bytes()) for p in files},
        observed=dict(exists=True, files=len(files), logical_bytes=sum(p.stat().st_size for p in files)),
    ))
    inventory = dict(schema_version=1, observed_utc=mapping['observed_utc'],
                     target_host='server197', target_root=str(TARGET_ROOT), entries=entries,
                     after_inventory_sha256=digest(reviewed_bytes),
                     required_verified_archive_entries=required,
                     checkpoint_bindings=binding,
                     scope=mapping['scope'], excluded=mapping['excluded'])
    path = args.output / 'archive_inventory.baseline_addendum.json'
    path.write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), sha256=digest(path.read_bytes()),
                          entries=len(entries), new_weight_count=len(new_weights),
                          new_weight_bytes=sum(r['bytes'] for r in new_weights),
                          total_bytes=sum(e['observed']['logical_bytes'] for e in entries))))


if __name__ == '__main__':
    main()

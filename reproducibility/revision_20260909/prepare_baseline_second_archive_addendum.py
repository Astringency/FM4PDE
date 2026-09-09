#!/usr/bin/env python3
"""Freeze exact Burgers weights, native caches, and completed consumer evidence."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

DEPENDENCY_SHA = '9cab4bcb8953cd1c476cc500042c52b46317d56b4ae5b16a86ea2129433b74b5'
PROVENANCE_SHA = '4405e026a318c737075b292bef8f5db70115007a2f2bfaa454c1ed9edae50939'
DELIVERY_SHA = '9113d1531d09ae0854f0a68abce9a5cbc714e80c0521d4524b798a4fdef44af1'
GATE_SHA = 'cdf3add29c92763c51014e3f31df5329b73a482acc276878a16ecdaa576f1f30'
FIRST_SHA = '20dd2b9635a0d7fe01cb2beed728ee33a6baa16ca4f7cf0fc927f68039468a4a'
TARGET = '/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs'
STUDY = Path('main/revision_20260909/legacy_baseline_extracts/dependencies')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_json(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise RuntimeError('Frozen owner evidence changed: ' + str(path))
    return json.loads(data), data


def relative_file(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise RuntimeError('Unsafe relative file: ' + str(name))
    return path


def check_local(root, rows, key, exact=False):
    selected = {}
    for row in rows:
        relative = relative_file(row[key])
        path = root / relative
        if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
            raise RuntimeError('Missing file or unexpected link: ' + str(path))
        data = path.read_bytes()
        if len(data) != row['bytes'] or digest(data) != row['sha256']:
            raise RuntimeError('Local evidence bytes changed: ' + str(path))
        if str(relative) in selected:
            raise RuntimeError('Duplicate source file')
        selected[str(relative)] = row
    if exact:
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        if actual != set(selected):
            raise RuntimeError('Consumer delivery has unlisted or missing files')
    return selected


def entry(identifier, host, source, destination, rows, **extra):
    return dict(id=identifier, source_host=host, source_path=str(source),
                destination_relative=str(destination), status='ready', role='dependency',
                source_kind='directory', include_files=sorted(rows),
                purpose='Frozen reproducibility dependencies; original files remain in place.',
                evidence_sha256={name: row['sha256'] for name, row in sorted(rows.items())},
                observed=dict(exists=True, files=len(rows), logical_bytes=sum(r['bytes'] for r in rows.values())),
                **extra)


def check_overlap(entries):
    seen = []
    identifiers = set()
    for row in entries:
        destination = relative_file(row['destination_relative'])
        if row['id'] in identifiers:
            raise RuntimeError('Duplicate archive entry ID: ' + row['id'])
        identifiers.add(row['id'])
        for previous in seen:
            if destination == previous or destination in previous.parents or previous in destination.parents:
                raise RuntimeError('Archive destination overlap: ' + str(destination))
        seen.append(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--consumer-evidence', type=Path, required=True)
    parser.add_argument('--first-inventory', type=Path, required=True)
    parser.add_argument('--other-inventory', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to replace an existing frozen addendum')
    owner, owner_bytes = checked_json(args.coverage / 'baseline_second_dependency_files.json', DEPENDENCY_SHA)
    provenance, provenance_bytes = checked_json(args.coverage / 'baseline_second_provenance_files.json', PROVENANCE_SHA)
    delivery, delivery_bytes = checked_json(args.consumer_evidence / 'delivery_manifest.json', DELIVERY_SHA)
    gate, _ = checked_json(args.consumer_evidence / 'validation_complete.json', GATE_SHA)
    first, _ = checked_json(args.first_inventory, FIRST_SHA)
    if not (delivery['status'] == gate['status'] == 'pass' and delivery['complete'] and gate['complete']
            and delivery['validation_complete_sha256'] == GATE_SHA
            and gate['cache_manifest_sha256'] == owner['frozen_input_manifest_sha256']
            and gate['data_contract_cells'] == 186 and gate['representative_field_cells'] == 10
            and gate['representative_inference_cells'] == 4 and gate['full_inference_cells'] == 0
            and all(row['returncode'] == 0 for row in gate['checks'])):
        raise RuntimeError('Native consumer completion gate is inconsistent')
    if len(delivery['entries']) != 228 or delivery['total_bytes'] != 131334608:
        raise RuntimeError('Unexpected consumer delivery scope')
    delivery_rows = delivery['entries'] + [dict(path='delivery_manifest.json', bytes=len(delivery_bytes), sha256=DELIVERY_SHA)]
    consumer_files = check_local(args.consumer_evidence, delivery_rows, 'path', exact=True)
    if sum(r['bytes'] for r in delivery['entries']) != delivery['total_bytes']:
        raise RuntimeError('Consumer total bytes disagree')
    provenance_files = check_local(Path(provenance['source_root']), provenance['files'], 'relative_path')
    if len(provenance_files) != 14 or sum(r['bytes'] for r in provenance_files.values()) != 9679500:
        raise RuntimeError('Unexpected provenance scope')
    first_weights = {r['sha256'] for r in first['checkpoint_bindings']}
    entries = []
    expected_counts = dict(checkpoints=6, metadata=168, frozen_inputs=19)
    expected_bytes = dict(checkpoints=49144970, metadata=3454371, frozen_inputs=3935091434)
    for group in expected_counts:
        source = Path(owner['source_roots'][group])
        rows = owner[group]
        if len(rows) != expected_counts[group] or sum(r['bytes'] for r in rows) != expected_bytes[group]:
            raise RuntimeError('Unexpected primary dependency scope: ' + group)
        selected = {str(Path(r['path']).relative_to(source)): r for r in rows}
        if len(selected) != len(rows) or any(len(r['sha256']) != 64 for r in rows):
            raise RuntimeError('Duplicate path or invalid SHA in primary dependency')
        if group == 'checkpoints' and any(not r['matches_expected'] or r['sha256'] != r['expected_sha256']
                                          or r['sha256'] in first_weights for r in rows):
            raise RuntimeError('Checkpoint mismatch or duplicate first-addendum weights')
        entries.append(entry('baseline_second_' + group, owner['source_host'], source,
                             owner['suggested_destinations'][group], selected,
                             owner_inventory_sha256=DEPENDENCY_SHA, native_consumer_gate_sha256=GATE_SHA))
    entries.append(entry('baseline_second_selected_provenance', 'local', provenance['source_root'],
                         STUDY / 'second_provenance', provenance_files, owner_inventory_sha256=PROVENANCE_SHA))
    entries.append(entry('baseline_native_consumer_evidence', 'local', args.consumer_evidence,
                         STUDY / 'native_consumer_evidence', consumer_files,
                         delivery_manifest_sha256=DELIVERY_SHA, native_consumer_gate_sha256=GATE_SHA))
    review = dict(status='pass', observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  dependency_manifest_sha256=DEPENDENCY_SHA, provenance_manifest_sha256=PROVENANCE_SHA,
                  delivery_manifest_sha256=DELIVERY_SHA, native_consumer_gate_sha256=GATE_SHA,
                  owner_proposal_status_preserved=owner['status'],
                  readiness_basis='The frozen external final consumer gate supersedes the proposal-time pending status; owner evidence is unchanged.',
                  validated_scope=dict(mask_contract_cells=186, saved_field_cells=10, samples_per_saved_field_cell=3,
                                       small_batch_model_inference_cells=4, full_model_inference_cells=0),
                  local_evidence_files_verified=len(consumer_files) + len(provenance_files),
                  remote_source_hashes='Frozen owner hashes; archive executor independently rereads source bytes before publication.',
                  source_roots=owner['source_roots'], exclusions=owner['excluded'])
    review_bytes = (json.dumps(review, indent=2) + '\n').encode()
    binding_files = {
        'baseline_second_dependency_files.json': owner_bytes,
        'baseline_second_provenance_files.json': provenance_bytes,
        'completion_review.json': review_bytes,
    }
    entries.append(entry('baseline_second_completion_provenance', 'local', args.output / 'evidence',
                         STUDY / 'second_completion_provenance',
                         {name: dict(bytes=len(data), sha256=digest(data)) for name, data in binding_files.items()}))
    other_entries = list(first['entries'])
    for path in args.other_inventory:
        other_entries.extend(json.loads(path.read_text())['entries'])
    check_overlap(other_entries + entries)
    evidence = args.output / 'evidence'
    evidence.mkdir(parents=True)
    for name, data in binding_files.items():
        (evidence / name).write_bytes(data)
    inventory = dict(schema_version=1, observed_utc=review['observed_utc'], target_host='server197',
                     target_root=TARGET, after_inventory_sha256=FIRST_SHA, entries=entries,
                     native_consumer_gate_sha256=GATE_SHA, delivery_manifest_sha256=DELIVERY_SHA,
                     completion_review_sha256=digest(review_bytes), exclusions=owner['excluded'])
    path = args.output / 'archive_inventory.baseline_second_addendum.json'
    path.write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), sha256=digest(path.read_bytes()), entries=len(entries),
                          total_bytes=sum(row['observed']['logical_bytes'] for row in entries))))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Reconcile recorded archive execute/verify outcomes against frozen inventories.

This reads receipts recorded by the archiver; it does not replace the archiver's
independent destination-file SHA256 verification or change any archived file.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re


def read_json(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def summarize(inventory_path, state_path, addendum_inventory=None, addendum_result=None):
    inventory, inventory_sha = read_json(inventory_path)
    state, state_sha = read_json(state_path)
    issues = []
    if state.get('inventory_sha256') != inventory_sha:
        issues.append('Execution state does not identify the supplied inventory SHA256')
    planned = [e for e in inventory['entries'] if e['status'] == 'ready'
               and e['role'] in ('primary', 'dependency')]
    records = dict(state['entries'])
    manifest_sources = [{'path': str(inventory_path), 'sha256': inventory_sha}]
    state_sources = [{'path': str(state_path), 'sha256': state_sha}]
    if bool(addendum_inventory) != bool(addendum_result):
        raise ValueError('Supply both addendum inventory and result')
    if addendum_inventory:
        addendum, addendum_sha = read_json(addendum_inventory)
        result, result_sha = read_json(addendum_result)
        if len(addendum['entries']) != 1:
            raise ValueError('The single-entry addendum result requires exactly one entry')
        entry = addendum['entries'][0]
        if entry['id'] in records or any(e['id'] == entry['id'] for e in planned):
            raise ValueError('Addendum ID duplicates the main batch')
        planned.append(entry)
        records[entry['id']] = {
            'status': result['status'], **result['entries'],
            'finished_utc': result.get('completed_utc'),
        }
        manifest_sources.append({'path': str(addendum_inventory), 'sha256': addendum_sha})
        state_sources.append({'path': str(addendum_result), 'sha256': result_sha})
    planned_ids = [e['id'] for e in planned]
    if len(set(planned_ids)) != len(planned_ids):
        issues.append('Duplicate entry IDs in approved inventories')
    extra = sorted(set(records) - set(planned_ids))
    if extra:
        issues.append(f'Execution state contains unapproved IDs: {extra}')
    root = Path(inventory['target_root'])
    rows = []
    destinations = set()
    for entry in planned:
        name = entry['id']
        record = records.get(name, {})
        destination = str(root / entry['destination_relative'])
        if destination in destinations:
            issues.append(f'{name}: repeated archive destination')
        destinations.add(destination)
        expected_bytes = entry['observed']['logical_bytes']
        row = {
            'id': name, 'status': record.get('status', 'not_started'),
            'source_host': entry['source_host'], 'source_path': entry['source_path'],
            'destination': destination, 'planned_bytes': expected_bytes,
            'planned_files': entry['observed'].get('files'),
        }
        if row['status'] == 'verified':
            executed, verified = record['execute'], record['verify']
            checks = {
                'execute_status': executed['status'] in ('archived', 'already_verified'),
                'verify_status': verified['status'] == 'verified',
                'destinations_match': executed['destination'] == verified['destination'] == destination,
                'bytes_match': executed['bytes'] == verified['bytes'] == expected_bytes,
                'receipt_hashes_match': executed['receipt_sha256'] == verified['receipt_sha256'],
                'receipt_hash_format': bool(re.fullmatch('[0-9a-f]{64}', verified['receipt_sha256'])),
                'state_receipt_matches': record.get('receipt_sha256', verified['receipt_sha256']) == verified['receipt_sha256'],
                'state_bytes_match': record.get('bytes', verified['bytes']) == verified['bytes'],
            }
            row.update(bytes=verified['bytes'], receipt_sha256=verified['receipt_sha256'],
                       finished_utc=record.get('finished_utc'), checks=checks)
            for check, passed in checks.items():
                if not passed:
                    issues.append(f'{name}: {check} failed')
        elif row['status'] == 'held':
            row['reason'] = record.get('reason', record.get('stderr'))
        rows.append(row)
    main_verified = [record for record in state['entries'].values() if record['status'] == 'verified']
    if state.get('verified_entries', 0) != len(main_verified):
        issues.append('Main-batch verified count differs from its per-entry records')
    if state.get('verified_bytes', 0) != sum(r['verify']['bytes'] for r in main_verified):
        issues.append('Main-batch verified bytes differ from its per-entry records')
    verified_rows = [row for row in rows if row['status'] == 'verified']
    complete = (len(verified_rows) == len(planned) and not issues
                and state['status'] == 'complete')
    return {
        'observed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'complete_and_consistent' if complete else 'inconsistent' if issues else 'in_progress',
        'scope': 'Recorded independent archive verification outcomes, with approved-entry and byte reconciliation.',
        'main_batch_status': state['status'],
        'inventories': manifest_sources, 'execution_records': state_sources,
        'target_host': inventory['target_host'], 'target_root': str(root),
        'planned_entries': len(planned), 'verified_entries': len(verified_rows),
        'planned_bytes': sum(row['planned_bytes'] for row in rows),
        'verified_bytes': sum(row['bytes'] for row in verified_rows),
        'held_entries': [row['id'] for row in rows if row['status'] == 'held'],
        'issues': issues, 'entries': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--addendum-inventory', type=Path)
    parser.add_argument('--addendum-result', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = summarize(args.inventory, args.state, args.addendum_inventory, args.addendum_result)
    serialized = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end='')
    if report['issues']:
        raise SystemExit(1)
    if args.require_complete and report['status'] != 'complete_and_consistent':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

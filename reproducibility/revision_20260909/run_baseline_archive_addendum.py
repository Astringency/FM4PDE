#!/usr/bin/env python3
"""Wait for the original serial archive, verify references, then copy its addendum."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import run_reviewed_archive as serial


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--after-state', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.inventory) == args.inventory_sha256
    inventory = json.loads(args.inventory.read_text())
    status_path = args.inventory.parent / 'queue_status.json'
    while True:
        state = json.loads(args.after_state.read_text())
        assert state['inventory_sha256'] == inventory['after_inventory_sha256']
        if state['status'] != 'running':
            if state['status'] != 'complete':
                raise RuntimeError('Prior archive has held entries; refusing to start the addendum')
            break
        serial.save(status_path, dict(status='waiting_for_prior_batch',
                    observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    prior_verified_entries=state.get('verified_entries'),
                    prior_current_entry=state.get('current_entry')))
        time.sleep(30)
    assert sha(args.inventory) == args.inventory_sha256
    required = []
    for name, relative in inventory['required_verified_archive_entries'].items():
        record = state['entries'][name]
        assert record['status'] == 'verified'
        assert record['execute']['receipt_sha256'] == record['verify']['receipt_sha256'] == record['receipt_sha256']
        destination = str(Path(inventory['target_root']) / relative)
        assert record['verify']['destination'] == destination
        required.append(dict(id=name, destination=destination,
                             receipt_sha256=record['receipt_sha256']))
    bindings = [row for row in inventory['checkpoint_bindings'] if 'existing_entry_id' in row]
    # Read only the two already independently verified archive receipts. Their
    # full-byte SHA binds the per-file hashes used by the cross-study references.
    reader = '''import hashlib,json,sys
from pathlib import Path
data=json.load(sys.stdin); receipts={}; checks=[]
for entry in data['entries']:
 p=Path(entry['destination'])/'.revision_archive_receipt.json'; raw=p.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==entry['receipt_sha256']
 receipts[entry['id']]=json.loads(raw)
for binding in data['bindings']:
 row=receipts[binding['existing_entry_id']]['files'][binding['existing_relative_file']]
 assert row['kind']=='file' and row['sha256']==binding['sha256'] and row['bytes']==binding['bytes']
 checks.append({'archive_file':binding['archive_file'],'sha256':row['sha256'],'bytes':row['bytes']})
print(json.dumps({'status':'pass','checked_references':checks}))
'''
    completed = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', inventory['target_host'],
         'python3 -c ' + shlex.quote(reader)],
        input=json.dumps(dict(entries=required, bindings=bindings)),
        text=True, capture_output=True, check=True, timeout=60)
    check = json.loads(completed.stdout)
    check.update(archive_inventory_sha256=args.inventory_sha256,
                 prior_execution_state_sha256=sha(args.after_state),
                 verified_archive_receipts=required,
                 observed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    serial.save(args.inventory.parent / 'existing_reference_verification.json', check)
    serial.save(status_path, dict(status='executing_addendum',
                observed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    serial.OUT = args.inventory.parent
    serial.INVENTORY = args.inventory
    serial.main()
    final = json.loads((serial.OUT / 'execution_status.json').read_text())
    serial.save(status_path, dict(status=final['status'],
                verified_entries=final['verified_entries'], verified_bytes=final['verified_bytes'],
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))


if __name__ == '__main__':
    main()

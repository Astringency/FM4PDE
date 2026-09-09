#!/usr/bin/env python3
"""Run a frozen archive addendum only after a separately frozen batch completes."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time

import run_reviewed_archive as serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--after-state', type=Path, required=True)
    parser.add_argument('--after-inventory-sha256', required=True)
    args = parser.parse_args()
    inventory_bytes = args.inventory.read_bytes()
    if hashlib.sha256(inventory_bytes).hexdigest() != args.inventory_sha256:
        raise RuntimeError('Frozen addendum inventory changed')
    inventory = json.loads(inventory_bytes)
    status_path = args.inventory.parent / 'queue_status.json'
    while True:
        state = json.loads(args.after_state.read_text()) if args.after_state.exists() else None
        if state is not None:
            if state['inventory_sha256'] != args.after_inventory_sha256:
                raise RuntimeError('Prior execution does not match its frozen inventory')
            if state['status'] != 'running':
                if state['status'] != 'complete':
                    raise RuntimeError('Prior archive did not complete cleanly; addendum not started')
                rows = list(state['entries'].values())
                if not rows or any(row['status'] != 'verified' for row in rows):
                    raise RuntimeError('Prior archive has an unverified entry')
                for row in rows:
                    if not (row['execute']['receipt_sha256'] == row['verify']['receipt_sha256']
                            == row['receipt_sha256']):
                        raise RuntimeError('Prior archive receipt hashes disagree')
                break
        serial.save(status_path, dict(status='waiting_for_prior_batch',
                    observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    prior_verified_entries=state.get('verified_entries') if state else 0,
                    prior_current_entry=state.get('current_entry') if state else None))
        time.sleep(30)
    if hashlib.sha256(args.inventory.read_bytes()).hexdigest() != args.inventory_sha256:
        raise RuntimeError('Addendum inventory changed while queued')
    serial.save(args.inventory.parent / 'prior_batch_verification.json', dict(
        status='pass', inventory_sha256=args.inventory_sha256,
        prior_inventory_sha256=args.after_inventory_sha256,
        prior_state_sha256=hashlib.sha256(args.after_state.read_bytes()).hexdigest(),
        prior_verified_entries=state['verified_entries'],
        observed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
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

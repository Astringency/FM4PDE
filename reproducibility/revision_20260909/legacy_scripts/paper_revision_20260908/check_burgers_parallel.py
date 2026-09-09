"""Verify retained predictions and the ownership of redistributed native calls.

This complements the trajectory/mask/error checks in export_burgers_revision.py.
It reads downloaded outputs only; no remote job is changed.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parent
HOSTS = {'193': 'user-NF5280M6', '197': 'user-R5300-G5', '216': 'bm-2208md2'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row['cell'], row['method'], row['steps'], row['sample_id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    assignment = ROOT/'burgers_parallel_assignment_2242.json'
    ledger = json.loads(assignment.read_text())
    protocol = json.loads((ROOT/'burgers_inputs/sampling_protocol_v3.json').read_text())
    assert ledger['original_workers_stopped']
    assert digest(ROOT/'burgers_inputs/sampling_protocol_v3.json') == ledger['protocol_sha256']
    jobs = {key(j): j for j in protocol['jobs']}
    original = {key(row['job']): row for row in ledger['completed']}
    owners = {key(job): worker['index'] for worker in ledger['workers'] for job in worker['jobs']}
    assert set(original).isdisjoint(owners)
    assert set(original) | set(owners) == set(jobs)
    assert len(original) == 4448 and len(owners) == 2552
    seen, new_counts, pilots = set(), {}, {}
    for server, host in HOSTS.items():
        root = ROOT/f'burgers_results_{server}'
        new_counts[server] = 0
        for worker in ledger['workers']:
            if worker['host'] != host:
                continue
            index = worker['index']
            tag = f'parallel_{index:02d}'
            environment = json.loads((root/f'environment_shard{tag}.json').read_text())
            assert environment['host'] == host
            assert environment['visible_devices'] == str(worker['gpu'])
            pilot_path = root/f'pilot_shard{tag}.json'
            if not pilot_path.exists():
                pilots[str(index)] = 'pending'
                continue
            pilot = json.loads(pilot_path.read_text())
            assert pilot['status'] == 'passed'
            assert pilot['protocol_sha256'] == ledger['protocol_sha256']
            assert len(pilot['checks']) == 4
            for row in pilot['checks']:
                assert row['repeat_max_abs'] == row['hidden_max_abs'] == row['reference_max_abs'] == 0
                assert row['observation_change_max_abs'] > 0
            pilots[str(index)] = 'passed'
        for path in (root/'results').glob('*/*/*.json'):
            row = json.loads(path.read_text())
            ident = key(row)
            assert ident in jobs and ident not in seen, (ident, path)
            seen.add(ident)
            assert all(row[k] == v for k, v in jobs[ident].items())
            assert row['protocol_sha256'] == ledger['protocol_sha256']
            assert digest(path.with_suffix('.pt')) == row['tensor_sha256']
            if ident in original:
                assert digest(path) == original[ident]['receipt_sha256']
                assert row['tensor_sha256'] == original[ident]['tensor_sha256']
                continue
            worker = ledger['workers'][owners[ident]]
            assert worker['host'] == host
            tag = f"parallel_{worker['index']:02d}"
            evidence = row['executor']
            assert evidence == json.loads((root/f'executor_{tag}.json').read_text())
            assert evidence['assignment_sha256'] == digest(assignment)
            assert evidence['executor_sha256'] == ledger['executor_sha256']
            assert evidence['reference_runner_sha256'] == protocol['runner_sha256']
            assert evidence['generated_worker_sha256'] == digest(root/f'worker_{tag}.py')
            assert not evidence['pilot_only']
            assert pilots[str(worker['index'])] == 'passed'
            new_counts[server] += 1
    assert set(original) <= seen
    if args.require_complete:
        assert seen == set(jobs), len(set(jobs)-seen)
    result = dict(checked_local=datetime.datetime.now().astimezone().isoformat(),
                  preserved_original_calls=len(original), new_calls_by_host=new_counts,
                  total_calls=len(seen), remaining_calls=len(set(jobs)-seen),
                  pilots=pilots, assignment_sha256=digest(assignment),
                  scope='Assignment, native reference pilots, unchanged prior receipts and tensor digests. '
                        'Physical masks, truth and error recomputation are checked by the trajectory exporter.')
    (ROOT/'burgers_parallel_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

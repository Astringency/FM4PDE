#!/usr/bin/env python3
"""Release the three K-study archive entries only after terminal scientific gates."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_archive_inventory import probe, PYTHON
from prepare_baseline_second_archive_addendum import check_overlap

COMMIT = '1f1573bfbc246b83e48a3b46402c8f2467488e0d'
PRODUCER_SHA = '1fc697c438cd52bc6b637d0159cfd7375a5cf762ff26870e0886b7c5f29c9606'
KS = [1, 3, 10, 100, 1000]
TASKS = ['forward', 'inverse', 'both']
SHARDS = {'server197': [0], 'server216': [1, 2, 3]}
IDS = ['conditional_scaling_raw_server197', 'conditional_scaling_raw_server216', 'conditional_scaling_local_audit']

REMOTE_METADATA = r'''
import hashlib,json,sys
from pathlib import Path
args=json.load(sys.stdin);root=Path(args['root'])/'production'
def record(path):
 data=path.read_bytes()
 return dict(path=str(path.relative_to(root.parent)),sha256=hashlib.sha256(data).hexdigest(),data=json.loads(data))
rows=[]
for shard in args['shards']:
 environment=record(root/f'environment_run_{shard}.json')
 complete=record(root/f'complete_{shard}.json')
 command_path=Path('/proc')/str(environment['data']['pid'])/'cmdline'
 command=command_path.read_bytes().decode(errors='replace') if command_path.exists() else ''
 rows.append(dict(shard=shard,environment=environment,complete=complete,
                  producer_live='run_conditional_sample_scaling.py' in command))
print(json.dumps(dict(shards=rows,receipts=[record(p) for p in sorted(root.glob('*/*/K*.json'))])))
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(data), sha(data)


def check_local_gate(audit, review_path, review_sha):
    review, actual_review_sha = read(review_path)
    if actual_review_sha != review_sha:
        raise RuntimeError('Scientific review differs from the explicitly bound SHA')
    ready, ready_sha = read(audit / 'READY_FOR_PAPER_REVIEW.json')
    final, final_sha = read(Path(ready['manifest']))
    csv_path = Path(ready['manifest']).parent / 'conditional_scaling_per_input.csv'
    csv_sha = sha(csv_path.read_bytes())
    if not (review['status'] == 'pass' and review['complete']
            and review['ready_for_paper_review_sha256'] == ready_sha
            and review['final_manifest_sha256'] == final_sha
            and review['per_input_csv_sha256'] == csv_sha):
        raise RuntimeError('Final scientific review has not approved these exact completion artifacts')
    if not (final['complete'] and final['physical_inputs'] == 32 and final['rows'] == 480
            and final['tasks'] == TASKS and final['K'] == KS
            and final['canonical_trajectories'] == 96000 and final['timed_trajectories'] == 106944):
        raise RuntimeError('Final K-study dimensions are incomplete or changed')
    state, state_sha = read(audit / 'STATUS.json')
    workers = [row for rows in state['hosts'].values() for row in rows]
    if not (state['status'] == 'terminal_complete' and len(workers) == 4
            and all(row['complete'] and not row['live'] for row in workers)):
        raise RuntimeError('Collector has not observed all four producers terminating completely')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            arguments = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(arg.endswith(b'/collect_conditional_sample_scaling.py')
               or arg == b'collect_conditional_sample_scaling.py' for arg in arguments):
            raise RuntimeError('Local collector is still active')
    return dict(scientific_review=review, scientific_review_sha256=review_sha,
                ready_for_paper_review_sha256=ready_sha, final_manifest_sha256=final_sha,
                per_input_csv_sha256=csv_sha, terminal_status_sha256=state_sha)


def approved_results(audit, source_entries):
    ready, _ = read(audit / 'READY_FOR_PAPER_REVIEW.json')
    final, _ = read(Path(ready['manifest']))
    approved = {host: {} for host in SHARDS}
    environments = {host: {} for host in SHARDS}
    with (Path(ready['manifest']).parent / 'conditional_scaling_per_input.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    timings = {(r['task'], int(r['offset']), int(r['K'])): r for r in rows}
    expected = {(task, offset, k) for task in TASKS for offset in range(1500, 1532) for k in KS}
    if len(rows) != 480 or set(timings) != expected or len(final['source_manifests']) != 2:
        raise RuntimeError('Approved CSV or source manifests do not contain the full cohort')
    for manifest in final['source_manifests']:
        if not manifest['complete'] or manifest['hash_verified_results'] != len(manifest['results']):
            raise RuntimeError('Scientific review contains an incomplete source manifest')
        hosts = set()
        for result in manifest['results']:
            owners = []
            for host in SHARDS:
                root = Path(source_entries['conditional_scaling_raw_' + host]['source_path'])
                try:
                    relative = str(Path(result['path']).relative_to(root))
                except ValueError:
                    continue
                owners.append((host, relative))
            if len(owners) != 1:
                raise RuntimeError('Approved tensor source has no unique host/root binding')
            host, relative = owners[0]
            key = result['task'], result['offset'], result['K']
            if relative != f'production/{key[0]}/offset{key[1]}/K{key[2]}.pt' or relative in approved[host]:
                raise RuntimeError('Approved tensor path or cohort identity is inconsistent')
            approved[host][relative] = result['sha256']
            hosts.add(host)
        if len(hosts) != 1:
            raise RuntimeError('Source manifest mixes result hosts')
        host = hosts.pop()
        for env in manifest['environments']:
            shard = env['args']['shard_index']
            if shard in environments[host]:
                raise RuntimeError('Duplicate approved shard environment')
            environments[host][shard] = env
    jobs = [(task, offset) for task in TASKS for offset in range(1500, 1532)]
    for host, shards in SHARDS.items():
        paths = {f'production/{task}/offset{offset}/K{k}.pt' for index, (task, offset) in enumerate(jobs)
                 if index % 4 in shards for k in KS}
        if set(approved[host]) != paths or set(environments[host]) != set(shards):
            raise RuntimeError('Approved source manifest differs from the frozen static shard cohort')
    return approved, environments, timings


def check_remote_gate(host, entry, approved, approved_environments, timings):
    result = subprocess.run(['ssh', host, shlex.join([PYTHON[host], '-c', REMOTE_METADATA])],
                            input=json.dumps(dict(root=entry['source_path'], shards=SHARDS[host])),
                            capture_output=True, text=True, check=True, timeout=45)
    metadata = json.loads(result.stdout)
    jobs = [(task, offset) for task in TASKS for offset in range(1500, 1532)]
    assigned = {job for index, job in enumerate(jobs) if index % 4 in SHARDS[host]}
    evidence = {}
    completed = set()
    for row in metadata['shards']:
        env = row['environment']['data']
        if env != approved_environments[row['shard']]:
            raise RuntimeError('Producer environment differs from the scientifically approved export')
        expected = [list(job) for index, job in enumerate(jobs) if index % 4 == row['shard']]
        if row['producer_live'] or row['complete']['data']['jobs'] != expected:
            raise RuntimeError('Live or incomplete production shard: ' + host)
        if not (env['commit'] == COMMIT and env['script_sha256'] == PRODUCER_SHA
                and env['K'] == KS and env['tf32'] and env['args']['batch_size'] == 64
                and env['args']['fused_guidance'] and env['args']['num_shards'] == 4
                and env['args']['shard_index'] == row['shard']):
            raise RuntimeError('Producer identity or frozen precision/batching differs')
        completed.update(tuple(job) for job in expected)
        for kind in ('environment', 'complete'):
            evidence[row[kind]['path']] = row[kind]['sha256']
    expected_receipts = {(task, offset, k) for task, offset in assigned for k in KS}
    observed = set()
    for row in metadata['receipts']:
        receipt = row['data']
        key = receipt['task'], receipt['offset'], receipt['K']
        expected_path = f'production/{key[0]}/offset{key[1]}/K{key[2]}.json'
        if key in observed or row['path'] != expected_path or not (
                receipt['num_steps'] == receipt['nfe_per_draw'] == 100
                and receipt['script_sha256'] == PRODUCER_SHA and receipt['seconds'] > 0
                and len(receipt['result_sha256']) == 64):
            raise RuntimeError('Duplicate or inconsistent production receipt')
        observed.add(key)
        tensor_path = str(Path(row['path']).with_suffix('.pt'))
        if receipt['result_sha256'] != approved.get(tensor_path):
            raise RuntimeError('Current tensor receipt differs from the scientifically approved tensor SHA')
        if any(float(receipt[field]) != float(timings[key][field])
               for field in ('seconds', 'compute_seconds', 'peak_bytes')):
            raise RuntimeError('Current timing receipt differs from the scientifically approved CSV')
        evidence[row['path']] = row['sha256']
        evidence[tensor_path] = receipt['result_sha256']
    if observed != expected_receipts or completed != assigned:
        raise RuntimeError('Receipt cohort does not match frozen static shard assignment')
    return metadata, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-inventory', type=Path, required=True)
    parser.add_argument('--scientific-review', type=Path, required=True)
    parser.add_argument('--scientific-review-sha256', required=True)
    parser.add_argument('--prior-inventory', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to replace a frozen K archive inventory')
    original, original_sha = read(args.original_inventory)
    source_entries = {entry['id']: entry for entry in original['entries']}
    audit = Path(source_entries['conditional_scaling_local_audit']['source_path'])
    gate = check_local_gate(audit, args.scientific_review, args.scientific_review_sha256)
    approved, environments, timings = approved_results(audit, source_entries)
    entries = []
    remote_metadata = {}
    for identifier in IDS:
        entry = dict(source_entries[identifier])
        entry.pop('observed', None)
        entry.pop('evidence_sha256', None)
        if entry['status'] != 'pending':
            raise RuntimeError('Unexpected original entry status')
        expected = {}
        if entry['source_host'] in SHARDS:
            host = entry['source_host']
            metadata, expected = check_remote_gate(host, entry, approved[host], environments[host], timings)
            remote_metadata[entry['source_host']] = metadata
        # Count bytes and metadata only here. The serial executor independently
        # hashes every tensor and checks the receipt's tensor SHA before publish.
        observed = probe(entry['source_host'], [entry])[0]
        if not observed['exists'] or observed.get('missing') or observed.get('symlinks'):
            raise RuntimeError('Missing or unexpectedly linked K archive source')
        metadata_hashes = observed['metadata_sha256']
        if any(metadata_hashes.get(path) != value for path, value in expected.items() if path.endswith('.json')):
            raise RuntimeError('Production metadata changed during inventory census')
        entry.update(status='ready', source_kind='directory', observed=observed,
                     evidence_sha256={**metadata_hashes, **expected}, completion_gate=gate,
                     purpose='Completed K-study raw results or frozen local audit; original source files retained.')
        entries.append(entry)
    # Detect updates to review, readiness, final plot manifest, or terminal state.
    if check_local_gate(audit, args.scientific_review, args.scientific_review_sha256) != gate:
        raise RuntimeError('Final local gate changed during source inventory')
    prior = [entry for path in args.prior_inventory for entry in json.loads(path.read_text())['entries']]
    check_overlap(prior + entries)
    args.output.mkdir(parents=True)
    evidence_dir = args.output / 'evidence'
    evidence_dir.mkdir()
    (evidence_dir / 'scientific_review.json').write_bytes(args.scientific_review.read_bytes())
    (evidence_dir / 'production_terminal_metadata.json').write_text(json.dumps(remote_metadata, indent=2) + '\n')
    inventory = dict(schema_version=1, observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     target_host=original['target_host'], target_root=original['target_root'],
                     original_inventory_sha256=original_sha, completion_gate=gate, entries=entries,
                     completion_evidence_package='This inventory and its sibling evidence directory are retained in the final reproducibility package.')
    path = args.output / 'archive_inventory.conditional_scaling_final.json'
    path.write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), sha256=sha(path.read_bytes()), entries=len(entries),
                          total_bytes=sum(entry['observed']['logical_bytes'] for entry in entries))))


if __name__ == '__main__':
    main()

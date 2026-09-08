"""Back up the running September 8 Burgers study and export when complete.

Run locally with the FM4PDE Python environment in a dedicated tmux session.
Remote access consists of read-only process/file queries and rsync downloads.
Sampling jobs and manuscript files are never changed by this collector.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


HOSTS = {
    '193': ('zhangxf@192.168.191.193', 9088,
            '/home/zhangxf/C01Python/paper_revision_20260908', 'user-NF5280M6'),
    '197': ('zhangxifeng@192.168.191.197', 22,
            '/research_data/users/zhangxifeng/C01Python/paper_revision_20260908', 'user-R5300-G5'),
    '216': ('zjinzxf2025@175.102.135.216', 22,
            '/data1/zjinzxf2025/C01Python/paper_revision_20260908', 'bm-2208md2'),
}
REMOTE = r'''
import json,pathlib,shlex,subprocess
root=pathlib.Path(__ROOT__)
out=root/'burgers_output_v3'
receipts=sorted(str(p.relative_to(out)) for p in (out/'results').glob('*/*/*.json'))
metadata=sorted(p.name for p in out.iterdir() if p.is_file() and
    (p.suffix=='.py' or (p.suffix=='.json' and not p.name.startswith('complete_'))))
complete=sorted(p.name for p in out.glob('complete_*.json'))
live=[]
driver=str(root/'BurgersParallelCode/plot/run_burgers_parallel.py')
for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
    columns=line.strip().split(None,1)
    if len(columns)!=2: continue
    try: args=shlex.split(columns[1])
    except ValueError: continue
    if not args or not args[0].startswith('/') or '/bin/python' not in args[0]: continue
    if driver in args and '--worker-index' in args:
        live.append(dict(pid=int(columns[0]),index=int(args[args.index('--worker-index')+1])))
print(json.dumps(dict(receipts=receipts,metadata=metadata,complete=complete,live=live)))
'''


def now():
    return datetime.datetime.now().astimezone().isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def log(event, **values):
    print(json.dumps(dict(time=now(), event=event, **values)), flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_path(job):
    return f"results/{job['cell']}/{job['method']}_{job['steps']}/sample{job['sample_id']}.json"


def ssh_args(server):
    host, port, _, _ = HOSTS[server]
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', '-p', str(port), host]


def probe(server):
    code = REMOTE.replace('__ROOT__', repr(HOSTS[server][2]))
    result = subprocess.run(ssh_args(server) + ['python3 -c ' + shlex.quote(code)],
                            capture_output=True, text=True, timeout=55, check=True)
    data = json.loads(result.stdout)
    data['checked_local'] = now()
    return data


def parallel(function, names):
    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(function, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as error:
                errors[name] = str(error)
                log('retry_needed', server=name, error=str(error))
    return results, errors


def mirror(server, snapshot, study, complete=False):
    host, _, remote, _ = HOSTS[server]
    destination = study / f'burgers_results_{server}'
    destination.mkdir(exist_ok=True)
    # Freeze the receipt list before copying tensors: a result completed during
    # rsync is picked up next cycle, never copied without its prediction.
    stages = [[str(Path(p).with_suffix('.pt')) for p in snapshot['receipts']] + snapshot['metadata'],
              snapshot['receipts'], snapshot['complete'] if complete else []]
    for paths in stages:
        if not paths:
            continue
        for path in paths:
            if Path(path).is_absolute() or '..' in Path(path).parts or '\n' in path:
                raise ValueError(f'Invalid relative path: {path!r}')
        result = subprocess.run(
            ['rsync', '-a', '--protect-args', '--timeout=45', '--files-from=-',
             '-e', shlex.join(ssh_args(server)[:-1]),
             f'{host}:{remote}/burgers_output_v3/', str(destination) + '/'],
            input='\n'.join(paths) + '\n', capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError(f'rsync {server}: {result.stderr[-2000:]}')
    log('backup_finished', server=server, receipts=len(snapshot['receipts']))
    return len(snapshot['receipts'])


def export_complete(study, guarded):
    for path, expected in guarded.items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f'Source changed while waiting: {path}')
    scripts = Path(__file__).resolve().parent
    commands = [
        [sys.executable, str(study / 'check_burgers_parallel.py'), '--require-complete'],
        [sys.executable, str(scripts / 'export_burgers_revision.py'), 'export',
         '--inputs', str(study / 'burgers_inputs'),
         '--sampling-protocol', str(study / 'burgers_inputs/sampling_protocol_v3.json'),
         '--fm-random', str(study / 'burgers_fm_random'), '--results',
         *(str(study / f'burgers_results_{s}') for s in HOSTS),
         '--output', str(study / 'burgers_complete')],
    ]
    for command in commands:
        log('validation_started', command=command)
        subprocess.run(command, check=True)
    output = study / 'burgers_complete'
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['final_ready'] and not manifest['pending_jobs']
    assert manifest['expected_new_calls'] == manifest['completed_new_calls'] == 7000
    assert manifest['rows'] == 40000
    with (output / 'summary.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 48
    assert sum(r['supported'] == 'True' for r in rows) == 40
    assert all(int(r['n']) == (1000 if r['supported'] == 'True' else 0) for r in rows)
    write(study / 'burgers_watch_ready.json', dict(
        ready_local=now(), output=str(output), guarded_sources=guarded,
        artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        manuscript_integration='Pending guarded main/response integration and final review.'))
    log('full_export_ready', output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--backup-seconds', type=int, default=900)
    parser.add_argument('--once', action='store_true', help='Run one monitor/backup cycle and exit.')
    args = parser.parse_args()
    study = args.study.resolve()
    if args.poll_seconds < 30 or args.backup_seconds < args.poll_seconds:
        parser.error('Use poll >= 30 seconds and backup >= poll.')
    # A lock prevents two local collectors from writing the same backup/state.
    with (study / 'burgers_watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assignment = study / 'burgers_parallel_assignment_2242.json'
        protocol = study / 'burgers_inputs/sampling_protocol_v3.json'
        scripts = Path(__file__).resolve().parent
        guarded = {str(p): sha(p) for p in [Path(__file__).resolve(), assignment, protocol,
            study / 'check_burgers_parallel.py', scripts / 'export_burgers_revision.py',
            scripts / 'run_paper_ablation_revision.py', study / 'burgers_inputs/protocol.json']}
        guard_file = study / 'burgers_watch_sources.json'
        if guard_file.exists():
            assert json.loads(guard_file.read_text()) == guarded, 'Collector sources changed since first launch'
        else:
            write(guard_file, guarded)
        jobs = json.loads(protocol.read_text())['jobs']
        expected = {receipt_path(job) for job in jobs}
        assert len(expected) == len(jobs) == 7000
        ledger = json.loads(assignment.read_text())
        last_backup = 0.0
        backed_up = None
        while True:
            snapshots, errors = parallel(probe, HOSTS)
            state = dict(checked_local=now(), status='running', errors=errors, hosts={},
                         last_backup=backed_up, expected=7000)
            seen = set()
            for server, snapshot in snapshots.items():
                found = set(snapshot['receipts'])
                assert len(found) == len(snapshot['receipts'])
                assert found <= expected and not (seen & found), ('unexpected/duplicate receipts', server)
                seen |= found
                live = {row['index'] for row in snapshot['live']}
                pending_workers = {w['index'] for w in ledger['workers']
                    if w['host'] == HOSTS[server][3] and
                    any(receipt_path(j) not in found for j in w['jobs'])}
                missing = sorted(pending_workers - live)
                state['hosts'][server] = dict(saved=len(found), live=snapshot['live'],
                    missing_workers=missing, checked_local=snapshot['checked_local'])
                if missing:
                    state['status'] = 'attention_needed'
            state['remote_saved'] = len(seen) if not errors else None
            if errors:
                state['status'] = 'connection_retry'
            complete = not errors and seen == expected
            if snapshots and (complete or time.monotonic() - last_backup >= args.backup_seconds):
                copied, copy_errors = parallel(lambda s: mirror(s, snapshots[s], study, complete), snapshots)
                state['backup_errors'] = copy_errors
                if copy_errors:
                    state['status'] = 'backup_retry'
                if not errors and not copy_errors:
                    backed_up = dict(finished_local=now(), receipts=sum(copied.values()))
                    last_backup = time.monotonic()
                    state['last_backup'] = backed_up
                    if complete:
                        try:
                            export_complete(study, guarded)
                        except Exception as error:
                            state.update(status='validation_failed', validation_error=str(error))
                            write(study / 'burgers_watch_state.json', state)
                            raise
                        state['status'] = 'ready_for_paper'
            write(study / 'burgers_watch_state.json', state)
            log('monitor', **state)
            if args.once or state['status'] == 'ready_for_paper':
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()

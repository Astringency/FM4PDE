"""Stage canonical training inputs on a compute host and verify every file.

Run one controller per PDE in tmux. Existing datasets and formal checkpoints on
both hosts are read only; staged copies live in this study's temporary cache.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def remote_python(host, program, arguments=(), payload=None):
    return subprocess.check_output(['ssh', '-T', '-o', 'BatchMode=yes', host,
        shlex.join(['/usr/bin/python3', '-c', program, *map(str, arguments)])],
        input=payload, text=True)


def remote_write(path, value):
    remote_python('server197',
        "from pathlib import Path; import sys,json; p=Path(sys.argv[1]); "
        "t=Path(str(p)+'.tmp'); t.write_text(json.dumps(json.load(sys.stdin),indent=2)+'\\n'); t.replace(p)",
        [path], json.dumps(value))


def verify(path, expected, destination=None):
    program = '''from pathlib import Path
import hashlib,json,sys,time
p=Path(sys.argv[1]); start=time.monotonic()
if not p.exists():
 print(json.dumps(dict(exists=False))); raise SystemExit(0)
h=hashlib.sha256()
with p.open('rb') as f:
 for block in iter(lambda:f.read(8<<20),b''): h.update(block)
ok=h.hexdigest()==sys.argv[2]
if ok and len(sys.argv)>3: p.replace(sys.argv[3])
print(json.dumps(dict(exists=True,verified=ok,sha256=h.hexdigest(),bytes=p.stat().st_size if p.exists() else Path(sys.argv[3]).stat().st_size,seconds=time.monotonic()-start)))
'''
    arguments = [path, expected] + ([destination] if destination else [])
    return json.loads(remote_python('server216', program, arguments))


def main(args):
    study = args.study.resolve()
    output = study / 'remote_execution'
    canonical = Path('/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/pretrained')
    remote_output = canonical / study.name / 'remote_execution'
    state_path = output / f'transfer_{args.pde}.json'
    if state_path.exists() and read(state_path).get('status') == 'complete':
        raise RuntimeError('This PDE already has a completed transfer manifest')
    data = [row for row in read(output / 'training_data_server197.json')['rows'] if row['pde'] == args.pde]
    assert len(data) == 5
    files = []
    for row in data:
        relative = Path(row['path']).relative_to('/large_storage/zhangxf/PDEdata')
        files.append(dict(kind='training_data', source=row['path'], sha256=row['sha256'],
                          destination=str(args.cache / 'training_data' / relative)))
    binding = next(row for row in read(study / 'main_checkpoint_binding.json')['models'] if row['pde'] == args.pde)
    for field in ['resume', 'inference']:
        source = Path(binding[field + '_checkpoint'])
        files.append(dict(kind=field + '_checkpoint', source=str(source), sha256=binding[field + '_sha256'],
                          destination=str(args.cache / 'pretrained' / source.relative_to(canonical))))
    remote_python('server216',
        'from pathlib import Path; import sys; [Path(p).mkdir(parents=True,exist_ok=True) for p in sys.argv[1:]]',
        sorted({str(Path(row['destination']).parent) for row in files}))
    state = dict(status='running', pde=args.pde, pid=os.getpid(), started_unix=time.time(),
                 cache=str(args.cache), files=[])
    for row in files:
        before = verify(row['destination'], row['sha256'])
        if before['exists']:
            assert before['verified'], 'An existing completed cache file differs from its canonical source'
            checked = before
            elapsed = 0.0
        else:
            pending = row['destination'] + '.transfer.part'
            partial = verify(pending, row['sha256'], row['destination'])
            if partial.get('verified'):
                checked, elapsed = partial, 0.0
            else:
                command = ['scp', '-3', '-B', 'server197:' + row['source'], 'server216:' + pending]
                started = time.monotonic()
                with (output / f'transfer_{args.pde}.log').open('a') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    state.update(current=row, child_pid=child.pid, command=command, checked_unix=time.time())
                    write(state_path, state)
                    remote_write(remote_output / state_path.name, state)
                    code = child.wait()
                assert code == 0, f'Input transfer exited with {code}; retain partial file and inspect the log'
                elapsed = time.monotonic() - started
                checked = verify(pending, row['sha256'], row['destination'])
                assert checked['verified']
        state['files'].append(dict(**row, verification=checked, transfer_seconds=elapsed))
        state.pop('current', None)
        state.pop('child_pid', None)
        state.pop('command', None)
        write(state_path, state)
        remote_write(remote_output / state_path.name, state)
        print('VERIFIED_REMOTE_INPUT', args.pde, row['destination'], checked['sha256'], flush=True)
    state.update(status='complete', ended_unix=time.time())
    with (output / 'transfer_manifest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write(state_path, state)
        remote_write(remote_output / state_path.name, state)
        completed = [read(path) for path in output.glob('transfer_*.json')
                     if read(path).get('status') == 'complete']
        receipt = dict(status='verified', pdes=sorted(row['pde'] for row in completed), files_per_pde=5,
                       cached_training_root=str(args.cache / 'training_data'),
                       cached_pretrained_root=str(args.cache / 'pretrained'),
                       source='Canonical server197 files copied unchanged; every staged file verified by SHA-256',
                       manifests={row['pde']: row['files'] for row in completed}, checked_unix=time.time())
        write(output / 'data_identity_verification.json', receipt)
        remote_write(remote_output / 'data_identity_verification.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--pde', choices=['poisson', 'darcy', 'burger'], required=True)
    parser.add_argument('--cache', type=Path, required=True)
    main(parser.parse_args())

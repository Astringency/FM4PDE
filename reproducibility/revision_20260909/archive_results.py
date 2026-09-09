#!/usr/bin/env python3
"""Copy completed study artifacts to server197 without replacing different data.

The public command defaults to a read-only plan. Remote helper modes are used
only after this exact script has been installed on server197 through git.
No producer, source directory, or existing archive is deleted or modified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

HERE = Path(__file__).resolve().parent
TARGET = Path('/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs')
REMOTE_SCRIPT = TARGET.parent / 'reproducibility/revision_20260909/archive_results.py'
RECEIPT = '.revision_archive_receipt.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, text=True, **kwargs)


def tree(path, dereference=False, ignore_receipt=False):
    """SHA-256 every file; reject source mutation and unsupported node types."""
    path = Path(path)
    result = {}

    def visit(p, relative, ancestors):
        if p.is_symlink() and not dereference:
            result[relative] = {'kind': 'symlink', 'target': os.readlink(p)}
            return
        if p.is_dir():
            resolved = p.resolve()
            if resolved in ancestors:
                raise RuntimeError(f'Symlink cycle: {p}')
            if relative:
                result[relative] = {'kind': 'directory'}
            for child in sorted(p.iterdir()):
                if child.name == RECEIPT and p == path and ignore_receipt:
                    continue
                visit(child, '/'.join(x for x in [relative, child.name] if x), ancestors | {resolved})
        elif p.is_file():
            before = p.stat()
            digest = sha(p)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise RuntimeError(f'Source changed while hashing: {p}')
            result[relative] = {'kind': 'file', 'bytes': after.st_size, 'sha256': digest}
        else:
            raise RuntimeError(f'Missing or unsupported source node: {p}')

    visit(path, '' if path.is_dir() else path.name, set())
    return result


def content_bytes(manifest):
    return sum(row.get('bytes', 0) for row in manifest.values())


def source_tree(entry, source):
    if Path(source).is_dir() and (Path(source) / RECEIPT).exists():
        raise ValueError('A source must not contain the reserved root archive receipt')
    selected = entry.get('include_files')
    if not selected:
        return tree(source, entry.get('dereference_symlinks', False))
    result = {}
    for relative in selected:
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Invalid selected relative file')
        full = Path(source) / p
        if full.is_dir():
            raise ValueError('include_files accepts files only, not directory trees')
        row = tree(full, entry.get('dereference_symlinks', False))[p.name]
        result[p.as_posix()] = row
        for parent in p.parents:
            if parent != Path('.'):
                result[parent.as_posix()] = {'kind': 'directory'}
    return result


def selection_flags(entry, directory):
    if not entry.get('include_files'):
        return []
    path = Path(directory) / 'selected-files.txt'
    if any('\n' in p or '\r' in p for p in entry['include_files']):
        raise ValueError('Newlines are not supported in selected file names')
    path.write_text('\n'.join(entry['include_files']) + '\n')
    return ['-r', '--files-from=' + str(path)]


def validate_entry(entry):
    if entry.get('status') != 'ready' or entry.get('role') not in {'primary', 'dependency'}:
        raise RuntimeError(f"Not an executable complete artifact: {entry['id']} ({entry.get('status')}/{entry.get('role')})")
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', entry['id']):
        raise ValueError('Invalid entry id')
    source = Path(entry['source_path'])
    if not source.is_absolute() or '..' in source.parts:
        raise ValueError('Source must be an absolute normalized path')
    relative = Path(entry['destination_relative'])
    if relative.is_absolute() or '..' in relative.parts or len(relative.parts) < 4:
        raise ValueError('Invalid destination')
    if relative.parts[0] not in {'main', 'ablations'} or relative.parts[1] != 'revision_20260909':
        raise ValueError('Destination must be inside this revision subtree')
    if entry['source_host'] != 'local' and not re.fullmatch(r'[a-zA-Z0-9_.-]+', entry['source_host']):
        raise ValueError('Invalid SSH alias')
    if 'include_files' in entry and not entry['include_files']:
        raise ValueError('An empty include_files list must never expand to an entire source tree')


def check_evidence(entry, source_root):
    for relative, expected in entry.get('evidence_sha256', {}).items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Invalid evidence relative path')
        path = Path(source_root) / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f'Frozen metadata mismatch: {entry["id"]}/{relative}')


def prepare(entry, root=TARGET):
    validate_entry(entry)
    final = root / entry['destination_relative']
    if final.resolve() != root.resolve() and root.resolve() not in final.resolve().parents:
        raise ValueError('Destination traverses a symlink outside the target root')
    # Each entry receives a new private staging directory on the same filesystem.
    staging = root / entry['destination_relative'].split('/')[0] / 'revision_20260909/.incoming'
    if root.resolve() not in staging.resolve().parents:
        raise ValueError('Staging directory traverses outside the target root')
    staging.mkdir(parents=True, exist_ok=True)
    stage = staging / (entry['id'] + '-' + uuid.uuid4().hex)
    stage.mkdir()
    (stage / 'payload').mkdir()
    (stage / 'entry.json').write_text(json.dumps(entry, indent=2) + '\n')
    return stage, final


def promote(stage, expected, root=TARGET):
    stage = Path(stage).resolve()
    allowed = [(root / k / 'revision_20260909/.incoming').resolve() for k in ['main', 'ablations']]
    if not any(stage.parent == p for p in allowed):
        raise ValueError('Staging path is outside the archive staging area')
    entry = json.loads((stage / 'entry.json').read_text())
    validate_entry(entry)
    payload = stage / 'payload'
    actual = tree(payload)
    if actual != expected:
        raise RuntimeError(f"Transfer hash mismatch: {entry['id']}")
    final = root / entry['destination_relative']
    if final.resolve() != root.resolve() and root.resolve() not in final.resolve().parents:
        raise ValueError('Destination traverses a symlink outside the target root')
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() or final.is_symlink():
        if final.is_symlink() or not final.is_dir() or tree(final, ignore_receipt=True) != expected:
            raise RuntimeError(f'Existing destination differs; left untouched: {final}')
        if not (final / RECEIPT).is_file():
            raise RuntimeError(f'Identical destination lacks an archive receipt; review manually: {final}')
        return {'status': 'already_verified', 'destination': str(final), 'bytes': content_bytes(expected),
                'receipt_sha256': sha(final / RECEIPT)}
    receipt = dict(version=1, archived_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   source=entry, files=expected, content_bytes=content_bytes(expected),
                   archive_script_sha256=sha(__file__), source_preserved=True)
    (payload / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    # Directory rename cannot replace a nonempty different archive directory.
    # No per-file writes are ever made into the final destination.
    try:
        payload.rename(final)
    except OSError as exc:
        raise RuntimeError(f'Atomic publication failed; destination untouched: {final}') from exc
    return {'status': 'archived', 'destination': str(final), 'bytes': content_bytes(expected),
            'receipt_sha256': sha(final / RECEIPT)}


def verify_entry(entry, root=TARGET):
    """Read only: recompute destination SHA-256 against its archive receipt."""
    validate_entry(entry)
    final = root / entry['destination_relative']
    if final.is_symlink() or root.resolve() not in final.resolve().parents:
        raise ValueError('Invalid archive destination')
    receipt = json.loads((final / RECEIPT).read_text())
    for key in ['id', 'source_host', 'source_path', 'destination_relative']:
        if receipt['source'].get(key) != entry.get(key):
            raise RuntimeError('Archive receipt source identity mismatch: ' + key)
    if tree(final, ignore_receipt=True) != receipt['files']:
        raise RuntimeError('Archived content differs from the recorded file hashes')
    check_evidence(entry, final)
    return {'status': 'verified', 'destination': str(final), 'bytes': content_bytes(receipt['files']),
            'receipt_sha256': sha(final / RECEIPT)}


def copy_local(entry, root=TARGET):
    """Server197 sources stay on that filesystem, avoiding a network round trip."""
    validate_entry(entry)
    source = Path(entry['source_path'])
    final = root / entry['destination_relative']
    if source.is_dir() and (final.resolve() == source.resolve() or source.resolve() in final.resolve().parents):
        raise ValueError('Destination would be inside source')
    check_evidence(entry, source if source.is_dir() else source.parent)
    expected = source_tree(entry, source)
    if shutil.disk_usage(root).free < content_bytes(expected) * 1.1 + (1 << 30):
        raise RuntimeError('Insufficient destination space with a 10% plus 1 GiB reserve')
    stage, _ = prepare(entry, root)
    flags = ['-a', '--checksum', '--protect-args']
    if entry.get('dereference_symlinks'):
        flags.append('--copy-links')
    src = str(source) + ('/' if source.is_dir() else '')
    run(['rsync', *flags, *selection_flags(entry, stage), '--', src, str(stage / 'payload') + '/'], stdout=sys.stderr)
    if source_tree(entry, source) != expected:
        raise RuntimeError('Source changed during copy; staging retained, nothing published')
    return promote(stage, expected, root)


def remote(mode, payload, target_host='server197', remote_script=REMOTE_SCRIPT):
    command = shlex.join(['/usr/bin/python3', str(remote_script), mode])
    response = run(['ssh', target_host, command], input=json.dumps(payload), capture_output=True)
    return json.loads(response.stdout)


def job_directory(entry, request, root=TARGET):
    validate_entry(entry)
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
    category = Path(entry['destination_relative']).parts[0]
    return root / category / 'revision_20260909/.archive_jobs' / digest


def job_state(directory, root=TARGET):
    directory = Path(directory).resolve()
    allowed = [(root / k / 'revision_20260909/.archive_jobs').resolve() for k in ['main', 'ablations']]
    if directory.parent not in allowed or not re.fullmatch('[0-9a-f]{24}', directory.name):
        raise ValueError('Invalid persistent archive job path')
    result = directory / 'result.json'
    if result.exists():
        return json.loads(result.read_text())
    session = 'archive0909_' + directory.name
    live = subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0
    return {'state': 'running' if live else 'interrupted', 'job': str(directory), 'session': session}


def submit_job(message, root=TARGET):
    """Launch only our own deterministic tmux session, or reattach by request hash."""
    mode = message['mode']
    if mode not in ['_local-copy', '_promote', '_verify']:
        raise ValueError('Unsupported persistent operation')
    request = dict(mode=mode, payload=message['payload'], entry=message['entry'], nonce=message['nonce'],
                   script=str(Path(__file__).resolve()), script_sha256=sha(__file__))
    directory = job_directory(message['entry'], request, root)
    if root.resolve() not in directory.resolve().parents:
        raise ValueError('Persistent job directory traverses outside the target root')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recorded = directory / 'request.json'
        if recorded.exists():
            if json.loads(recorded.read_text()) != request:
                raise RuntimeError('Persistent job request identity mismatch')
            state = job_state(directory, root)
            if state['state'] == 'interrupted':
                raise RuntimeError('Previous archive job was interrupted; inspect its private logs/staging before retrying: ' + str(directory))
            return {'job': str(directory), **state}
        recorded.write_text(json.dumps(request, indent=2) + '\n')
        session = 'archive0909_' + directory.name
        command = shlex.join(['tmux', 'set-window-option', '-t', session, 'remain-on-exit', 'off'])
        command += ' && exec ' + shlex.join([sys.executable, str(Path(__file__).resolve()), '_run-job', str(directory)])
        run(['tmux', 'new-session', '-d', '-s', session, command], capture_output=True)
        return {'job': str(directory), 'session': session, 'state': 'running'}


def run_job(directory, root=TARGET):
    directory = Path(directory).resolve()
    # Validate the job namespace without depending on whether tmux is visible.
    allowed = [(root / k / 'revision_20260909/.archive_jobs').resolve() for k in ['main', 'ablations']]
    if directory.parent not in allowed or not re.fullmatch('[0-9a-f]{24}', directory.name):
        raise ValueError('Invalid persistent archive job path')
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / 'result.json').exists():
            return
        request = json.loads((directory / 'request.json').read_text())
        started = dict(pid=os.getpid(), started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (directory / 'started.json').write_text(json.dumps(started) + '\n')
        try:
            if sha(request['script']) != request['script_sha256']:
                raise RuntimeError('Persistent archive job script changed after submission')
            with (directory / 'stdout.json').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
                result = subprocess.run([sys.executable, request['script'], request['mode']],
                    input=json.dumps(request['payload']), text=True, stdout=stdout, stderr=stderr)
            if result.returncode:
                outcome = {'state':'failed','returncode':result.returncode,'job':str(directory), 'error':'Inspect stderr.log; archive staging was retained.'}
            else:
                outcome = {'state':'complete','job':str(directory),'result':json.loads((directory / 'stdout.json').read_text())}
        except Exception as exc:
            outcome = {'state':'failed','job':str(directory),'error':str(exc)}
        outcome['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        temporary = directory / 'result.json.tmp'
        temporary.write_text(json.dumps(outcome, indent=2) + '\n')
        temporary.replace(directory / 'result.json')
    # The tmux command uses exec, so this process exiting closes only its session.


def persistent_remote(mode, payload, entry, target_host='server197', remote_script=REMOTE_SCRIPT):
    nonce = uuid.uuid4().hex
    print(json.dumps({'archive_request_nonce':nonce,'entry':entry['id'],'mode':mode}), flush=True)
    status = remote('_submit-job', {'mode':mode,'payload':payload,'entry':entry,'nonce':nonce}, target_host, remote_script)
    print(json.dumps({'persistent_archive_job':status['job'], 'state':status['state']}), flush=True)
    return wait_remote_job(status, target_host, remote_script)


def wait_remote_job(status, target_host='server197', remote_script=REMOTE_SCRIPT):
    while status['state'] == 'running':
        time.sleep(5)
        status = remote('_job-status', {'job':status['job']}, target_host, remote_script)
    if status['state'] != 'complete':
        raise RuntimeError('Persistent archive job did not complete: ' + json.dumps(status))
    return status['result']


def archive_entry(entry, relay_root, target_host='server197', remote_script=REMOTE_SCRIPT):
    validate_entry(entry)
    if entry['source_host'] == target_host:
        return persistent_remote('_local-copy', entry, entry, target_host, remote_script)
    relay_root = Path(relay_root)
    relay_root.mkdir(parents=True, exist_ok=True)
    estimate = entry.get('observed', {}).get('logical_bytes', 0)
    if shutil.disk_usage(relay_root).free < estimate * 1.1 + (1 << 30):
        raise RuntimeError('Insufficient local relay space')
    # Each relay is isolated; a failed transfer is retained for inspection.
    relay = Path(tempfile.mkdtemp(prefix=entry['id'] + '-', dir=relay_root))
    payload = relay / 'payload'
    payload.mkdir()
    source = entry['source_path']
    source_is_dir = entry.get('source_kind', 'directory') == 'directory'
    source_arg = source + ('/' if source_is_dir else '')
    if entry['source_host'] != 'local':
        source_arg = entry['source_host'] + ':' + source_arg
    flags = ['-a', '--checksum', '--protect-args', '--partial-dir=.rsync-partial']
    if entry.get('dereference_symlinks'):
        flags.append('--copy-links')
    selected_flags = selection_flags(entry, relay)
    run(['rsync', *flags, *selected_flags, '--', source_arg, str(payload) + '/'])
    # Rsync independently rereads source checksums after the transfer. --delete
    # occurs only in dry-run mode and detects extra files; it deletes nothing.
    check = run(['rsync', '-rcln', '--delete', '--itemize-changes', '--protect-args',
                 *(['--copy-links'] if entry.get('dereference_symlinks') else []),
                 *selected_flags,
                 '--', source_arg, str(payload) + '/'], capture_output=True)
    if check.stdout.strip():
        raise RuntimeError(f'Source/relay differ after transfer: {check.stdout[:1000]}')
    check_evidence(entry, payload)
    if (payload / RECEIPT).exists():
        raise ValueError('A source must not contain the reserved root archive receipt')
    expected = tree(payload)
    stage = remote('_prepare', {'entry': entry, 'bytes': content_bytes(expected)}, target_host, remote_script)['stage']
    run(['rsync', '-a', '--checksum', '--protect-args', '--partial-dir=.rsync-partial',
         '--', str(payload) + '/', target_host + ':' + stage + '/payload/'])
    outcome = persistent_remote('_promote', {'stage': stage, 'expected': expected}, entry, target_host, remote_script)
    # Only this invocation's private relay is removed after complete verification.
    shutil.rmtree(relay)
    return outcome


def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith('_'):
        mode = sys.argv[1]
        if mode == '_run-job':
            run_job(sys.argv[2]); return
        if mode == '_version':
            print(json.dumps({'sha256': sha(__file__)})); return
        payload = json.load(sys.stdin)
        if mode == '_local-copy':
            result = copy_local(payload)
        elif mode == '_submit-job':
            result = submit_job(payload)
        elif mode == '_job-status':
            result = job_state(payload['job'])
        elif mode == '_verify':
            result = verify_entry(payload)
        elif mode == '_prepare':
            if shutil.disk_usage(TARGET).free < payload['bytes'] * 1.1 + (1 << 30):
                raise RuntimeError('Insufficient target disk space')
            stage, final = prepare(payload['entry'])
            result = {'stage': str(stage), 'destination': str(final)}
        elif mode == '_promote':
            result = promote(payload['stage'], payload['expected'])
        else:
            raise ValueError(mode)
        print(json.dumps(result)); return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=HERE / 'archive_inventory.json')
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--all-ready', action='store_true')
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--execute', action='store_true', help='Actually copy only ready primary/dependency entries')
    operation.add_argument('--verify', action='store_true', help='Read-only SHA-256 verification of already archived entries')
    operation.add_argument('--resume-job', type=Path, help='Observe/wait on an existing persistent archive helper without restarting it')
    parser.add_argument('--remote-script', type=Path, default=REMOTE_SCRIPT,
                        help='Absolute script path in a git-synchronized independent checkout on server197, or the main FM4PDE checkout')
    parser.add_argument('--relay-root', type=Path, default=Path('/home/tat512/C01Python/audit/revision_archive_relay'))
    args = parser.parse_args()
    if args.resume_job:
        if remote('_version', {}, remote_script=args.remote_script)['sha256'] != sha(__file__):
            raise RuntimeError('Local and remote archive scripts differ')
        status = remote('_job-status', {'job':str(args.resume_job)}, remote_script=args.remote_script)
        print(json.dumps(wait_remote_job(status, remote_script=args.remote_script)), flush=True)
        return
    inventory = json.loads(args.inventory.read_text())
    selected = [e for e in inventory['entries'] if not args.ids or e['id'] in args.ids]
    if args.ids and {e['id'] for e in selected} != set(args.ids):
        raise ValueError('Unknown entry id')
    if args.all_ready:
        selected = [e for e in selected if e['status'] == 'ready' and e['role'] in {'primary', 'dependency'}]
    for entry in selected:
        print(json.dumps({k: entry.get(k) for k in ['id', 'status', 'role', 'source_host', 'source_path', 'destination_relative']}), flush=True)
    if not args.execute and not args.verify:
        print('PLAN ONLY: no transfers or remote mutations performed.'); return
    if not args.ids and not args.all_ready:
        raise ValueError('--execute/--verify requires --ids or --all-ready')
    if not args.remote_script.is_absolute() or '..' in args.remote_script.parts:
        raise ValueError('--remote-script must be an absolute normalized path')
    for entry in selected:
        validate_entry(entry)
    if remote('_version', {}, remote_script=args.remote_script)['sha256'] != sha(__file__):
        raise RuntimeError('Install this exact archive script on server197 through git before execution')
    for entry in selected:
        if args.verify:
            result = persistent_remote('_verify', entry, entry, remote_script=args.remote_script)
        else:
            result = archive_entry(entry, args.relay_root, remote_script=args.remote_script)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

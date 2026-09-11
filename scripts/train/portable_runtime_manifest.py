"""Hash the copied Python runtime without importing or executing that runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time


MEMBERS = ('bin/python', 'bin/python3', 'bin/python3.12', 'lib')


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def walk(path, relative):
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        yield relative, dict(type='symlink', target=os.readlink(path))
    elif stat.S_ISDIR(metadata.st_mode):
        yield relative, dict(type='directory', mode=mode)
        for child in sorted(path.iterdir()):
            yield from walk(child, relative + '/' + child.name)
    elif stat.S_ISREG(metadata.st_mode):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        after = path.lstat()
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size,
                                 item.st_mtime_ns, item.st_ctime_ns, item.st_mode)
        assert identity(metadata) == identity(after), f'Changed during hashing: {path}'
        yield relative, dict(type='file', mode=mode, bytes=metadata.st_size,
                             sha256=digest.hexdigest())
    else:
        raise ValueError(f'Unsupported runtime member: {path}')


def manifest(args):
    started = time.monotonic()
    entries = {}
    last_progress = started
    for member in MEMBERS:
        for relative, metadata in walk(args.root / member, member):
            assert relative not in entries
            entries[relative] = metadata
            if time.monotonic() - last_progress >= 10:
                write(args.output / 'progress.json', dict(
                    pid=os.getpid(), entries=len(entries), last=relative,
                    updated_unix=time.time()))
                last_progress = time.monotonic()
    # Detect additions or removals during the traversal, without rehashing files.
    current = set()
    def names(path, relative):
        current.add(relative)
        if path.is_dir() and not path.is_symlink():
            for child in path.iterdir():
                names(child, relative + '/' + child.name)
    for member in MEMBERS:
        names(args.root / member, member)
    assert current == set(entries), 'Runtime membership changed during hashing'
    result = dict(status='complete', root=str(args.root), members=MEMBERS,
        host=socket.gethostname(), pid=os.getpid(), entries=entries,
        regular_files=sum(v['type'] == 'file' for v in entries.values()),
        regular_bytes=sum(v.get('bytes', 0) for v in entries.values()),
        seconds=time.monotonic() - started, ended_unix=time.time(),
        scope='All copied library members and three interpreter entries; symlink targets preserved; no symlink directory traversal; ownership and timestamps excluded from cross-host comparison')
    write(args.output / 'manifest.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'entries'}), flush=True)


def supervise(args):
    assert args.root.is_absolute() and args.output.is_absolute()
    assert '/outputs/pretrained/' in str(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
               '--root', str(args.root), '--output', str(args.output)]
    start = time.time()
    with (args.output / 'manifest.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write(args.output / 'process.json', dict(pid=os.getpid(), child_pid=child.pid,
              host=socket.gethostname(), command=command, started_unix=start))
        code = child.wait()
    write(args.output / 'exit.json', dict(exit_code=code, child_pid=child.pid,
          host=socket.gethostname(), wait_returned=True, ended_unix=time.time()))
    raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        manifest(arguments)
    else:
        supervise(arguments)

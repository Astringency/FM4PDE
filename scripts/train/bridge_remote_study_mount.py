"""Keep a task-scoped SSHFS connection between two authorized SSH hosts.

Both SSH authentications stay on the controller. The remote SSHFS client speaks
SFTP over stdin/stdout; no private key or persistent SSH configuration is copied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'mount_bridge.json'
    if state_path.exists():
        old = json.loads(state_path.read_text())
        cmdline = Path('/proc', str(old['pid']), 'cmdline')
        if cmdline.exists() and b'bridge_remote_study_mount' in cmdline.read_bytes():
            raise RuntimeError('The recorded mount bridge is still running')
    options = 'passive,idmap=user,dir_cache=no,attr_timeout=0,entry_timeout=0'
    if args.read_only:
        options += ',ro'
    client = shlex.join([args.sshfs, ':' + args.source_root, args.mountpoint,
                        '-f', '-o', options])
    common = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30',
              '-o', 'ServerAliveCountMax=3']
    commands = [common + ['-s', args.source_host, 'sftp'],
                common + [args.compute_host, 'exec ' + client]]
    sockets = socket.socketpair()
    children = []
    started = time.time()

    def record(state, **extra):
        value = dict(state=state, pid=os.getpid(), child_pids=[p.pid for p in children],
                     commands=commands, source_root=args.source_root, mountpoint=args.mountpoint,
                     started_unix=started, checked_unix=time.time(), **extra)
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(state_path)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (out / 'sftp_transport.log').open('ab', buffering=0) as source_log, \
                (out / 'sshfs_transport.log').open('ab', buffering=0) as compute_log:
            for command, stream, log in zip(commands, sockets, [source_log, compute_log]):
                children.append(subprocess.Popen(command, stdin=stream, stdout=stream, stderr=log))
            for stream in sockets:
                stream.close()
            while all(p.poll() is None for p in children):
                record('running')
                time.sleep(10)
            record('transport_exited', exit_codes=[p.poll() for p in children])
            raise RuntimeError('An SSH transport exited; inspect transport logs and the mount')
    except KeyboardInterrupt:
        record('controller_stopped')
    finally:
        for stream in sockets:
            stream.close()
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-host', default='server197')
    parser.add_argument('--compute-host', default='server216')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--mountpoint', required=True)
    parser.add_argument('--sshfs', required=True)
    parser.add_argument('--read-only', action='store_true', help='Mount runtime dependencies read-only')
    main(parser.parse_args())

"""Run one declared experiment stage and retain its log and exit status."""
import argparse
import datetime
import json
from pathlib import Path
import socket
import subprocess
import time


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('An executable command is required')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    start_path = args.status.with_suffix('.started.json')
    if args.status.exists() or start_path.exists() or args.log.exists():
        raise FileExistsError('Stage records already exist; use a new stage identifier')
    started = time.time()
    record = dict(command=command, host=socket.gethostname(),
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  code_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], text=True).strip())
    write_new(start_path, record)
    with args.log.open('x') as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            code = result.returncode
        except Exception as exc:
            log.write(repr(exc) + '\n')
            code = 127
    record.update(exit_code=code, elapsed_seconds=time.time()-started,
                  completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_new(args.status, record)
    raise SystemExit(code)


if __name__ == '__main__':
    main()

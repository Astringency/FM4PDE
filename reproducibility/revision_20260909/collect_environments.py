#!/usr/bin/env python3
"""Read installed versions on an existing experiment host; write records locally.

This does not install packages, change a remote environment, or initialize CUDA.
The original per-run environment records remain authoritative for each run.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

QUERY = r'''
import importlib.metadata as metadata
import ast, json, platform, socket, subprocess, sys
from pathlib import Path
torch_dist = metadata.distribution("torch")
version_source = Path(torch_dist.locate_file("torch/version.py")).read_text()
torch_build = {}
for node in ast.parse(version_source).body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {"__version__", "cuda", "git_version"}:
                try: torch_build[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError): pass
packages = sorted([{'name': d.metadata.get('Name', ''), 'version': d.version}
                   for d in metadata.distributions()], key=lambda d: (d['name'].lower(), d['version']))
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                      '--format=csv,noheader,nounits'], capture_output=True, text=True)
print(json.dumps(dict(hostname=socket.gethostname(), python_executable=sys.executable,
      python=sys.version, platform=platform.platform(), packages=packages,
      torch=torch_build.get("__version__", torch_dist.version),
      cuda_toolkit=torch_build.get("cuda"), torch_git_version=torch_build.get("git_version"),
      gpu_inventory=gpu.stdout.splitlines(), gpu_query_status=gpu.returncode,
      cuda_initialized=False, torch_imported=False)))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'environments')
    a = p.parse_args()
    if Path(a.name).name != a.name:
        p.error('--name must be a single filename component')
    dest = a.output / (a.name + '.json')
    if dest.exists():
        raise FileExistsError(f'Use a new observation name; retain existing record: {dest}')
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', a.host,
               shlex.join([a.python, '-c', QUERY])]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=55)
    record = json.loads(result.stdout)
    record.update(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  ssh_alias=a.host, query_sha256=hashlib.sha256(QUERY.encode()).hexdigest(),
                  scope='Read-only installed-package observation after launch; original run receipts govern historical settings')
    if record['cuda_initialized']:
        raise RuntimeError('Unexpected CUDA initialization during environment collection')
    a.output.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
    # No direct URLs, credentials, or site-package filesystem paths are exported.
    versions = a.output / (a.name + '.requirements.txt')
    with versions.open('x') as f:
        f.write('# Observed installed packages; see adjacent JSON for hardware and scope.\n')
        for item in record['packages']:
            f.write(item['name'] + '==' + item['version'] + '\n')
    print(json.dumps(dict(path=str(dest), packages=len(record['packages']),
                         torch=record['torch'], cuda=record['cuda_toolkit'])))


if __name__ == '__main__':
    main()

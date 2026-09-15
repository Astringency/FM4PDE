"""Run a declared disjoint part of the existing 20-input Burgers trace study."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--diffusion-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--partition', choices=['0', '1'], required=True)
    p.add_argument('--mode', choices=['pilot', 'run'], required=True)
    a = p.parse_args()
    root, code = a.root.resolve(), a.code.resolve()
    plan = json.loads(a.plan.read_text())
    assert sha(root/'manifest.json') == plan['manifest_sha256']
    manifest = json.loads((root/'manifest.json').read_text())
    original_ids = manifest['cells']['burger']['evaluation_ids']
    assert plan['partitions']['0']['sample_ids'] + plan['partitions']['1']['sample_ids'] == original_ids
    assert len(original_ids) == len(set(original_ids)) == 20
    item = plan['partitions'][a.partition]
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(item['gpu'])
    transition = json.loads((root/'queue_control/after_darcy.json').read_text())
    assert transition['next_pdes'] == ['helmholtz']
    assert transition['status'] in ['running_replacement_queue', 'complete']
    assert transition['sampler_exit_code'] == 0
    out = root/'queue_control'
    with (out/f'burger_{a.mode}_partition{a.partition}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = [('pilot',100),('pilot',1000)] if a.mode == 'pilot' else [('run',1000)]
        for mode, steps in stages:
            command = [sys.executable, '-u', str(code/'collect_trace.py'), '--root', str(root),
                       '--diffusion-root', str(a.diffusion_root.resolve()), '--pde', 'burger',
                       '--mode', mode, '--steps', str(steps)]
            if mode == 'run':
                for sid in item['sample_ids']:
                    command.extend(['--sample-id', str(sid)])
            name = f'burger_tail_{mode}_{steps}_partition{a.partition}'
            before = time.time()
            with (root/'logs'/f'{name}.log').open('a') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record = {'argv': command, 'started_unix': before, 'finished_unix': time.time(),
                      'exit_code': result.returncode, 'plan_sha256': sha(a.plan),
                      'script_sha256': sha(__file__), 'sample_ids': item['sample_ids'] if mode == 'run' else [manifest['cells']['burger']['pilot_id']]}
            (root/'logs'/f'{name}.stage.json').write_text(json.dumps(record, indent=2)+'\n')
            print(name, 'EXIT', result.returncode, flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

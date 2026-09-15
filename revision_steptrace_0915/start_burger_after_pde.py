"""Start one frozen Burgers partition after this card's current PDE has finished."""
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


def write(path, record):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def predecessor_complete(root, pde, ids):
    stage = root/'logs'/f'{pde}_run_1000.stage.json'
    if not stage.exists():
        return False
    result = json.loads(stage.read_text())
    assert result['exit_code'] == 0, result
    for method in ['FM4PDE', 'DiffusionPDE']:
        for sid in ids:
            stem = root/'traces'/pde/f'{method}_1000_{sid}'
            record = json.loads(stem.with_suffix('.json').read_text())
            assert record['status'] == 'complete'
            assert (record['pde'], record['method'], record['sample_id']) == (pde, method, sid)
            assert sha(stem.with_suffix('.csv')) == record['trace_sha256']
            assert sha(stem.with_suffix('.pt')) == record['prediction_sha256']
    return True


def gpu_is_idle(gpu, expected_uuid):
    actual = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid',
                                      '--format=csv,noheader'], text=True).strip()
    assert actual == expected_uuid, (actual, expected_uuid)
    running = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader'], text=True)
    return not any(line.split(',')[0].strip() == actual for line in running.splitlines())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--diffusion-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--partition', choices=['0', '1'], required=True)
    p.add_argument('--gpu-uuid', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    plan = json.loads(a.plan.read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert sha(root/'manifest.json') == plan['manifest_sha256']
    gpu = plan['partitions'][a.partition]['gpu']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
    predecessor = {'0': 'nsnonbounded', '1': 'helmholtz'}[a.partition]
    control = root/'queue_control'
    status = control/f'burger_driver_partition{a.partition}.json'
    record = {'status': 'waiting', 'partition': a.partition, 'gpu': gpu,
              'gpu_uuid': a.gpu_uuid, 'predecessor': predecessor,
              'sample_ids': plan['partitions'][a.partition]['sample_ids'],
              'plan_sha256': sha(a.plan), 'script_sha256': sha(__file__),
              'python': sys.executable, 'started_unix': time.time()}
    with (control/f'burger_driver_partition{a.partition}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                write(status, record)
                if predecessor_complete(root, predecessor, manifest['cells'][predecessor]['evaluation_ids']) and gpu_is_idle(gpu, a.gpu_uuid):
                    break
                time.sleep(30)
            wrapper = Path(__file__).with_name('run_burger_partition.py')
            common = [sys.executable, '-u', str(wrapper), '--root', str(root), '--code', str(a.code),
                      '--diffusion-root', str(a.diffusion_root), '--plan', str(a.plan), '--partition', a.partition]
            with (control/'burger_pilot_global.lock').open('a+') as pilot_lock:
                fcntl.flock(pilot_lock, fcntl.LOCK_EX)
                failure = control/'burger_pilot_failed.json'
                assert not failure.exists(), 'Earlier Burgers pilot failed; review before resuming'
                certificate = root/'pilots/burger/certificate_1000.json'
                if not certificate.exists():
                    record.update(status='pilot', pilot_started_unix=time.time());write(status, record)
                    result = subprocess.run(common + ['--mode', 'pilot'])
                    if result.returncode:
                        write(failure, {'exit_code': result.returncode, 'partition': a.partition})
                    assert result.returncode == 0, result.returncode
                for steps in [100, 1000]:
                    certificate = json.loads((root/f'pilots/burger/certificate_{steps}.json').read_text())
                    assert certificate['status'] == 'pass'
                    assert certificate['environment']['collector_sha256'] == sha(a.code/'collect_trace.py')
                    assert certificate['environment']['manifest_sha256'] == plan['manifest_sha256']
                    assert certificate['environment']['torch'] == plan['torch']
            assert gpu_is_idle(gpu, a.gpu_uuid), 'GPU acquired another process before production'
            record.update(status='running', production_started_unix=time.time());write(status, record)
            result = subprocess.run(common + ['--mode', 'run'])
            assert result.returncode == 0, result.returncode
            record.update(status='complete', finished_unix=time.time());write(status, record)
        except BaseException as exc:
            record.update(status='failed', error=repr(exc), finished_unix=time.time());write(status, record)
            raise


if __name__ == '__main__':
    main()

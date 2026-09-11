"""Run a queued main-model continuation on a second host, saving to server197."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts.train.resume_study import ROOT, write


def translated(path, source_root, target_root):
    return str(Path(target_root) / Path(path).relative_to(source_root))


def canonical_completion(value, mounted_root, canonical_root):
    result = dict(value)
    result['checkpoint_candidates'] = [translated(p, mounted_root, canonical_root)
                                       for p in value['checkpoint_candidates']]
    result['best_fm_checkpoint'] = translated(value['best_fm_checkpoint'], mounted_root, canonical_root)
    return result


def main(args):
    mounted = args.mounted_pretrained.resolve()
    canonical = args.canonical_pretrained
    study = mounted / args.study_name
    plan = json.loads((study / 'study_plan.json').read_text())
    job = next(row for row in plan['jobs'] if row['pde'] == args.pde)
    out = study / args.pde
    assert not (out / 'exit.json').exists()
    preflight = json.loads((study / 'remote_execution/data_identity_verification.json').read_text())
    assert preflight['status'] == 'verified' and args.pde in preflight['pdes']
    assert preflight['files_per_pde'] == 5
    assert os.path.ismount(mounted), 'The canonical result storage is not mounted'
    args.lock_directory.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    command = [sys.executable, '-u', '-m', 'scripts.train.resume_long_study',
               '--pde', args.pde, '--checkpoint', translated(job['source'], canonical, mounted),
               '--inference-checkpoint', translated(job['inference'], canonical, mounted),
               '--output', str(out), '--data-root', str(args.data_root),
               '--epochs', str(job['epochs']), '--lr', str(job['lr'])]
    with (args.lock_directory / f'gpu{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
                   PYTHONUNBUFFERED='1')
        execution = dict(host=host, gpu=args.gpu, canonical_pretrained=str(canonical),
                         mounted_pretrained=str(mounted), canonical_output=job['output'],
                         mounted_output=str(out), command=command, started_unix=time.time())
        write(out / 'remote_execution.json', execution)
        with (out / 'run.log').open('a') as log:
            log.write('REMOTE_DISPATCH ' + json.dumps(execution) + '\n')
            log.flush()
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(out / 'queue_state.json', dict(state='running_remote', pid=os.getpid(),
                  child_pid=child.pid, **execution))
            code = child.wait()
        if code == 0:
            done_path = out / 'training_complete.json'
            value = json.loads(done_path.read_text())
            assert value['completed_epochs'] == job['epochs'] == 50 and value['updates'] == 35200
            write(out / 'training_complete_execution_paths.json', value)
            write(done_path, canonical_completion(value, mounted, canonical))
        write(out / 'exit.json', dict(exit_code=code, child_pid=child.pid, host=host,
                                     wait_returned=True, ended_unix=time.time()))
        write(out / 'queue_state.json', dict(state='training_complete' if code == 0 else 'failed',
              host=host, gpu=args.gpu, child_pid=child.pid, exit_code=code))
        raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pde', required=True, choices=['poisson', 'darcy', 'burger'])
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--mounted-pretrained', type=Path, required=True)
    parser.add_argument('--canonical-pretrained', type=Path, required=True)
    parser.add_argument('--study-name', default='resume_main_20260911')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--lock-directory', type=Path, required=True)
    main(parser.parse_args())

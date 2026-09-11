"""Separate portable GPU evaluation from training-exit proof and CPU audit on 197."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def terminal_ready(terminal, expected_host):
    assert terminal['exit_code'] == 0, 'Training failed'
    assert terminal.get('host', expected_host) == expected_host
    if expected_host == socket.gethostname():
        return not Path('/proc', str(terminal['child_pid'])).exists()
    return terminal.get('host') == expected_host and terminal.get('wait_returned') is True


def verified_proof(out, dispatch_sha):
    path = out / 'portable_training_exit_proof.json'
    if not path.exists():
        return False
    proof = read(path)
    assert proof['training_reaped'] is True
    assert proof['dispatch_sha256'] == dispatch_sha
    assert proof['terminal_sha256'] == sha(out / 'exit.json')
    assert proof['training_complete_sha256'] == sha(out / 'training_complete.json')
    assert read(out / 'exit.json')['exit_code'] == 0
    done = read(out / 'training_complete.json')
    assert done['completed_epochs'] == 50 and done['updates'] == 35200
    return True


def relay(study, pde, dispatch, dispatch_sha):
    out = study / pde
    assert socket.gethostname() == dispatch['audit_host']
    while True:
        terminal = out / 'exit.json'
        if terminal.exists() and terminal_ready(read(terminal), dispatch['training_host']):
            done = read(out / 'training_complete.json')
            assert done['completed_epochs'] == 50 and done['updates'] == 35200
            write(out / 'portable_training_exit_proof.json', dict(
                training_reaped=True, training_host=dispatch['training_host'],
                checked_on=socket.gethostname(), checked_unix=time.time(),
                method='local PID absent after supervisor exit' if dispatch['training_host'] == socket.gethostname()
                       else 'remote supervisor returned from child.wait',
                terminal_sha256=sha(terminal),
                training_complete_sha256=sha(out / 'training_complete.json'),
                dispatch_sha256=dispatch_sha))
            break
        write(out / 'portable_relay_status.json', dict(state='waiting_for_training',
              pid=os.getpid(), host=socket.gethostname(), checked_unix=time.time()))
        time.sleep(30)
    while not (out / 'evaluation.exit.json').exists():
        write(out / 'portable_relay_status.json', dict(state='waiting_for_sampling',
              pid=os.getpid(), host=socket.gethostname(), checked_unix=time.time()))
        time.sleep(30)
    terminal = read(out / 'evaluation.exit.json')
    assert terminal['exit_code'] == 0 and terminal['wait_returned'] is True
    assert terminal['host'] == dispatch['sampling_host']
    assert terminal['dispatch_sha256'] == dispatch_sha
    assert read(out / 'evaluation/complete.json')['status'] == 'complete'
    with (out / 'portable_audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out / 'final_audit.exit.json').exists()
        command = [sys.executable, '-u', '-m', 'scripts.train.audit_long_resume',
                   '--study', str(study), '--pde', pde, '--require-evaluation']
        with (out / 'final_audit.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
                                              OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'))
            write(out / 'evaluation_queue.json', dict(state='auditing', pid=os.getpid(),
                  child_pid=child.pid, host=socket.gethostname(), command=command))
            code = child.wait()
        write(out / 'final_audit.exit.json', dict(exit_code=code, child_pid=child.pid,
              host=socket.gethostname(), wait_returned=True, ended_unix=time.time()))
        write(out / 'evaluation_queue.json', dict(state='complete' if code == 0 else 'failed',
              pid=os.getpid(), host=socket.gethostname(), exit_code=code))
        raise SystemExit(code)


def worker(study, pde, dispatch, dispatch_sha):
    out = study / pde
    assert socket.gethostname() == dispatch['sampling_host']
    process = dict(pid=os.getpid(), host=socket.gethostname(), pde=pde,
                   dispatch_sha256=dispatch_sha)
    while not verified_proof(out, dispatch_sha):
        # Fail instead of waiting forever if the training supervisor reported failure.
        if (out / 'exit.json').exists():
            assert read(out / 'exit.json')['exit_code'] == 0, 'Training failed'
        write(out / 'evaluation_queue.json', dict(**process, state='waiting_for_training_proof',
                                                  checked_unix=time.time()))
        time.sleep(30)
    assert read(study / 'sampling_smoke.exit.json')['exit_code'] == 0
    assert pde in read(study / 'evaluation_smoke/complete.json')['pdes']
    lock_directory = Path(dispatch['lock_directory'])
    assert lock_directory.is_absolute()
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock = None
    while lock is None:
        for gpu in dispatch['gpus']:
            candidate = (lock_directory / f'gpu{gpu}.lock').open('a')
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            lock = candidate
            break
        if lock is None:
            write(out / 'evaluation_queue.json', dict(**process, state='waiting_for_gpu',
                                                      checked_unix=time.time()))
            time.sleep(30)
    with lock:
        command = [sys.executable, '-u', '-m', 'scripts.train.evaluate_long_resume',
                   '--study', str(study), '--pde', pde,
                   '--memory-limit-gib', str(dispatch['memory_limit_gib']),
                   '--minimum-free-gib', str(dispatch['minimum_free_gib']),
                   '--maximum-batch', str(dispatch['maximum_batch'])]
        if dispatch['buffer_step_metrics']:
            command.append('--buffer-step-metrics')
        with (out / 'evaluation.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                         OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4'))
            write(out / 'evaluation_queue.json', dict(**process, state='running', gpu=gpu,
                  child_pid=child.pid, command=command))
            code = child.wait()
        # Publish this state before the exit receipt that wakes the CPU auditor.
        write(out / 'evaluation_queue.json', dict(**process,
              state='waiting_for_audit' if code == 0 else 'failed', exit_code=code, gpu=gpu))
        write(out / 'evaluation.exit.json', dict(exit_code=code, child_pid=child.pid,
              host=socket.gethostname(), wait_returned=True, gpu=gpu,
              dispatch_sha256=dispatch_sha, ended_unix=time.time()))
        raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['worker', 'relay'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--pde', required=True)
    args = parser.parse_args()
    assert args.study.is_absolute() and '/outputs/pretrained/' in str(args.study)
    path = args.study / args.pde / 'portable_evaluation_dispatch.json'
    dispatch = read(path)
    assert dispatch['pde'] == args.pde and dispatch['previous_worker_reaped'] is True
    assert dispatch['evaluation_had_not_started'] is True
    assert dispatch['gpus'] and set(dispatch['gpus']) <= {0, 1}
    assert 2 < dispatch['memory_limit_gib'] < dispatch['minimum_free_gib']
    assert dispatch['maximum_batch'] in [1, 2, 4, 8, 16]
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert actual == dispatch[args.mode + '_git_commit']
    (worker if args.mode == 'worker' else relay)(args.study, args.pde, dispatch, sha(path))

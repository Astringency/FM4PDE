"""Start one visible tmux session per PDE, with exclusive per-GPU queues."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'poisson': 'poisson/260621-104344-poisson-batch4-epoch300-accum8-float32/fm4poisson-checkpoint.pth',
    'darcy': 'darcy/260625-150746-darcy-batch4-epoch300-accum8-float32/fm4darcy-checkpoint.pth',
    'helmholtz': 'helmholtz/260625-150653-helmholtz-batch4-epoch300-accum8-float32/fm4helmholtz-checkpoint.pth',
    'nsnonbounded': 'nsnonbounded/260712-121023-nsnonbounded-batch4-epoch300-accum8-float32/fm4nsnonbounded-checkpoint.pth',
    'burger': 'burger/260625-150439-burger-batch4-epoch300-accum8-float32/fm4burger-checkpoint.pth',
}
ASSIGNMENTS = [('poisson', 0, 90), ('nsnonbounded', 1, 200),
               ('darcy', 0, 100), ('helmholtz', 0, 100), ('burger', 1, 90)]


def write(path, x):
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(x, indent=2) + '\n')
    temporary.replace(path)


def init(study):
    assert '/outputs/pretrained/' in str(study)
    study.mkdir(parents=True, exist_ok=True)
    assert not (study / 'study_plan.json').exists(), 'Study already initialized'
    jobs = []
    for pde, gpu, minutes in ASSIGNMENTS:
        checkpoint = ROOT / 'outputs/pretrained/formal' / SOURCES[pde]
        assert checkpoint.is_file()
        jobs.append(dict(pde=pde, gpu=gpu, minutes=minutes, checkpoint=str(checkpoint),
                         output=str(study/pde), session='fm_resume_0910_'+pde))
    write(study/'study_plan.json', dict(jobs=jobs, created_unix=time.time(),
          goal='Resume five existing checkpoints and compare precision within approximately eight hours',
          training_budget_minutes_per_gpu=290, downstream_evaluation_required=True,
          checkpoints_selected='original clean formal runs; exclude Poisson Aug31 trained on test files'))
    for job in jobs:
        Path(job['output']).mkdir()
        command = shlex.join([sys.executable, '-u', '-m', 'scripts.train.launch_resume_study',
                             'worker', '--study', str(study), '--pde', job['pde']])
        subprocess.run(['tmux', 'new-session', '-d', '-s', job['session'], '-c', str(ROOT), command], check=True)
        print('LAUNCHED', job['pde'], job['session'], 'GPU', job['gpu'], flush=True)


def worker(study, pde):
    plan = json.loads((study/'study_plan.json').read_text())
    jobs = plan['jobs']
    job = next(j for j in jobs if j['pde'] == pde)
    out = Path(job['output'])
    write(out/'queue_state.json', dict(state='waiting_for_gpu', pid=os.getpid(), gpu=job['gpu']))
    lock_directory = Path(plan.get('gpu_lock_directory', study)).resolve()
    assert '/outputs/pretrained/' in str(lock_directory)
    with (lock_directory / ('gpu' + str(job['gpu']) + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Acquisition is exclusive for this study; no other user's process is stopped.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
                   PYTHONUNBUFFERED='1')
        command = [sys.executable, '-u', '-m', 'scripts.train.resume_study', '--pde', pde,
                   '--checkpoint', job['checkpoint'], '--output', str(out),
                   '--minutes', str(job['minutes'])]
        command.extend(job.get('extra_training_args', []))
        begin = time.monotonic()
        with (out/'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(out/'queue_state.json', dict(state='running', pid=os.getpid(), child_pid=process.pid,
                  gpu=job['gpu'], started_unix=time.time(), command=command))
            code = process.wait()
        write(out/'exit.json', dict(exit_code=code, seconds=time.monotonic()-begin,
                                   child_pid=process.pid, ended_unix=time.time()))
        write(out/'queue_state.json', dict(state='complete' if code==0 else 'failed',
              pid=os.getpid(), child_pid=process.pid, gpu=job['gpu'], exit_code=code))
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['init','worker'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--pde', choices=SOURCES)
    args = parser.parse_args()
    if args.mode == 'init':
        init(args.study.resolve())
    else:
        worker(args.study.resolve(), args.pde)

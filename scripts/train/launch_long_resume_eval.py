"""Queue independent evaluation workers after their training exits successfully."""
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

from scripts.train.resume_study import ROOT, write


def worker(study,pde):
    out=study/pde
    process=dict(pid=os.getpid(),pde=pde,started_unix=time.time())
    while True:
        exit_path=out/'exit.json'
        if exit_path.exists():
            terminal=json.loads(exit_path.read_text())
            if terminal['exit_code']!=0:
                write(out/'evaluation_queue.json',dict(**process,state='training_failed',training_exit=terminal))
                raise SystemExit(1)
            if not Path('/proc',str(terminal['child_pid'])).exists():
                assert (out/'training_complete.json').is_file()
                break
        write(out/'evaluation_queue.json',dict(**process,state='waiting_for_training',checked_unix=time.time()))
        time.sleep(30)
    lock=None;gpu=None
    while lock is None:
        for number in [0,1]:
            candidate=(study/f'gpu{number}.lock').open('a')
            try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:candidate.close();continue
            lock,gpu=candidate,number;break
        if lock is None:
            write(out/'evaluation_queue.json',dict(**process,state='waiting_for_gpu',checked_unix=time.time()))
            time.sleep(30)
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                 MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        command=[sys.executable,'-u','-m','scripts.train.evaluate_long_resume','--study',str(study),'--pde',pde]
        with (out/'evaluation.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(out/'evaluation_queue.json',dict(**process,state='running',gpu=gpu,child_pid=child.pid,command=command))
            code=child.wait()
        write(out/'evaluation.exit.json',dict(exit_code=code,child_pid=child.pid,gpu=gpu,ended_unix=time.time()))
        lock.close();lock=None
        if code==0:
            audit_command=[sys.executable,'-u','-m','scripts.train.audit_long_resume','--study',str(study),
                           '--pde',pde,'--require-evaluation']
            with (out/'final_audit.log').open('a') as log:
                child=subprocess.Popen(audit_command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),
                                       stdout=log,stderr=subprocess.STDOUT)
                write(out/'evaluation_queue.json',dict(**process,state='auditing',child_pid=child.pid,command=audit_command))
                code=child.wait()
            write(out/'final_audit.exit.json',dict(exit_code=code,child_pid=child.pid,ended_unix=time.time()))
        write(out/'evaluation_queue.json',dict(**process,state='complete' if code==0 else 'failed',gpu=gpu,exit_code=code))
        raise SystemExit(code)
    finally:
        if lock is not None:lock.close()


def init(study):
    assert study.is_absolute() and '/outputs/pretrained/' in str(study)
    jobs=json.loads((study/'study_plan.json').read_text())['jobs']
    sessions=set(subprocess.check_output(['tmux','list-sessions','-F','#{session_name}'],text=True).splitlines())
    for job in jobs:
        session='fm_long_eval_0911_'+job['pde']
        assert session not in sessions, f'Worker already exists: {session}'
        assert not (study/job['pde']/'evaluation/complete.json').exists()
        command=shlex.join([sys.executable,'-u','-m','scripts.train.launch_long_resume_eval','worker',
                           '--study',str(study),'--pde',job['pde']])
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),command],check=True)
        print('QUEUED_EVALUATION',job['pde'],session,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['init','worker']);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--pde');args=p.parse_args()
    if args.mode=='init':init(args.study.resolve())
    else:worker(args.study.resolve(),args.pde)

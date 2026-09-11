"""One visible tmux worker per PDE; two exclusive GPU training queues."""
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

from scripts.train.launch_resume_study import ROOT, SOURCES
from scripts.train.resume_study import write

NS_MAIN = ROOT/'outputs/main/revision_20260909/ns_main_revision_0909/local/complete_local_ns_study'


def init(study):
    assert study.is_absolute() and '/outputs/pretrained/' in str(study)
    study.mkdir(parents=True,exist_ok=True)
    assert not (study/'study_plan.json').exists()
    sources=dict(SOURCES)
    sources['nsnonbounded']='nsnonbounded/260904-170053-nsnonbounded-batch32-epoch300-accum1-bfloat16/fm4nsnonbounded-checkpoint.pth'
    jobs=[]
    for pde,gpu in [('nsnonbounded',0),('poisson',1),('darcy',0),('helmholtz',1),('burger',0)]:
        source=ROOT/'outputs/pretrained/formal'/sources[pde]
        inference=NS_MAIN/'inputs/weights.pth' if pde=='nsnonbounded' else source.with_name('fm4'+pde+'.pth')
        assert source.is_file() and inference.is_file()
        jobs.append(dict(pde=pde,gpu=gpu,source=str(source),inference=str(inference),
            lr=1e-5 if pde=='nsnonbounded' else 3e-6,epochs=50,
            session='fm_long_0911_'+pde,output=str(study/pde)))
    resources={name:subprocess.check_output(command,text=True) for name,command in {
        'gpu':['nvidia-smi'],'memory':['free','-h'],'disk':['df','-h',str(ROOT)],
        'load':['uptime'],'existing_sessions':['tmux','list-sessions']}.items()}
    write(study/'resource_preflight.json',dict(unix=time.time(),**resources))
    write(study/'study_plan.json',dict(jobs=jobs,created_unix=time.time(),
        goal='50 additional epochs for the five exact main-experiment checkpoints; final evaluation of every main cell on 1000 inputs',
        final_evaluation=dict(samples_per_cell=1000,distributions=['id','rough','smooth'],
            cells=66,steps=100,source='frozen main_hyperparameters_verified.csv and archived effective configurations',
            scope='P/NS/D/H full forward/inverse, sparse forward/inverse/joint; Burgers random and sensor-column',
            checkpoint_selection='32 training-validation inputs, disjoint from continuation updates',
            hard_diagnostics='25 largest baseline errors in each archived main cell, never used for model selection'),
        precision='FP32 with TF32',effective_batch=64,save_every=5))
    for job in jobs:
        Path(job['output']).mkdir()
        command=shlex.join([sys.executable,'-u','-m','scripts.train.launch_long_resume','worker',
                           '--study',str(study),'--pde',job['pde']])
        subprocess.run(['tmux','new-session','-d','-s',job['session'],'-c',str(ROOT),command],check=True)
        print('LAUNCHED',job['pde'],job['session'],flush=True)


def worker(study,pde):
    plan=json.loads((study/'study_plan.json').read_text())
    job=next(j for j in plan['jobs'] if j['pde']==pde)
    out=Path(job['output'])
    write(out/'queue_state.json',dict(state='waiting_for_gpu',pid=os.getpid(),gpu=job['gpu']))
    with (study/f"gpu{job['gpu']}.lock").open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        command=[sys.executable,'-u','-m','scripts.train.resume_long_study','--pde',pde,
            '--checkpoint',job['source'],'--inference-checkpoint',job['inference'],
            '--output',str(out),'--epochs',str(job['epochs']),'--lr',str(job['lr'])]
        with (out/'run.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(out/'queue_state.json',dict(state='running',pid=os.getpid(),child_pid=process.pid,
                gpu=job['gpu'],command=command,started_unix=time.time()))
            code=process.wait()
        write(out/'exit.json',dict(exit_code=code,child_pid=process.pid,ended_unix=time.time()))
        write(out/'queue_state.json',dict(state='training_complete' if code==0 else 'failed',
            gpu=job['gpu'],exit_code=code,child_pid=process.pid))
        raise SystemExit(code)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['init','worker'])
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--pde')
    a=p.parse_args()
    (init(a.study.resolve()) if a.mode=='init' else worker(a.study.resolve(),a.pde))

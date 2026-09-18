"""Tmux workers claim distinct cells using atomic directories on shared storage."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from experiments.rough_stress.run import METHODS, TASKS, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--input-root")
    p.add_argument("--baseline-root")
    p.add_argument("--method-group",choices=["baseline","fm"],default="baseline")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--min-free-mib",type=int,default=10000)
    p.add_argument("--fm-protocol")
    p.add_argument("--fm-checkpoint")
    p.add_argument("--noise-root")
    args=p.parse_args()
    root=Path(args.root).resolve()
    input_root=Path(args.input_root or args.root).resolve()
    state=root/"queue_state"; state.mkdir(parents=True,exist_ok=True)
    logs=root/"logs"; logs.mkdir(exist_ok=True)
    methods=METHODS if args.method_group=="baseline" else ("fm4pde",)
    cells=[dict(method=m,task=t,distribution=d,num_obs=n)
           for d,n in [("rough2",500),("rough3",500),("id",500),("smooth",500),("rough",500),("id",400),("rough",400)]
           for t in TASKS for m in methods]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    worker=f"{socket.gethostname()}-gpu{args.gpu}-{os.getpid()}"
    for cell in cells:
        key=f"{cell['method']}_{cell['task']}_{cell['distribution']}_obs{cell['num_obs']}"
        job=state/f"{key}.json"
        if job.exists():
            continue
        # mkdir is atomic at the storage server, including through SSHFS;
        # a local advisory lock need not coordinate two different hosts.
        try:
            (state/f"{key}.claim").mkdir()
        except FileExistsError:
            continue
        if job.exists():
            continue
        write_json(job,dict(status="claimed",worker=worker,cell=cell,claimed_at=time.time()))
        while True:
            free=int(subprocess.check_output(["nvidia-smi",f"--id={args.gpu}","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
            if free>=args.min_free_mib:
                break
            print(f"WAIT GPU {args.gpu}: {free} MiB free; need {args.min_free_mib}",flush=True)
            time.sleep(30)
        command=[sys.executable,"-u","-m","experiments.rough_stress.run","run",
            "--output-root",str(root),"--input-root",str(input_root),"--pde","poisson",
            "--batch-size",str(args.batch_size),"--device","cuda:0"]
        for key2,value in cell.items():
            command += ["--"+key2.replace("_","-"),str(value)]
        for option in ("baseline_root","fm_protocol","fm_checkpoint","noise_root"):
            if getattr(args,option):
                command += ["--"+option.replace("_","-"),getattr(args,option)]
        with (logs/f"{key}.log").open("a",buffering=1) as log:
            log.write(json.dumps(dict(command=command,worker=worker))+"\n")
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            write_json(job,dict(status="running",worker=worker,child_pid=process.pid,cell=cell,
                                command=command,started_at=time.time()))
            code=process.wait()
        write_json(job,dict(status="complete" if code==0 else "failed",worker=worker,
            cell=cell,command=command,exit_code=code,finished_at=time.time()))
        print(f"CELL EXIT {key} = {code}",flush=True)
        if code:
            raise SystemExit(code)
    print("WORKER COMPLETE",flush=True)


if __name__=="__main__":
    main()

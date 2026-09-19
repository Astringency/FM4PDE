"""Durable per-GPU queue with resource checks and authoritative child exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.optimizer_diagnostics.study import write


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["prepare","run","inspect"])
    p.add_argument("--pdes",nargs="+",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--inputs")
    p.add_argument("--pretrained-root")
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--minimum-free-mib",type=int,default=42000)
    p.add_argument("--max-memory-gib",type=float,default=26)
    p.add_argument("--steps",type=int,default=128)
    args=p.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",NUMPY_MADVISE_HUGEPAGE="0",
        CUDA_VISIBLE_DEVICES=str(args.gpu))
    queue_key=f"queue_{args.mode}_gpu{args.gpu}"
    for name in args.pdes:
        target=out/name
        target.mkdir(parents=True,exist_ok=True)
        if args.mode in ("run","inspect"):
            while True:
                gpu=subprocess.check_output(["nvidia-smi",f"--id={args.gpu}",
                    "--query-gpu=memory.free,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
                free=int(gpu.split(",")[0])
                write(out/f"{queue_key}.json",dict(state="resource_check",pde=name,gpu=args.gpu,
                    memory_and_utilization=gpu,pid=os.getpid(),time=time.time()))
                if free>=args.minimum_free_mib:
                    break
                time.sleep(30)
        command=[sys.executable,"-u","-m","experiments.optimizer_diagnostics.study",args.mode,
            "--pde",name,"--output",str(out),"--steps",str(args.steps),"--max-memory-gib",str(args.max_memory_gib)]
        if args.inputs:
            command.extend(["--inputs",args.inputs])
        if args.pretrained_root:
            command.extend(["--pretrained-root",args.pretrained_root])
        with (target/f"{args.mode}.log").open("a") as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(out/f"{queue_key}.json",dict(state="running",pde=name,gpu=args.gpu,
                pid=os.getpid(),child_pid=child.pid,command=command,time=time.time()))
            code=child.wait()
        write(target/f"{args.mode}.exit.json",dict(exit_code=code,child_pid=child.pid,command=command,time=time.time()))
        if code:
            write(out/f"{queue_key}.json",dict(state="failed",pde=name,exit_code=code,time=time.time()))
            raise SystemExit(code)
    write(out/f"{queue_key}.json",dict(state="complete",pdes=args.pdes,time=time.time()))


if __name__=="__main__":
    main()

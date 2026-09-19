"""Inspect, audit and summarize each published run after its training exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.optimizer_diagnostics.study import PDES, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    args=parser.parse_args()
    root=args.root.resolve()
    env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
    pending=list(PDES)
    while pending:
        for pde in list(pending):
            run=root/"runs"/pde
            exit_path=run/"run.exit.json"
            if not exit_path.exists():
                continue
            status=json.loads(exit_path.read_text())
            if status["exit_code"]:
                raise RuntimeError(f"Training failed for {pde}: {status}")
            if not (run/"inspect.exit.json").exists():
                subprocess.run([sys.executable,"-u","-m","experiments.optimizer_diagnostics.queue","inspect",
                    "--pdes",pde,"--gpu",str(args.gpu),"--inputs",str(root/"inputs"),
                    "--output",str(root/"runs"),"--max-memory-gib","26","--minimum-free-mib","42000"],
                    env=env,check=True)
            assert json.loads((run/"inspect.exit.json").read_text())["exit_code"]==0
            if not (run/"audit.json").exists():
                subprocess.run([sys.executable,"-u","-m","experiments.optimizer_diagnostics.audit",
                    "--root",str(root),"--pdes",pde],env=env,check=True)
            subprocess.run([sys.executable,"-u","-m","experiments.optimizer_diagnostics.report",
                "--root",str(root)],env=env,check=True)
            pending.remove(pde)
            write(root/"finalization.json",dict(state="running" if pending else "complete",pending=pending,time=time.time()))
        if pending:
            time.sleep(30)


if __name__=="__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise

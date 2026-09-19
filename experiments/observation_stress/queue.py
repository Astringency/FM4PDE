"""Run a frozen experiment manifest, with atomic claims across GPU workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from experiments.rough_stress.run import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--group", choices=["baseline", "fm"], required=True)
    p.add_argument("--min-free-mib", type=int, default=18000)
    args = p.parse_args()
    root = Path(args.root)
    state = root / "queue_state"
    logs = root / "logs"
    state.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    jobs = [job for job in manifest["jobs"] if job["group"] == args.group]
    worker = f"{socket.gethostname()}-gpu{args.gpu}-{os.getpid()}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    for job in jobs:
        key = job["key"]
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in key):
            raise ValueError(f"Unsafe job key: {key}")
        record = state / f"{key}.json"
        if record.exists():
            continue
        try:
            (state / f"{key}.claim").mkdir()
        except FileExistsError:
            continue
        write_json(record, dict(status="claimed", worker=worker, job=job, claimed_at=time.time()))
        free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        while free < args.min_free_mib:
            print(f"WAIT GPU {args.gpu}: {free} MiB free", flush=True)
            time.sleep(30)
            free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        command = [sys.executable, "-u", "-m", job["module"], *job["argv"]]
        with (logs / f"{key}.log").open("a", buffering=1) as log:
            log.write(json.dumps(dict(command=command, worker=worker)) + "\n")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            write_json(record, dict(status="running", worker=worker, child_pid=process.pid,
                job=job, command=command, started_at=time.time()))
            code = process.wait()
        write_json(record, dict(status="complete" if code == 0 else "failed", worker=worker,
            job=job, command=command, exit_code=code, finished_at=time.time()))
        print(f"CELL EXIT {key} = {code}", flush=True)
        if code:
            raise SystemExit(code)
    print("WORKER COMPLETE", flush=True)


if __name__ == "__main__":
    main()

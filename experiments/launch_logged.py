"""Run one independent tmux worker, retaining logs and its actual exit status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command
    if command[0] == "--":
        command = command[1:]
    dest = args.output / "execution"
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONUNBUFFERED="1")
    receipt = dict(command=command, gpu=args.gpu, launcher_pid=os.getpid(), start_unix=time.time())
    path = dest / (args.name + ".json")
    with (dest / (args.name + ".log")).open("w") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["pid"] = child.pid
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["exit_code"] = child.wait()
    receipt["end_unix"] = time.time()
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(receipt["exit_code"])


if __name__ == "__main__":
    main()

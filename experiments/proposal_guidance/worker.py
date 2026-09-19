"""Run each task in an externally named tmux session and persist exit receipts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", default=["pilot", "run"])
    args = parser.parse_args()
    logs = args.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for mode in args.modes:
            command = [sys.executable, "-u", "-m", "experiments.proposal_guidance.study", mode,
                       "--root", str(args.root), "--task", task]
            started = time.time()
            with (logs / f"{task}_{mode}.log").open("a") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            (logs / f"{task}_{mode}.exit.json").write_text(json.dumps(dict(
                argv=command, started_unix=started, finished_unix=time.time(), exit_code=result.returncode), indent=2) + "\n")
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

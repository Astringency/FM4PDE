"""Use the inverse worker's GPU after it exits, preserving logs and exit codes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    logs = args.root / "logs"
    receipt = logs / "sparse_inverse_run.exit.json"
    while not receipt.exists():
        time.sleep(10)
    assert json.loads(receipt.read_text())["exit_code"] == 0
    # No second inference process shares this GPU. Recheck available memory.
    import torch
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    assert free > 12 * 2**30, (free, total)
    jobs = [(f"diagnose_{task}", "diagnose", ["--task", task])
            for task in ["sparse_forward", "sparse_inverse", "sparse_joint"]]
    jobs.append(("budget_control", "budget_control", []))
    for name, module, extra in jobs:
        command = [sys.executable, "-u", "-m", f"experiments.proposal_guidance.{module}",
                   "--root", str(args.root), *extra]
        started = time.time()
        with (logs / f"{name}.log").open("a") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        (logs / f"{name}.exit.json").write_text(json.dumps(dict(argv=command,
            started_unix=started, finished_unix=time.time(), exit_code=result.returncode), indent=2)+"\n")
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

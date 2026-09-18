"""Wait for all independent workers, then verify the complete 84-cell study."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.rough_stress.audit import SETTINGS, TASK_FIELDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--fm-root", required=True)
    args = p.parse_args()
    last = None
    while True:
        counts = dict(complete=0, running=0, claimed=0, unclaimed=0)
        for method in ("recfno", "senseiver", "voronoicnn", "fm4pde"):
            root = Path(args.fm_root if method == "fm4pde" else args.baseline_root)
            for dist, n in SETTINGS:
                for task in TASK_FIELDS:
                    job = root/"queue_state"/f"{method}_{task}_{dist}_obs{n}.json"
                    record = json.loads(job.read_text()) if job.exists() else {"status":"unclaimed"}
                    status = record["status"]
                    if status not in counts:
                        raise RuntimeError(f"Worker failed or invalid status: {job}: {record}")
                    counts[status] += 1
        if counts != last:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), counts, flush=True)
            last = counts
        if counts["complete"] == 84:
            break
        time.sleep(30)
    root = Path(args.fm_root)
    source = json.loads((root/"audits/source_verification.json").read_text())
    if source["status"] != "complete" or len(source["distributions"]) != 5:
        raise RuntimeError("Original source verification is not complete")
    subprocess.run([sys.executable, "-u", "-m", "experiments.rough_stress.audit",
        "--baseline-root", args.baseline_root, "--fm-root", args.fm_root,
        "--output-root", str(root/"audits/final")], check=True)
    print("ALL 84 CELLS AND SOURCE DATA VERIFIED", flush=True)


if __name__ == "__main__":
    main()

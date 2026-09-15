"""Start a second stock NS worker after its owned Poisson worker releases a GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision_ns44m_0915.run_ablations import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release-record", type=Path, required=True)
    p.add_argument("--released-pid", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--worker", required=True)
    args = p.parse_args()
    root = args.output
    state = root/"workers"/("launch_"+args.worker+".json")
    try:
        deadline = time.monotonic()+3600
        while not args.release_record.exists():
            os.kill(args.released_pid, 0)
            assert time.monotonic() < deadline
            atomic_json(state, dict(status="running", stage="waiting_for_owned_worker_release", pid=os.getpid()))
            time.sleep(1)
        record = json.loads(args.release_record.read_text())
        assert record["pid"] == args.released_pid and record["status"] == "released_after_complete_pool"
        for _ in range(120):
            if not Path(f"/proc/{args.released_pid}").exists():break
            time.sleep(.25)
        assert not Path(f"/proc/{args.released_pid}").exists()
        free = int(subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        assert free > 32768, ("Insufficient free memory for the measured B1/B4 worker", free)
        cert = root/"pilot_v2/pilot_complete.json"
        assert json.loads(cert.read_text())["status"] == "pass"
        argv = [sys.executable, str(ROOT/"revision_ns44m_0915/run_ablations.py"), "run",
            "--protocol", str(root/"protocol.json"), "--output", str(root),
            "--checkpoint", str(root/"inputs/weights/nsnonbounded.pth"),
            "--worker", args.worker, "--pilot-certificate", str(cert)]
        atomic_json(root/"provenance"/("worker_"+args.worker+"_launch.json"), dict(argv=argv,
            gpu=args.gpu, free_mib=free, release_record=record, started_unix=time.time()))
        atomic_json(state, dict(status="running", stage="main", pid=os.getpid(), argv=argv))
        with (root/"logs"/(args.worker+"_main.log")).open("w") as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        (root/"logs"/(args.worker+"_main.exit")).write_text(str(result.returncode)+"\n")
        assert result.returncode == 0
        atomic_json(state, dict(status="complete", pid=os.getpid(), finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(state, dict(status="error", pid=os.getpid(), error=repr(exc), failed_unix=time.time()))
        raise


if __name__ == "__main__":
    main()

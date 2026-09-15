"""Finish one owned Poisson pool, then run the prior and NS stages on that GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision_ns44m_0915.run_ablations import PILOT_JOB_IDS, atomic_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--handoff-pid", type=int, required=True)
    p.add_argument("--handoff-completion", type=Path, required=True)
    p.add_argument("--prior-checkpoint", type=Path, required=True)
    p.add_argument("--prior-output", type=Path, required=True)
    p.add_argument("--worker", default="g7")
    p.add_argument("--pilot-name", default="pilot")
    args = p.parse_args()
    root = args.output.resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state = root / "workers" / ("chain_"+args.worker+".json")
    try:
        deadline = time.monotonic()+3600
        while not args.handoff_completion.exists():
            os.kill(args.handoff_pid, 0)
            assert time.monotonic() < deadline, "The expected complete Poisson pool did not arrive"
            atomic_json(state, dict(status="running", stage="waiting_for_complete_poisson_pool", pid=os.getpid()))
            time.sleep(.25)
        if Path(f"/proc/{args.handoff_pid}/cmdline").exists():
            cmdline = Path(f"/proc/{args.handoff_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            assert "run_conditional_sample_scaling.py" in cmdline
            assert "conditional_fixed_observations_20260915" in cmdline and "--worker "+args.worker in cmdline
            record = dict(completed_pool=str(args.handoff_completion), completed_pool_sha256=file_hash(args.handoff_completion),
                          terminated_own_pid=args.handoff_pid, cmdline=cmdline, unix=time.time(),
                          policy="Completed pool retained; any later partial batch remains resumable by the other fixed-observation workers")
            os.kill(args.handoff_pid, signal.SIGTERM)
        else:
            record = json.loads((root/"provenance/gpu_handoff.json").read_text())
            assert record["terminated_own_pid"] == args.handoff_pid
            assert record["completed_pool_sha256"] == file_hash(args.handoff_completion)
        for _ in range(120):
            if not Path(f"/proc/{args.handoff_pid}").exists():
                break
            time.sleep(.25)
        assert not Path(f"/proc/{args.handoff_pid}").exists()
        record["gpu_processes_after_handoff"] = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True)
        if not (root/"provenance/gpu_handoff.json").exists():
            atomic_json(root/"provenance/gpu_handoff.json", record)
        commands = []
        def run(label, argv):
            atomic_json(state, dict(status="running", stage=label, pid=os.getpid(), argv=argv))
            start = time.time()
            with (logs/(label+".log")).open("w") as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
            (logs/(label+".exit")).write_text(str(result.returncode)+"\n")
            commands.append(dict(label=label, argv=argv, seconds=time.time()-start, returncode=result.returncode))
            atomic_json(root/"provenance/chain_commands.json", commands)
            if result.returncode:
                raise RuntimeError(f"{label} failed: {result.returncode}")
        import torch
        free, _ = torch.cuda.mem_get_info()
        assert free > 32*2**30, ("Insufficient memory after handoff", free)
        prior = [sys.executable, str(ROOT/"revision_ns44m_0915/run_poisson_prior.py"),
            "--checkpoint", str(args.prior_checkpoint), "--checkpoint-sha256",
            "d17e9a9e755769a51bec04a2ac079b6ca68f7ee5c95c244d7a4ac7730ff924b8",
            "--fm-code", str(ROOT), "--output", str(args.prior_output), "--seed", "20260911"]
        if (args.prior_output/"profile.json").exists():
            old_prior = json.loads((args.prior_output/"profile.json").read_text())
            assert old_prior["status"] == "complete" and old_prior["seed"] == 20260911
            assert old_prior["sample_sha256"] == file_hash(args.prior_output/"sample.pt")
        else:
            run("poisson_prior", prior)
        common = [sys.executable, str(ROOT/"revision_ns44m_0915/run_ablations.py")]
        flags = ["--protocol", str(root/"protocol.json"), "--checkpoint", str(root/"inputs/weights/nsnonbounded.pth"),
                 "--worker", args.worker]
        pilotroot = root/args.pilot_name
        run(args.pilot_name, common+["pilot", "--output", str(pilotroot)]+flags+["--job-ids"]+sorted(PILOT_JOB_IDS))
        run("main", common+["run", "--output", str(root)]+flags+
            ["--pilot-certificate", str(pilotroot/"pilot_complete.json")])
        run("weight_study", [sys.executable, str(ROOT/"revision_ns44m_0915/run_weight_selection.py"), "run",
            "--output", str(root), "--inventory", str(root/"provenance/ns_scope_inventory.json"),
            "--checkpoint", str(root/"inputs/weights/nsnonbounded.pth"),
            "--pilot-certificate", str(pilotroot/"pilot_complete.json"), "--worker", args.worker])
        atomic_json(state, dict(status="complete", pid=os.getpid(), finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(state, dict(status="error", error=repr(exc), pid=os.getpid(), failed_unix=time.time()))
        raise


if __name__ == "__main__":
    main()

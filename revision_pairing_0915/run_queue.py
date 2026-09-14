"""Static, inspectable GPU queues for the 57,000-prediction paired rerun.

No SSH, tmux creation, automatic migration, tuning, or inference implementation is
included.  The operator starts one ``worker`` per GPU in an existing tmux session.
Each worker calls run_stage.py around the unchanged run_inference.py command.
To revise assignments, stop the old workers, collect a stopped snapshot from each
server, and use ``revise``; started slices cannot change owner or job ID.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

TASK = "baseline_pairing_20260915_57k"
SERVERS = {"server197": [0, 1], "server216": list(range(8))}
PDE_GPUS = {"poisson": [0, 1], "burger": [0, 1],
            "darcy": [2, 3, 4], "helmholtz": [5, 6, 7]}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def jhash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fhash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def write_state(path, value):
    """Replace this task's mutable progress record, never an inference result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    write_new(tmp, value)
    tmp.replace(path)


def checked_document(path):
    doc = read(path)
    content = dict(doc)
    expected = content.pop("content_sha256")
    if jhash(content) != expected:
        raise ValueError(f"Document content hash mismatch: {path}")
    return doc


def finish_document(path, doc):
    doc = dict(doc)
    doc["content_sha256"] = jhash(doc)
    if Path(path).exists():
        if read(path) != doc:
            raise FileExistsError(f"Refusing to replace a different immutable document: {path}")
    else:
        write_new(path, doc)
    return doc


def worker_id(server, gpu):
    return f"{server}-gpu{gpu}"


def job_id(job):
    fields = ["prod", job["server"], f"g{job['gpu']}", *job["cell_id"].split("/"),
              f"{job['start']:04d}", f"{job['stop']:04d}"]
    return "_".join(fields)


def validate_plan(plan):
    jobs = plan["jobs"]
    if plan["task"] != TASK or len(jobs) != 114:
        raise ValueError("A production queue must contain all 114 slices")
    if len({j["slice_id"] for j in jobs}) != 114 or len({j["job_id"] for j in jobs}) != 114:
        raise ValueError("Duplicate slice or job ID in queue")
    cells = collections.defaultdict(list)
    for job in jobs:
        if job["server"] not in SERVERS or job["gpu"] not in SERVERS[job["server"]]:
            raise ValueError("Unknown static worker")
        if (job["pde"] == "nsnonbounded") != (job["server"] == "server197"):
            raise ValueError("A100 workers are reserved for NS; A800 workers handle the other PDEs")
        if job["worker"] != worker_id(job["server"], job["gpu"]) or job["job_id"] != job_id(job):
            raise ValueError("Worker/job identity is inconsistent")
        if job["slice_id"] != f"{job['cell_id']}/{job['start']:04d}-{job['stop']:04d}":
            raise ValueError("Slice ID is inconsistent")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job["job_id"]):
            raise ValueError("Unsafe job ID")
        if len(job["cell_id"].split("/")) != 4 or ".." in job["cell_id"].split("/"):
            raise ValueError("Unsafe cell ID")
        cells[job["cell_id"]].append((job["start"], job["stop"]))
    if len(cells) != 57 or any(sorted(v) != [(0, 500), (500, 1000)] for v in cells.values()):
        raise ValueError("Every cell must have exactly its two disjoint 500-row slices")
    counts = collections.Counter()
    for j in jobs:
        counts[j["server"]] += j["stop"] - j["start"]
    if counts != {"server197": 18000, "server216": 39000}:
        raise ValueError("Unexpected server workload")


def plan_summary(plan):
    groups = collections.defaultdict(lambda: dict(jobs=0, predictions=0, pdes=collections.Counter()))
    for job in plan["jobs"]:
        row = groups[job["worker"]]
        row["jobs"] += 1
        row["predictions"] += job["stop"] - job["start"]
        row["pdes"][job["pde"]] += job["stop"] - job["start"]
    return {key: dict(value, pdes=dict(value["pdes"])) for key, value in sorted(groups.items())}


def make_plan(args):
    inventory = checked_document(args.config_inventory)
    cells = sorted(inventory["cells"], key=lambda c: c["cell_id"])
    if len(cells) != 57 or any(c["count"] != 1000 for c in cells):
        raise ValueError("The full verified configuration inventory is required")
    cursor = collections.Counter()
    jobs = []
    for cell in cells:
        pde = cell["pde"]
        server = "server197" if pde == "nsnonbounded" else "server216"
        candidates = [0, 1] if server == "server197" else PDE_GPUS[pde]
        for start in (0, 500):
            gpu = candidates[cursor[pde] % len(candidates)]
            cursor[pde] += 1
            job = dict(slice_id=f"{cell['cell_id']}/{start:04d}-{start+500:04d}",
                       cell_id=cell["cell_id"], pde=pde, cohort=cell["cohort"],
                       distribution=cell["distribution"], setting=cell["setting"],
                       start=start, stop=start+500, server=server, gpu=gpu,
                       worker=worker_id(server, gpu), config_sha256=cell["config_sha256"])
            job["job_id"] = job_id(job)
            jobs.append(job)
    # Keep model/task groups together in each worker; the small Burgers workload
    # goes first on the two Poisson/Burgers workers, providing early completions.
    jobs.sort(key=lambda j: (j["worker"], 0 if j["pde"] == "burger" else 1,
                             j["pde"], j["setting"], j["cohort"], j["distribution"], j["start"]))
    plan = dict(version=1, task=TASK, status="static_production_queue", revision=1,
                config_inventory_sha256=fhash(args.config_inventory), jobs=jobs,
                expected_predictions=57000, expected_slices=114,
                batch_policy="PDE batch sizes are explicit worker launch arguments and must have a successful matching pilot certificate.",
                reassignment_policy="Stop old workers on both servers, collect their snapshots, and revise only slices with no persisted started record. No automatic migration.")
    validate_plan(plan)
    plan["worker_summary"] = plan_summary(plan)
    result = finish_document(args.output, plan)
    print(json.dumps(dict(status="plan_ready", output=str(args.output), sha256=fhash(args.output), workers=result["worker_summary"]), indent=2))


def state_paths(root, job):
    folder = root / "queue_state" / "slices" / jhash(job["slice_id"])
    return folder / "started.json", folder / "completed.json"


def lock_path(root, worker):
    return root / "queue_state" / "workers" / f"{worker}.lock"


def is_locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
    return False


def snapshot(args):
    plan = checked_document(args.queue)
    validate_plan(plan)
    jobs = [j for j in plan["jobs"] if j["server"] == args.server]
    records = []
    for job in jobs:
        started_path, completed_path = state_paths(args.root, job)
        started = read(started_path) if started_path.exists() else None
        complete = read(completed_path) if completed_path.exists() else None
        records.append(dict(slice_id=job["slice_id"], job_id=job["job_id"], worker=job["worker"],
                            state="completed" if complete else "started" if started else "not_started",
                            started_record=started,
                            started_sha256=fhash(started_path) if started else None,
                            completed_sha256=fhash(completed_path) if complete else None))
    active = [worker_id(args.server, gpu) for gpu in SERVERS[args.server]
              if is_locked(lock_path(args.root, worker_id(args.server, gpu)))]
    doc = dict(version=1, task=TASK, server=args.server, host=socket.gethostname(), collected_utc=now(),
               queue_sha256=fhash(args.queue), active_workers=active, records=records,
               counts=dict(collections.Counter(r["state"] for r in records)))
    if args.output:
        finish_document(args.output, doc)
    print(json.dumps(doc if not args.output else dict(server=args.server, counts=doc["counts"], active_workers=active, output=str(args.output)), indent=2))


def revise(args):
    if not args.workers_stopped:
        raise ValueError("Explicit --workers-stopped confirmation is required after stopping the old workers")
    old = checked_document(args.previous_queue)
    validate_plan(old)
    snapshots = [checked_document(p) for p in args.state_snapshot]
    if {s["server"] for s in snapshots} != set(SERVERS) or len(snapshots) != 2:
        raise ValueError("One freshly collected stopped snapshot from each server is required")
    old_sha = fhash(args.previous_queue)
    if any(s["queue_sha256"] != old_sha or s["active_workers"] for s in snapshots):
        raise ValueError("Snapshots must refer to this queue and contain no active workers")
    states = {r["slice_id"]: r for s in snapshots for r in s["records"]}
    if set(states) != {j["slice_id"] for j in old["jobs"]}:
        raise ValueError("Stopped snapshots do not cover every slice")
    moves = read(args.reassign)
    if not isinstance(moves, dict):
        raise ValueError("Reassignment JSON must map slice_id to {server, gpu}")
    unknown = set(moves) - set(states)
    if unknown:
        raise ValueError(f"Unknown slices: {sorted(unknown)}")
    revised = json.loads(json.dumps(old))
    revised.pop("content_sha256")
    for job in revised["jobs"]:
        if job["slice_id"] not in moves:
            continue
        if states[job["slice_id"]]["state"] != "not_started":
            raise ValueError(f"Started slice cannot be reassigned: {job['slice_id']}")
        target = moves[job["slice_id"]]
        job.update(server=target["server"], gpu=int(target["gpu"]))
        job["worker"] = worker_id(job["server"], job["gpu"])
        job["job_id"] = job_id(job)
    revised.update(revision=old["revision"]+1, previous_queue_sha256=old_sha,
                   stopped_snapshot_sha256=[fhash(p) for p in args.state_snapshot],
                   reassignment_file_sha256=fhash(args.reassign))
    validate_plan(revised)
    revised["worker_summary"] = plan_summary(revised)
    finish_document(args.output, revised)
    print(json.dumps(dict(status="revised_queue_ready", moved_slices=len(moves), output=str(args.output)), indent=2))


def completed_coverage(root, job, protocol_sha):
    folder = root / "jobs" / job["job_id"] / job["cell_id"]
    expected = set(range(job["start"], job["stop"]))
    found = set()
    receipts = []
    for path in sorted(folder.glob("batch_*/receipt.json")):
        rec = read(path)
        if rec["cell_id"] != job["cell_id"] or rec["protocol_sha256"] != protocol_sha:
            raise ValueError(f"Completed receipt identity mismatch: {path}")
        ids = rec["indices"]
        if len(ids) != len(set(ids)) or not set(ids) <= expected or found & set(ids):
            raise ValueError(f"Duplicate/out-of-range completed rows: {path}")
        if not (path.parent / "prediction.pt").is_file():
            raise ValueError(f"Completed receipt has no stored prediction: {path}")
        found.update(ids)
        receipts.append(dict(path=str(path), sha256=fhash(path)))
    if found != expected:
        raise ValueError(f"Successful runner did not produce this complete slice: {len(found)}/500")
    return dict(predictions=len(found), receipts=receipts)


def batch_mapping(values):
    result = {}
    for value in values:
        name, count = value.split("=", 1)
        if name in result or int(count) < 1:
            raise ValueError("Repeated PDE or invalid batch size")
        result[name] = int(count)
    return result


def run_worker(args):
    plan = checked_document(args.queue)
    validate_plan(plan)
    protocol = checked_document(args.protocol)
    if protocol.get("status") != "frozen" or len(protocol["cells"]) != 57:
        raise ValueError("Workers require the final complete frozen protocol")
    cells = {c["cell_id"]: c for c in protocol["cells"]}
    who = worker_id(args.server, args.gpu)
    jobs = [j for j in plan["jobs"] if j["worker"] == who]
    if not jobs:
        raise ValueError(f"No jobs assigned to {who}")
    batches = batch_mapping(args.batch)
    if not {j["pde"] for j in jobs} <= set(batches):
        raise ValueError("Supply an explicit --batch PDE=N for every assigned PDE")
    for job in jobs:
        if jhash(cells[job["cell_id"]]["config"]) != job["config_sha256"]:
            raise ValueError("Queue scientific configuration differs from the final protocol")
    for p in [args.root, args.code_root, args.python, args.protocol, args.queue, *args.pilot_certificate]:
        if not p.is_absolute() or not p.exists():
            raise ValueError(f"Existing explicit absolute path required: {p}")
    if args.root.resolve().name != TASK:
        raise ValueError("The explicit output root must be this task's isolated directory")
    stage = args.code_root / "revision_pairing_0915/run_stage.py"
    runner = args.code_root / "revision_pairing_0915/run_inference.py"
    if not stage.is_file() or not runner.is_file():
        raise ValueError("Missing stage or inference wrapper in the declared code checkout")
    queue_sha, protocol_sha = fhash(args.queue), fhash(args.protocol)
    worker_state = args.root / "queue_state/workers" / f"{who}.json"
    worker_lock = lock_path(args.root, who)
    worker_lock.parent.mkdir(parents=True, exist_ok=True)
    with worker_lock.open("a+") as locked:
        fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = dict(task=TASK, worker=who, host=socket.gethostname(), pid=os.getpid(),
                        queue_sha256=queue_sha, protocol_sha256=protocol_sha, started_utc=now(),
                        status="running", current_job=None, completed_jobs=0, total_jobs=len(jobs))
        write_state(worker_state, progress)
        for job in jobs:
            started_path, completed_path = state_paths(args.root, job)
            binding = dict(slice_id=job["slice_id"], job_id=job["job_id"], worker=who,
                           protocol_sha256=protocol_sha, config_sha256=job["config_sha256"])
            if started_path.exists():
                started = read(started_path)
                if any(started.get(k) != v for k, v in binding.items()):
                    raise ValueError("A persisted started slice belongs to a different job/worker/protocol")
            else:
                write_new(started_path, dict(binding, queue_sha256=queue_sha, started_utc=now()))
            if completed_path.exists():
                completed_coverage(args.root, job, protocol_sha)
                progress["completed_jobs"] += 1
                write_state(worker_state, progress)
                print("QUEUE_SKIP_COMPLETE", job["job_id"], flush=True)
                continue
            attempt_dir = args.root / "queue_state/attempts" / job["job_id"]
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt = 1
            while any(attempt_dir.glob(f"attempt_{attempt:04d}.*")):
                attempt += 1
            log = attempt_dir / f"attempt_{attempt:04d}.log"
            status = attempt_dir / f"attempt_{attempt:04d}.json"
            command = [str(args.python), str(runner), "--protocol", str(args.protocol),
                       "--output-root", str(args.root), "--job-id", job["job_id"],
                       "--cell", job["cell_id"], "--mode", "run", "--batch-size", str(batches[job["pde"]]),
                       "--device", "cuda:0", "--tf32" if args.tf32 else "--no-tf32",
                       "--threads", str(args.threads), "--start", str(job["start"]), "--stop", str(job["stop"])]
            for certificate in args.pilot_certificate:
                command.extend(["--pilot-certificate", str(certificate)])
            stage_command = [str(args.python), str(stage), "--log", str(log), "--status", str(status), "--", *command]
            progress.update(current_job=job["job_id"], attempt=attempt, current_log=str(log), updated_utc=now())
            write_state(worker_state, progress)
            print("QUEUE_START", job["job_id"], "batch", batches[job["pde"]], "log", log, flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
            result = subprocess.run(stage_command, cwd=args.code_root, env=env)
            if result.returncode:
                progress.update(status="failed", exit_code=result.returncode, updated_utc=now())
                write_state(worker_state, progress)
                print("QUEUE_STOP_FAILED", job["job_id"], result.returncode, flush=True)
                return result.returncode if 0 < result.returncode < 256 else 1
            coverage = completed_coverage(args.root, job, protocol_sha)
            write_new(completed_path, dict(binding, completed_utc=now(), stage_status_sha256=fhash(status), **coverage))
            progress["completed_jobs"] += 1
            progress.update(current_job=None, updated_utc=now())
            write_state(worker_state, progress)
            print("QUEUE_COMPLETE", job["job_id"], "500/500", flush=True)
        progress.update(status="complete", completed_utc=now())
        write_state(worker_state, progress)
    return 0


def worker(args):
    try:
        return run_worker(args)
    except (Exception, KeyboardInterrupt) as exc:
        # Preserve a useful status even when preflight or output coverage checks
        # fail outside the child stage. The worker lock is released on unwinding.
        if (args.root.is_absolute() and args.root.name == TASK and args.root.exists()
                and not is_locked(lock_path(args.root, worker_id(args.server, args.gpu)))):
            path = args.root / "queue_state/workers" / f"{worker_id(args.server, args.gpu)}.json"
            state = read(path) if path.exists() else dict(worker=worker_id(args.server, args.gpu))
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=str(exc), updated_utc=now())
            write_state(path, state)
        print("QUEUE_STOP_ERROR", repr(exc), file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config-inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("worker")
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--server", choices=SERVERS, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--batch", action="append", required=True, metavar="PDE=N")
    p.add_argument("--pilot-certificate", type=Path, action="append", required=True)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("snapshot")
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--server", choices=SERVERS, required=True)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("revise")
    p.add_argument("--previous-queue", type=Path, required=True)
    p.add_argument("--state-snapshot", type=Path, action="append", required=True)
    p.add_argument("--reassign", type=Path, required=True)
    p.add_argument("--workers-stopped", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "worker":
        raise SystemExit(worker(args))
    {"plan": make_plan, "snapshot": snapshot, "revise": revise}[args.mode](args)


if __name__ == "__main__":
    main()

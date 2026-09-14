"""Local relay: wait for all workers, archive, strictly audit, then compare.

Never starts/stops inference or removes data. The collector must use the version
that honors STOP_AFTER_CYCLE. All paths are explicit; this program is launched
locally only after the audit/compare scripts have been deployed through Git.
Restarting resumes successful stages; failed/lost stages require investigation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import time

from collect_remote import TASK, REMOTE197, REMOTE216, rsync_command, transfer_statistics


def remote_action(operation, root, payload):
    """Self-contained stdlib helper sent to SSH; snapshot is entirely read-only."""
    import hashlib
    import json
    from pathlib import Path
    import shlex
    import subprocess

    root = Path(root)

    def read(path):
        return json.loads(path.read_text())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def new(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = read(path)
            if old != value:
                raise ValueError(f"Existing stage/marker has different contents: {path}")
            return
        # Publish a complete new file without replacing any existing artifact.
        import os
        import tempfile
        fd, temporary = tempfile.mkstemp(prefix=".finalizer_", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    if operation == "snapshot":
        protocol_path = root / "protocol.json"
        protocol = read(protocol_path)
        cells = {c["cell_id"]: c for c in protocol["cells"]}
        if (protocol["status"] != "frozen" or len(cells) != len(protocol["cells"])
                or len(cells) != 57 or any(c["count"] != 1000 for c in cells.values())
                or protocol["expected_predictions"] != 57000):
            raise ValueError("Requires the final frozen 57-cell/57000-input protocol")
        protocol_sha = sha(protocol_path)
        queue_path = root / payload.get("queue_relative", "provenance/production_queue_v1.json")
        queue = read(queue_path)
        queue_sha = sha(queue_path)
        jobs = queue["jobs"]
        expected_workers = {f"{s}-gpu{g}" for s, n in (("server197", 2), ("server216", 8)) for g in range(n)}
        if len(jobs) != 114 or {j["worker"] for j in jobs} != expected_workers:
            raise ValueError("Queue does not contain exactly 114 slices on the expected 10 workers")
        found = {c: set() for c in cells}
        errors, pending, batches = [], 0, 0
        for path in sorted((root / "jobs").glob("prod_*/**/batch_*/receipt.json")):
            if any(p.startswith(".partial_") or p in (".rsync-partial", ".~tmp~") for p in path.parts):
                continue
            if not (path.parent / "prediction.pt").is_file():
                pending += 1
                continue
            receipt = read(path)
            cell, indices = receipt.get("cell_id"), receipt.get("indices", [])
            if (cell not in cells or receipt.get("protocol_sha256") != protocol_sha
                    or not indices or any(type(i) is not int or not 0 <= i < 1000 for i in indices)):
                errors.append(f"Invalid committed receipt identity/indices: {path}")
                continue
            if len(set(indices)) != len(indices) or found[cell].intersection(indices):
                errors.append(f"Duplicate committed cell/index: {path}")
            found[cell].update(indices)
            batches += 1
        workers = []
        for server, ngpu in (("server197", 2), ("server216", 8)):
            prefix = root if server == "server197" else root / "incoming/server216"
            for gpu in range(ngpu):
                who = f"{server}-gpu{gpu}"
                path = prefix / "queue_state/workers" / f"{who}.json"
                state = read(path) if path.exists() else {"status": "missing"}
                status = state.get("status", "missing")
                if status in ("failed", "interrupted", "draining_after_failure") or state.get("failed_jobs"):
                    errors.append(f"Worker {who} is {status}: {state.get('failed_jobs', state.get('error'))}")
                if path.exists() and (state.get("worker") != who or state.get("protocol_sha256") != protocol_sha
                                      or state.get("queue_sha256") != queue_sha):
                    errors.append(f"Worker identity/protocol/queue mismatch: {path}")
                assigned = sum(j["worker"] == who for j in jobs)
                complete = (status == "complete" and state.get("completed_jobs", 0) == state.get("total_jobs", -1)
                            and state.get("total_jobs", 0) == assigned and not state.get("active_jobs")
                            and not state.get("failed_jobs") and state.get("exit_code", 0) == 0)
                workers.append(dict(worker=who, status=status, complete=complete,
                                    completed_jobs=state.get("completed_jobs"), total_jobs=state.get("total_jobs")))
        counts = {cell: len(ids) for cell, ids in found.items()}
        return dict(protocol_sha256=protocol_sha, queue_sha256=queue_sha, unique_predictions=sum(counts.values()),
                    committed_batches=batches, counts=counts, workers=workers, errors=errors,
                    pending_receipts_without_predictions=pending,
                    ready=not errors and not pending and all(w["complete"] for w in workers)
                    and all(n == 1000 for n in counts.values()))
    if operation == "directories":
        return {name: (root / name).is_dir() for name in payload["names"]}
    if operation in ("stage", "stage_status"):
        name = payload["name"]
        session = payload["session"]
        folder = root / "completion"
        exit_path = folder / f"{name}.exit.json"
        started = exit_path.with_suffix(".started.json")
        alive = subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode == 0
        if operation == "stage":
            command, code = payload["command"], Path(payload["code_root"])
            launch_path = folder / f"{name}.launch.json"
            if alive and not launch_path.exists():
                raise ValueError(f"Existing tmux session is not owned by this finalizer: {session}")
            binding = dict(command=command, code_root=str(code), session=session,
                           protocol_sha256=sha(root / "protocol.json"),
                           script_sha256=sha(Path(command[1])),
                           stage_script_sha256=sha(code / "revision_pairing_0915/run_stage.py"))
            new(launch_path, binding)
            if not exit_path.exists() and not alive:
                if started.exists():
                    raise ValueError(f"Lost {name} stage: started receipt exists without tmux/exit status")
                if Path(payload["output"]).exists():
                    raise ValueError(f"Unowned existing output directory; refusing to overwrite: {payload['output']}")
                stage = [command[0], str(code / "revision_pairing_0915/run_stage.py"),
                         "--log", str(folder / f"{name}.log"), "--status", str(exit_path), "--", *command]
                shell = "cd " + shlex.quote(str(code)) + " && exec env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 " + shlex.join(stage)
                result = subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], capture_output=True, text=True)
                if result.returncode:
                    raise ValueError(f"tmux launch failed ({result.returncode}): {result.stderr}")
                alive = True
        try:
            exit_record = read(exit_path) if exit_path.exists() else None
        except json.JSONDecodeError:
            if not alive:
                raise
            exit_record = None  # run_stage is still writing its terminal JSON.
        return dict(alive=alive, started=started.exists(), exit=exit_record)
    if operation == "artifact":
        path = root / payload["path"]
        return dict(document=read(path), sha256=sha(path))
    if operation == "publish":
        path = root / "completion" / payload["name"]
        value = payload["document"]
        if path.exists():
            old = read(path)
            if {k: v for k, v in old.items() if k != "verified_utc"} != {k: v for k, v in value.items() if k != "verified_utc"}:
                raise ValueError(f"Existing completion marker has different bindings: {path}")
            value = old
        new(path, value)
        return dict(document=value, sha256=sha(path))
    raise ValueError(f"Unknown operation: {operation}")


def validate_audit(result, protocol_sha):
    report = result["document"]
    coverage = report.get("coverage", {})
    if not (report.get("status") == "pass" and report.get("complete") is True
            and report.get("expected_predictions") == report.get("verified_predictions") == 57000
            and report.get("protocol_sha256") == protocol_sha and not report.get("failed_batches")
            and report.get("predictions_by_cohort") == {"supervised": 45000, "diffusion": 12000}
            and len(coverage) == 57 and all(c.get("expected") == c.get("found") == 1000
                                             and c.get("missing") == [] for c in coverage.values())):
        raise ValueError("Strict final audit did not pass all 57 cells / 57000 unique predictions")


def validate_comparison(result, protocol_sha, audit_sha):
    report = result["document"]
    if not (report.get("status") == "complete" and report.get("allow_partial") is False
            and report.get("new_predictions") == report.get("expected_predictions") == 57000
            and report.get("complete_cells") == report.get("expected_cells") == 57
            and report.get("historical_comparison_rows") == report.get("formal_historical_rows") == 73
            and report.get("baseline_comparison_rows") == report.get("formal_baseline_rows") == 203
            and report.get("sources", {}).get("protocol", {}).get("sha256") == protocol_sha
            and report.get("sources", {}).get("audit", {}).get("sha256") == audit_sha):
        raise ValueError("Comparison is incomplete or is not bound to the strict final audit")


def validate_sensitivity(result, audit, protocol_sha):
    report = result["document"]
    binding = audit["document"].get("ns_smooth_selection_overlap_sensitivity", {})
    cells = report.get("cells", [])
    if not (binding.get("status") == "complete" and binding.get("complete") is True
            and binding.get("json", {}).get("sha256") == result["sha256"]
            and binding.get("csv") == report.get("csv")
            and report.get("status") == "complete" and report.get("complete") is True
            and report.get("primary_audit_status") == "pass" and report.get("primary_audit_complete") is True
            and report.get("protocol_sha256") == protocol_sha and report.get("expected_cells") == 8
            and report.get("verified_primary_predictions") == 8000
            and report.get("verified_sensitivity_predictions") == 7968
            and report.get("exclude_source_indices") == [197, 251, 364, 878]
            and len(report.get("frozen_physical_overlap_checks", [])) == 32
            and len(cells) == 8 and len({c["cell_id"] for c in cells}) == 8
            and all(c.get("status") == "complete" and c.get("primary_n") == 1000
                    and c.get("n") == 996 and c.get("excluded_verified_n") == 4 for c in cells)):
        raise ValueError("NS Smooth 996-input sensitivity is incomplete or does not match the final audit")


class NetworkError(RuntimeError):
    pass


class Finalizer:
    def __init__(self, args):
        self.a = args
        self.folder = args.local_root / "finalizer"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + args.timeout_hours * 3600
        self.state = dict(task=TASK, pid=os.getpid(), started_utc=self.now(), arguments=vars(args).copy())
        self.state["arguments"]["local_root"] = str(args.local_root)
        self.helper = inspect.getsource(remote_action) + "\nimport sys, json\ntry:\n print(json.dumps(remote_action(sys.argv[1],sys.argv[2],json.loads(sys.argv[3]))))\nexcept Exception as exc:\n print(json.dumps({'remote_error':repr(exc)}))\n"

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def save(self, **values):
        self.state.update(updated_utc=self.now(), **values)
        temporary = self.folder / f".status.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.folder / "status.json")
        print(json.dumps({k: self.state[k] for k in ("updated_utc", "status", "phase") if k in self.state}), flush=True)

    def command(self, name, command, timeout=180, input_text=None):
        started = time.monotonic()
        try:
            result = subprocess.run(command, input=input_text, capture_output=True, text=True,
                                    env=dict(os.environ, LC_ALL="C"), timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = subprocess.CompletedProcess(command, 124, "", repr(exc))
        record = dict(time=self.now(), name=name, command=command, returncode=result.returncode,
                      seconds=time.monotonic() - started, stdout_tail=result.stdout[-6000:], stderr_tail=result.stderr[-4000:])
        if command[0] == "rsync":
            record.update(transfer_statistics(result.stdout))
        with (self.folder / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if result.returncode:
            raise NetworkError(f"{name} exited {result.returncode}: {result.stderr[-1500:]}")
        return result.stdout

    def remote(self, operation, payload=None, *, server=None, root=None):
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20",
                   "-o", "ServerAliveCountMax=3", server or self.a.server197,
                   shlex.join([self.a.remote_python if not server else "python3", "-c", self.helper,
                               operation, root or self.a.remote197_root, json.dumps(payload or {})])]
        result = json.loads(self.command("ssh_" + operation, command))
        if "remote_error" in result:
            raise ValueError(result["remote_error"])
        return result

    def pause(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Finalization timeout: required worker/results/network/stage completion is still missing")
        time.sleep(min(self.a.interval, remaining))

    def retry(self, action):
        while True:
            try:
                return action()
            except NetworkError as exc:
                self.save(status="waiting_network", reason=str(exc))
                self.pause()

    def wait_ready(self):
        self.save(status="waiting", phase="workers_and_coverage")
        while True:
            snapshot = self.retry(lambda: self.remote("snapshot", {"queue_relative": self.a.queue_relative}))
            self.save(status="waiting", snapshot=snapshot)
            if snapshot["errors"]:
                raise ValueError("; ".join(snapshot["errors"][:10]))
            if snapshot["ready"]:
                return snapshot
            self.pause()

    def stop_collector(self):
        self.save(status="running", phase="collector_stop_after_cycle")
        folder = self.a.local_root / "collector"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "STOP_AFTER_CYCLE").touch(exist_ok=True)
        stop_deadline = time.monotonic() + self.a.collector_stop_timeout
        while True:
            with (folder / "collector.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
            if time.monotonic() >= stop_deadline:
                raise TimeoutError("Collector did not release its lock; watch must be restarted with STOP_AFTER_CYCLE support")
            self.pause()
        command = [sys.executable, str(Path(__file__).with_name("collect_remote.py")), "--mode", "once",
                   "--local-root", str(self.a.local_root), "--remote197-root", self.a.remote197_root,
                   "--remote216-root", self.a.remote216_root, "--server197", self.a.server197,
                   "--server216", self.a.server216]
        self.retry(lambda: self.command("collector_final_once", command, timeout=3600))

    def sync(self, name, source, target, *, patterns=None, immutable=False):
        command = rsync_command(source, target, immutable=False)
        if immutable:
            command[1:1] = ["--ignore-existing"]
        if patterns:
            command[-2:-2] = ["--prune-empty-dirs", "--include=*/",
                              *["--include=" + pattern for pattern in patterns], "--exclude=*"]
        self.retry(lambda: self.command(name, command, timeout=3600))

    def archive216(self):
        self.save(status="running", phase="archive216_metadata")
        a = self.a
        local = a.local_root / "incoming/server216"
        r216, r197 = f"{a.server216}:{a.remote216_root}", f"{a.server197}:{a.remote197_root}"
        patterns = ["invocation_*.json", "pilot_complete.json"]
        self.sync("pull216_job_metadata", r216 + "/jobs", local / "jobs", patterns=patterns, immutable=True)
        self.sync("push197_job_metadata", local / "jobs", r197 + "/jobs", patterns=patterns, immutable=True)
        exists = self.retry(lambda: self.remote("directories", {"names": ["audits", "logs", "queue_state"]},
                                               server=a.server216, root=a.remote216_root))
        self.save(archived216_directories=exists)
        for name, present in exists.items():
            if not present:
                if name != "audits":
                    raise ValueError(f"Required server216 metadata directory missing: {name}")
                continue
            self.sync("pull216_final_" + name, r216 + "/" + name, local / name)
            self.sync("push197_final_" + name, local / name, r197 + "/incoming/server216/" + name)

    def stage(self, name, script, extra, output):
        self.save(status="running", phase=name)
        a = self.a
        command = [a.remote_python, a.remote_code_root + "/revision_pairing_0915/" + script,
                   "--protocol", a.remote197_root + "/protocol.json", *extra,
                   "--output", a.remote197_root + "/" + output]
        payload = dict(name=name, session=a.tmux_prefix + "_" + name, command=command,
                       code_root=a.remote_code_root, output=a.remote197_root + "/" + output)
        result = self.retry(lambda: self.remote("stage", payload))
        while result["exit"] is None:
            if not result["alive"]:
                raise ValueError(f"{name} tmux ended without a recorded exit code")
            self.pause()
            result = self.retry(lambda: self.remote("stage_status", payload))
        self.save(stage_exit=result["exit"])
        if result["exit"]["exit_code"] != 0:
            raise ValueError(f"{name} failed with exit code {result['exit']['exit_code']}; see canonical completion/{name}.log")
        return result["exit"]

    def publish(self, name, document):
        document = dict(task=TASK, verified_utc=self.now(), **document)
        result = self.retry(lambda: self.remote("publish", dict(name=name, document=document)))
        folder = self.a.local_root / "completion"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        if path.exists() and json.loads(path.read_text()) != result["document"]:
            raise ValueError(f"Local completion marker differs from canonical: {path}")
        if not path.exists():
            with path.open("x") as stream:
                json.dump(result["document"], stream, indent=2)
                stream.write("\n")

    def run(self):
        initial = self.wait_ready()
        self.stop_collector()
        self.archive216()
        snapshot = self.retry(lambda: self.remote("snapshot", {"queue_relative": self.a.queue_relative}))
        if (not snapshot["ready"] or snapshot["protocol_sha256"] != initial["protocol_sha256"]
                or snapshot["queue_sha256"] != initial["queue_sha256"]):
            raise ValueError("Canonical completion/identity changed during final archive")
        protocol_sha = snapshot["protocol_sha256"]
        audit_exit = self.stage("final_audit", "audit_outputs.py", ["--results", self.a.remote197_root + "/jobs"], "audits/final")
        audit = self.retry(lambda: self.remote("artifact", {"path": "audits/final/audit.json"}))
        validate_audit(audit, protocol_sha)
        sensitivity = self.retry(lambda: self.remote("artifact", {"path": "audits/final/ns_smooth_996_sensitivity.json"}))
        validate_sensitivity(sensitivity, audit, protocol_sha)
        proof = dict(protocol_sha256=protocol_sha, audit_sha256=audit["sha256"], predictions=57000,
                     complete_cells=57, audit_exit_code=audit_exit["exit_code"],
                     ns_smooth_996_sensitivity_sha256=sensitivity["sha256"])
        self.publish("computation_verified.json", dict(status="computation_verified", **proof))
        comparison_exit = self.stage("final_compare", "compare_results.py",
                                     ["--audit-dir", self.a.remote197_root + "/audits/final", "--references",
                                      self.a.remote197_root + "/provenance/reference_metrics.json"], "comparisons/final")
        comparison = self.retry(lambda: self.remote("artifact", {"path": "comparisons/final/comparison_summary.json"}))
        validate_comparison(comparison, protocol_sha, audit["sha256"])
        self.save(status="running", phase="archive_final_reports")
        for name in ("audits/final", "comparisons/final", "completion"):
            self.sync("pull197_" + name, f"{self.a.server197}:{self.a.remote197_root}/{name}",
                      self.a.local_root / name, immutable=True)
        expected_local = {"audits/final/audit.json": audit["sha256"],
                          "audits/final/ns_smooth_996_sensitivity.json": sensitivity["sha256"],
                          "audits/final/ns_smooth_996_summary.csv": sensitivity["document"]["csv"]["sha256"],
                          "comparisons/final/comparison_summary.json": comparison["sha256"]}
        for key in ("new_summary", "new_per_sample"):
            source = comparison["document"]["sources"][key]
            expected_local["audits/final/" + PurePosixPath(source["path"]).name] = source["sha256"]
        for artifact in comparison["document"]["outputs"].values():
            expected_local["comparisons/final/" + PurePosixPath(artifact["path"]).name] = artifact["sha256"]
        for relative, digest in expected_local.items():
            with (self.a.local_root / relative).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"Local archive differs from canonical final report: {relative}")
        self.publish("finalized.json", dict(status="complete", **proof, comparison_sha256=comparison["sha256"],
                                            comparison_exit_code=comparison_exit["exit_code"],
                                            historical_rows=73, baseline_rows=203, cleanup_performed=False))
        self.save(status="complete", phase="finalized", reason=None, snapshot=snapshot)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--remote-code-root", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--queue-relative", default="provenance/production_queue_v1.json")
    parser.add_argument("--remote197-root", default=REMOTE197)
    parser.add_argument("--remote216-root", default=REMOTE216)
    parser.add_argument("--server197", default="server197")
    parser.add_argument("--server216", default="server216")
    parser.add_argument("--tmux-prefix", default="baseline_pairing_20260915_finalize")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--timeout-hours", type=float, default=72)
    parser.add_argument("--collector-stop-timeout", type=float, default=600)
    args = parser.parse_args(argv)
    args.local_root = args.local_root.resolve()
    if args.local_root.name != TASK or not args.local_root.is_dir():
        parser.error("--local-root must be the existing isolated task directory on the local relay")
    for path in (args.remote197_root, args.remote216_root, args.remote_code_root, args.remote_python):
        if not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            parser.error("All remote paths must be explicit absolute paths without '..'")
    if any(PurePosixPath(p).name != TASK for p in (args.remote197_root, args.remote216_root)):
        parser.error("Remote task roots must name this isolated task")
    if PurePosixPath(args.queue_relative).is_absolute() or ".." in PurePosixPath(args.queue_relative).parts:
        parser.error("--queue-relative must name a queue file inside the isolated task root")
    for name in (args.server197, args.server216, args.tmux_prefix):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            parser.error("SSH aliases/tmux prefix must be simple names")
    if min(args.interval, args.timeout_hours, args.collector_stop_timeout) <= 0:
        parser.error("Intervals/timeouts must be positive")
    finalizer = Finalizer(args)
    with (finalizer.folder / "finalizer.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another finalizer already holds this task's lock")
        try:
            return finalizer.run()
        except (ValueError, TimeoutError) as exc:
            finalizer.save(status="blocked", reason=str(exc))
            return 2
        except KeyboardInterrupt:
            finalizer.save(status="stopped", reason="Local finalizer interrupted; remote stages and workers were not stopped")
            return 130
        except Exception as exc:
            finalizer.save(status="error", reason=repr(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())

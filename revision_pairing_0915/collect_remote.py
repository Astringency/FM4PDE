"""Local-only rsync relay for baseline_pairing_20260915_57k.

Stages: pull server216 into local incoming/server216; append completed batches
to server197/jobs; update server216 logs/queue state only inside server197's
incoming/server216 namespace. No deletion, in-place writes, or replacement of
existing prediction/receipt files is permitted. Failed cycles retry in watch
mode. Partial transfers remain outside the final filenames.

Run once with --mode once, or periodically with --mode watch --interval 60.
The collector must run on the local relay host, never on either GPU server.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time

TASK = "baseline_pairing_20260915_57k"
REMOTE216 = f"/data1/zjinzxf2025/C01Python/FM4PDE/outputs/{TASK}"
REMOTE197 = f"/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/{TASK}"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rsync_command(source, destination, *, immutable, io_timeout=60):
    command = [
        "rsync", "-rt", "--protect-args", "--mkpath", "--stats",
        "--partial-dir=.rsync-partial", "--delay-updates",
        f"--timeout={io_timeout}",
        "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=3",
        "--exclude=.partial_*", "--exclude=.rsync-partial", "--exclude=.~tmp~",
    ]
    if immutable:
        command += [
            "--ignore-existing", "--prune-empty-dirs", "--include=*/",
            "--include=**/batch_*/prediction.pt", "--include=**/batch_*/receipt.json",
            "--exclude=*",
        ]
    return command + [str(source).rstrip("/") + "/", str(destination).rstrip("/") + "/"]


def transfer_statistics(stdout):
    fields = {
        "copied_file_bytes": "Total transferred file size",
        "bytes_sent": "Total bytes sent",
        "bytes_received": "Total bytes received",
        "regular_files_transferred": "Number of regular files transferred",
    }
    result = {}
    for key, label in fields.items():
        match = re.search(r"^" + re.escape(label) + r": ([0-9,]+)", stdout, re.MULTILINE)
        result[key] = int(match[1].replace(",", "")) if match else None
    return result


class Collector:
    def __init__(self, args):
        self.args = args
        self.incoming = args.local_root / "incoming" / "server216"
        self.log_dir = args.local_root / "collector"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for name in ("jobs", "logs", "queue_state"):
            (self.incoming / name).mkdir(parents=True, exist_ok=True)
        self.status = {}

    def log(self, entry):
        with (self.log_dir / "collector.log").open("a") as stream:
            stream.write(json.dumps(dict(time=now(), **entry), ensure_ascii=False) + "\n")
            stream.flush()

    def save_status(self):
        path = self.log_dir / "status.json"
        temporary = path.with_name(f".status.{os.getpid()}.tmp")
        with temporary.open("w") as stream:
            json.dump(self.status, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def transfer(self, name, source, destination, *, immutable):
        command = rsync_command(source, destination, immutable=immutable,
                                io_timeout=self.args.io_timeout)
        start = time.monotonic()
        self.status.update(active_transfer=name, active_transfer_started=now())
        self.save_status()
        self.log(dict(event="transfer_started", name=name, command=command))
        environment = dict(os.environ, LC_ALL="C")
        try:
            completed = subprocess.run(command, capture_output=True, text=True, env=environment)
            result = dict(name=name, returncode=completed.returncode,
                          seconds=time.monotonic() - start,
                          **transfer_statistics(completed.stdout))
            if completed.returncode:
                result["stderr"] = completed.stderr[-4000:]
                result["stdout_tail"] = completed.stdout[-2000:]
        except OSError as exc:
            result = dict(name=name, returncode=127, seconds=time.monotonic() - start,
                          error=str(exc), copied_file_bytes=None)
        self.log(dict(event="transfer_finished", **result))
        return result

    def cycle(self, number):
        self.status = dict(task=TASK, pid=os.getpid(), cycle=number, started=now(),
                           state="running", stages=[])
        self.save_status()
        a = self.args
        remote216 = f"{a.server216}:{a.remote216_root}"
        remote197 = f"{a.server197}:{a.remote197_root}"
        pulled = {}
        # Stage 1: only server216 writes into the relay's incoming namespace.
        for name in ("jobs", "logs", "queue_state"):
            pulled[name] = self.transfer(
                "pull216_" + name, remote216 + "/" + name, self.incoming / name,
                immutable=name == "jobs")
        self.record_stage("pull216", list(pulled.values()))

        # Stage 2: only a completely successful jobs pull may feed the canonical
        # archive. A failed pull may contain staged files; do not forward those.
        if pulled["jobs"]["returncode"] == 0:
            jobs = self.transfer("push197_jobs", self.incoming / "jobs",
                                 remote197 + "/jobs", immutable=True)
        else:
            jobs = self.skipped("push197_jobs", "server216 jobs pull did not complete")
        self.record_stage("push197_jobs", [jobs])

        # Stage 3: mutable remote216 metadata never targets remote197/logs or
        # remote197/queue_state. Do not push an older local copy after pull fails.
        metadata = []
        for name in ("logs", "queue_state"):
            if pulled[name]["returncode"] == 0:
                metadata.append(self.transfer(
                    "push197_" + name, self.incoming / name,
                    remote197 + "/incoming/server216/" + name, immutable=False))
            else:
                metadata.append(self.skipped("push197_" + name,
                                             "latest server216 pull did not complete"))
        self.record_stage("push197_metadata", metadata)
        good = all(stage["returncode"] == 0 for stage in self.status["stages"])
        self.status.update(state="complete" if good else "retry_needed", finished=now(),
                           active_transfer=None,
                           copied_file_bytes_by_stage={stage["name"]: stage["copied_file_bytes"]
                                                       for stage in self.status["stages"]})
        self.save_status()
        self.log(dict(event="cycle_finished", **self.status))
        print(json.dumps(dict(time=now(), cycle=number, state=self.status["state"],
                              stages=[dict(name=s["name"], returncode=s["returncode"],
                                           copied_file_bytes=s["copied_file_bytes"])
                                      for s in self.status["stages"]])), flush=True)
        return 0 if good else 1

    def skipped(self, name, reason):
        result = dict(name=name, returncode=None, skipped=reason, copied_file_bytes=0)
        self.log(dict(event="transfer_skipped", **result))
        return result

    def record_stage(self, name, transfers):
        codes = [item["returncode"] for item in transfers]
        stage = dict(name=name, returncode=0 if all(code == 0 for code in codes) else
                     next((code for code in codes if code not in (0, None)), None),
                     copied_file_bytes=sum(item.get("copied_file_bytes") or 0 for item in transfers),
                     transfers=transfers)
        self.status["stages"].append(stage)
        self.save_status()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--remote216-root", default=REMOTE216)
    parser.add_argument("--remote197-root", default=REMOTE197)
    parser.add_argument("--server216", default="server216")
    parser.add_argument("--server197", default="server197")
    parser.add_argument("--mode", choices=["once", "watch"], required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--io-timeout", type=int, default=60)
    args = parser.parse_args(argv)
    args.local_root = args.local_root.resolve()
    if args.local_root.name != TASK:
        parser.error(f"--local-root must name this isolated task: {TASK}")
    for root in (args.remote216_root, args.remote197_root):
        p = PurePosixPath(root)
        if not p.is_absolute() or ".." in p.parts or p.name != TASK:
            parser.error("Remote roots must be absolute directories for this isolated task")
    for host in (args.server216, args.server197):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host):
            parser.error("Servers must be SSH configuration aliases")
    if args.interval <= 0 or args.io_timeout <= 0:
        parser.error("Interval and I/O timeout must be positive")
    if shutil.which("rsync") is None:
        parser.error("rsync is not installed on the local relay")
    collector = Collector(args)
    with (collector.log_dir / "collector.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        number = 0
        try:
            while True:
                number += 1
                result = collector.cycle(number)
                if args.mode == "once":
                    return result
                time.sleep(args.interval)
        except KeyboardInterrupt:
            collector.status.update(state="stopped", finished=now())
            collector.save_status()
            collector.log(dict(event="stopped"))
            return 130


if __name__ == "__main__":
    sys.exit(main())

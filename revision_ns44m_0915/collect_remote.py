"""Local relay of committed NS44M jobs, followed by the strict canonical audit."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision_ns44m_0915.run_ablations import atomic_json, file_hash, load_json


def command(args, argv, label, *, capture=False, timeout=1800):
    started = time.time()
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout)
    with (args.local / "collector/commands.log").open("a") as f:
        f.write(json.dumps(dict(label=label, argv=argv, started_unix=started,
                               seconds=time.time()-started, returncode=result.returncode)) + "\n")
        f.write(result.stdout + "\n")
    if result.returncode:
        raise RuntimeError(f"{label} returned {result.returncode}: {result.stdout[-1500:]}")
    return result.stdout if capture else None


def snapshot(args):
    script = '''import json,pathlib,os
r=pathlib.Path(ROOT)
receipts=[]
files=[]
for p in (r/'jobs').rglob('receipt.json'):
 d=json.loads(p.read_text())
 if d.get('status')!='complete':raise RuntimeError(str(p))
 receipts.append({'job_id':d['job_id'],'sample_ids':d['sample_ids'],'protocol_sha256':d['protocol_sha256']})
 files.extend(str(f.relative_to(r)) for f in p.parent.rglob('*') if f.is_file() and not any(x.startswith('.partial_') for x in f.parts) and f.name!='job.lock')
for folder in ['inputs','provenance','invocations','pilot']:
 files.extend(str(f.relative_to(r)) for f in (r/folder).rglob('*') if f.is_file() and not any(x.startswith('.partial_') for x in f.parts))
files.append('protocol.json')
workers=[]
for p in (r/'workers').glob('*.json'):
 d=json.loads(p.read_text());d['file']=p.name
 try:os.kill(d['pid'],0);d['live']=True
 except ProcessLookupError:d['live']=False
 workers.append(d)
print(json.dumps({'receipts':receipts,'files':sorted(set(files)),'workers':workers}))
'''.replace("ROOT", repr(args.remote))
    value = command(args, ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.source_host,
                           "python3 -c " + shlex.quote(script)], "snapshot", capture=True)
    return json.loads(value)


def cycle(args, protocol):
    state = snapshot(args)
    expected = {j["job_id"]: j for j in protocol["jobs"]}
    seen = set()
    ph = file_hash(args.local / "protocol.json")
    for r in state["receipts"]:
        assert r["job_id"] in expected and r["job_id"] not in seen
        assert r["sample_ids"] == expected[r["job_id"]]["sample_ids"]
        assert r["protocol_sha256"] == ph
        seen.add(r["job_id"])
    failures = [w for w in state["workers"] if w["status"] == "error" or
                (w["status"] == "running" and not w["live"])]
    if failures:
        raise RuntimeError("Failed/dead NS44M workers: " + json.dumps(failures))
    files = args.local / "collector/committed_files.txt"
    files.write_text("\n".join(state["files"]) + "\n")
    common = ["rsync", "-a", "--ignore-existing", "--stats", "--timeout=120", "--exclude=.partial_*",
              "--files-from="+str(files), "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15"]
    command(args, common + [f"{args.source_host}:{args.remote}/", str(args.local)+"/"], "216-to-local")
    command(args, common + [str(args.local)+"/", f"{args.canonical_host}:{args.canonical}/"], "local-to-197")
    mutable = args.local / "incoming/server216"
    mutable.mkdir(parents=True, exist_ok=True)
    for folder in ("workers", "logs"):
        (mutable / folder).mkdir(exist_ok=True)
        command(args, ["rsync", "-a", "--timeout=120", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
                       f"{args.source_host}:{args.remote}/{folder}/", str(mutable/folder)+"/"], "mutable-"+folder)
    command(args, ["rsync", "-a", "--timeout=120", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
                   str(mutable)+"/", f"{args.canonical_host}:{args.canonical}/incoming/server216/"], "mutable-to-197")
    status = dict(status="collecting", completed_jobs=len(seen),
                  completed_predictions=sum(len(expected[j]["sample_ids"]) for j in seen),
                  expected_jobs=len(expected), workers=state["workers"], updated_unix=time.time())
    atomic_json(args.local / "collector/status.json", status)
    print(json.dumps(status), flush=True)
    return len(seen) == len(expected)


def final_audit(args, protocol):
    script = Path(args.code) / "revision_ns44m_0915/run_ablations.py"
    argv = [args.python, str(script), "audit", "--protocol", args.canonical+"/protocol.json",
            "--output", args.canonical]
    command(args, ["ssh", "-o", "BatchMode=yes", args.canonical_host,
                   "test -f " + shlex.quote(args.canonical+"/audit/audit.json") +
                   " || CUDA_VISIBLE_DEVICES='' " + shlex.join(argv)], "final-audit")
    command(args, ["rsync", "-a", "--ignore-existing", f"{args.canonical_host}:{args.canonical}/audit/",
                   str(args.local / "audit")+"/"], "audit-to-local")
    remote_sha = command(args, ["ssh", args.canonical_host, "sha256sum "+shlex.quote(args.canonical+"/audit/audit.json")],
                         "audit-hash", capture=True).split()[0]
    assert file_hash(args.local / "audit/audit.json") == remote_sha
    result = load_json(args.local / "audit/audit.json")
    assert result["status"] == "pass" and result["complete"] is True
    assert result["jobs"] == protocol["expected_jobs"] and result["predictions"] == protocol["expected_predictions"]
    assert result["per_sample_sha256"] == file_hash(args.local / "audit/per_sample.csv")
    atomic_json(args.local / "completion/computation_verified.json", result)
    command(args, ["rsync", "-a", "--ignore-existing", str(args.local / "completion")+"/",
                   f"{args.canonical_host}:{args.canonical}/completion/"], "completion-to-197")
    atomic_json(args.local / "collector/status.json", dict(result, status="audited"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--canonical", required=True)
    p.add_argument("--code", required=True, help="Canonical Git checkout")
    p.add_argument("--python", required=True, help="Canonical Python interpreter")
    p.add_argument("--source-host", default="server216")
    p.add_argument("--canonical-host", default="server197")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    args.local = args.local.resolve()
    (args.local / "collector").mkdir(parents=True, exist_ok=True)
    lock = (args.local / "collector/collector.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = load_json(args.local / "protocol.json")
    assert protocol["status"] == "frozen"
    while True:
        try:
            complete = cycle(args, protocol)
            if complete:
                final_audit(args, protocol); return
        except Exception as exc:
            atomic_json(args.local / "collector/status.json", dict(status="error_retry", reason=repr(exc),
                        updated_unix=time.time()))
            print("RETRY", repr(exc), flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

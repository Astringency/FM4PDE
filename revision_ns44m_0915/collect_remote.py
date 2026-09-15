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
for folder in ['inputs','provenance','invocations']:
 files.extend(str(f.relative_to(r)) for f in (r/folder).rglob('*') if f.is_file() and f.name!='chain_commands.json' and not any(x.startswith('.partial_') for x in f.parts))
for pilot in r.glob('pilot*'):
 if (pilot/'pilot_complete.json').exists():files.extend(str(f.relative_to(r)) for f in pilot.rglob('*') if f.is_file() and f.name!='job.lock')
for stage in ['development','confirmation']:
 s=r/'weight_study'/stage
 for p in (s/'jobs').rglob('receipt.json'):
  d=json.loads(p.read_text())
  if d.get('status')!='complete':raise RuntimeError(str(p))
  files.extend(str(f.relative_to(r)) for f in p.parent.rglob('*') if f.is_file() and not any(x.startswith('.partial_') for x in f.parts) and f.name!='job.lock')
 for p in (s/'inputs').rglob('*.pt'):files.append(str(p.relative_to(r)))
 if (s/'protocol.json').exists():files.append(str((s/'protocol.json').relative_to(r)))
for name in ['confirmation_templates.json','selection.json','completion.json']:
 p=r/'weight_study'/name
 if p.exists():files.append(str(p.relative_to(r)))
files.append('protocol.json')
workers=[]
for p in (r/'workers').glob('*.json'):
 d=json.loads(p.read_text());d['file']=p.name
 try:os.kill(d['pid'],0);d['live']=True
 except ProcessLookupError:d['live']=False
 workers.append(d)
weight_complete=json.loads((r/'weight_study/completion.json').read_text()) if (r/'weight_study/completion.json').exists() else None
print(json.dumps({'receipts':receipts,'files':sorted(set(files)),'workers':workers,'weight_complete':weight_complete}))
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
    for folder in ("workers", "logs", "provenance"):
        (mutable / folder).mkdir(exist_ok=True)
        command(args, ["rsync", "-a", "--timeout=120", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
                       f"{args.source_host}:{args.remote}/{folder}/", str(mutable/folder)+"/"], "mutable-"+folder)
    command(args, ["rsync", "-a", "--timeout=120", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
                   str(mutable)+"/", f"{args.canonical_host}:{args.canonical}/incoming/server216/"], "mutable-to-197")
    status = dict(status="collecting", completed_jobs=len(seen),
                  completed_predictions=sum(len(expected[j]["sample_ids"]) for j in seen),
                  expected_jobs=len(expected), workers=state["workers"], weight_complete=state["weight_complete"], updated_unix=time.time())
    atomic_json(args.local / "collector/status.json", status)
    print(json.dumps(status), flush=True)
    return len(seen) == len(expected) and state["weight_complete"] is not None


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
    weight = load_json(args.local / "weight_study/completion.json")
    assert weight["status"] == "complete" and weight["development_predictions"] == 28
    assert weight["confirmation_predictions"] == (16 if weight["selected_multiplier"] is None else 24)
    assert weight["selection_sha256"] == file_hash(args.local/"weight_study/selection.json")
    assert load_json(args.local/"weight_study/selection.json")["selected_multiplier"] == weight["selected_multiplier"]
    for stage in ("development", "confirmation"):
        original_audit = args.local / "incoming/server216/weight_study" / stage / "audit"
        original_audit.mkdir(parents=True, exist_ok=True)
        command(args, ["rsync", "-a", "--ignore-existing", f"{args.source_host}:{args.remote}/weight_study/{stage}/audit/",
                       str(original_audit)+"/"], "original-weight-audit-"+stage)
        assert file_hash(original_audit/"audit.json") == weight[stage+"_audit_sha256"]
        remote_stage = args.canonical+"/weight_study/"+stage
        argv = [args.python, str(script), "audit", "--protocol", remote_stage+"/protocol.json", "--output", remote_stage]
        command(args, ["ssh", args.canonical_host, "test -f "+shlex.quote(remote_stage+"/audit/audit.json")+
            " || CUDA_VISIBLE_DEVICES='' "+shlex.join(argv)], "weight-audit-"+stage)
        target = args.local / "weight_study" / stage / "audit"
        target.mkdir(parents=True, exist_ok=True)
        command(args, ["rsync", "-a", "--ignore-existing", f"{args.canonical_host}:{remote_stage}/audit/", str(target)+"/"], "weight-audit-download-"+stage)
        a = load_json(target/"audit.json")
        assert a["status"] == "pass" and a["complete"] is True
        assert a["predictions"] == weight[stage+"_predictions"]
        assert a["protocol_sha256"] == file_hash(target.parent/"protocol.json")
        assert a["per_sample_sha256"] == file_hash(target/"per_sample.csv")
    command(args, ["rsync", "-a", "--ignore-existing", str(args.local/"incoming/server216/weight_study")+"/",
                   f"{args.canonical_host}:{args.canonical}/incoming/server216/weight_study/"], "original-weight-audits-to-197")
    final = dict(result, fixed_configuration_jobs=result["jobs"],
                 jobs=result["jobs"]+7+weight["confirmation_predictions"]//4,
                 fixed_configuration_predictions=result["predictions"],
                 development_predictions=28, confirmation_predictions=weight["confirmation_predictions"],
                 predictions=result["predictions"]+28+weight["confirmation_predictions"],
                 weight_selection_sha256=file_hash(args.local/"weight_study/selection.json"),
                 selected_multiplier=weight["selected_multiplier"])
    atomic_json(args.local / "completion/computation_verified.json", final)
    command(args, ["rsync", "-a", "--ignore-existing", str(args.local / "completion")+"/",
                   f"{args.canonical_host}:{args.canonical}/completion/"], "completion-to-197")
    atomic_json(args.local / "collector/status.json", dict(final, status="audited"))


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

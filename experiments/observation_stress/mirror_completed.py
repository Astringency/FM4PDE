"""Copy only terminal, complete A800 cells into the canonical 197 result tree.

Run on the local workstation, which has authorized SSH access to both hosts.
No server credentials are copied. Repeated calls skip already mirrored cells.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess

SOURCE = "/data1/zjinzxf2025/C01Python/FM4PDE/outputs/observation_stress_20260919"
DEST = "/large_storage/zhangxf/outputs/FM4PDE/observation_stress_20260919"


def ssh(host, command, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["observations", "distribution"], default="observations")
    args = parser.parse_args()
    queue = "observation_queue" if args.study == "observations" else "distribution_queue"
    inventory = (
        "import json,pathlib; r=pathlib.Path("+repr(SOURCE)+"); "
        "v=[json.loads(p.read_text()) for p in (r/"+repr(queue+"/queue_state")+").glob('*.json')]; "
        "print(json.dumps([x for x in v if x['status']=='complete']))"
    )
    completed = json.loads(ssh("server216", shlex.join([
        "/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python", "-c", inventory]),
        capture_output=True, text=True).stdout)
    receipt = DEST+("/setup/mirrored216.json" if args.study == "observations" else "/setup/mirrored_distribution216.json")
    read_receipt = "import pathlib; p=pathlib.Path("+repr(receipt)+"); print(p.read_text() if p.exists() else '{}')"
    mirrored = json.loads(ssh("server197", shlex.join(["python3", "-c", read_receipt]),
                             capture_output=True, text=True).stdout)
    pending = [c for c in completed if c["job"]["key"] not in mirrored]
    if not pending:
        print(f"MIRROR {args.study}: no new cells; {len(mirrored)} already copied", flush=True)
        return
    folders = []
    for record in pending:
        argv = record["job"]["argv"]
        get = lambda flag: argv[argv.index(flag)+1]
        if get("--method") != "fm4pde" or get("--output-root") != SOURCE+"/"+args.study:
            raise ValueError("Unexpected output binding")
        if args.study == "observations":
            components = ["poisson", "fm4pde", get("--task"), get("--distribution"), get("--case")]
        else:
            components = [get("--pde"), "fm4pde", get("--task"), get("--distribution"), "obs"+get("--num-obs")]
        if any(not all(x.isalnum() or x == "_" for x in c) for c in components):
            raise ValueError("Unsafe result path")
        folders.append(args.study+"/results/"+"/".join(components))
    source_command = shlex.join(["tar", "-C", SOURCE, "-cf", "-", *folders])
    destination_command = shlex.join(["tar", "--warning=no-timestamp", "-C", DEST, "-xf", "-"])
    producer = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "server216", source_command], stdout=subprocess.PIPE)
    try:
        ssh("server197", destination_command, stdin=producer.stdout)
    finally:
        producer.stdout.close()
    if producer.wait() != 0:
        raise RuntimeError("Source transfer failed")
    for record in pending:
        mirrored[record["job"]["key"]] = dict(source="server216", source_record=record)
    save = "import sys,pathlib; p=pathlib.Path("+repr(receipt)+"); t=p.with_suffix('.tmp'); t.write_text(sys.stdin.read()); t.replace(p)"
    ssh("server197", shlex.join(["python3", "-c", save]), input=json.dumps(mirrored, indent=2), text=True)
    if args.study == "distribution":
        update = """import json,pathlib,sys
r=pathlib.Path(ROOT)/'distribution_queue/queue_state'
for source in json.load(sys.stdin):
 key=source['job']['key']; p=r/(key+'.json'); prior=json.loads(p.read_text())
 assert prior.get('delegated_to')=='server216', key
 assert prior['status'] in ['delegated','complete'], (key,prior['status'])
 prior.update(status='complete',remote_result=source,exit_code=0)
 tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(prior,indent=2)+'\\n'); tmp.replace(p)
""".replace("ROOT", repr(DEST))
        ssh("server197", shlex.join(["python3", "-c", update]), input=json.dumps(completed), text=True)
    print(f"MIRROR {args.study}: copied {len(pending)} complete cells; {len(mirrored)} total. Prediction hashes remain subject to the audit.", flush=True)


if __name__ == "__main__":
    main()

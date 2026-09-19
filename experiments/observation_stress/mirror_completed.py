"""Copy only terminal, complete A800 cells into the canonical 197 result tree.

Run on the local workstation, which has authorized SSH access to both hosts.
No server credentials are copied. Repeated calls skip already mirrored cells.
"""
from __future__ import annotations

import json
import shlex
import subprocess

SOURCE = "/data1/zjinzxf2025/C01Python/FM4PDE/outputs/observation_stress_20260919"
DEST = "/large_storage/zhangxf/outputs/FM4PDE/observation_stress_20260919"


def ssh(host, command, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], check=True, **kwargs)


def main():
    inventory = (
        "import json,pathlib; r=pathlib.Path("+repr(SOURCE)+"); "
        "v=[json.loads(p.read_text()) for p in (r/'observation_queue/queue_state').glob('*.json')]; "
        "print(json.dumps([x for x in v if x['status']=='complete']))"
    )
    completed = json.loads(ssh("server216", shlex.join([
        "/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python", "-c", inventory]),
        capture_output=True, text=True).stdout)
    receipt = DEST+"/setup/mirrored216.json"
    read_receipt = "import pathlib; p=pathlib.Path("+repr(receipt)+"); print(p.read_text() if p.exists() else '{}')"
    mirrored = json.loads(ssh("server197", shlex.join(["python3", "-c", read_receipt]),
                             capture_output=True, text=True).stdout)
    pending = [c for c in completed if c["job"]["key"] not in mirrored]
    if not pending:
        print(f"MIRROR no new cells; {len(mirrored)}/78 already copied", flush=True)
        return
    folders = []
    for record in pending:
        argv = record["job"]["argv"]
        get = lambda flag: argv[argv.index(flag)+1]
        if get("--method") != "fm4pde" or get("--output-root") != SOURCE+"/observations":
            raise ValueError("Unexpected output binding")
        components = [get("--task"), get("--distribution"), get("--case")]
        if any(not all(x.isalnum() or x == "_" for x in c) for c in components):
            raise ValueError("Unsafe result path")
        folders.append("observations/results/poisson/fm4pde/"+"/".join(components))
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
    print(f"MIRROR copied {len(pending)} complete cells; {len(mirrored)}/78 total. Prediction hashes remain subject to the audit.", flush=True)


if __name__ == "__main__":
    main()

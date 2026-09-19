"""Reserve unstarted NS cells for RTX 4090 workers after same-batch validation.

The owning queue is claimed atomically before publishing the receiving manifest.
Repeated calls recover reservations from per-host receipts. Completed predictions
are written directly to the canonical mounted result tree.
"""
from __future__ import annotations

import argparse
import json
import shlex

from experiments.observation_stress.mirror_completed import SOURCE, DEST, ssh
from experiments.observation_stress.delegate_observations193 import MOUNT, MOUNTED


def remote_json(host, code, payload=None):
    result = ssh(host, shlex.join(["python3", "-c", code]),
                 input=None if payload is None else json.dumps(payload),
                 capture_output=True, text=True)
    return json.loads(result.stdout)


def reconcile():
    code = """import json,pathlib
r=pathlib.Path(ROOT)
records=[json.loads(p.read_text()) for p in (r/'distribution193_queue/queue_state').glob('*.json')]
receipt=json.loads((r/'setup/distribution_193_ns.json').read_text())
owners=receipt['source_hosts']
print(json.dumps([dict(record=x,owner=owners[x['job']['key']]) for x in records if x['status']=='complete']))
""".replace("ROOT", repr(DEST))
    completed = remote_json("server197", code)
    for host, root in [("server197", DEST), ("server216", SOURCE)]:
        code = """import json,pathlib,sys
r=pathlib.Path(ROOT)/'distribution_queue/queue_state'
count=0
for item in json.load(sys.stdin):
 if HOST=='server216' and item['owner']!='server216':continue
 source=item['record'];p=r/(source['job']['key']+'.json');old=json.loads(p.read_text())
 expected='server193' if HOST=='server216' or item['owner']=='server197' else 'server216'
 assert old.get('delegated_to')==expected,(p,old)
 assert old['status'] in ['delegated','complete','complete_external'],(p,old['status'])
 old.update(status='complete' if HOST=='server197' else 'complete_external',
            completed_on='server193',external_result=source,exit_code=0)
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(old,indent=2)+'\\n');tmp.replace(p);count+=1
print(json.dumps(dict(updated=count)))
""".replace("ROOT", repr(root)).replace("HOST", repr(host))
        print(host, remote_json(host, code, completed), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, default=6)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if args.reconcile:
        reconcile()
        return
    if not 0 <= args.target_total <= 21:
        parser.error("Target total must be between 0 and 21")
    code = """import json,pathlib
p=pathlib.Path(ROOT)/'setup/rtx4090_ns_pilot_equivalence.json'
receipt=json.loads(p.read_text())
assert receipt['status']=='passed'
assert set(receipt['batch_sizes'])=={4,16}
print(json.dumps(receipt))
""".replace("ROOT", repr(DEST))
    validation = remote_json("server197", code)
    jobs, owners = [], {}
    for host, root, manifest in [("server197", DEST, "distribution_197.json"),
                                 ("server216", SOURCE, "distribution_216.json")]:
        code = """import json,pathlib,time
r=pathlib.Path(ROOT);state=r/'distribution_queue/queue_state'
receipt=r/'setup/ns_delegation193_claims.json'
selected=json.loads(receipt.read_text())['jobs'] if receipt.exists() else []
alljobs=json.loads((r/'setup'/MANIFEST).read_text())['jobs']
# Recover a reservation even if the prior invocation stopped before saving its receipt.
keys={j['key'] for j in selected}
for job in alljobs:
 p=state/(job['key']+'.json')
 if job['key'] not in keys and p.exists() and json.loads(p.read_text()).get('delegated_to')=='server193':
  selected.append(job);keys.add(job['key'])
for job in reversed(alljobs):
 if len(selected)>=TARGET:break
 if not job['key'].startswith('nsnonbounded_fm4pde_'):continue
 p=state/(job['key']+'.json')
 if p.exists():continue
 try:(state/(job['key']+'.claim')).mkdir()
 except FileExistsError:continue
 record=dict(status='delegated',delegated_to='server193',job=job,claimed_at=time.time())
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\\n');tmp.replace(p)
 selected.append(job)
result=dict(jobs=selected)
tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\\n');tmp.replace(receipt)
print(json.dumps(result))
""".replace("ROOT", repr(root)).replace("MANIFEST", repr(manifest)).replace(
            "TARGET", repr(max(0, args.target_total-len(jobs))))
        selected = remote_json(host, code)["jobs"]
        for job in selected:
            assert job["key"] not in owners
            owners[job["key"]] = host
            argv = job["argv"]
            replacements = {
                "--output-root": MOUNTED+"/distribution",
                "--input-root": MOUNT+"/outputs/FM4PDEbaseline/observation_stress_20260919",
                "--fm-protocol": MOUNT+"/outputs/FM4PDE/rough_stress_20260918/setup/reference_protocol.json",
                "--fm-checkpoint": MOUNT+"/outputs/FM4PDE/audit/ns_main_revision_0909/weights.pth",
                "--noise-root": MOUNT+"/outputs/FM4PDE/rough_stress_20260918/setup/noise_a100_seed0",
            }
            for flag, value in replacements.items():
                argv[argv.index(flag)+1] = value
            jobs.append(job)
    result = dict(jobs=jobs, source_hosts=owners, validation=validation,
                  reason="Same-batch B4/B16 validation passed; only unclaimed NS cells moved.")
    save = "import pathlib,sys; p=pathlib.Path("+repr(DEST+"/setup/distribution_193_ns.json")+"); t=p.with_suffix('.tmp'); t.write_text(sys.stdin.read()); t.replace(p)"
    ssh("server197", shlex.join(["python3", "-c", save]), input=json.dumps(result,indent=2), text=True)
    print(f"RTX 4090 NS allocation: {len(jobs)} cells", flush=True)


if __name__ == "__main__":
    main()

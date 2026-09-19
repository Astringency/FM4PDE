"""Reserve unstarted A800 Poisson cells for two validated RTX 4090 workers."""
from __future__ import annotations

import argparse
import json
import shlex

from experiments.observation_stress.mirror_completed import SOURCE, DEST, ssh

MOUNT = "/home/zhangxf/share/zhangxfA100/large_storage"
MOUNTED = DEST.replace("/large_storage/zhangxf", MOUNT)


def reconcile():
    code = """import json,pathlib
r=pathlib.Path(ROOT)
v=[json.loads(p.read_text()) for p in (r/'observation193_queue/queue_state').glob('*.json')]
print(json.dumps([x for x in v if x['status']=='complete']))
""".replace("ROOT", repr(DEST))
    completed = json.loads(ssh("server197", shlex.join(["python3", "-c", code]), capture_output=True, text=True).stdout)
    code = """import json,pathlib,sys
r=pathlib.Path(ROOT)/'observation_queue/queue_state'
for record in json.load(sys.stdin):
 p=r/(record['job']['key']+'.json'); old=json.loads(p.read_text())
 assert old['delegated_to']=='server193'
 assert old['status'] in ['delegated','complete_external']
 old.update(status='complete_external',external_result=record,exit_code=0)
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(old,indent=2)+'\\n');tmp.replace(p)
""".replace("ROOT", repr(SOURCE))
    ssh("server216", shlex.join(["python3", "-c", code]), input=json.dumps(completed), text=True)
    print(f"Reconciled {len(completed)} externally completed Poisson cells")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if args.reconcile:
        reconcile()
        return
    check = """import json,pathlib
r=pathlib.Path(ROOT)/'setup'
for name in ['rtx4090_torch25_pilot_equivalence.json','rtx4090_b16_pilot_equivalence.json']:
 assert json.loads((r/name).read_text())['status']=='passed',name
""".replace("ROOT", repr(DEST))
    ssh("server197", shlex.join(["python3", "-c", check]))
    code = """import json,pathlib,time
r=pathlib.Path(ROOT);state=r/'observation_queue/queue_state'
receipt=r/'setup/observation_delegation193.json'
if receipt.exists():
 print(receipt.read_text())
else:
 jobs=[]
 for dist in ['id','rough']:
  jobs.extend(json.loads((r/f'setup/observations_216_{dist}.json').read_text())['jobs'])
 selected=[json.loads(p.read_text())['job'] for p in state.glob('*.json') if json.loads(p.read_text()).get('delegated_to')=='server193']
 available=[j for j in jobs if not (state/(j['key']+'.json')).exists() and not (state/(j['key']+'.claim')).exists()]
 for i,job in enumerate(available):
  if i%3==2:continue
  key=job['key']
  try:(state/(key+'.claim')).mkdir()
  except FileExistsError:continue
  p=state/(key+'.json');tmp=p.with_suffix('.tmp')
  tmp.write_text(json.dumps(dict(status='delegated',delegated_to='server193',job=job,claimed_at=time.time()),indent=2)+'\\n');tmp.replace(p)
  selected.append(job)
 result=dict(jobs=selected,reason='Validated B4 and B16; exact archived noise replay; two idle RTX 4090 GPUs.')
 receipt.write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
""".replace("ROOT", repr(SOURCE))
    result = json.loads(ssh("server216", shlex.join(["python3", "-c", code]), capture_output=True, text=True).stdout)
    old = MOUNT+"/outputs/FM4PDE/rough_stress_20260918"
    for job in result["jobs"]:
        argv = job["argv"]
        replacements = {"--output-root": MOUNTED+"/observations",
            "--input-root": MOUNT+"/outputs/FM4PDEbaseline/observation_stress_20260919/measurements",
            "--fm-protocol": old+"/setup/reference_protocol.json",
            "--fm-checkpoint": old+"/setup/fm4poisson.pth",
            "--noise-root": old+"/setup/noise_a100_seed0"}
        for flag, value in replacements.items():
            argv[argv.index(flag)+1] = value
    ssh("server197", "cat > "+shlex.quote(DEST+"/setup/observations_193_fm.json"), input=json.dumps(result,indent=2), text=True)
    print(f"Assigned {len(result['jobs'])} previously unclaimed Poisson cells to server193", flush=True)


if __name__ == "__main__":
    main()

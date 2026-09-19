"""Return unstarted A800 cells to A100 GPUs as their original queues finish."""
from __future__ import annotations

import argparse
import json
import shlex

from experiments.observation_stress.mirror_completed import SOURCE, DEST, ssh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.count <= 63:
        parser.error("Count must be between 0 and 63")
    if args.reconcile:
        code = """import pathlib,json
r=pathlib.Path(ROOT)
records=[json.loads(p.read_text()) for p in (r/'distribution_return_queue/queue_state').glob('*.json')]
print(json.dumps([x for x in records if x['status']=='complete']))
""".replace("ROOT", repr(DEST))
        records = json.loads(ssh("server197", shlex.join(["python3", "-c", code]), capture_output=True, text=True).stdout)
        for host, root, status, owner in [("server197", DEST, "complete", "server216"),
                                          ("server216", SOURCE, "complete_external", "server197")]:
            code = """import pathlib,json,sys
r=pathlib.Path(ROOT)/'distribution_queue/queue_state'
for record in json.load(sys.stdin):
 p=r/(record['job']['key']+'.json');old=json.loads(p.read_text())
 assert old.get('delegated_to')==OWNER and old['status'] in ['delegated','complete','complete_external'],p
 old.update(status=STATUS,completed_on='server197_return_queue',external_result=record,exit_code=0)
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(old,indent=2)+'\\n');tmp.replace(p)
""".replace("ROOT", repr(root)).replace("STATUS", repr(status)).replace("OWNER", repr(owner))
            ssh(host, shlex.join(["python3", "-c", code]), input=json.dumps(records), text=True)
        print(f"Reconciled {len(records)} returned distribution cells", flush=True)
        return
    code = """import pathlib,json,time
r=pathlib.Path(ROOT);state=r/'distribution_queue/queue_state'
jobs=json.loads((r/'setup/distribution_216.json').read_text())['jobs']
selected=[]
for job in jobs:
 p=state/(job['key']+'.json')
 if p.exists() and json.loads(p.read_text()).get('delegated_to')=='server197':selected.append(job['key'])
# Favor the memory-light NS cells, while preserving all running and terminal cells.
ordered=sorted(reversed(jobs),key=lambda job:not job['key'].startswith('nsnonbounded_'))
for job in ordered:
 if len(selected)>=COUNT:break
 p=state/(job['key']+'.json')
 if p.exists():continue
 try:(state/(job['key']+'.claim')).mkdir()
 except FileExistsError:continue
 record=dict(status='delegated',delegated_to='server197',job=job,claimed_at=time.time())
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\\n');tmp.replace(p)
 selected.append(job['key'])
print(json.dumps(selected))
""".replace("ROOT", repr(SOURCE)).replace("COUNT", repr(args.count))
    keys = json.loads(ssh("server216", shlex.join(["python3", "-c", code]), capture_output=True, text=True).stdout)
    code = """import pathlib,json,sys
r=pathlib.Path(ROOT);keys=json.load(sys.stdin)
original=json.loads((r/'setup/distribution_197.json').read_text())['jobs']
jobs=[job for job in original if job['key'] in keys]
assert len(jobs)==len(set(keys))
for job in jobs:
 state=json.loads((r/'distribution_queue/queue_state'/(job['key']+'.json')).read_text())
 assert state.get('delegated_to')=='server216'
receipt=dict(jobs=jobs,reason='Atomic reservations on A800; run the original frozen A100 commands after its workers finish.')
p=r/'setup/distribution_return197.json';tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps(receipt,indent=2)+'\\n');tmp.replace(p)
print('Returned allocation:',len(jobs),'cells')
""".replace("ROOT", repr(DEST))
    ssh("server197", shlex.join(["python3", "-c", code]), input=json.dumps(keys), text=True)


if __name__ == "__main__":
    main()

"""Assign unclaimed distribution cells to the validated A800 workers.

The atomic claim on canonical 197 storage prevents its existing workers from
starting the same logical cell. Running or terminal work is never reassigned.
"""
from __future__ import annotations

import argparse
import json
import shlex

from experiments.observation_stress.mirror_completed import SOURCE, DEST, ssh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int,
                        help="Raise the total A800 allocation without moving running cells")
    args = parser.parse_args()
    if args.target_total is not None and not 0 <= args.target_total <= 63:
        parser.error("Target total must be between 0 and 63")
    code = """import json,pathlib,time
r=pathlib.Path(ROOT)
assert json.loads((r/'setup/a800_distribution_pilot_equivalence.json').read_text())['status']=='passed'
receipt=r/'setup/distribution_delegation216.json'
target=TARGET
if receipt.exists() and target is None:
 print(receipt.read_text())
else:
 jobs=json.loads((r/'setup/distribution_197.json').read_text())['jobs']
 state=r/'distribution_queue/queue_state'
 available=[j for j in jobs if not (state/(j['key']+'.json')).exists() and not (state/(j['key']+'.claim')).exists()]
 selected=[]
 for job in jobs:
  path=state/(job['key']+'.json')
  if path.exists() and json.loads(path.read_text()).get('delegated_to')=='server216':selected.append(job)
 if target is None:target=len(selected)+len(available[::2])
 for job in available[::2]+available[1::2]:
  if len(selected)>=target:break
  key=job['key']
  try:(state/(key+'.claim')).mkdir()
  except FileExistsError:continue
  record=dict(status='delegated',delegated_to='server216',job=job,claimed_at=time.time())
  path=state/(key+'.json'); tmp=path.with_suffix('.tmp')
  tmp.write_text(json.dumps(record,indent=2)+'\\n');tmp.replace(path)
  selected.append(job)
 result=dict(jobs=selected,target_total=target,reason='Validated A800 workers including scheduled successors of Poisson workers; only previously unclaimed cells reassigned.')
 tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\\n');tmp.replace(receipt);print(json.dumps(result))
""".replace("ROOT", repr(DEST)).replace("TARGET", repr(args.target_total))
    result = json.loads(ssh("server197", shlex.join(["python3", "-c", code]), capture_output=True, text=True).stdout)
    for job in result["jobs"]:
        argv = job["argv"]
        pde = argv[argv.index("--pde")+1]
        replacements = {"--output-root": SOURCE+"/distribution",
            "--input-root": SOURCE+"/distribution_inputs",
            "--fm-protocol": SOURCE+"/setup/protocol_frozen.json",
            "--fm-checkpoint": "/data1/zjinzxf2025/C01Python/FM4PDE/outputs/baseline_pairing_20260915_57k/inputs/weights/"+pde+".pth",
            "--noise-root": SOURCE+"/setup/native_noise_seed0"}
        for flag, value in replacements.items():
            argv[argv.index(flag)+1] = value
    save = "import pathlib,sys; p=pathlib.Path("+repr(SOURCE+"/setup/distribution_216.json")+"); t=p.with_suffix('.tmp'); t.write_text(sys.stdin.read()); t.replace(p)"
    ssh("server216", shlex.join(["python3", "-c", save]), input=json.dumps(result,indent=2), text=True)
    print(f"A800 distribution allocation: {len(result['jobs'])} cells total", flush=True)


if __name__ == "__main__":
    main()

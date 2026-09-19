"""Assign unclaimed distribution cells to the two additional free A800s.

The atomic claim on canonical 197 storage prevents its existing workers from
starting the same logical cell. Running or terminal work is never reassigned.
"""
from __future__ import annotations

import json
import shlex
import subprocess

from experiments.observation_stress.mirror_completed import SOURCE, DEST, ssh


def main():
    code = """import json,pathlib,time
r=pathlib.Path(ROOT)
assert json.loads((r/'setup/a800_distribution_pilot_equivalence.json').read_text())['status']=='passed'
receipt=r/'setup/distribution_delegation216.json'
if receipt.exists():
 print(receipt.read_text())
else:
 jobs=json.loads((r/'setup/distribution_197.json').read_text())['jobs']
 state=r/'distribution_queue/queue_state'
 available=[j for j in jobs if not (state/(j['key']+'.json')).exists() and not (state/(j['key']+'.claim')).exists()]
 selected=[]
 for job in available[::2]:
  key=job['key']
  try:(state/(key+'.claim')).mkdir()
  except FileExistsError:continue
  record=dict(status='delegated',delegated_to='server216',job=job,claimed_at=time.time())
  path=state/(key+'.json'); tmp=path.with_suffix('.tmp')
  tmp.write_text(json.dumps(record,indent=2)+'\\n');tmp.replace(path)
  selected.append(job)
 result=dict(jobs=selected,reason='Two newly idle A800 GPUs; matching checkpoints, noise, and physical prediction pilots verified.')
 receipt.write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
""".replace("ROOT", repr(DEST))
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
    ssh("server216", "cat > "+shlex.quote(SOURCE+"/setup/distribution_216.json"), input=json.dumps(result,indent=2), text=True)
    print(f"Assigned {len(result['jobs'])} previously unclaimed distribution cells to server216", flush=True)


if __name__ == "__main__":
    main()

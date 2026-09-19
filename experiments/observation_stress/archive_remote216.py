"""Archive terminal A800 metadata on canonical storage with byte verification.

Prediction tensors, frozen inputs, and noise arrays are retained through their
canonical counterparts and checked separately by the independent result audit.
This command never removes the source tree.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess

from experiments.observation_stress.mirror_completed import SOURCE, DEST, ssh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["observations", "all"], required=True)
    args = parser.parse_args()
    inventory = """import pathlib,json,hashlib
r=pathlib.Path(ROOT);phase=PHASE
states=[json.loads(p.read_text()) for p in (r/'observation_queue/queue_state').glob('*.json')]
assert len(states)==78 and all(x['status'] in ['complete','complete_external'] for x in states)
if phase=='all':
 jobs=json.loads((r/'setup/distribution_216.json').read_text())['jobs']
 for job in jobs:
  record=json.loads((r/'distribution_queue/queue_state'/(job['key']+'.json')).read_text())
  assert record['status'] in ['complete','complete_external'],job['key']
stable_setup={'protocol_frozen.json','measurement_transfer_verification.json','poisson_checkpoint_equivalence.json',
 'observation_delegation193.json','observations_216_id.json','observations_216_rough.json'}
files=[]
for p in sorted(r.rglob('*')):
 if not p.is_file():continue
 assert not p.is_symlink(),p
 rel=p.relative_to(r)
 if p.suffix in ['.pt','.npy']:continue
 assert p.suffix in ['.json','.log','.exit'],p
 selected=(phase=='all' or rel.parts[0] in ['observations','observation_queue','observation_pilot_queue','observation_pilots','inputs']
  or rel.parts[0]=='setup' and (p.name in stable_setup or 'native_noise_seed0' in rel.parts)
  or rel.parts[0]=='logs' and p.name.startswith('obs_poisson_216_'))
 if selected:files.append(dict(path=str(rel),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
assert files
print(json.dumps(dict(source_host='server216',source_root=str(r),phase=phase,files=files)))
""".replace("ROOT", repr(SOURCE)).replace("PHASE", repr(args.phase))
    receipt = json.loads(ssh("server216", shlex.join(["python3", "-c", inventory]),
                            capture_output=True, text=True).stdout)
    destination = DEST+"/remote216/runtime"
    ssh("server197", shlex.join(["mkdir", "-p", destination]))
    source_command = shlex.join(["tar", "-C", SOURCE, "-cf", "-", *[x["path"] for x in receipt["files"]]])
    destination_command = shlex.join(["tar", "--warning=no-timestamp", "-C", destination, "-xf", "-"])
    producer = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "server216", source_command], stdout=subprocess.PIPE)
    try:
        ssh("server197", destination_command, stdin=producer.stdout)
    finally:
        producer.stdout.close()
    if producer.wait() != 0:
        raise RuntimeError("Source metadata transfer failed")
    verify = """import pathlib,sys,json,hashlib
r=pathlib.Path(ROOT);receipt=json.load(sys.stdin)
for entry in receipt['files']:
 p=r/entry['path']
 assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],str(p)
receipt.update(destination_host='server197',destination_root=str(r),status='verified')
p=r.parent/('metadata_'+receipt['phase']+'_verification.json');tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps(receipt,indent=2)+'\\n');tmp.replace(p)
print('VERIFIED',len(receipt['files']),'metadata files',sum(x['bytes'] for x in receipt['files']),'bytes')
""".replace("ROOT", repr(destination))
    ssh("server197", shlex.join(["python3", "-c", verify]), input=json.dumps(receipt), text=True)


if __name__ == "__main__":
    main()

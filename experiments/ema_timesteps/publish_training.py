"""Return immutable checkpoint snapshots and exact log prefixes from server216."""
import argparse
import json
import subprocess
import time

from experiments.ema_timesteps.relay_sampling import remote_python
from experiments.ema_timesteps.transfer_inputs import CANONICAL, CACHE
from experiments.optimizer_diagnostics.relay import ssh, pipe


def publish(pde, arm, epoch):
    tag = f'{pde}_{arm}_{epoch:04d}'
    receipt = f'{CANONICAL}/training_publications/{tag}.json'
    existing = json.loads(remote_python('server197', f'''
from pathlib import Path
p=Path({receipt!r})
print(p.read_text() if p.exists() else 'null')
'''))
    if existing is not None:
        assert existing['status'] == 'verified'
        return True
    relative = f'runs/{pde}/{arm}'
    package = f'{CACHE}/training_publications/{tag}'
    manifest = json.loads(remote_python('server216', f'''
from pathlib import Path
import hashlib,json,shutil
source=Path({(CACHE+'/'+relative)!r});package=Path({package!r})
stem='epoch_{epoch:04d}';ready=source/(stem+'.ready.json')
if not ready.exists():
 print('null');raise SystemExit(0)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
record=json.loads(ready.read_text())
assert record['pde']=={pde!r} and record['arm']=={arm!r} and record['epoch']=={epoch}
assert record['steps']=={epoch}*703
manifest_path=package/'manifest.json'
if not manifest_path.exists():
 package.mkdir(parents=True,exist_ok=True)
 checkpoint=package/(stem+'.pth')
 if not checkpoint.exists():checkpoint.hardlink_to(source/checkpoint.name)
 assert sha(checkpoint)==record['sha256']
 names=['protocol.json',stem+'.ready.json']+[f'epoch_{{e:04d}}.json' for e in range(1,{epoch}+1)]
 for name in names:shutil.copyfile(source/name,package/name)
 selected=[];steps=[]
 with (source/'training.jsonl').open() as f:
  for line in f:
   row=json.loads(line)
   if row['step']>{epoch}*703:break
   selected.append(line);steps.append(row['step'])
 assert steps==list(range(1,{epoch}*703+1))
 (package/'training.jsonl').write_text(''.join(selected))
 files={{p.name:sha(p) for p in package.iterdir() if p.name not in ['manifest.json','manifest.tmp']}}
 manifest=dict(pde={pde!r},arm={arm!r},epoch={epoch},steps={epoch}*703,
               source={CACHE!r},checkpoint_sha256=record['sha256'],files=files)
 temporary=manifest_path.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2)+'\\n');temporary.replace(manifest_path)
print(manifest_path.read_text())
'''))
    if manifest is None:
        return False
    stage = f'{CANONICAL}/incoming_training_{tag}'
    ssh('server197', f'mkdir -p {stage}')
    print('TRAINING_TRANSFER', tag, flush=True)
    pipe('server216', f'tar -C {package} -cf - .',
         'server197', f'tar -C {stage} -xf -')
    print(remote_python('server197', f'''
from pathlib import Path
import hashlib,json,os,shutil
stage=Path({stage!r});dest=Path({(CANONICAL+'/'+relative)!r});manifest={manifest!r}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
assert json.loads((stage/'manifest.json').read_text())==manifest
assert {{p.name for p in stage.iterdir()}}==set(manifest['files'])|{{'manifest.json'}}
for name,digest in manifest['files'].items():assert sha(stage/name)==digest,name
dest.mkdir(parents=True,exist_ok=True)
ready='epoch_{epoch:04d}.ready.json'
names=[n for n in manifest['files'] if n!=ready]+[ready]
for name in names:
 incoming=stage/name;target=dest/name
 if target.exists():
  if name=='training.jsonl':
   old=target.read_bytes();new=incoming.read_bytes()
   assert old.startswith(new) or new.startswith(old),'Published training logs diverge'
   if len(new)>len(old):incoming.replace(target)
  else:assert sha(target)==manifest['files'][name],name
 else:
  # Keep the staging package intact until the final receipt makes retry safe.
  temporary=target.with_suffix(target.suffix+'.publishing')
  if temporary.exists():temporary.unlink()
  os.link(incoming,temporary);temporary.replace(target)
receipt=Path({receipt!r});receipt.parent.mkdir(exist_ok=True)
temporary=receipt.with_suffix('.tmp')
temporary.write_text(json.dumps(dict(status='verified',**manifest),indent=2)+'\\n');temporary.replace(receipt)
shutil.rmtree(stage)
print('TRAINING_PUBLISHED',{tag!r},manifest['checkpoint_sha256'])
'''), flush=True)
    return True


def run(pdes, arms, epochs, watch):
    pending = [(pde, arm, epoch) for epoch in epochs for arm in arms for pde in pdes]
    while pending:
        try:
            for pde, arm, epoch in pending[:]:
                if publish(pde, arm, epoch):
                    pending.remove((pde, arm, epoch))
            if not watch:
                break
        except subprocess.CalledProcessError as exc:
            if not watch or exc.returncode != 255:
                raise
            print('SSH_DISCONNECTED_RETRY_FROM_PUBLICATION_MARKERS', str(exc), flush=True)
        except RuntimeError as exc:
            if not watch or not str(exc).startswith('Transfer failed source='):
                raise
            print('TRANSFER_INTERRUPTED_RETRY_VERIFIED_PUBLICATION', str(exc), flush=True)
        if pending:
            time.sleep(30)
    print('TRAINING_PUBLICATION_FINISHED', dict(pending=pending), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdes', nargs='+', choices=['helmholtz','darcy','burger'],
                        default=['helmholtz','darcy','burger'])
    parser.add_argument('--arms', nargs='+', choices=['uniform','stratified_uniform','logit_normal','beta1_05'],
                        default=['uniform','stratified_uniform','logit_normal','beta1_05'])
    parser.add_argument('--epochs', nargs='+', type=int, choices=[2,5,10], default=[2])
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    run(args.pdes, args.arms, args.epochs, args.watch)

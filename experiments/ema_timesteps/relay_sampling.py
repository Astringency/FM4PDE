"""Relay frozen sampling inputs and immutable NS/Poisson checkpoints via SSH."""
import argparse
import json
import shlex
import time

from experiments.optimizer_diagnostics.relay import ssh, pipe
from experiments.ema_timesteps.transfer_inputs import CANONICAL, CACHE, PYTHON


def remote_python(host, script):
    return ssh(host, "python3 - <<'PY'\n" + script + "\nPY", capture_output=True, text=True).stdout


def inputs(pdes, prepare_only=False):
    for pde in pdes:
        stage = f'{CACHE}/incoming_evaluation_{pde}'
        ssh('server216', f'mkdir -p {stage}')
        print('SAMPLING_INPUT_TRANSFER', pde, flush=True)
        reuse_source = pde in ('helmholtz', 'darcy', 'burger')
        files = ' '.join(f'{pde}/{name}' for name in ['selection.json','truth.pt','masks.pt']) if reuse_source else pde
        pipe('server197', f'tar -C {CANONICAL}/evaluation_inputs -chf - {files}',
             'server216', f'tar -C {stage} -xf -')
        if reuse_source:
            while not json.loads(remote_python('server216', f'''
from pathlib import Path
import json
print(json.dumps(Path({(CACHE+'/inputs/'+pde+'/transfer_verified.json')!r}).exists()))
''')):
                print('WAIT_VERIFIED_TRAINING_SOURCE',pde,flush=True)
                time.sleep(30)
        script = f'''
from pathlib import Path
import hashlib,json,subprocess,shlex
r=Path({CACHE!r});stage=Path({stage!r})/{pde!r}
if {reuse_source!r}:
 (stage/'source.pth').symlink_to(r/'inputs'/{pde!r}/'source.pth')
s=json.loads((stage/'selection.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
for name,key in [('truth.pt','truth_sha256'),('masks.pt','masks_sha256')]:
 assert sha(stage/name)==s['cell'][key]
assert sha(stage/'source.pth')==s['original_checkpoint_sha256']
dest=r/'evaluation_inputs'/{pde!r};dest.parent.mkdir(exist_ok=True)
assert not dest.exists()
stage.replace(dest);stage.parent.rmdir()
(dest/'transfer_verified.json').write_text(json.dumps(dict(status='verified',source={CANONICAL!r})))
if {prepare_only!r}:
 print('SAMPLING_INPUTS_VERIFIED',{pde!r})
 raise SystemExit(0)
gpu={dict(helmholtz=3,darcy=4,burger=5,nsnonbounded=6,poisson=7)[pde]}
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
assert int(free[gpu])>70000, 'Sampling GPU is no longer free; review before launching'
command=[{PYTHON!r},'-u','-m','experiments.ema_timesteps.sampling','queue','--root',str(r),'--pde',{pde!r},'--epoch','2']
script=r/'sample_{pde}_02.sh'
lines=['#!/bin/bash','set -u','cd '+shlex.quote(str(r/'code_sampling')),
 'export CUDA_VISIBLE_DEVICES={dict(helmholtz=3,darcy=4,burger=5,nsnonbounded=6,poisson=7)[pde]} OMP_NUM_THREADS=4',
 shlex.join(command)+' > '+str(r/'sample_{pde}_02.log')+' 2>&1',
 'code=$?','printf "%s\\n" "$code" > '+str(r/'sample_{pde}_02.exit'),'exit "$code"']
script.write_text('\\n'.join(lines)+'\\n')
subprocess.run(['tmux','new-session','-d','-s','fm_ema_sample_{pde}','bash',str(script)],check=True)
print('SAMPLING_INPUT_VERIFIED_QUEUE_LAUNCHED',{pde!r})
'''
        print(remote_python('server216', script), flush=True)


def checkpoints(epochs, pdes, arms):
    pending = [(pde, arm, epoch) for epoch in epochs for arm in arms
               for pde in pdes if pde in ('nsnonbounded', 'poisson')]
    while pending:
        for pde, arm, epoch in pending[:]:
            relative = f'runs/{pde}/{arm}/epoch_{epoch:04d}'
            ready = json.loads(remote_python('server197', f'''
from pathlib import Path
import json
p=Path({(CANONICAL+'/'+relative+'.ready.json')!r})
print(p.read_text() if p.exists() else 'null')
'''))
            if ready is None:
                continue
            print('CHECKPOINT_TRANSFER', pde, arm, epoch, flush=True)
            stage = f'{CACHE}/incoming_checkpoint_{pde}_{arm}_{epoch}'
            ssh('server216', f'mkdir -p {stage}')
            pipe('server197', f'tar -C {CANONICAL} -cf - {relative}.pth {relative}.ready.json',
                 'server216', f'tar -C {stage} -xf -')
            print(remote_python('server216', f'''
from pathlib import Path
import hashlib,json,shutil
stage=Path({stage!r});relative={relative!r}
p=stage/(relative+'.pth');ready=p.with_suffix('.ready.json')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
assert h.hexdigest()==json.loads(ready.read_text())['sha256']
dest=Path({CACHE!r})/(relative+'.pth');dest.parent.mkdir(parents=True,exist_ok=True)
assert not dest.exists()
p.replace(dest);ready.replace(dest.with_suffix('.ready.json'))
shutil.rmtree(stage)
print('CHECKPOINT_VERIFIED',relative,h.hexdigest())
'''), flush=True)
            pending.remove((pde, arm, epoch))
        if pending:
            time.sleep(30)
    print('ALL_CHECKPOINTS_TRANSFERRED', epochs, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['inputs', 'checkpoints'])
    parser.add_argument('--pdes', nargs='+', default=['nsnonbounded','poisson','helmholtz','darcy','burger'])
    parser.add_argument('--epochs', nargs='+', type=int, default=[2])
    parser.add_argument('--prepare-only',action='store_true',help='Verify inputs; launch separately after a fresh resource check')
    parser.add_argument('--arms', nargs='+', choices=['uniform','stratified_uniform','logit_normal','beta1_05'],
                        default=['uniform','stratified_uniform','logit_normal','beta1_05'])
    args = parser.parse_args()
    if args.mode == 'inputs':
        inputs(args.pdes,args.prepare_only)
    else:
        checkpoints(args.epochs, args.pdes, args.arms)

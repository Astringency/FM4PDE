"""Evaluate later snapshots after the initial paired sampling queue finishes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ARMS=['uniform','stratified_uniform','logit_normal','beta1_05']


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def run(root,pde,gpu,epochs,arms,checkout,min_free_mib):
    tag='_'.join(arms)+'_'+'_'.join(map(str,epochs))
    out=root/'extended_sampling'/pde/tag
    out.mkdir(parents=True,exist_ok=True)
    folder=root/'evaluation'/pde
    locks=root/'runtime_locks';locks.mkdir(exist_ok=True)
    def status(state,**kwargs):
        write(out/'progress.json',dict(state=state,pid=os.getpid(),pde=pde,gpu=gpu,epochs=epochs,arms=arms,**kwargs))
    with (locks/f'sampling_{pde}.lock').open('a+') as lock:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status('waiting_other_sampling_for_pde');time.sleep(30)
        required=['original']+[f'{a}_e02_{w}' for a in ARMS for w in ['raw','ema']]
        while True:
            for variant in required:
                exit_path=folder/f'{variant}.exit.json'
                if exit_path.exists() and json.loads(exit_path.read_text())['exit_code'] != 0:
                    raise RuntimeError(f'Initial sampling failed for {pde}/{variant}; inspect its log')
            session=subprocess.run(['tmux','has-session','-t',f'fm_ema_sample_{pde}'],capture_output=True)
            if session.returncode != 0 and all((folder/'variants'/v/'complete.json').exists() for v in required):
                break
            status('waiting_stage1_sampling');time.sleep(30)
        selection_sha=hashlib.sha256((root/'evaluation_inputs'/pde/'selection.json').read_bytes()).hexdigest()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4')
        for epoch in epochs:
            for arm in arms:
                checkpoint=root/'runs'/pde/arm/f'epoch_{epoch:04d}.pth'
                ready=checkpoint.with_suffix('.ready.json')
                while not checkpoint.exists() or not ready.exists():
                    status('waiting_checkpoint',checkpoint=str(checkpoint));time.sleep(30)
                digest=json.loads(ready.read_text())['sha256']
                actual=hashlib.sha256()
                with checkpoint.open('rb') as stream:
                    for block in iter(lambda:stream.read(8<<20),b''):actual.update(block)
                assert actual.hexdigest()==digest
                for weight in ['raw','ema']:
                    variant=f'{arm}_e{epoch:02d}_{weight}'
                    dest=folder/'variants'/variant
                    if (dest/'complete.json').exists():
                        identity=json.loads((dest/'identity.json').read_text())
                        assert identity['checkpoint_sha256']==digest and identity['selection_sha256']==selection_sha
                        assert identity['weight']==weight
                        continue
                    while True:
                        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                '--format=csv,noheader,nounits'],text=True).splitlines()
                        if int(free[gpu]) >= min_free_mib:break
                        status('waiting_gpu_memory',variant=variant,free_mib=int(free[gpu]));time.sleep(30)
                    command=[sys.executable,'-u','-m','experiments.ema_timesteps.sampling','run',
                        '--root',str(root),'--pde',pde,'--variant',variant,'--checkpoint',str(checkpoint),'--weight',weight]
                    with (out/f'{variant}.log').open('a') as log:
                        child=subprocess.Popen(command,cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT,
                                               pass_fds=(lock.fileno(),))
                        status('sampling',variant=variant,child_pid=child.pid,observed_free_mib=int(free[gpu]))
                        code=child.wait()
                    write(out/f'{variant}.exit.json',dict(exit_code=code,child_pid=child.pid))
                    if code:raise RuntimeError(f'Sampling failed for {pde}/{variant}: {code}')
                    completed=json.loads((dest/'complete.json').read_text())
                    assert completed['checkpoint_sha256']==digest
                    subprocess.run([sys.executable,'-m','experiments.ema_timesteps.sampling','report',
                                    '--root',str(root),'--pde',pde],cwd=checkout,env=env,check=True)
        status('complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pde',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--epochs',nargs='+',type=int,choices=[5,10],default=[5,10])
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=['uniform'])
    parser.add_argument('--checkout',type=Path,required=True)
    parser.add_argument('--min-free-mib',type=int,default=16384)
    args=parser.parse_args()
    run(args.root,args.pde,args.gpu,args.epochs,args.arms,args.checkout,args.min_free_mib)

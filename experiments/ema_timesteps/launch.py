"""Launch an experiment queue on an explicitly inspected GPU in named tmux."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def launch(root, pde, mode, gpu, epochs, arms, min_free_mib):
    free = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    assert int(free[gpu]) >= min_free_mib, 'Insufficient measured free memory for requested queue'
    label = 'sample' if mode == 'sampling' else 'train'
    session = f'fm_ema_{label}_{pde}'
    existing = subprocess.run(['tmux','has-session','-t',session], capture_output=True)
    assert existing.returncode != 0, f'Queue session already exists: {session}'
    module = 'sampling' if mode == 'sampling' else 'study'
    command = [sys.executable,'-u','-m',f'experiments.ema_timesteps.{module}',
               'queue','--root',str(root),'--pde',pde,
               '--epoch' if mode == 'sampling' else '--epochs',str(epochs),'--arms',*arms]
    checkout = Path(__file__).resolve().parents[2]
    stem = root/f'{label}_{pde}_{epochs:02d}'
    script = stem.with_suffix('.sh')
    lines = ['#!/bin/bash','set -u','cd '+shlex.quote(str(checkout)),
             f'export CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=4',
             shlex.join(command)+' > '+shlex.quote(str(stem.with_suffix('.log')))+' 2>&1',
             'code=$?', 'printf "%s\\n" "$code" > '+shlex.quote(str(stem.with_suffix('.exit'))),
             'exit "$code"']
    script.write_text('\n'.join(lines)+'\n')
    record=dict(gpu=gpu,observed_free_mib=int(free[gpu]),minimum_free_mib=min_free_mib,
                pde=pde,mode=mode,epochs=epochs,arms=arms,command=command,session=session,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip())
    stem.with_suffix('.launch.json').write_text(json.dumps(record,indent=2))
    subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(script)],check=True)
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pde',required=True)
    parser.add_argument('--mode',choices=['sampling','training'],required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--arms',nargs='+',default=['uniform','stratified_uniform','logit_normal','beta1_05'])
    parser.add_argument('--min-free-mib',type=int,default=70000)
    args=parser.parse_args()
    launch(args.root,args.pde,args.mode,args.gpu,args.epochs,args.arms,args.min_free_mib)

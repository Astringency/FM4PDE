"""Transfer immutable prepared pools to the A800 cache and verify before profiling."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from experiments.optimizer_diagnostics.relay import ssh, pipe

CANONICAL='/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/pretrained/ema_timesteps_20260919'
CACHE='/data1/zjinzxf2025/C01Python/FM4PDE/reproducibility/ema_timesteps_20260919'
PYTHON='/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python'


def transfer(pde,gpu,compression=True):
    while True:
        check=subprocess.run(['ssh','server197','test','-f',f'{CANONICAL}/inputs/{pde}/prepared.json'])
        if check.returncode==0:break
        print('WAIT_PREPARATION',pde,flush=True);time.sleep(30)
    print('TRANSFER_START',pde,flush=True)
    stage=f'{CACHE}/incoming_{pde}'
    ssh('server216',f'mkdir -p {stage}')
    compressor="-I 'gzip -1' " if compression else ''
    extract='-xzf' if compression else '-xf'
    pipe('server197',f'tar -C {CANONICAL}/inputs {compressor}-chf - {pde}',
         'server216',f'tar -C {stage} {extract} -')
    script=f'''
from pathlib import Path
import hashlib,json,subprocess,shlex
r=Path({CACHE!r});stage=Path({stage!r})/{pde!r}
d=json.loads((stage/'prepared.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
assert sha(stage/'full_data.pt')==d['data_sha256']
assert sha(stage/'source.pth')==d['source_checkpoint_sha256']
destination=r/'inputs'/{pde!r}
assert not destination.exists()
stage.replace(destination);stage.parent.rmdir()
(destination/'transfer_verified.json').write_text(json.dumps(dict(data_sha256=d['data_sha256'],source_sha256=d['source_checkpoint_sha256'],canonical={CANONICAL!r},cache={CACHE!r})))
command=[{PYTHON!r},'-u','-m','experiments.ema_timesteps.study','profile','--root',str(r),'--pde',{pde!r}]
script=r/'profile_{pde}.sh'
script.write_text('#!/bin/bash\\nset -u\\ncd '+shlex.quote(str(r/'code_training'))+'\\nexport CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=4\\n'+shlex.join(command)+' > '+str(r/'profile_{pde}.log')+' 2>&1\\ncode=$?\\nprintf \\'%s\\\\n\\' "$code" > '+str(r/'profile_{pde}.exit')+'\\nexit "$code"\\n')
subprocess.run(['tmux','new-session','-d','-s','fm_ema_profile_{pde}','bash',str(script)],check=True)
print('TRANSFER_VERIFIED_PROFILE_LAUNCHED',{pde!r})
'''
    ssh('server216',"python3 - <<'PY'\n"+script+"\nPY")


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pdes',nargs='+',default=['helmholtz','darcy','burger'])
    p.add_argument('--compression',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()
    for name in a.pdes:transfer(name,{'helmholtz':0,'darcy':1,'burger':2}[name],a.compression)

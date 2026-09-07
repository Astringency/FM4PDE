"""Start reserved-input pilots when four A800 GPUs become idle.

This watcher never stops current sampling or deploys a continuation plan.
Pilot processes retain their CUDA context for at most 15 minutes afterward
so the verified handoff can be prepared before those processes are replaced.
"""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time

from run_ns_loss_study import write


REMOTE = r'''
from pathlib import Path
import hashlib,json,shlex,subprocess
ALLOW_LAUNCH = __ALLOW_LAUNCH__
base=Path('/data1/zjinzxf2025/C01Python/ns_loss_extension_20260907')
pilot=base/'pilot_v1'
existing=list(pilot.glob('environment_*.json'))+list(pilot.glob('pilot_[0-3].json'))
if existing or (base/'pilot_launch.json').exists():
 print(json.dumps(dict(status='existing_pilots_require_review',files=[p.name for p in existing])))
 raise SystemExit(0)
def gpu_state():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
 return [dict(index=int(i),name=n.strip(),uuid=u.strip(),memory_mib=int(m)) for i,n,u,m in (line.split(',') for line in text.splitlines())]
gpus=gpu_state();idle=[g['index'] for g in gpus if 'A800' in g['name'] and g['memory_mib']<20]
if len(idle)<4 or not ALLOW_LAUNCH:
 print(json.dumps(dict(status='waiting_for_idle_gpus' if len(idle)<4 else 'idle_probe_only',idle_indices=idle,gpus=gpus)))
 raise SystemExit(0)
head=subprocess.check_output(['git','-C',str(base/'FM4PDE'),'rev-parse','HEAD'],text=True).strip()
assert head.startswith('e8b29f6'),head
protocol=json.loads((pilot/'protocol.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
assert sha(pilot/'protocol.json')=='2e724eff2b5e9e694927d0071a8f71070b8c885133370323b2f24a14e1a3a854'
for name,h in protocol['code_sha256'].items():assert sha(base/'FM4PDE'/name)==h,name
assert sha(base/'inputs_v2/source.json')==protocol['source_sha256']
source=json.loads((base/'inputs_v2/source.json').read_text())
assert sha(base/'inputs_v2/fields_masks.npz')==source['fields_sha256']
assert sha(base/'weights/fm_nsnonbounded.pth')==protocol['fm_checkpoint_sha256']
dm=Path('/data0/zhangxf/Models/pretrained-models/pretrained-ns-nonbounded.pkl')
assert sha(dm)==protocol['diffusion_checkpoint_sha256']
driver=Path('/data1/zjinzxf2025/C01Python/DiffusionPDE')
assert sha(driver/'scripts/generate_ns_nonbounded.py')==protocol['diffusion_source_sha256']
# Recheck after hashing; another job may have claimed a device meanwhile.
gpus=gpu_state();idle=[g['index'] for g in gpus if 'A800' in g['name'] and g['memory_mib']<20]
if len(idle)<4:
 print(json.dumps(dict(status='waiting_for_idle_gpus',idle_indices=idle,gpus=gpus)))
 raise SystemExit(0)
hold="import pathlib,runpy,sys,time; driver=sys.argv[1]; sys.path.insert(0,str(pathlib.Path(driver).resolve().parent)); sys.argv=sys.argv[1:]; runpy.run_path(driver,run_name='__main__'); print('PILOT_COMPLETE_HOLDING_GPU_MAX_900_SECONDS',flush=True); time.sleep(900)"
launched=[]
for shard,gpu in enumerate(idle[:4]):
 session=f'ns_ext_pilot{shard}_0907'
 args=['env',f'CUDA_VISIBLE_DEVICES={gpu}','CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=2',
 '/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python','-u','-c',hold,'plot/run_ns_loss_study.py','run',
 '--inputs',str(base/'inputs_v2'),'--output',str(pilot),'--diffusion-root',str(driver),
 '--fm-checkpoint',str(base/'weights/fm_nsnonbounded.pth'),'--dm-checkpoint',str(dm),
 '--shard',str(shard),'--shards','4','--pilot-only']
 cmd=shlex.join(args)+' >> '+shlex.quote(str(base/f'pilot_{shard}.log'))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(base/'FM4PDE'),cmd],check=True)
 launched.append(dict(shard=shard,gpu=gpu,session=session,command=args))
 (base/'pilot_launch.json').write_text(json.dumps(dict(status='launching' if shard<3 else 'pilots_started',workers=launched),indent=2)+'\n')
print(json.dumps(dict(status='pilots_started',workers=launched,gpus=gpus,hold_seconds=900)))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--launch',action='store_true',help='Launch pilots after fresh resource and source checks.')
    args=parser.parse_args()
    out=args.study/'acceleration_v2'
    out.mkdir(exist_ok=True)
    lock=(out/'capacity_watch.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        sampling=json.loads((args.study/'sampling_progress.json').read_text())
        checked=datetime.datetime.strptime(sampling['checked_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc)
        age=time.time()-checked.timestamp()
        remaining_long=576-len(list((args.study/'ns_results_v3/results').glob('*/DiffusionPDE_1000/*/*.json')))
        if sampling['sampling_complete'] or sampling['calls']>=1500 or remaining_long<64:
            result=dict(status='finished_without_extra_pilots',reason='Remaining work is below the pilot-start threshold.')
        elif age>360 or set(sampling['outcomes'])!={'complete'}:
            result=dict(status='waiting_for_current_collection',collection_age_seconds=age)
        else:
            try:
                raw=subprocess.check_output(['ssh','-o','ConnectTimeout=20','zjinzxf2025@175.102.135.216','python3','-'],
                     input=REMOTE.replace('__ALLOW_LAUNCH__',str(args.launch)),text=True,timeout=60)
                result=json.loads(raw)
            except (subprocess.SubprocessError,ValueError,OSError) as exc:
                # Observe again; a transport timeout does not justify relaunch.
                result=dict(status='observation_error_requires_recheck',error=repr(exc))
        result.update(checked_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
                      calls_collected=sampling['calls'],remaining_diffusion_1000=remaining_long)
        write(out/'capacity_watch_progress.json',result)
        print(json.dumps(result),flush=True)
        if result['status'] in ['pilots_started','existing_pilots_require_review','finished_without_extra_pilots'] or not args.watch:
            return
        time.sleep(120)


if __name__=='__main__':
    main()

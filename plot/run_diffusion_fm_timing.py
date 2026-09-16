"""Controlled, resident-model FM4PDE/DiffusionPDE sampling latency.

Frozen twenty-case design; no fitting or tuning. Run one worker per idle GPU.
The preselected extra case is used only for validation and warm-up. Every
reported setting receives the same twenty physical inputs and sparse masks.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import socket
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_diffusion_comparison import PDES, digest
from diffusion_timing_adapter import build, MODULES


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def prepare(args):
    import numpy as np
    from sampling.config import load_config
    dest=args.inputs
    assert not (dest/'protocol.json').exists(), 'Do not replace a frozen protocol'
    inventory=json.loads((dest/'source/input_inventory.json').read_text())
    assert digest(dest/'source/timing_truths.npz')==inventory['timing_truths_sha256']
    configs={p:load_config(ROOT/f'configs/main/both/{p}.yaml').asdict() for p in PDES}
    masks={}
    ids=inventory['evaluation_ids']+[inventory['pilot_id']]
    for p in PDES:
        for i in ids:
            rng=np.random.default_rng(20260907+i)
            a=np.zeros((128,128),dtype='float32');u=a.copy()
            if p=='burger':
                u[:,rng.choice(128,5,replace=False)]=1;a=u.copy()
            else:
                a.flat[rng.choice(16384,500,replace=False)]=1
                u.flat[rng.choice(16384,500,replace=False)]=1
            masks[f'{p}_{i}_a']=a;masks[f'{p}_{i}_u']=u
    np.savez_compressed(dest/'masks.npz',**masks)
    artifacts=[dest/'source/input_inventory.json',dest/'source/timing_truths.npz',dest/'masks.npz']
    artifacts+=list((dest/'weights').glob('*.pth'))+list((dest/'weights').glob('*.pkl'))
    assert len([p for p in artifacts if p.suffix in {'.pth','.pkl'}])==10
    protocol=dict(version=1,seed=20260907,evaluation_ids=ids[:20],pilot_id=ids[-1],
        pdes=PDES,methods=['FM4PDE','DiffusionPDE'],steps=[100,1000],batch_size=1,
        examples_per_setting=20,distribution='legacy Smooth',task='sparse joint',
        observations='500 independent point observations of each paired field; Burgers: five spatial columns across all 128 time slices',
        selection=inventory['selection'],fm_configs=configs,diffusion_configs=inventory['diffusion_configs'],
        dtype='float32 for both networks, sampler states and residual arithmetic; DiffusionPDE native float64 state/residual cast to float32 explicitly',
        boundary='Resident GPU observations and masks to detached physical GPU fields; synchronized perf_counter. Includes initialization, time grid, network calls, guidance/autograd and physical decoding. Excludes model/data loading, input preparation/transfers, warm-up, scoring, progress diagnostics and output writes.',
        gpu_control='One worker per initially idle GPU; both methods for each PDE share one physical GPU. Background one-second GPU/process telemetry plus boundary snapshots. Any foreign compute PID invalidates the affected attempt; retain that attempt and wait for idle before retrying.',
        order='For each of the same twenty inputs, independently shuffle the four method/budget settings using the frozen seed.',
        warmup='One complete unreported call per method/budget on the separate pilot input; validation calls are also unreported.',
        errors='Sample standard deviation across twenty single-sample wall times (ddof=1), not standard error.',
        precision_scope='Controlled-precision latency experiment only; archived accuracy tables retain their original arithmetic and observation draws. No guidance retuning for 1000 steps or this mask draw.',
        expected_nfe={'FM4PDE':'N','DiffusionPDE':'2N-1 (Heun)'},
        cpu_threads=2,tf32=False,deterministic_algorithms=True,
        fm_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        diffusion_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.diffusion_root,text=True).strip(),
        artifacts=[dict(path=str(p.relative_to(dest)),sha256=digest(p)) for p in artifacts])
    write(dest/'protocol.json',protocol)
    print('FROZEN',len(artifacts),'inputs',flush=True)


class Monitor:
    def __init__(self,uuid,path):
        self.uuid=uuid;self.path=path;self.rows=[];self.stop=threading.Event();self.lock=threading.Lock()
    def snapshot(self):
        metrics=subprocess.check_output(['nvidia-smi','-i',self.uuid,'--query-gpu=uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True).strip()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
        processes=[x.strip() for x in apps.splitlines() if x.startswith(self.uuid)]
        foreign=[x for x in processes if int(x.split(',')[1])!=os.getpid()]
        row=dict(monotonic=time.perf_counter(),wall_time=time.time(),gpu=metrics,processes=processes,foreign=foreign)
        with self.lock:
            self.rows.append(row)
            with self.path.open('a') as f:f.write(json.dumps(row)+'\n')
        return row
    def run(self):
        while not self.stop.is_set():
            try:self.snapshot()
            except Exception as e:
                with self.lock:self.rows.append(dict(monotonic=time.perf_counter(),monitor_error=repr(e),foreign=['monitor failure']))
            self.stop.wait(1.)
    def wait_idle(self):
        while self.snapshot()['foreign']:
            print('WAIT: foreign compute process on assigned GPU',flush=True)
            time.sleep(10)
    def between(self,start,end):
        with self.lock:return [r for r in self.rows if start<=r['monotonic']<=end]


def run(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from data.specs import get_pde_spec
    from run_matched_timing import fm_predict
    import sampling.runner as runner
    sys.path.append(str(args.diffusion_root))
    protocol=json.loads((args.inputs/'protocol.json').read_text());ph=digest(args.inputs/'protocol.json')
    for item in protocol['artifacts']:assert digest(args.inputs/item['path'])==item['sha256'],item
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==protocol['fm_commit']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.diffusion_root,text=True).strip()==protocol['diffusion_commit']
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device='cuda:0';uuid=str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith('GPU-'):uuid='GPU-'+uuid
    args.output.mkdir(parents=True,exist_ok=True)
    env=dict(host=socket.gethostname(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0),uuid=uuid,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
             pid=os.getpid(),protocol_sha256=ph,tf32=False,cpu_threads=2,deterministic=True)
    write(args.output/f'environment_{uuid}.json',env)
    monitor=Monitor(uuid,args.output/f'telemetry_{uuid}.jsonl');monitor.wait_idle()
    thread=threading.Thread(target=monitor.run,daemon=True);thread.start()
    data=np.load(args.inputs/'source/timing_truths.npz');mask_data=np.load(args.inputs/'masks.npz')
    ids=protocol['evaluation_ids']+[protocol['pilot_id']]
    try:
        for pde in args.pdes:
            target=args.output/pde;target.mkdir(parents=True,exist_ok=True)
            monitor.wait_idle()
            fm=load_fm4pde_checkpoint_bundle(str(args.inputs/'weights'/f'fm_{pde}.pth'),pde,device,
                model_profile=protocol['fm_configs'][pde].get('model_profile', 'auto'))
            with (args.inputs/'weights'/f'pretrained-{MODULES[pde].replace("_","-")}.pkl').open('rb') as f:dm=pickle.load(f)['ema'].to(device).eval()
            dm.requires_grad_(False)
            fast,source=build(args.diffusion_root,pde)
            reference,reference_source=build(args.diffusion_root,pde,diagnostics=True)
            (target/'diffusion_effective_sampler.py').write_text(source)
            (target/'diffusion_diagnostic_reference.py').write_text(reference_source)
            calls={'FM4PDE':0,'DiffusionPDE':0}
            def hook_fm(*_):calls['FM4PDE']+=1
            def hook_dm(*_):calls['DiffusionPDE']+=1
            handles=[fm[0].model.register_forward_hook(hook_fm),dm.register_forward_hook(hook_dm)]
            pairs={};truths={};masks={}
            spec=get_pde_spec(pde)
            for j,i in enumerate(ids):
                a=torch.from_numpy(data[pde+'_a'][j:j+1]).to(device)
                u=torch.from_numpy(data[pde+'_u'][j:j+1]).to(device)
                ma=torch.from_numpy(mask_data[f'{pde}_{i}_a'])[None,None].to(device)
                mu=torch.from_numpy(mask_data[f'{pde}_{i}_u'])[None,None].to(device)
                masks[i]=PairMasks(ma,mu,dict(num_observations_coef=int(ma.sum()),num_observations_sol=int(mu.sum()),source='frozen common masks'))
                meta=dict(sample_ids=[str(i)],sample_offsets=[i],offset=i,batch_size=1,synthetic=False,
                          endpoint_pair=pde!='burger',data_path=protocol['diffusion_configs'][pde]['data']['datapath'])
                def gt(aa,uu):
                    return PDEGroundTruth(pde,aa,uu,aa if pde=='burger' else torch.cat([aa,uu],1),{},list(spec.coef_channel_names),list(spec.sol_channel_names),meta)
                truths[i]=gt(a,u);pairs[i]=gt(a*ma,u*mu)
            def predict(method,n,i,*,hidden=False,scale=1.,reference_dm=False):
                g=pairs[i];pm=masks[i]
                if hidden or scale!=1:
                    a=g.coef*scale+((1-pm.coef)*3. if hidden else 0.)
                    u=g.sol*scale+((1-pm.sol)*7. if hidden else 0.)
                    g=dataclasses.replace(g,coef=a,sol=u,pair=a if pde=='burger' else torch.cat([a,u],1))
                seed=protocol['seed']+i
                if method=='FM4PDE':
                    c=copy.deepcopy(protocol['fm_configs'][pde]);c.update(device=device,num_steps=n,batch_size=1,
                        offset=i,sample_seed=seed,save_plots=False,save_intermediate=False,save_per_sample_curves=False,
                        output_dir=str(target/'reference'),checkpoint_path=str(args.inputs/'weights'/f'fm_{pde}.pth'),
                        initial_noise_source_indices=[],initial_noise_source_batch_size=None)
                    if pde=='burger':c.update(sensor_mode='sensor_column',num_sensor_columns=5,num_obs=640)
                    cfg=AblationConfig(**c);prediction=fm_predict(cfg,fm,g,pm)
                    return prediction,(cfg,g,pm)
                c=copy.deepcopy(protocol['diffusion_configs'][pde]);c['generate'].update(device=device,seed=seed,batch_size=1)
                c['test']['iterations']=n
                prediction=(reference if reference_dm else fast)(c,dm,g.coef,g.sol,pm.coef[0,0],pm.sol[0,0])
                return prediction,None
            pilot=protocol['pilot_id'];checks=[]
            for method in protocol['methods']:
                monitor.wait_idle();base,extra=predict(method,100,pilot)
                assert all(torch.isfinite(t).all() for t in base),(pde,method,'nonfinite pilot')
                repeated,_=predict(method,100,pilot);hidden,_=predict(method,100,pilot,hidden=True)
                changed,_=predict(method,100,pilot,scale=.83)
                delta=lambda other:max(float((a-b).abs().max()) for a,b in zip(base,other))
                row=dict(method=method,pilot_id=pilot,repeat_max_abs=delta(repeated),hidden_max_abs=delta(hidden),observation_change_max_abs=delta(changed))
                assert row['repeat_max_abs']==0 and row['hidden_max_abs']==0 and row['observation_change_max_abs']>0,row
                if method=='FM4PDE':
                    cfg,g,pm=extra
                    receipt=runner.run_single_ablation(cfg,fm,ground_truth=g,observation_masks=pm)
                    saved=torch.load(Path(receipt['run_dir'])/'result.pt',map_location=device,weights_only=False)
                    row['reference_max_abs']=delta((saved['coef_final'],saved['sol_final']))
                    del saved
                else:
                    ref,_=predict(method,100,pilot,reference_dm=True);row['reference_max_abs']=delta(ref)
                    del ref
                assert row['reference_max_abs']==0,row
                checks.append(row);del base,repeated,hidden,changed
            write(target/'pilot.json',dict(status='pass',protocol_sha256=ph,checks=checks))
            variants=[(m,n) for m in protocol['methods'] for n in protocol['steps']]
            warmups=[]
            for method,n in variants:
                monitor.wait_idle();torch.cuda.synchronize();t=time.perf_counter()
                prediction,_=predict(method,n,pilot);torch.cuda.synchronize()
                warmups.append(dict(method=method,steps=n,seconds=time.perf_counter()-t,finite=all(bool(torch.isfinite(x).all()) for x in prediction)))
                del prediction
            write(target/'warmups.json',warmups)
            assert all(r['finite'] for r in warmups)
            for i in protocol['evaluation_ids']:
                order=variants.copy();random.Random(protocol['seed']+i).shuffle(order)
                for method,n in order:
                    stem=f'{method}_{n}_{i}';receipt_path=target/(stem+'.json')
                    if receipt_path.exists():
                        old=json.loads(receipt_path.read_text());assert old['protocol_sha256']==ph and old['uncontended']
                        continue
                    attempt=0
                    while True:
                        monitor.wait_idle();before=monitor.snapshot();calls[method]=0
                        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                        start=time.perf_counter();prediction,_=predict(method,n,i);torch.cuda.synchronize();end=time.perf_counter()
                        after=monitor.snapshot();telemetry=[before]+monitor.between(start,end)+[after]
                        uncontended=all(not r['foreign'] for r in telemetry)
                        row=dict(pde=pde,method=method,steps=n,sample_id=i,seconds=end-start,nfe=calls[method],
                            peak_memory_bytes=torch.cuda.max_memory_allocated(),protocol_sha256=ph,uuid=uuid,
                            uncontended=uncontended,telemetry_samples=len(telemetry),start_monotonic=start,end_monotonic=end,
                            attempt=attempt,finite=all(bool(torch.isfinite(t).all()) for t in prediction))
                        assert calls[method]==(n if method=='FM4PDE' else 2*n-1),row
                        if not uncontended:
                            write(target/(stem+f'_contended_{attempt}.json'),row);attempt+=1;del prediction;continue
                        # Preserve unstable predictions rather than selecting favorable cases.
                        truth=truths[i];pm=masks[i]
                        row['relative_l2']=[float((v-t).norm()/t.norm()) if bool(torch.isfinite(v).all()) else None for v,t in zip(prediction,(truth.coef,truth.sol))]
                        output=target/(stem+'.pt')
                        torch.save(dict(coef=prediction[0].cpu(),sol=prediction[1].cpu(),coef_truth=truth.coef.cpu(),sol_truth=truth.sol.cpu(),mask_a=pm.coef.cpu(),mask_u=pm.sol.cpu(),receipt=row),output)
                        row['prediction_sha256']=digest(output);write(receipt_path,row)
                        print('TIMED',pde,method,n,i,round(row['seconds'],3),'seconds',flush=True)
                        del prediction
                        break
            for handle in handles:handle.remove()
            del fm,dm,pairs,truths,masks,fast,reference
            gc.collect();torch.cuda.empty_cache()
            write(target/'complete.json',dict(protocol_sha256=ph,calls=80,status='complete'))
    finally:
        monitor.stop.set();thread.join(timeout=10)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run']);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--diffusion-root',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--pdes',nargs='+',default=PDES,choices=PDES)
    a=p.parse_args();prepare(a) if a.mode=='prepare' else run(a)


if __name__=='__main__':main()

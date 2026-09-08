"""Complete missing Burgers cells using frozen baseline observations.

Native DiffusionPDE arithmetic and guidance; FM4PDE's archived 100-step
configuration. Resident networks avoid per-example checkpoint loading.
Every physical prediction and its input/mask hashes are retained.
"""
from __future__ import annotations
import argparse
import copy
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diffusion_timing_adapter import build
from run_paper_ablation_revision import digest,write


def freeze(args):
    from sampling.config import load_config
    inputs=args.inputs
    protocol_path=args.sampling_protocol or inputs/'sampling_protocol.json'
    assert not protocol_path.exists()
    source=json.loads((inputs/'protocol.json').read_text())
    jobs=[]
    for cell in source['new_fm_cells']:
        jobs.extend(dict(cell=cell,method='FM4PDE',steps=100,sample_id=i) for i in range(1000))
    for cell in source['new_diffusion_cells']:
        for n in [100,1000]:
            jobs.extend(dict(cell=cell,method='DiffusionPDE',steps=n,sample_id=i) for i in range(1000))
    c=load_config(ROOT/'configs/main/both/burger.yaml').asdict()
    # Per-example job assignment is independent of results, balancing both costs.
    for j in jobs:j['shard']=j['sample_id']%args.shards
    protocol=dict(version=1,input_protocol_sha256=digest(inputs/'protocol.json'),
        fm_config=c,diffusion_config=source['source_configs']['diffusion'],jobs=jobs,
        shards=args.shards,seed=20260913,diffusion_native_precision=True,
        fm_weights_sha256=digest(args.fm_weights),dm_weights_sha256=digest(args.dm_weights),
        diffusion_source_sha256=digest(args.diffusion_root/'scripts/generate_burgers.py'),
        adapter_sha256=digest(ROOT/'plot/diffusion_timing_adapter.py'),
        runner_sha256=digest(Path(__file__)),
        scope='FM100: ID/Smooth/Rough structured; DM100 and DM1000: Smooth random and structured. '
              'Existing FM random thousand-input results are retained with their original random masks.',
        pilot='Input 0 with a separate seed for implementation equivalence, repeatability, hidden-field invariance, '
              'and observation sensitivity only; fixed archived guidance is not tuned.',
        metrics='Full 128x128 physical trajectory relative L2; per-input values, mean, n-1 standard deviation.',
        precision='FM network/state float32; Diffusion network float32, native state/time/residual float64; TF32 disabled.')
    write(protocol_path,protocol)
    print('FROZEN',len(jobs),'calls',args.shards,'shards',flush=True)


def worker(args):
    import numpy as np
    import torch
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from run_matched_timing import fm_predict
    import sampling.runner as runner
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    source=json.loads((args.inputs/'protocol.json').read_text())
    protocol_path=args.sampling_protocol or args.inputs/'sampling_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    ph=digest(protocol_path)
    assert digest(args.inputs/'protocol.json')==protocol['input_protocol_sha256']
    assert digest(ROOT/'plot/diffusion_timing_adapter.py')==protocol['adapter_sha256']
    assert digest(Path(__file__))==protocol['runner_sha256']
    assert digest(args.diffusion_root/'scripts/generate_burgers.py')==protocol['diffusion_source_sha256']
    assert digest(args.fm_weights)==protocol['fm_weights_sha256']
    assert digest(args.dm_weights)==protocol['dm_weights_sha256']
    target=args.output;target.mkdir(parents=True,exist_ok=True)
    with (target/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(target/f'environment_shard{args.shard}.json',dict(protocol_sha256=ph,
            host=socket.gethostname(),pid=os.getpid(),torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        # Both repositories have a top-level data package. Keep FM imports
        # first while making DiffusionPDE's pickle classes available.
        sys.path.append(str(args.diffusion_root))
        fm=load_fm4pde_checkpoint_bundle(str(args.fm_weights),'burger','cuda:0',model_profile='recommended')
        with args.dm_weights.open('rb') as f:dm=pickle.load(f)['ema'].to('cuda:0').eval()
        dm.requires_grad_(False)
        fast,src=build(args.diffusion_root,'burger',native_precision=True)
        ref,_=build(args.diffusion_root,'burger',diagnostics=True,native_precision=True)
        (target/f'diffusion_native_shard{args.shard}.py').write_text(src)
        counters={'FM4PDE':0,'DiffusionPDE':0}
        def fmh(*_):counters['FM4PDE']+=1
        def dmh(*_):counters['DiffusionPDE']+=1
        handles=[fm[0].model.register_forward_hook(fmh),dm.register_forward_hook(dmh)]
        cells={}
        for name,h in source['artifacts'].items():
            assert digest(args.inputs/name)==h,name
            cells[name[:-3]]=torch.load(args.inputs/name,map_location='cpu',weights_only=False)

        def predict(job,hidden=False,scale=1.,reference=False,pilot=False):
            cell=cells[job['cell']];i=job['sample_id']
            truth=cell['truth'][i:i+1].to('cuda:0')
            mask=cell['mask'][i:i+1].to('cuda:0',dtype=torch.float32)
            obs=truth*mask*scale
            if hidden:obs=obs+7*(1-mask)
            pm=PairMasks(mask,mask,dict(source='frozen baseline observations',num_obs=int(mask.sum())))
            seed=protocol['seed']+i+(1000000 if pilot else 0)
            if job['method']=='FM4PDE':
                c=copy.deepcopy(protocol['fm_config'])
                c.update(device='cuda:0',batch_size=1,offset=i,sample_seed=seed,num_steps=job['steps'],
                    num_obs=int(mask.sum()),checkpoint_path=str(args.fm_weights),
                    output_dir=str(target/f'pilot_reference_{args.shard}'),
                    save_plots=False,save_intermediate=False,save_per_sample_curves=False)
                cfg=AblationConfig(**c)
                gt=PDEGroundTruth('burger',obs,obs,obs,{},['u'],['u'],dict(synthetic=False,
                    sample_ids=[str(i)],sample_offsets=[i],offset=i,batch_size=1,endpoint_pair=False))
                if reference:
                    result=runner.run_single_ablation(cfg,fm,ground_truth=gt,observation_masks=pm)
                    d=torch.load(Path(result['run_dir'])/'result.pt',map_location='cuda:0',weights_only=False)
                    pred=d['sol_final']
                else:pred=fm_predict(cfg,fm,gt,pm)[1]
            else:
                c=copy.deepcopy(protocol['diffusion_config'])
                c['generate'].update(seed=seed,device='cuda:0',batch_size=1)
                c['test']['iterations']=job['steps']
                pred=(ref if reference else fast)(c,dm,obs,obs,mask[0,0],mask[0,0])[1]
            return pred.detach(),truth,mask

        # This pilot validates both geometries before any formal shard result.
        checks=[]
        for cell in ['smooth_random','smooth_structured']:
            for method in ['FM4PDE','DiffusionPDE']:
                job=dict(cell=cell,method=method,steps=100,sample_id=0)
                base,_,_=predict(job,pilot=True)
                assert torch.isfinite(base).all()
                row=dict(cell=cell,method=method)
                for name,kw in [('repeat',{}),('hidden',{'hidden':True}),('observation_change',{'scale':.83}),('reference',{'reference':True})]:
                    other,_,_=predict(job,pilot=True,**kw)
                    row[name+'_max_abs']=float((other-base).abs().max())
                assert row['repeat_max_abs']==0 and row['hidden_max_abs']==0 and row['reference_max_abs']==0,row
                assert row['observation_change_max_abs']>0,row
                checks.append(row)
        write(target/f'pilot_shard{args.shard}.json',dict(protocol_sha256=ph,checks=checks,status='passed'))
        for job in [j for j in protocol['jobs'] if j['shard']==args.shard]:
            folder=target/'results'/job['cell']/f'{job["method"]}_{job["steps"]}'
            folder.mkdir(parents=True,exist_ok=True)
            receipt=folder/f'sample{job["sample_id"]}.json'
            if receipt.exists():
                previous=json.loads(receipt.read_text())
                assert previous['protocol_sha256']==ph
                continue
            counters[job['method']]=0
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            pred,truth,mask=predict(job)
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            assert torch.isfinite(pred).all(),job
            nfe=counters[job['method']]
            assert nfe==(job['steps'] if job['method']=='FM4PDE' else 2*job['steps']-1),(job,nfe)
            pred=pred.cpu();truth=truth.cpu();mask=mask.cpu()
            error=float(torch.linalg.vector_norm(pred.double()-truth.double())/torch.linalg.vector_norm(truth.double()))
            tensor_path=folder/f'sample{job["sample_id"]}.pt'
            torch.save(dict(prediction=pred,truth=truth,mask=mask,job=job,protocol_sha256=ph),tensor_path)
            write(receipt,dict(**job,protocol_sha256=ph,rel_l2_u=error,seconds=seconds,nfe=nfe,
                peak_bytes=torch.cuda.max_memory_allocated(),tensor_sha256=digest(tensor_path),
                mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                truth_sha256=hashlib.sha256(truth.numpy().tobytes()).hexdigest()))
            print('DONE',job['cell'],job['method'],job['steps'],job['sample_id'],f'{seconds:.2f}s',error,flush=True)
        for h in handles:h.remove()
        write(target/f'complete_shard{args.shard}.json',dict(protocol_sha256=ph,finished_unix=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','worker'])
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--sampling-protocol',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--fm-weights',type=Path,required=True)
    p.add_argument('--dm-weights',type=Path,required=True)
    p.add_argument('--diffusion-root',type=Path,required=True)
    p.add_argument('--shards',type=int,default=5)
    p.add_argument('--shard',type=int)
    a=p.parse_args()
    if a.mode=='freeze':freeze(a)
    else:
        assert a.output is not None and a.shard is not None
        worker(a)


if __name__=='__main__':main()

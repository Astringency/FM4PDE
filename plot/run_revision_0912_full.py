"""Matched layout/co-location and temporal-residual controls for the manuscript."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from sampling.config import AblationConfig
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.runner import run_single_ablation
from sampling.masks import PairMasks, make_mask
from sampling.batching import combine_truths
from run_paper_ablation_revision import digest


def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def layout_masks(shape,layout,shared,seed):
    assert tuple(shape)==(1,1,128,128)
    def one(s):
        if layout=='fixed':
            mask=torch.zeros(shape)
            pool=torch.arange(128*128).reshape(128,128)[:,:64].reshape(-1)
            ids=pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(s))[:500]]
            mask.reshape(-1)[ids]=1
            return mask
        return make_mask(shape,500,{'columns':'sensor_column'}.get(layout,layout),s,
                         num_sensor_columns=5,device='cpu')
    first_seed=0 if layout in ['fixed','grid'] else seed
    ma=one(first_seed)
    mu=ma.clone() if shared else (torch.roll(ma,(3,3),(-2,-1)) if layout=='grid' else one(first_seed+1))
    count=640 if layout=='columns' else 500
    assert ma.sum()==mu.sum()==count
    overlap=int((ma*mu).sum())
    assert (overlap==count)==shared
    if layout=='fixed':assert ma[...,64:].sum()==mu[...,64:].sum()==0
    return PairMasks(ma.cuda(),mu.cuda(),dict(layout=layout,shared_mask=shared,
        seed_a=first_seed,seed_u=first_seed if shared else first_seed+1,
        grid_translation=[0,0] if shared else ([3,3] if layout=='grid' else None),
        locations_per_field=count,overlap=overlap))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['layouts','temporal'])
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert args.output.is_absolute()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan_path=ROOT/'plot/revision_0912_full_plan.json';plan=json.loads(plan_path.read_text())
    write(args.output/'plan.json',plan)
    env=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        device=os.environ.get('CUDA_VISIBLE_DEVICES'),tf32=False,batch_size=1,plan_sha256=digest(plan_path))
    write(args.output/'environment.json',env)
    pdes=['helmholtz'] if args.mode=='layouts' else plan['temporal_pdes']
    rows=[]
    for pde in pdes:
        source=args.inputs/pde;protocol=json.loads((source/'protocol.json').read_text())
        assert digest(source/'weights.pth')==protocol['weights_sha256']
        assert digest(source/'truths.pt')==protocol['truth_sha256']
        truths=torch.load(source/'truths.pt',map_location='cpu',weights_only=False)
        bundle=load_fm4pde_checkpoint_bundle(str(source/'weights.pth'),pde,'cuda:0',model_profile='recommended')
        base=plan['base_records'][pde]['config']
        conditions=([(layout,shared,seed) for seed in plan['layout_seeds'] for layout in ['random','fixed','grid','columns'] for shared in [False,True]]
                    if args.mode=='layouts' else [(mode,None,0) for mode in plan['temporal_modes']])
        for condition,shared,seed in conditions:
            label=f'{condition}/seed{seed}/'+('shared' if shared else 'separate') if args.mode=='layouts' else condition
            folder=args.output/pde/label;receipt=folder/'receipt.json'
            if receipt.exists():
                old=json.loads(receipt.read_text());assert digest(Path(old['result_path']))==old['result_sha256']
                assert old['plan_sha256']==env['plan_sha256'];rows.append(old);continue
            folder.mkdir(parents=True,exist_ok=True)
            c=copy.deepcopy(base)
            c.update(output_dir=str(folder),checkpoint_path=str(source/'weights.pth'),device='cuda:0',
                batch_size=1,offset=0,sample_seed=seed,mask_seed=seed,num_steps=100,
                save_plots=False,save_intermediate=False,save_per_sample_curves=True,
                ablation_name='revision_0912_full')
            gt=combine_truths(truths,[0],'cuda:0')
            masks=None
            if args.mode=='layouts':
                c.update(sensor_mode={'columns':'sensor_column'}.get(condition,condition),shared_mask=shared)
                masks=layout_masks(gt.coef.shape,condition,shared,seed)
            else:
                c['residual_mode']=condition
                if condition=='near_endpoint_temporal':
                    assert Path(c['data_path']).exists(),c['data_path']
            cfg=AblationConfig(**c);cfg.validate()
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with (folder/'run.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
                result=run_single_ablation(cfg,checkpoint_bundle=bundle,ground_truth=gt,observation_masks=masks)
            torch.cuda.synchronize();path=Path(result['run_dir'])/'result.pt'
            payload=torch.load(path,map_location='cpu',weights_only=False)
            saved_masks=torch.load(path.parent/'masks.pt',map_location='cpu',weights_only=False)
            errors={}
            for f,prefix in [('a','coef'),('u','sol')]:
                t=payload[prefix+'_ground_truth'].cpu().double();v=payload[prefix+'_final'].cpu().double()
                mask=saved_masks[prefix].cpu().double()
                assert torch.isfinite(v).all()
                errors[f]=dict(full=float((v-t).norm()/t.norm()),observed=float(((v-t)*mask).norm()/(t*mask).norm()))
            row=dict(pde=pde,condition=condition,shared=shared,seed=seed,sample_ids=[0],
                config=dataclasses.asdict(cfg),plan_sha256=env['plan_sha256'],input_protocol_sha256=digest(source/'protocol.json'),
                weights_sha256=protocol['weights_sha256'],truth_sha256=protocol['truth_sha256'],
                errors=errors,result_path=str(path),result_sha256=digest(path),mask_sha256=digest(path.parent/'masks.pt'),
                seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),status=result['status'])
            write(receipt,row);rows.append(row)
            print('DONE',pde,label,errors,'peak GiB',row['peak_bytes']/2**30,flush=True)
            assert row['peak_bytes'] < .7*torch.cuda.get_device_properties(0).total_memory
        del bundle,truths,gt,payload,saved_masks
        torch.cuda.empty_cache()
    write(args.output/'complete.json',dict(environment=env,rows=rows))


if __name__=='__main__':main()

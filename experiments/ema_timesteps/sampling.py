"""Paired physical sampling for raw and EMA on difficult and ordinary ID inputs."""
import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.optimizer_diagnostics.study import write, sha
from experiments.optimizer_diagnostics.hard_sampling import paired_infer
from experiments.aligned_sampling.input_sources import load_cell
from experiments.aligned_sampling.run_inference import (
    configure_runtime, effective_config, observation_batch, score,
    relative_differences, runtime_environment,
)
from sampling.model_io import load_fm4pde_checkpoint_bundle


def setup(root, previous, reference, pde):
    folder=root/'evaluation_inputs'/pde;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'selection.json'
    if path.exists():return
    old=previous/'hard_sampling' if pde=='nsnonbounded' else previous/'hard_sampling'/pde
    selection=json.loads((old/'selection.json').read_text())
    cell=copy.deepcopy(selection['cell'])
    hard=selection['indices']
    available=np.array(sorted(set(range(1000))-set(hard)))
    ordinary=np.random.default_rng(20260924).choice(available,size=16,replace=False).tolist()
    for key,target in [('truth_file','truth.pt'),('masks_file','masks.pt')]:
        source=Path(cell[key])
        if not source.is_absolute():source=reference/source
        assert sha(source)==cell[key.replace('_file','_sha256')]
        dest=folder/target;shutil.copyfile(source,dest)
        assert sha(dest)==sha(source)
        cell[key]=str(dest.relative_to(root))
    source=root/'inputs'/pde/'source.pth'
    if not (folder/'source.pth').exists():
        (folder/'source.pth').symlink_to(source)
    write(path,dict(pde=pde,cell=cell,hard=hard,ordinary=ordinary,indices=hard+ordinary,
        seeds=[0,1],batch_size=1 if pde=='helmholtz' else 4,
        selection_policy='Original 16 hard cases retained; 16 ordinary ID cases randomly drawn before evaluating any new checkpoint; both exploratory validation cohorts',
        previous_selection_sha256=sha(old/'selection.json'),original_checkpoint_sha256=sha(source),
        primary_metric='physical-space solution relative L2',source_count=1000))
    print('SAMPLING_CASES_FROZEN',pde,hard,ordinary,flush=True)


def sample(root,pde,variant,checkpoint,weight):
    selection=json.loads((root/'evaluation_inputs'/pde/'selection.json').read_text())
    data=load_cell(root,selection['cell'])
    ids=selection['indices'];batch=selection['batch_size']
    out=root/'evaluation'/pde/'variants'/variant;out.mkdir(parents=True,exist_ok=True)
    configure_runtime('cuda:0',False,4)
    identity=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),weight=weight,variant=variant,
        selection_sha256=sha(root/'evaluation_inputs'/pde/'selection.json'),batch_size=batch,
        environment=runtime_environment('cuda:0',False))
    if (out/'identity.json').exists():
        old=json.loads((out/'identity.json').read_text())
        assert all(old[k]==identity[k] for k in ['checkpoint_sha256','weight','selection_sha256','batch_size'])
    if (out/'complete.json').exists():
        return
    bundle=load_fm4pde_checkpoint_bundle(str(checkpoint),pde,'cuda:0',prefer_ema=weight=='ema')
    assert bundle[2]['selected_inference_weight']==weight
    identity['selected_inference_weight']=bundle[2]['selected_inference_weight']
    write(out/'identity.json',identity)
    results={}
    for seed in selection['seeds']:
        rows=[]
        for start in range(0,len(ids),batch):
            group=ids[start:start+batch]
            path=out/f'seed{seed}_batch{start:02d}.pt';receipt=path.with_suffix('.json')
            if path.exists() and receipt.exists():
                assert sha(path)==json.loads(receipt.read_text())['sha256']
                payload=torch.load(path,map_location='cpu',weights_only=False)
                assert payload['indices']==group and payload['weight']==weight
            else:
                cell=copy.deepcopy(selection['cell']);cell['config']['sample_seed']=seed
                cfg=effective_config(cell,checkpoint,'cuda:0',group,1000,out)
                gt,masks,hashes=observation_batch(data,cfg,group,'cuda:0')
                pred,meta=paired_infer(cfg,bundle,gt,masks,group)
                if variant=='original' and seed==selection['seeds'][0] and start==0:
                    repeat,_=paired_infer(cfg,bundle,gt,masks,group)
                    diff=relative_differences(pred,repeat)
                    assert diff['max_relative']<1e-6
                    write(out/'execution_gate.json',dict(passed=True,difference=diff))
                    del repeat
                payload=dict(indices=group,prediction=pred,metrics=score(pred,data,group,cfg),
                    weight=weight,input_hashes=hashes,runtime=meta,effective_config=cfg.asdict())
                if variant!='original':
                    original=torch.load(root/'evaluation'/pde/'variants/original'/path.name,
                                        map_location='cpu',weights_only=False)
                    assert original['indices']==group and original['input_hashes']==hashes
                    for key in ['initial_noise_sha256','bridge_noise_sha256']:
                        assert original['runtime'][key]==meta[key]
                    left=dict(original['effective_config']);right=cfg.asdict()
                    for d in [left,right]:
                        for k in ['checkpoint_path','output_dir']:d.pop(k,None)
                    assert left==right
                temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
                write(receipt,dict(sha256=sha(path),indices=group,seed=seed,weight=weight))
                del pred,gt,masks
            rows.extend(payload['metrics'])
            print('SAMPLE',pde,variant,seed,start+len(group),len(ids),flush=True)
        results[str(seed)]=rows
    write(out/'complete.json',dict(pde=pde,variant=variant,weight=weight,results=results,
        checkpoint_sha256=identity['checkpoint_sha256'],inputs_masks_all_noise_and_config_paired=True))
    del bundle
    gc.collect();torch.cuda.empty_cache()


def report(root,pde):
    folder=root/'evaluation'/pde
    selection=json.loads((root/'evaluation_inputs'/pde/'selection.json').read_text())
    original=json.loads((folder/'variants/original/complete.json').read_text())
    variants=[]
    for path in sorted((folder/'variants').glob('*/complete.json')):
        candidate=json.loads(path.read_text())
        if candidate['variant']=='original':continue
        result=dict(variant=candidate['variant'],weight=candidate['weight'],cohorts={})
        for cohort in ['hard','ordinary']:
            ids=selection[cohort];metrics={}
            for field in ['u'] if pde=='burger' else ['u','a']:
                def matrix(record):
                    return np.array([[next(r[f'rel_l2_{field}'] for r in record['results'][str(seed)] if r['index']==i)
                                      for i in ids] for seed in selection['seeds']])
                b=matrix(original);c=matrix(candidate)
                delta=(c-b).mean(0)
                boot=np.random.default_rng(20260924).integers(0,len(ids),size=(10000,len(ids)))
                metrics[field]=dict(original_mean=float(b.mean()),candidate_mean=float(c.mean()),
                    improvement_pct=float(100*(1-c.mean()/b.mean())),
                    improved_cases=int((c.mean(0)<b.mean(0)).sum()),
                    paired_difference_ci95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),
                    seed_improvement_pct=(100*(1-c.mean(1)/b.mean(1))).tolist())
            result['cohorts'][cohort]=metrics
        variants.append(result)
    write(folder/'summary.json',dict(pde=pde,variants=variants,seeds=selection['seeds'],
        scope='Two exploratory ID cohorts of 16 inputs each; not population/OOD inference',
        sign='positive improvement means lower physical relative L2'))


def queue(root,pde,epoch,arms):
    import subprocess,sys
    folder=root/'evaluation'/pde;folder.mkdir(parents=True,exist_ok=True)
    jobs=[('original',root/'evaluation_inputs'/pde/'source.pth','raw')]
    jobs += [(f'{arm}_e{epoch:02d}_{weight}',root/'runs'/pde/arm/f'epoch_{epoch:04d}.pth',weight)
             for arm in arms for weight in ['raw','ema']]
    for variant,checkpoint,weight in jobs:
        ready=checkpoint.with_suffix('.ready.json')
        while not checkpoint.exists() or (variant!='original' and not ready.exists()):
            write(folder/'queue.json',dict(state='waiting_checkpoint',variant=variant,checkpoint=str(checkpoint)))
            time.sleep(30)
        if variant!='original':assert sha(checkpoint)==json.loads(ready.read_text())['sha256']
        command=[sys.executable,'-u','-m','experiments.ema_timesteps.sampling','run','--root',str(root),
                 '--pde',pde,'--variant',variant,'--checkpoint',str(checkpoint),'--weight',weight]
        with (folder/f'{variant}.log').open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            write(folder/'queue.json',dict(state='running',variant=variant,child_pid=child.pid))
            status=child.wait()
        write(folder/f'{variant}.exit.json',dict(exit_code=status,child_pid=child.pid))
        if status:raise RuntimeError(f'Sampling failed: {pde}/{variant}, exit {status}')
        report(root,pde)
    write(folder/'queue.json',dict(state='complete',epoch=epoch,arms=arms))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['setup','run','queue','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--pde',required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--reference',type=Path)
    p.add_argument('--variant');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--weight',choices=['raw','ema'],default='raw')
    p.add_argument('--epoch',type=int,default=2)
    p.add_argument('--arms',nargs='*',default=['uniform','stratified_uniform','logit_normal','beta1_05'])
    a=p.parse_args()
    if a.mode=='setup':setup(a.root,a.previous,a.reference,a.pde)
    elif a.mode=='run':sample(a.root,a.pde,a.variant,a.checkpoint,a.weight)
    elif a.mode=='report':report(a.root,a.pde)
    else:queue(a.root,a.pde,a.epoch,a.arms)

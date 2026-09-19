"""Continue the paired NS arms on the full original training split to 512 updates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from data.load import PDEloader
from data.specs import get_pde_spec
from data.training_manifest import load_training_file_manifest
from data.transform import PDEStandardizer
from models.model_configs import instantiate_model
from train import _train_val_split_indices
from experiments.optimizer_diagnostics.study import (
    ROOT, write, sha, tensor_sha, restore, probe_batch, draw_batch, backward_batch,
    layer_stats, evaluate, paired, save_resume,
)


def prepare(root):
    out=root/'continuation/nsnonbounded'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'training_pool.json').exists(): return
    source=torch.load(root/'inputs/nsnonbounded/source.pth',map_location='cpu',weights_only=False,mmap=True)
    saved=source['args'] if isinstance(source['args'],dict) else vars(source['args'])
    files=load_training_file_manifest(ROOT/'configs/training_data.yaml',data_root='/large_storage/zhangxf/PDEdata',
        pde_names=['nsnonbounded'])['nsnonbounded']
    raw,_=PDEloader('nsnonbounded').load_data_files(files)
    assert len(raw)==50000
    train,val=_train_val_split_indices(50000,seed=int(saved.get('seed',0))+get_pde_spec('nsnonbounded').label_id*1009,val_ratio=.1)
    screen=torch.load(root/'inputs/nsnonbounded/data.pt',map_location='cpu',weights_only=False,mmap=True)
    assert not set(train.tolist()) & set(screen['development_ids'].tolist()+screen['confirmation_ids'].tolist())
    normalized=PDEStandardizer.from_state_dict(source['normalizer']).transform(raw[train]).contiguous()
    path=out/'full_train.pt'
    temporary=path.with_suffix('.tmp')
    torch.save(dict(train=normalized,train_ids=train),temporary);temporary.replace(path)
    write(out/'training_pool.json',dict(count=len(train),sha256=sha(path),tensor_sha256=tensor_sha(normalized),
        train_ids_sha256=tensor_sha(train),files=[dict(path=str(p),sha256=sha(p)) for p in files],
        validation_membership_preserved=True,scope='Original 45000 training inputs; original normalizer retained'))
    print('FULL_TRAINING_POOL_READY',len(train),flush=True)


def run(root,arm):
    out=root/'continuation/nsnonbounded'/arm
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'complete.json').exists(),'Refusing to overwrite a completed continuation'
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    source_path=root/'runs/nsnonbounded'/f'{arm}.pth'
    source=torch.load(source_path,map_location='cpu',weights_only=False,mmap=True)
    assert source['optimizer_diagnostics']['additional_updates']==128
    source['continuation_parent']=dict(path=str(source_path),sha256=sha(source_path),start_update=128)
    prepared=json.loads((root/'inputs/nsnonbounded/prepared.json').read_text())
    pool=root/'continuation/nsnonbounded/full_train.pt'
    assert sha(pool)==json.loads(pool.with_name('training_pool.json').read_text())['sha256']
    data=torch.load(pool,map_location='cpu',weights_only=False,mmap=True)['train']
    screen=torch.load(root/'inputs/nsnonbounded/data.pt',map_location='cpu',weights_only=False,mmap=True)
    model=instantiate_model('nsnonbounded',use_ema=False,model_config=source['model_config']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,fused=True)
    restore(model,optimizer,source)
    batch=probe_batch(model,optimizer,source,data,20,out)
    restore(model,optimizer,source)
    cfg=source['optimizer_diagnostics']['config']
    assert optimizer.param_groups[0]['lr']==1e-5
    baseline=evaluate(model,screen['development'],batch)
    write(out/'baseline_development.json',baseline)
    write(out/'protocol.json',dict(parent=source['continuation_parent'],config=cfg,
        start_update=128,final_update=512,additional_updates=384,microbatch=batch,effective_batch=64,
        training_pool_sha256=sha(pool),common_rng_seed=20260920,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        validation='Development set monitored; original confirmation set is not used to select these snapshots'))
    torch.manual_seed(20260920)
    gen=torch.Generator().manual_seed(20260920)
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(4919)).tolist()
    assert len(order)>384*64
    records=[];snapshots=[]
    started=time.monotonic()
    for step in range(1,385):
        model.train();optimizer.zero_grad(set_to_none=True)
        xt,t,target=draw_batch(data,order[(step-1)*64:step*64],gen,torch.device('cuda:0'))
        loss=backward_batch(model,xt,t,target,batch)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
        diagnose=step in [1,128,256,384]
        before={n:p.detach().clone() for n,p in model.named_parameters()} if diagnose else None
        optimizer.step()
        row=dict(step=step,total_updates=128+step,loss=loss,grad_norm=float(norm),
            lr=optimizer.param_groups[0]['lr'],betas=list(optimizer.param_groups[0]['betas']),elapsed=time.monotonic()-started)
        if diagnose:
            stats=layer_stats(model,optimizer,before);row.update(stats['summary'])
            write(out/f'layers_step_{128+step:04d}.json',stats)
            del before
        records.append(row)
        if step%16==0 or step==1:
            write(out/'progress.json',row)
            print('CONTINUATION',arm,128+step,'loss',loss,flush=True)
        del xt,t,target
        if step in [128,384]:
            val=evaluate(model,screen['development'],batch)
            path=out/f'step_{128+step:04d}.pth'
            save_resume(path,source,model,optimizer,cfg,128+step,prepared)
            record=dict(total_updates=128+step,development=val,paired_start=paired(val,baseline),
                checkpoint=str(path),sha256=sha(path))
            write(out/f'step_{128+step:04d}.json',record);snapshots.append(record)
    write(out/'complete.json',dict(arm=arm,training=records,snapshots=snapshots,additional_updates=384,
        total_updates=512,seconds=time.monotonic()-started))
    print('CONTINUATION_COMPLETE',arm,flush=True)


def queue(root):
    out=root/'continuation/nsnonbounded';out.mkdir(parents=True,exist_ok=True)
    stages=[('prepare',None),('run','selected_resume'),('run','lr_control')]
    for mode,arm in stages:
        name=arm or mode
        cmd=[sys.executable,'-u','-m','experiments.optimizer_diagnostics.extend_ns',mode,'--root',str(root)]
        if arm: cmd+=['--arm',arm]
        with (out/f'{name}.log').open('a') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            write(out/'queue.json',dict(state='running',stage=name,child_pid=child.pid,time=time.time()))
            code=child.wait()
        write(out/f'{name}.exit.json',dict(exit_code=code,child_pid=child.pid,time=time.time()))
        if code: raise RuntimeError(f'Continuation failed: {name}: {code}')
    write(out/'queue.json',dict(state='complete',time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','run','queue'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=['selected_resume','lr_control'])
    args=p.parse_args();torch.set_num_threads(4)
    {'prepare':lambda:prepare(args.root),'run':lambda:run(args.root,args.arm),'queue':lambda:queue(args.root)}[args.mode]()

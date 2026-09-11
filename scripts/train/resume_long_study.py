"""Fixed-length continuation of the exact main-experiment model and Adam state.

Epoch-boundary recovery preserves optimizer/scheduler state. Test examples never
enter training or checkpoint selection; a physical validation cache is retained
for the separate conditional-sampling selector.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch

from data.specs import get_pde_spec
from data.training_manifest import load_training_file_manifest
from data.transform import PDEStandardizer
from models.legacy_checkpoint import read_checkpoint
from models.model_configs import instantiate_model
from train import _load_training_and_validation_data, _train_val_split_indices, _slice_loader_metadata
from scripts.train.resume_study import (
    ROOT, cpu_tree, evaluate, file_sha, optimizer_steps, probe_batch, restore,
    save_checkpoint, train_group, write,
)


def assert_same_inference(source, inference):
    """The resume state must be the actual main-experiment inference weights."""
    assert source['model_config'] == inference['model_config']
    assert source['model_for_resume'].keys() == inference['model'].keys()
    assert all(torch.equal(v, inference['model'][k]) for k, v in source['model_for_resume'].items())
    a, b = source['normalizer'], inference['normalizer']
    assert a.keys() == b.keys()
    for k, v in a.items():
        assert torch.equal(v, b[k]) if torch.is_tensor(v) else v == b[k], k


def protocol_arguments(args):
    values = dict(vars(args))
    if not values.get('checkpoint_spool'):
        values.pop('checkpoint_spool', None)
    return values


def main(args):
    out = Path(args.output).resolve()
    assert out.is_absolute() and '/outputs/pretrained/' in str(out)
    out.mkdir(parents=True, exist_ok=True)
    assert args.epochs >= 20 and args.lr > args.min_lr == 1e-6
    assert args.save_every > 0
    if (out/'training_complete.json').exists():
        raise RuntimeError('Training is already complete; use a new output directory')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    assert torch.cuda.mem_get_info()[0] > 70 * 2**30
    started = time.monotonic()
    source = read_checkpoint(args.checkpoint, args.pde)
    inference = read_checkpoint(args.inference_checkpoint, args.pde)
    assert_same_inference(source, inference)
    del inference
    assert source['checkpoint_schema_version'] == 3 and source.get('optimizer')
    assert not source.get('use_ema')
    saved_args = source['args'] if isinstance(source['args'], dict) else vars(source['args'])
    assert not saved_args.get('skewed_timesteps') and not saved_args.get('edm_schedule')
    assert not source['model_config'].get('scalar_conditioning')
    source_sha = file_sha(args.checkpoint)
    protocol = dict(arguments=protocol_arguments(args), source_sha256=source_sha,
        inference_checkpoint_sha256=file_sha(args.inference_checkpoint),
        main_inference_weights_and_normalizer_equal=True, source_epoch=int(source['epoch']),
        source_model_config=source['model_config'], effective_batch_size=64,
        planned_additional_epochs=args.epochs, planned_updates=args.epochs*704,
        objective='native uniform-time conditional flow matching velocity MSE; no extra coarse loss',
        precision='FP32 parameters/forward/backward/Adam; TF32 enabled',
        lr_policy='restore Adam moments/counters; 128-update warmup, then epoch cosine to 1e-6',
        checkpoint_policy='last every epoch; immutable full resume checkpoint every 5 epochs; best FM separately',
        sampling_selection_policy='separate physical training-validation inputs; historical hard test cases diagnostic only',
        source_seed=int(saved_args.get('seed', 0)), host=socket.gethostname(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), started_unix=time.time())
    if (out/'protocol.json').exists():
        previous = json.loads((out/'protocol.json').read_text())
        assert previous['arguments'] == protocol['arguments'] and previous['source_sha256'] == source_sha
    else:
        write(out/'protocol.json', protocol)
    write(out/'process.json', dict(pid=os.getpid(), host=socket.gethostname(), started_unix=time.time()))
    publisher = None
    save_model_checkpoint = save_checkpoint
    if getattr(args, 'checkpoint_spool', None):
        from scripts.train.checkpoint_publisher import CheckpointPublisher
        publisher = CheckpointPublisher(out, args.checkpoint_spool, source_sha)
        save_model_checkpoint = publisher.save_checkpoint
    manifest = load_training_file_manifest(ROOT/'configs/training_data.yaml',
                                          data_root=args.data_root, pde_names=[args.pde])
    assert all('test' not in str(p).lower() for p in manifest[args.pde])
    print('LOAD_DATA', args.pde, flush=True)
    raw, _, _, raw_val, _, val_meta, split = _load_training_and_validation_data(
        [args.pde], args.data_root, data_size=5, seed=int(saved_args.get('seed',0)),
        train_files_by_pde=manifest)
    assert len(raw)==45000 and len(raw_val)==5000
    train_idx, val_idx = _train_val_split_indices(50000,
        seed=int(saved_args.get('seed',0))+int(get_pde_spec(args.pde).label_id)*1009, val_ratio=.1)
    assert not set(train_idx.tolist()) & set(val_idx.tolist())
    permutation = torch.randperm(5000, generator=torch.Generator().manual_seed(args.seed+81))
    development, confirmation = permutation[:512], permutation[512:]
    if not (out/'split_indices.pt').exists():
        torch.save(dict(train_indices=train_idx, val_indices=val_idx,
            development_positions=development, confirmation_positions=confirmation), out/'split_indices.pt')
        sampling_positions = development[:32]
        torch.save(dict(pde=args.pde, fields=raw_val[sampling_positions].clone(),
            validation_positions=sampling_positions, original_training_file_pool_ids=val_idx[sampling_positions],
            loader_metadata={args.pde:_slice_loader_metadata(val_meta.get(args.pde,{}),sampling_positions)},
            source='training-pool validation partition; excluded from all continuation updates'),
            out/'sampling_validation_inputs.pt')
    normalizer = PDEStandardizer.from_state_dict(source['normalizer'])
    train, val = normalizer.transform(raw).contiguous(), normalizer.transform(raw_val).contiguous()
    write(out/'data_check.json', dict(split=split, normalizer_unchanged=True,
        files=[dict(path=str(p),bytes=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in manifest[args.pde]],
        train_ids_sha256=hashlib.sha256(train_idx.numpy().tobytes()).hexdigest(),
        val_ids_sha256=hashlib.sha256(val_idx.numpy().tobytes()).hexdigest(),
        sampling_validation_ids=val_idx[development[:32]].tolist(),
        historical_validation_caveat='Older June checkpoints may have seen validation inputs during earlier pretraining; current continuation excludes them. NS 260904 was trained from scratch with the same split.'))
    del raw, raw_val, val_meta
    gc.collect()
    model = instantiate_model(args.pde,use_ema=False,model_config=source['model_config']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,fused=True)
    restore(model,optimizer,source,args.lr)
    old_steps = optimizer_steps(optimizer)
    assert len(set(old_steps))==1
    source_step = old_steps[0]
    if (out/'resume_verification.json').exists():
        microbatch = json.loads((out/'resume_verification.json').read_text())['microbatch']
    else:
        microbatch = probe_batch(model,optimizer,source,tuple(train.shape[1:]),out,amp=False)
        assert optimizer_steps(optimizer)==old_steps
        assert all(torch.equal(v.cpu(),source['model_for_resume'][k]) for k,v in model.state_dict().items())
        write(out/'resume_verification.json',dict(source_optimizer_step=source_step,
            optimizer_states=len(old_steps),model_restored_after_probe=True,adam_restored_after_probe=True,
            source_sha256=source_sha,microbatch=microbatch,effective_batch_size=64))
    restore(model,optimizer,source,args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.min_lr)
    if not (out/'baseline_development.json').exists():
        write(out/'baseline_development.json',evaluate(model,val,development,args.seed+9001,repeats=2,batch_size=microbatch))
        write(out/'baseline_confirmation.json',evaluate(model,val,confirmation,args.seed+19001,batch_size=microbatch))
    history=[]
    best_score=math.inf
    recovery_path = (publisher.latest_local_checkpoint() if publisher else None) or out/'last_resume.pth'
    if recovery_path.exists():
        saved=read_checkpoint(recovery_path,args.pde)
        restore(model,optimizer,saved,saved['optimizer']['param_groups'][0]['lr'])
        scheduler.load_state_dict(saved['lr_schedule'])
        metadata=saved['resume_study']
        assert metadata['source_sha256']==source_sha and metadata['planned_epochs']==args.epochs
        history=metadata['history']
        best_score=min(row['validation_mse'] for row in history)
        assert set(optimizer_steps(optimizer))=={source_step+704*len(history)}
        del saved
    for additional in range(len(history)+1,args.epochs+1):
        if publisher: publisher.check()
        epoch=int(source['epoch'])+additional
        torch.manual_seed(args.seed+epoch)
        model.train()
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train),batch_size=64,
            sampler=order,num_workers=4,pin_memory=True,drop_last=False)
        total_loss,total_grad,count=0.,0.,0
        begin=time.monotonic()
        lr=optimizer.param_groups[0]['lr']
        for i,(samples,) in enumerate(loader):
            if additional==1:
                for group in optimizer.param_groups:
                    group['lr']=args.lr*(.2+.8*min((i+1)/128,1.))
            loss,grad=train_group(model,optimizer,samples,microbatch,0.,1.,amp=False)
            total_loss+=loss*len(samples);total_grad+=grad;count+=len(samples)
            if i%100==0:
                if publisher: publisher.check()
                print('TRAIN',args.pde,'additional_epoch',additional,'/',args.epochs,'batch',i,'/',len(loader),
                    'loss',loss,'lr',optimizer.param_groups[0]['lr'],flush=True)
        assert count==45000 and len(loader)==704
        assert set(optimizer_steps(optimizer))=={source_step+704*additional}
        scheduler.step()
        score=evaluate(model,val,development,args.seed+9001,repeats=2,batch_size=microbatch)
        row=dict(additional_epoch=additional,epoch=epoch,updates=704*additional,learning_rate=lr,
            next_learning_rate=optimizer.param_groups[0]['lr'],train_mse=total_loss/count,
            mean_gradient_norm=total_grad/704,validation_mse=score['mean'],validation_coarse=score['coarse_mean'],
            seconds=time.monotonic()-begin,peak_bytes=torch.cuda.max_memory_allocated(),samples=count)
        history.append(row)
        metadata=dict(pde=args.pde,source_checkpoint=str(Path(args.checkpoint).resolve()),source_sha256=source_sha,
            source_epoch=int(source['epoch']),source_optimizer_step=source_step,updates=704*additional,
            learning_rate=args.lr,batch_size=microbatch,effective_batch_size=64,training_dtype='float32',
            planned_epochs=args.epochs,completed_epochs=additional,history=history,
            validation_score=score['mean'],scheduler_step_unit='epoch',coarse_weight=0.)
        write(out/'validation'/f'epoch_{additional:03d}.json',score)
        if score['mean']<best_score:
            save_model_checkpoint(out/'best_fm_resume.pth',source,model,optimizer,scheduler,epoch,metadata)
            best_score=score['mean']
            write(out/'best_fm_selection.json',row)
        if additional%args.save_every==0 or additional==args.epochs:
            save_model_checkpoint(out/'checkpoints'/f'resume_epoch_{additional:03d}.pth',source,model,optimizer,scheduler,epoch,metadata)
        save_model_checkpoint(out/'last_resume.pth',source,model,optimizer,scheduler,epoch,metadata)
        write(out/'history.json',history)
        progress=dict(status='training',**row,planned_epochs=args.epochs,pid=os.getpid())
        if publisher: progress['checkpoint_publication']=publisher.state()
        write(out/'progress.json',progress)
        print('EPOCH_COMPLETE',args.pde,json.dumps(row),flush=True)
    assert len(history)==args.epochs
    write(out/'last_confirmation.json',evaluate(model,val,confirmation,args.seed+19001,batch_size=microbatch))
    if publisher:
        write(out/'progress.json',dict(status='publishing_checkpoints',additional_epoch=len(history),
              updates=704*len(history),planned_epochs=args.epochs,pid=os.getpid(),publication=publisher.state()))
        del model, optimizer, scheduler
        gc.collect(); torch.cuda.empty_cache()
        publisher.finish()
    assert file_sha(args.checkpoint)==source_sha
    write(out/'training_complete.json',dict(status='complete',completed_epochs=len(history),
        updates=704*len(history),source_sha256=source_sha,last_sha256=file_sha(out/'last_resume.pth'),
        checkpoint_candidates=[str(p) for p in sorted((out/'checkpoints').glob('*.pth'))],
        best_fm_checkpoint=str(out/'best_fm_resume.pth'),seconds_this_process=time.monotonic()-started,
        sampling_evaluation='pending validation selection and full main1000 evaluation'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pde',required=True,choices=['poisson','nsnonbounded','darcy','helmholtz','burger'])
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--inference-checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--data-root',default='/large_storage/zhangxf/PDEdata')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--save-every',type=int,default=5)
    p.add_argument('--lr',type=float,required=True)
    p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--seed',type=int,default=20260911)
    p.add_argument('--checkpoint-spool',help='Temporary local recovery cache; all versions publish before completion')
    main(p.parse_args())

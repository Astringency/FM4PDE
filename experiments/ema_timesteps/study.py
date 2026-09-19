"""Full-pool continuation with paired raw/EMA evaluation and resumable epochs."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import socket
import subprocess
import time

import torch

from models.ema import EMA
from models.model_configs import instantiate_model
from training.load_and_save import _load_resume_state, _load_optimizer_state_preserving_runtime_options
from training.timesteps import sample_timesteps
from experiments.optimizer_diagnostics.study import ROOT, cpu_tree, sha, tensor_sha, write, paired
from experiments.ema_timesteps.evaluate import fixed_metrics, evaluate_pair

ARMS = ('uniform', 'stratified_uniform', 'logit_normal', 'beta1_05')
LR = dict(poisson=3e-6, helmholtz=1e-6, darcy=1e-6, nsnonbounded=3e-6, burger=1e-6)
SEED = 20260923
BATCH = 64


def runtime():
    return dict(host=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), tf32=True, precision='float32',
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())


def initialize(root, pde):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    folder = root/'inputs'/pde
    record = json.loads((folder/'prepared.json').read_text())
    assert sha(folder/'source.pth') == record['source_checkpoint_sha256']
    source = torch.load(folder/'source.pth', map_location='cpu', weights_only=False, mmap=True)
    data = torch.load(folder/'full_data.pt', map_location='cpu', weights_only=False, mmap=True)
    assert len(data['train']) == record['counts']['train'] == 45000
    assert tensor_sha(data['train_ids']) == record['tensor_sha256']['train_ids']
    model = EMA(instantiate_model(pde, use_ema=False, model_config=source['model_config']),
                decay=.999, warmup=False).cuda()
    _load_resume_state(model, source, add_ema=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), fused=True)
    _load_optimizer_state_preserving_runtime_options(optimizer, source['optimizer'])
    return source, data, record, model, optimizer


def draw(data, ids, step, config):
    sample = data[ids].cuda()
    # Separate streams keep noise and dropout matched when time sampling changes.
    noise_gen = torch.Generator().manual_seed(SEED + 3 * step)
    time_gen = torch.Generator().manual_seed(SEED + 3 * step + 1)
    noise = torch.randn(sample.shape, generator=noise_gen).cuda()
    t = sample_timesteps(len(sample), 'cpu', config['time_sampling'], generator=time_gen,
                         logit_mean=config.get('logit_mean', 0.0), logit_std=config.get('logit_std', 1.0)).cuda()
    return noise.lerp(sample, t[:,None,None,None]), t, sample-noise


def backward(model, xt, t, target, microbatch):
    total = 0.0
    for start in range(0, len(xt), microbatch):
        end = min(start+microbatch, len(xt))
        error = model(xt[start:end], t[start:end], extra={}) - target[start:end]
        loss = error.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        (loss*((end-start)/len(xt))).backward()
        total += float(loss.detach())*((end-start)/len(xt))
    return total


def profile(root, pde):
    out = root/'profiles'/pde
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        print('PROFILE_ALREADY_COMPLETE',pde,flush=True)
        return
    source,data,record,model,opt = initialize(root,pde)
    assert sha(root/'inputs'/pde/'full_data.pt') == record['data_sha256']
    free,total = torch.cuda.mem_get_info()
    budget = min(50*2**30, .75*free)
    config = dict(time_sampling='uniform')
    xt,t,target=draw(data['train'],list(range(BATCH)),0,config)
    probes=[]; chosen=4
    # Profile gradient memory without mutating weights or Adam moments.
    for micro in (4,8,16,32,64):
        if probes:
            base=probes[-1]['base_bytes']; prev=probes[-1]
            estimate=base+(prev['peak_bytes']-base)*micro/prev['microbatch']
            if estimate > budget:
                break
        opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        base=torch.cuda.memory_allocated();start=time.monotonic()
        backward(model,xt,t,target,micro)
        torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()
        probes.append(dict(microbatch=micro,base_bytes=base,peak_bytes=peak,seconds=time.monotonic()-start))
        assert peak < budget, 'Measured profile exceeded conservative budget'
        chosen=micro
    opt.zero_grad(set_to_none=True)
    del xt,t,target
    gc.collect();torch.cuda.empty_cache()
    _load_resume_state(model,source,add_ema=True)
    metrics=evaluate_pair(model,data['development'],chosen)
    assert metrics['raw']==metrics['ema'], 'New EMA must equal source weights before first update'
    write(out/'baseline_development.json',metrics['raw'])
    write(out/'complete.json',dict(pde=pde,microbatch=chosen,probes=probes,budget_bytes=budget,
        original_raw_equals_initial_ema=True,environment=runtime(),input_metadata=record,
        measured_before_timestep_plan=True))
    print('PROFILE_COMPLETE',pde,'microbatch',chosen,'bins',metrics['raw']['by_time_bin'],flush=True)


def configuration(root,pde,arm):
    plan=json.loads((root/'time_sampling_plan.json').read_text())
    assert plan['profiles_reviewed_before_training']
    p=plan['pdes'][pde]
    config=dict(arm=arm,lr=LR[pde],betas=[.5,.999] if arm=='beta1_05' else [.9,.999],
                time_sampling='uniform' if arm=='beta1_05' else arm,
                logit_mean=p['logit_mean'],logit_std=p['logit_std'],
                ema_decay=.999,ema_warmup=False,effective_batch=BATCH,warmup_updates=128)
    return config


def checkpoint_payload(source,model,opt,epoch,steps,config,record,microbatch):
    assert model.training and model.model.training
    # Share the CPU tensor objects between inference and resume dictionaries.
    resume=cpu_tree(model.state_dict())
    raw={k.removeprefix('model.'):v for k,v in resume.items() if k.startswith('model.')}
    ema=dict(raw)
    for i,(name,param) in enumerate((x for x in model.model.named_parameters() if x[1].requires_grad)):
        ema[name]=resume[f'shadow_params.{i}']
    result=dict(source)
    result.update(model=raw,model_ema=ema,model_for_resume=resume,optimizer=cpu_tree(opt.state_dict()),
                  has_ema=True,use_ema=True,inference_weight='ema',ema_decay=model.decay,ema_warmup=model.warmup,
                  epoch=int(source['epoch'])+epoch,lr_schedule=None,lr_scheduler='constant',resolved_lr_scheduler='constant')
    result['ema_timesteps']=dict(config=config,completed_epochs=epoch,additional_updates=steps,
        source_epoch=int(source['epoch']),source_optimizer_steps=record['source_optimizer_steps'],
        input_data_sha256=record['data_sha256'],source_checkpoint_sha256=record['source_checkpoint_sha256'],
        microbatch=microbatch,environment=runtime(),dropped_inputs_per_epoch=8,
        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state())
    return result


def run(root,pde,arm,epochs):
    out=root/'runs'/pde/arm;out.mkdir(parents=True,exist_ok=True)
    source,data,record,model,opt=initialize(root,pde)
    config=configuration(root,pde,arm)
    profile_record=json.loads((root/'profiles'/pde/'complete.json').read_text())
    micro=profile_record['microbatch']
    assert profile_record['environment']['gpu']==runtime()['gpu']
    steps_per_epoch=len(data['train'])//BATCH
    assert steps_per_epoch==703
    base=json.loads((root/'profiles'/pde/'baseline_development.json').read_text())
    start_epoch=0
    last=out/'last.pth'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False,mmap=True)
        meta=saved['ema_timesteps']
        assert meta['config']==config and meta['microbatch']==micro
        assert meta['input_data_sha256']==record['data_sha256']
        model.load_state_dict(saved['model_for_resume'],strict=True)
        _load_optimizer_state_preserving_runtime_options(opt,saved['optimizer'])
        torch.set_rng_state(meta['rng_cpu']);torch.cuda.set_rng_state(meta['rng_cuda'])
        start_epoch=int(meta['completed_epochs'])
        assert int(model.num_updates)==start_epoch*steps_per_epoch
        del saved
        if start_epoch in (2,5,10):
            snapshot=out/f'epoch_{start_epoch:04d}.pth'
            if not snapshot.exists():
                snapshot.hardlink_to(last)
            ready=snapshot.with_suffix('.ready.json')
            if not ready.exists():
                assert sha(snapshot)==sha(last)
                write(ready,dict(checkpoint=str(snapshot),sha256=sha(snapshot),pde=pde,arm=arm,
                                epoch=start_epoch,steps=start_epoch*steps_per_epoch))
    else:
        torch.manual_seed(SEED)
        write(out/'protocol.json',dict(pde=pde,config=config,input_metadata=record,
            environment=runtime(),microbatch=micro,steps_per_epoch=steps_per_epoch,
            initializer='original weights + original Adam moments/step; EMA initialized from raw',
            validation='fixed uniform ten-bin development; confirmation reserved until branch choices are frozen',
            random_policy='matched data/noise/dropout streams; time RNG separate; fixed profile microbatch',
            source_precision_caveat='Original NS BF16; all continuation arms use FP32/TF32'))
    if start_epoch>=epochs:
        complete=out/f'complete_{epochs:02d}.json'
        if not complete.exists():
            write(complete,dict(pde=pde,arm=arm,epochs=start_epoch,steps=start_epoch*steps_per_epoch,
                                latest_checkpoint=str(last),latest_sha256=sha(last),
                                ema_updates=int(model.num_updates),recovered_after_checkpoint_save=True))
        print('ALREADY_AT_REQUESTED_EPOCH',pde,arm,start_epoch,flush=True)
        return
    for group in opt.param_groups:
        group['betas']=tuple(config['betas'])
    started=time.monotonic()
    for epoch in range(start_epoch+1,epochs+1):
        order=torch.randperm(len(data['train']),generator=torch.Generator().manual_seed(SEED+epoch+10000))
        rows=[];time_hist=torch.zeros(10,dtype=torch.long)
        for local_step in range(steps_per_epoch):
            step=(epoch-1)*steps_per_epoch+local_step+1
            ids=order[local_step*BATCH:(local_step+1)*BATCH]
            model.train(True);opt.zero_grad(set_to_none=True)
            xt,t,target=draw(data['train'],ids,step,config)
            loss=backward(model,xt,t,target,micro)
            lr=config['lr']*min(1.0,.1+.9*step/config['warmup_updates'])
            for group in opt.param_groups:group['lr']=lr;group['initial_lr']=config['lr']
            diagnose=local_step%64==0 or local_step==steps_per_epoch-1
            grad=None;before=None
            if diagnose:
                grad=float(torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),float('inf'),error_if_nonfinite=True))
                before=next(model.model.parameters()).detach().clone()
            opt.step();model.update_ema()
            time_hist+=torch.bincount((t.detach().cpu()*10).long().clamp_max(9),minlength=10)
            row=dict(step=step,epoch=epoch,loss=loss,lr=lr,grad_norm=grad,elapsed_seconds=time.monotonic()-started)
            if before is not None:
                param=next(model.model.parameters()).detach()
                row['first_parameter_relative_update']=float((param-before).norm()/before.norm().clamp_min(1e-12))
                row['input_ids_sha256']=tensor_sha(data['train_ids'][ids])
                row['target_sha256']=tensor_sha(target)
            rows.append(row)
            if diagnose:
                write(out/'progress.json',dict(state='training',**row,arm=arm,pde=pde,ema_updates=int(model.num_updates)))
                print('UPDATE',pde,arm,step,'epoch',epoch,'loss',loss,'grad',grad,flush=True)
            del xt,t,target,before
        with (out/'training.jsonl').open('a') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        metrics=evaluate_pair(model,data['development'],micro)
        for name in ('raw','ema'):
            metrics[f'{name}_vs_original']=paired(metrics[name],base)
        metrics['ema_vs_raw']=paired(metrics['ema'],metrics['raw'])
        metrics.update(epoch=epoch,steps=epoch*steps_per_epoch,training_time_histogram=time_hist.tolist(),
                       training_loss_mean=sum(r['loss'] for r in rows)/len(rows))
        write(out/f'epoch_{epoch:04d}.json',metrics)
        payload=checkpoint_payload(source,model,opt,epoch,epoch*steps_per_epoch,config,record,micro)
        tmp=out/'last.tmp';torch.save(payload,tmp);tmp.replace(last)
        del payload
        if epoch in (2,5,10):
            snapshot=out/f'epoch_{epoch:04d}.pth'
            if snapshot.exists():
                raise RuntimeError(f'Refusing to overwrite snapshot {snapshot}')
            snapshot.hardlink_to(last)
            write(snapshot.with_suffix('.ready.json'),dict(checkpoint=str(snapshot),sha256=sha(snapshot),
                pde=pde,arm=arm,epoch=epoch,steps=epoch*steps_per_epoch))
        write(out/'progress.json',dict(state='epoch_complete',pde=pde,arm=arm,epoch=epoch,
            steps=epoch*steps_per_epoch,raw_relative_change_pct=metrics['raw_vs_original']['relative_change_pct'],
            ema_relative_change_pct=metrics['ema_vs_original']['relative_change_pct']))
        print('EPOCH_COMPLETE',pde,arm,epoch,metrics['raw_vs_original'],metrics['ema_vs_original'],flush=True)
    write(out/f'complete_{epochs:02d}.json',dict(pde=pde,arm=arm,epochs=epochs,steps=epochs*steps_per_epoch,
        latest_checkpoint=str(last),latest_sha256=sha(last),ema_updates=int(model.num_updates),seconds=time.monotonic()-started))


def queue(root,pde,epochs,arms):
    folder=root/'queues'/pde;folder.mkdir(parents=True,exist_ok=True)
    import sys
    for arm in arms:
        cmd=[sys.executable,'-u','-m','experiments.ema_timesteps.study','run','--root',str(root),
             '--pde',pde,'--arm',arm,'--epochs',str(epochs)]
        with (folder/f'{arm}_{epochs:02d}.log').open('a') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            write(folder/'progress.json',dict(state='running',arm=arm,child_pid=child.pid,epochs=epochs))
            code=child.wait()
        write(folder/f'{arm}_{epochs:02d}.exit.json',dict(exit_code=code,child_pid=child.pid))
        if code:raise RuntimeError(f'Training failed: {pde}/{arm}, exit {code}')
    write(folder/'progress.json',dict(state='complete',epochs=epochs,arms=arms))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['profile','run','queue'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pde',choices=list(LR),required=True)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--epochs',type=int,default=2)
    a=p.parse_args()
    if a.mode=='profile':profile(a.root,a.pde)
    elif a.mode=='run':run(a.root,a.pde,a.arm,a.epochs)
    else:queue(a.root,a.pde,a.epochs,a.arms)

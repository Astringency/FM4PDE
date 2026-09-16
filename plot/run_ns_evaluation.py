"""Evaluate Navier–Stokes endpoint reconstruction and controlled sampling latency.

Formal targets are used only by score(); infer() receives masked observations.
Canonical Gaussian rows keep each stochastic path invariant to batch partition.
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
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'plot')]
DEV = [251, 878, 197, 364]
EVAL = list(range(2000, 3000))
SETTINGS = {'full_forward':('forward',16384), 'full_inverse':('inverse',16384),
            'sparse_forward':('forward',500), 'sparse_inverse':('inverse',500),
            'sparse_joint':('both',500)}


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def prepare(a):
    import h5py
    import numpy as np
    import torch
    from sampling.config import load_config
    from models.legacy_checkpoint import read_checkpoint
    from run_ns_checkpoint_comparison import inference_signature, simple
    assert not (a.inputs/'protocol.json').exists()
    a.inputs.mkdir(parents=True,exist_ok=True)
    ids=sorted(DEV+EVAL)
    config={name:load_config(ROOT/f'configs/main/{task}/nsnonbounded.yaml').asdict()
            for name,(task,_) in SETTINGS.items()}
    for name,(_,n) in SETTINGS.items():
        config[name].update(num_obs=n,model_profile='auto',num_steps=100,
                            initial_noise_source_batch_size=1000)
    sources={}
    for dist in ['id','smooth','rough']:
        source=Path(config['sparse_joint']['data_paths'][dist])
        with h5py.File(source,'r') as f:
            assert f['w0'].shape==(10000,128,128) and f['w'].shape==(10000,128,128,10)
            attrs={k:simple(v.item() if isinstance(v,np.generic) else v) for k,v in f.attrs.items()}
            assert float(attrs['T'])==1. and float(attrs['viscosity'])==.001
            fields=np.stack([f['w0'][ids],f['w'][ids,:,:,-1]],axis=1).astype('float32')
            assert np.isfinite(fields).all()
            np.savez(a.inputs/f'{dist}.npz',ids=ids,fields=fields)
            sources[dist]=dict(path=str(source),size=source.stat().st_size,
                mtime_ns=source.stat().st_mtime_ns,attrs=attrs,t=f['t'][:].tolist(),
                cache_sha256=digest(a.inputs/f'{dist}.npz'))
        print('PREPARED',dist,flush=True)
    shutil.copy2(a.weights,a.inputs/'weights.pth')
    payload=read_checkpoint(a.inputs/'weights.pth','nsnonbounded')
    keys=['epoch','model_profile','model_config','model_config_metadata','normalizer','data_metadata','slimmed_from']
    model={k:simple(payload.get(k)) for k in keys}
    model.update(inference_signature=inference_signature(payload),weights_sha256=digest(a.inputs/'weights.pth'),
                 parameter_count=sum(v.numel() for v in payload['model'].values()))
    assert model['model_profile']=='light'
    protocol=dict(version=1,development_ids=DEV,evaluation_ids=EVAL,configs=config,
        sources=sources,model=model,steps=100,nfe=100,seed=0,mask_seed=0,
        random_source='1000-row Gaussian pool regenerated with seed 0 at every call; retain each evaluation row offset-2000 at initialization and every stochastic update; development rows 0..3',
        residual='current mean squared endpoint-secant residual; public nu=0.001,T=1 only',
        pde_params=dict(nu=.001,T=1.,solver_dt=.0001),
        selection_rule='For full and sparse inverse separately, compare existing weights and observation x10 / PDE x1000 on the four Smooth development IDs, seed 0. Choose lower mean relative L2 initial-field error. No other candidates; no evaluation targets used.',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(a.inputs/'protocol.json',protocol)


def configuration(protocol,setting,dist,weights,scaled=False):
    from sampling.config import AblationConfig
    c=copy.deepcopy(protocol['configs'][setting])
    c.update(device='cuda:0',dtype='float32',model_profile='auto',checkpoint_path=str(weights),
        data_path=protocol['sources'][dist]['path'],test_type=dist,save_plots=False,
        save_intermediate=False,save_per_sample_curves=False,empty_cache_each_step=False,
        allow_synthetic_data=False,initial_noise_source_batch_size=1000)
    if scaled:
        assert c['task']=='inverse'
        c['zeta_obs_u']*=10;c['zeta_pde']*=1000
    cfg=AblationConfig(**c);cfg.validate()
    assert cfg.num_steps==100 and cfg.residual_mode=='endpoint_secant'
    assert cfg.gradient_target=='current_state_chain_rule' and cfg.noise_level==0
    assert cfg.clip_mode=='global_norm' and cfg.pde_guidance_start_ratio==.8
    return cfg


def observations(fields,ids,cfg,params,device='cuda:0'):
    import torch
    from sampling.masks import make_pair_masks
    from sampling.data import PDEGroundTruth
    x=fields.to(device)
    masks=make_pair_masks(x[:,0:1].shape,x[:,1:2].shape,cfg.num_obs,cfg.sensor_mode,
                          cfg.shared_mask,cfg.mask_seed,device=device)
    if cfg.task=='forward':masks.sol.zero_()
    if cfg.task=='inverse':masks.coef.zero_()
    aa,uu=x[:,0:1]*masks.coef,x[:,1:2]*masks.sol
    public={k:torch.full((len(ids),),v,device=device) for k,v in params.items()}
    gt=PDEGroundTruth('nsnonbounded',aa,uu,torch.cat([aa,uu],1),public,['w0'],['wT'],
         dict(sample_ids=[str(i) for i in ids],sample_offsets=ids,offset=ids[0],batch_size=len(ids),
              synthetic=False,endpoint_pair=True,data_path=cfg.data_path))
    return gt,masks


def infer(cfg,bundle,gt,masks,indices,*,steps=100,fused=True):
    import torch
    import sampling.runner as r
    from sampling.guidance import _clip_per_sample
    from types import SimpleNamespace
    c=copy.deepcopy(cfg);c.batch_size=len(indices);c.initial_noise_source_indices=list(indices)
    c=r.finalize_ground_truth_config(c);c.validate();r._disable_unreliable_pde_guidance(c)
    r._set_seed(c.sample_seed)
    net,normalizer,payload=bundle
    scalar,_=r._scalar_conditioning_for_sampling(checkpoint_payload=payload,gt=gt,config=c,device='cuda:0')
    classes,_=r._class_conditioning_for_sampling(checkpoint_payload=payload,pde=c.pde,batch_size=len(indices),device='cuda:0',cfg_scale=c.cfg_scale)
    extra={**classes,**(scalar or {})} or None
    r._check_sampling_channels(gt,normalizer,payload)
    assert c.obs_l2_reference_mse_zeta_a is None and c.obs_l2_reference_mse_zeta_u is None
    count=[0]
    def hook(*_):count[0]+=1
    handle=net.model.register_forward_hook(hook)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    grid=r.make_time_grid(c.time_grid,c.num_steps,device='cuda:0',eta=c.time_grid_eta)
    x=r._sample_initial_noise(c,gt,'cuda:0')
    for step in range(steps):
        cur=x.detach().clone().requires_grad_(True);t,tn=grid[step],grid[step+1]
        out=r.sampler_step(net,cur,t,tn,c.sampler_phase,c.step_method,c.loss_state,device='cuda:0',
            model_extra=extra,stochastic_noise_source_batch_size=1000,stochastic_noise_source_indices=list(indices))
        physical=r._physical_from_model_state(out.x_loss_state,c,normalizer)
        losses=r.compute_guidance_losses(physical,gt,masks,c)
        assert losses.pde_residual_status!='error'
        affine=r.affine_coefficients(r.scheduler_coefficients(t,scheduler='CondOT'),training='velocity')
        schedule=r.make_zeta_schedule(c,t,tn,affine.b_t)
        target=r._gradient_target_tensor(c,cur,out)
        if fused:
            loss=(schedule.zeta_obs_a_t*losses.guidance_L_obs_a+schedule.zeta_obs_u_t*losses.guidance_L_obs_u+schedule.zeta_pde_t*losses.guidance_L_pde)
            grad=torch.autograd.grad(loss*len(indices),target)[0]
            grad,_=_clip_per_sample(grad,c.clip_threshold,True)
            gradient=SimpleNamespace(grad_total=grad,metadata={})
        else:gradient=r.compute_guidance_gradient(losses,target,schedule,c)
        x=r.apply_guidance_update(out.x_raw_next,gradient,out,schedule,c).detach()
    with torch.no_grad():
        physical=r._physical_from_model_state(x,c,normalizer)
        pred=torch.cat([physical.coef,physical.sol],1)
    torch.cuda.synchronize();seconds=time.perf_counter()-start;handle.remove()
    assert count[0]==steps and torch.isfinite(pred).all()
    return pred.detach().cpu(),dict(seconds=seconds,nfe=count[0],batch_size=len(indices),
                                  peak_bytes=torch.cuda.max_memory_allocated())


def score(pred,truth,gt,masks,cfg):
    import torch
    from sampling.losses import compute_guidance_losses
    from sampling.state import SplitState
    error=torch.linalg.vector_norm((pred-truth).double().flatten(2),dim=2)/torch.linalg.vector_norm(truth.double().flatten(2),dim=2)
    rows=[]
    for j in range(len(pred)):
        g=dataclasses.replace(gt,coef=gt.coef[j:j+1],sol=gt.sol[j:j+1],pair=gt.pair[j:j+1],
                             pde_params={k:v[j:j+1] for k,v in gt.pde_params.items()})
        m=dataclasses.replace(masks,coef=masks.coef[j:j+1],sol=masks.sol[j:j+1])
        p=pred[j:j+1].to(gt.coef.device)
        with torch.no_grad():l=compute_guidance_losses(SplitState(p[:,0:1],p[:,1:2]),g,m,cfg)
        row=dict(error_a=float(error[j,0]),error_u=float(error[j,1]),obs_a=float(l.L_obs_a),obs_u=float(l.L_obs_u),pde_mse=float(l.L_pde))
        assert all(torch.isfinite(torch.tensor(v)) for v in row.values())
        rows.append(row)
    return rows


def worker(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=a.tf32;torch.backends.cudnn.allow_tf32=a.tf32
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((a.inputs/'protocol.json').read_text());ph=digest(a.inputs/'protocol.json')
    assert digest(a.inputs/'weights.pth')==protocol['model']['weights_sha256']
    bundle=load_fm4pde_checkpoint_bundle(str(a.inputs/'weights.pth'),'nsnonbounded','cuda:0',model_profile='auto')
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in bundle[0].model.modules())
    write(a.output/f'environment_{a.mode}_{a.shard}.json',dict(host=socket.gethostname(),pid=os.getpid(),
         torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
         gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
         tf32=a.tf32,precision='float32',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         script_sha256=digest(__file__),protocol_sha256=ph,batch_size=a.batch_size))
    cache={}
    for dist in a.dists:
        assert digest(a.inputs/f'{dist}.npz')==protocol['sources'][dist]['cache_sha256']
        data=np.load(a.inputs/f'{dist}.npz');cache[dist]={int(i):torch.from_numpy(x.copy()) for i,x in zip(data['ids'],data['fields'])}
    def run(setting,dist,ids,scaled=False,fused=True,steps=100,hidden=False):
        c=configuration(protocol,setting,dist,a.inputs/'weights.pth',scaled)
        fields=torch.stack([cache[dist][i] for i in ids]);c.offset=ids[0]
        gt,masks=observations(fields,ids,c,protocol['pde_params'])
        if hidden:
            aa=gt.coef+3*(1-masks.coef);uu=gt.sol-7*(1-masks.sol)
            gt=dataclasses.replace(gt,coef=aa,sol=uu,pair=torch.cat([aa,uu],1))
        indices=[EVAL.index(i) if i in EVAL else DEV.index(i) for i in ids]
        pred,receipt=infer(c,bundle,gt,masks,indices,steps=steps,fused=fused)
        return pred,receipt,c,fields,gt,masks
    if a.mode=='batch_check':
        reference=run('sparse_joint','smooth',[DEV[0]],fused=False)[0]
        if a.tf32:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            strict=run('sparse_joint','smooth',[DEV[0]],fused=False)[0]
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            precision_difference=float((reference-strict).norm()/strict.norm())
            assert precision_difference<.005,precision_difference
        else:precision_difference=0.
        ids=(DEV*((a.batch_size+3)//4))[:a.batch_size]
        pred,receipt,*_=run('sparse_joint','smooth',ids)
        difference=float((pred[0]-reference[0]).norm()/reference[0].norm())
        assert difference<(.005 if a.tf32 else .0003),difference
        receipt.update(batch_vs_single_relative=difference,tf32_vs_strict_relative=precision_difference,
                       status='pass',tf32=a.tf32)
        torch.save(dict(prediction=pred,reference=reference,receipt=receipt),a.output/'batch_check.pt')
        write(a.output/'batch_check.json',receipt);print('BATCH_CHECK',json.dumps(receipt),flush=True)
        return
    if a.mode=='pilot':
        rows=[]
        # All full 100-step checks use development inputs only.
        base=run('sparse_joint','smooth',[DEV[0]],fused=False)
        ref=base[0]
        # Full production runner reference includes its conditioning preparation.
        import sampling.runner as runner
        from contextlib import redirect_stdout, redirect_stderr
        native_cfg=copy.deepcopy(base[2]);native_cfg.batch_size=1
        native_cfg.initial_noise_source_indices=[0]
        native_cfg.output_dir=str(a.output/'native_reference')
        with (a.output/'native_reference.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
            result=runner.run_single_ablation(native_cfg,bundle,ground_truth=base[4],observation_masks=base[5])
        saved=torch.load(Path(result['run_dir'])/'result.pt',map_location='cpu',weights_only=False)
        native=torch.cat([saved['coef_final'],saved['sol_final']],1)
        native_difference=float((ref-native).norm()/native.norm())
        assert native_difference<1e-7,native_difference
        torch.save(dict(prediction=ref,native_difference=native_difference),a.output/'reference_prediction.pt')
        for setting in ['sparse_joint','sparse_inverse','full_inverse']:
            reference=run(setting,'smooth',[DEV[0]],fused=False)[0]
            for b in [1,4]:
                out=run(setting,'smooth',DEV[:b]);pred,receipt=out[:2]
                receipt.update(setting=setting,fused_vs_stock_relative=float((pred[0]-reference[0]).norm()/reference[0].norm()))
                assert receipt['fused_vs_stock_relative']<(.005 if a.tf32 else .0003),receipt
                rows.append(receipt);print('EQUIVALENCE',json.dumps(receipt),flush=True)
            hidden=run(setting,'smooth',[DEV[0]],hidden=True)[0]
            visible=run(setting,'smooth',[DEV[0]])[0]
            assert torch.equal(hidden,visible),(setting,'hidden leakage')
        # Repeat development examples only for memory/throughput measurement.
        for b in [8,16,32,64,96]:
            if b>a.batch_size:break
            ids=(DEV*((b+3)//4))[:b]
            pred,receipt,*_=run('sparse_joint','smooth',ids,steps=5)
            rows.append(receipt);print('THROUGHPUT',json.dumps(receipt),flush=True)
            if receipt['peak_bytes']>.65*torch.cuda.get_device_properties(0).total_memory:break
        b=rows[-1]['batch_size'];out=run('sparse_joint','smooth',(DEV*((b+3)//4))[:b])
        rows.append(out[1]);print('FULL_THROUGHPUT',json.dumps(out[1]),flush=True)
        write(a.output/'pilot_complete.json',dict(status='pass',checks=rows,hidden_target_invariance=True,
             native_relative_difference=native_difference,selected_batch_size=b,estimated_formal_seconds=15000/b*out[1]['seconds']*1.12))
        return
    if a.mode=='develop':
        rows=[];selected={}
        for setting in ['full_inverse','sparse_inverse']:
            for scaled in [False,True]:
                pred,receipt,c,fields,gt,masks=run(setting,'smooth',DEV,scaled)
                metrics=score(pred,fields,gt,masks,c)
                row=dict(setting=setting,scaled=scaled,rows=metrics,mean_error_a=float(np.mean([x['error_a'] for x in metrics])),receipt=receipt)
                rows.append(row);torch.save(dict(predictions=pred,truths=fields,receipt=row),a.output/f'{setting}_{scaled}.pt')
                print('DEVELOPMENT',json.dumps(row),flush=True)
            selected[setting]=min(rows[-2:],key=lambda x:x['mean_error_a'])['scaled']
        write(a.output/'selection.json',dict(protocol_sha256=ph,development_ids=DEV,rows=rows,scaled_inverse=selected,
             rule=protocol['selection_rule'],frozen_before_formal_evaluation=True))
        return
    assert a.selection and a.selection.exists()
    selection=json.loads(a.selection.read_text());assert selection['protocol_sha256']==ph
    sh=digest(a.selection)
    lock=(a.output/f'worker_{a.shard}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert 0<=a.start<a.stop<=1000
    jobs=[(d,s,start) for d in a.dists for s in a.settings for start in range(a.start,a.stop,a.batch_size)]
    for j,(dist,setting,start) in enumerate(jobs):
        if a.shards>1 and j%a.shards!=a.shard:continue
        ids=EVAL[start:min(a.stop,start+a.batch_size)];folder=a.output/dist/setting;folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'offset{ids[0]}.pt';rp=path.with_suffix('.json')
        if rp.exists():
            old=json.loads(rp.read_text());assert old['selection_sha256']==sh and old['result_sha256']==digest(path);continue
        pred,receipt,c,fields,gt,masks=run(setting,dist,ids,selection['scaled_inverse'].get(setting,False))
        rows=score(pred,fields,gt,masks,c)
        receipt.update(ids=ids,dist=dist,setting=setting,rows=rows,protocol_sha256=ph,selection_sha256=sh,
                       tf32=a.tf32,gpu=torch.cuda.get_device_name(),worker=a.shard)
        torch.save(dict(predictions=pred,truths=fields,masks=torch.cat([masks.coef,masks.sol],1).cpu(),config=c.asdict(),receipt=receipt),path)
        receipt['result_sha256']=digest(path);write(rp,receipt)
        print('BATCH',dist,setting,ids[0],len(ids),round(receipt['seconds'],3),'seconds',flush=True)
    write(a.output/f'complete_{a.shard}.json',dict(status='complete',protocol_sha256=ph,selection_sha256=sh))


def timing_prepare(a):
    from sampling.config import load_config
    from diffusion_timing_adapter import MODULES
    assert not (a.inputs/'protocol.json').exists()
    a.inputs.mkdir(parents=True,exist_ok=True);(a.inputs/'weights').mkdir()
    previous=json.loads((a.timing_source/'protocol.json').read_text())
    protocol=copy.deepcopy(previous)
    for name in ['source','masks.npz']:(a.inputs/name).symlink_to((a.timing_source/name).resolve())
    dm=f'pretrained-{MODULES["nsnonbounded"].replace("_","-")}.pkl'
    (a.inputs/'weights'/dm).symlink_to((a.timing_source/'weights'/dm).resolve())
    shutil.copy2(a.weights,a.inputs/'weights/fm_nsnonbounded.pth')
    protocol['fm_configs']['nsnonbounded']=load_config(ROOT/'configs/main/both/nsnonbounded.yaml').asdict()
    protocol['fm_configs']['nsnonbounded']['model_profile']='auto'
    protocol.update(version=3,pdes=['nsnonbounded'],fm_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        previous_protocol_sha256=digest(a.timing_source/'protocol.json'),checkpoint_update='260904 light checkpoint, stored normalizer; same NS sparse-joint parameters, observations, precision, timing boundary, budgets, and twenty inputs')
    paths=['source/input_inventory.json','source/timing_truths.npz','masks.npz','weights/'+dm,'weights/fm_nsnonbounded.pth']
    protocol['artifacts']=[dict(path=p,sha256=digest(a.inputs/p)) for p in paths]
    write(a.inputs/'protocol.json',protocol)


def timing_run(a):
    # Reuse the validated timing protocol and its complete pilot/telemetry checks.
    # The sole loader adaptation permits the explicitly stored light architecture.
    import sampling.model_io as io
    original=io.load_fm4pde_checkpoint_bundle
    def load(*args,**kwargs):
        kwargs['model_profile']='auto'
        return original(*args,**kwargs)
    io.load_fm4pde_checkpoint_bundle=load
    from run_diffusion_fm_timing import run
    a.pdes=['nsnonbounded'];run(a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','pilot','batch_check','develop','run','timing_prepare','timing_run'])
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--weights',type=Path);p.add_argument('--selection',type=Path)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--tf32',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=1000)
    p.add_argument('--dists',nargs='+',default=['id','smooth','rough']);p.add_argument('--settings',nargs='+',default=list(SETTINGS))
    p.add_argument('--timing-source',type=Path);p.add_argument('--diffusion-root',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='timing_prepare':timing_prepare(a)
    elif a.mode=='timing_run':timing_run(a)
    else:worker(a)


if __name__=='__main__':main()

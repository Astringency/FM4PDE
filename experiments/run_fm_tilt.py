"""Resume-safe, batched, same-checkpoint FM terminal-tilt PDE evaluation.

Each input has >=4 independent latent chains. The public evaluation cohort is
exactly the 1000 IDs in inputs/protocol.json. Pilot IDs are outside that cohort.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from experiments.fm_tilt_adapter import FMTiltTarget
from experiments.fm_tilt_mh import transition, adapt_beta, chain_diagnostics
from sampling.config import AblationConfig
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.masks import PairMasks, make_pair_masks
from scripts.train.main_resume_sampling import ground_truth, tensor_sha
from scripts.train.resume_study import file_sha, write


def repeated_case(ip, cases, ids, chains, device):
    # Each input followed by its independently initialized chains.
    expanded = [sid for sid in ids for _ in range(chains)]
    truth = torch.cat([cases[i]['truth'] for i in expanded]).to(device)
    masks = PairMasks(*[torch.cat([cases[i]['masks'][f] for i in expanded]).to(device) for f in ['coef','sol']],
        metadata=dict(source='frozen actual main masks, repeated across independent chains'))
    keys = set().union(*(cases[i]['params'] for i in ids))
    params = {}
    for key in keys:
        values = [cases[i]['params'][key] for i in expanded]
        if all(torch.is_tensor(v) and v.numel()==1 for v in values):
            params[key] = torch.cat([v.reshape(1) for v in values]).to(device)
        elif all(v==values[0] for v in values):
            params[key] = values[0]
        else:
            raise ValueError(f'Unsupported batched parameter {key}; bind explicitly before evaluating this PDE')
    gt = ground_truth(ip['pde'], truth, expanded, params, device, source=ip['config']['data_path'])
    return gt, masks


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--ids',type=int,nargs='+')
    p.add_argument('--pilot-inputs',type=Path)
    p.add_argument('--batch-inputs',type=int,default=1)
    p.add_argument('--chains',type=int,default=4)
    p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--keep',type=int,default=256)
    p.add_argument('--force-steps',type=int,default=100)
    p.add_argument('--force-kind',choices=['coarse_ode','trajectory_adjoint'],default='trajectory_adjoint')
    p.add_argument('--force-scale',type=float,default=1.)
    p.add_argument('--beta',type=float,default=.008)
    p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--probe-only',action='store_true')
    p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    args=p.parse_args()
    assert args.output.is_absolute() and '/outputs/' in str(args.output)
    assert args.chains>=4 and args.keep>=8 and 1<=args.force_steps<=100 and 0<args.beta<1
    assert 0<=args.shard_index<args.shard_count
    assert np.isfinite(args.force_scale) and args.force_scale>0
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    ip=json.loads((args.output/'inputs/protocol.json').read_text())
    assert file_sha(args.output/'inputs/cases.pt')==ip['cases_sha256']
    assert file_sha(ip['checkpoint_path'])==ip['checkpoint_sha256']
    cases=torch.load(args.output/'inputs/cases.pt',map_location='cpu',weights_only=False)
    ids=args.ids or ip['sample_ids']
    assert len(ids)==len(set(ids))
    if args.pilot_inputs:
        assert set(ids).isdisjoint(ip['sample_ids']), 'Pilot must not tune on formal test cohort'
        pp=json.loads((args.pilot_inputs/'protocol.json').read_text())
        assert file_sha(args.pilot_inputs/'truths.pt')==pp['truth_sha256']
        cache=torch.load(args.pilot_inputs/'truths.pt',map_location='cpu',weights_only=False)
        for sid in ids:
            gt=cache[sid]
            assert not gt.metadata['synthetic']
            masks=make_pair_masks(gt.coef.shape,gt.sol.shape,ip['config']['num_obs'],ip['config']['sensor_mode'],
                ip['config']['shared_mask'],ip['config']['mask_seed'],device='cpu')
            cases[sid]=dict(truth=gt.pair,params=gt.pde_params,masks=dict(coef=masks.coef,sol=masks.sol))
    else:
        assert set(ids).issubset(ip['sample_ids'])
    out=args.output/args.name
    out.mkdir(parents=True,exist_ok=True)
    spec=dict(pde=ip['pde'],task=ip['task'],sample_ids=ids,chains=args.chains,warmup=args.warmup,keep=args.keep,
        force_steps=args.force_steps,force_kind=args.force_kind,force_scale=args.force_scale,
        initial_beta=args.beta,seed=args.seed,batch_inputs=args.batch_inputs,
        pilot=bool(args.pilot_inputs),probe_only=args.probe_only,shard_count=args.shard_count,
        inputs_protocol_sha256=file_sha(args.output/'inputs/protocol.json'),checkpoint_sha256=ip['checkpoint_sha256'],
        source_native_config=ip['config'], target='pi(z|y) proportional to exp(-||z||^2/2 - L_phys(G_100(z);y)); frozen actual FM checkpoint',
        proposal='Prior-preserving Gaussian proposal plus deterministic approximate force; both proposal densities enter MH ratio',
        full_generator_steps=100, full_generator='deterministic unguided Euler',
        finite_accuracy='Exact MH target up to floating-point arithmetic; finite chain output is not an exact posterior draw.',
        comparison_scope='Direct terminal tilt versus original FM guided sampler; does not isolate only the stepwise velocity.',
        estimators='One terminal draw per input is primary; chain mean is a separate point estimator.',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    with (out/'protocol.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'protocol.json').exists():
            assert json.loads((out/'protocol.json').read_text())==spec
        else:
            write(out/'protocol.json',spec)
    bundle=load_fm4pde_checkpoint_bundle(ip['checkpoint_path'],ip['pde'],'cuda:0',model_profile=ip['config']['model_profile'])
    suffix='' if args.shard_count==1 else f'_shard{args.shard_index}'
    write(out/f'model{suffix}.json',dict(selected_weight=bundle[2]['selected_inference_weight'],
        checkpoint_sha256=ip['checkpoint_sha256'],torch=str(torch.__version__),cuda=torch.version.cuda))
    completed=[]
    assigned=[]
    for start in range(0,len(ids),args.batch_inputs):
        if (start//args.batch_inputs)%args.shard_count!=args.shard_index:
            continue
        subset=ids[start:start+args.batch_inputs]
        assigned.extend(subset)
        dest=out/f'ids_{subset[0]}_{subset[-1]}'
        dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            done=json.loads((dest/'complete.json').read_text())
            assert done['protocol_sha256']==file_sha(out/'protocol.json')
            assert file_sha(dest/'result.pt')==done['result_sha256']
            completed.extend(subset);continue
        gt,masks=repeated_case(ip,cases,subset,args.chains,'cuda:0')
        conf=deepcopy(ip['config'])
        conf.update(checkpoint_path=ip['checkpoint_path'],output_dir=str(dest),device='cuda:0',batch_size=len(gt.pair),
            save_plots=False,save_intermediate=False,allow_synthetic_data=False)
        cfg=AblationConfig(**conf);cfg.validate()
        write(dest/'effective_config.json',dict(loss_and_conditioning_config=cfg.asdict(),
            actual_chain_seed=args.seed+subset[0]*1009,sample_ids=subset,chains_per_input=args.chains,
            generator_steps=100,force_kind=args.force_kind,force_steps=args.force_steps,force_scale=args.force_scale,
            native_sampler_fields_role='Only PDE/loss/conditioning fields are consumed by the adapter. Native stochastic/guidance schedules and sample_seed do not govern MCMC.'))
        adapter=FMTiltTarget(bundle,cfg,gt,masks,force_steps=args.force_steps,force_kind=args.force_kind)
        force=lambda value: args.force_scale*adapter.force(value)
        rng=torch.Generator(device='cuda:0').manual_seed(args.seed+subset[0]*1009)
        z=torch.randn(gt.pair.shape,device='cuda:0',generator=rng)
        torch.cuda.reset_peak_memory_stats()
        tic=time.monotonic()
        r,values=adapter.target(z)
        target_seconds=time.monotonic()-tic
        tic=time.monotonic();grad=force(z);force_seconds=time.monotonic()-tic
        assert torch.isfinite(grad).all() and torch.isfinite(r).all() and torch.isfinite(values).all()
        peak=torch.cuda.max_memory_allocated()
        write(dest/'probe.json',dict(target_seconds=target_seconds,force_seconds=force_seconds,peak_bytes=peak,
            energy=values.tolist(),force_norm=torch.linalg.vector_norm(grad.flatten(1),dim=1).tolist(),
            total_chains=len(z),truth_sha256=tensor_sha(gt.pair),mask_sha256=tensor_sha(torch.cat([masks.coef,masks.sol],1))))
        print('PROBE',subset,target_seconds,force_seconds,peak,values.tolist(),flush=True)
        assert peak<48*2**30, 'Measured footprint too large; choose a smaller batch before production'
        if args.probe_only:
            continue
        beta=torch.full((len(z),),args.beta,device=z.device)
        trace=[];means=torch.zeros_like(r,dtype=torch.float64);accepted=torch.zeros_like(values)
        iteration=0;elapsed=0.
        state=dest/'state.pt'
        if state.exists():
            saved=torch.load(state,map_location='cpu',weights_only=False)
            assert saved['protocol_sha256']==file_sha(out/'protocol.json')
            z,r,values,grad,beta,means,accepted=[saved[k].to('cuda:0') for k in ['z','r','values','grad','beta','means','accepted']]
            rng.set_state(saved['rng_state']);trace=saved['trace'];iteration=saved['iteration'];elapsed=saved['seconds']
        tic=time.monotonic()
        for it in range(iteration,args.warmup+args.keep):
            z,r,values,grad,accept,prob=transition(z,r,values,grad,beta,target=adapter.target,force=force,rng=rng)
            if it<args.warmup:
                beta=adapt_beta(beta,prob,it)
            else:
                trace.append(adapter.monitor(r,values).reshape(len(subset),args.chains,-1))
                means+=r.double();accepted+=accept
            if (it+1)%16==0 or it+1==args.warmup+args.keep:
                seconds=elapsed+time.monotonic()-tic
                monitors=np.asarray(trace)
                diag=[chain_diagnostics(monitors[:,j]) for j in range(len(subset))] if len(trace)>=8 else []
                progress=dict(sample_ids=subset,iteration=it+1,total_iterations=args.warmup+args.keep,seconds=seconds,
                    retained_draws=len(trace),beta=beta.tolist(),energy=values.tolist(),
                    acceptance=(accepted/max(1,len(trace))).tolist(),diagnostics=diag,peak_bytes=torch.cuda.max_memory_allocated())
                tmp=state.with_suffix('.writing')
                torch.save(dict(z=z.cpu(),r=r.cpu(),values=values.cpu(),grad=grad.cpu(),beta=beta.cpu(),means=means.cpu(),
                    accepted=accepted.cpu(),rng_state=rng.get_state(),trace=trace,iteration=it+1,seconds=seconds,
                    protocol_sha256=file_sha(out/'protocol.json')),tmp);tmp.replace(state)
                write(dest/'progress.json',progress)
                print('STEP',subset,it+1,'energy',values.tolist(),'accept',progress['acceptance'],'seconds',seconds,flush=True)
        draw=r.reshape(len(subset),args.chains,*r.shape[1:])[:,0]
        chain_mean=(means/args.keep).reshape(len(subset),args.chains,*r.shape[1:]).mean(1)
        # Affine decoding remains valid on the per-input posterior mean.
        physical_draw=adapter.normalizer.inverse_transform(draw).cpu()
        physical_mean=adapter.normalizer.inverse_transform(chain_mean).cpu()
        truth=torch.cat([cases[i]['truth'] for i in subset])
        torch.save(dict(sample_ids=subset,truth=truth,draw=physical_draw,finite_chain_mean=physical_mean,
            all_last_draws=adapter.normalizer.inverse_transform(r).cpu(),
            masks={f:torch.cat([cases[i]['masks'][f] for i in subset]) for f in ['coef','sol']},
            trace=np.asarray(trace)),dest/'result.pt')
        errors={}
        for name,prediction in [('draw',physical_draw),('finite_chain_mean',physical_mean)]:
            errors[name]=(torch.linalg.vector_norm((prediction-truth).double().flatten(2),dim=2)
                /torch.linalg.vector_norm(truth.double().flatten(2),dim=2)).tolist()
        done=dict(**progress,errors=errors,protocol_sha256=file_sha(out/'protocol.json'),
            result_sha256=file_sha(dest/'result.pt'),all_monitored_inputs_passed=all(d['passed'] for d in progress['diagnostics']))
        write(dest/'complete.json',done)
        completed.extend(subset)
        write(out/f'progress{suffix}.json',dict(completed_inputs=len(completed),requested_cohort_inputs=len(ids),
            shard_index=args.shard_index,shard_count=args.shard_count,completed_ids=completed))
    if not args.probe_only:
        assert completed==assigned
        write(out/f'complete{suffix}.json',dict(completed_inputs=len(completed),sample_ids=completed,
            shard_index=args.shard_index,shard_count=args.shard_count,protocol_sha256=file_sha(out/'protocol.json')))


if __name__=='__main__':
    main()

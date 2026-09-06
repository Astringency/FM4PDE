"""Freeze and run a batch-one, matched-observation inverse timing study.

No training, tuning or historical output mutation. Prediction timers include
CPU input construction, Voronoi fill where required, transfer and decoding;
checkpoint loading, warmup, scoring and artifact writes are outside timers.
"""
from __future__ import annotations
import argparse
import copy
import csv
import dataclasses
import fcntl
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_revision_sampling import digest, write_json

METHODS = {'RecFNO': 'recfno', 'Senseiver': 'senseiver',
           'VoronoiCNN': 'voronoicnn', 'PDE-Opt': 'pde_opt'}
CLASSES = {'recfno': 'RecFNOBaseline', 'senseiver': 'SenseiverBaseline',
           'voronoicnn': 'VoronoiCNNBaseline', 'pde_opt': 'PDEOptBaseline'}


def prepare(args):
    import torch
    from sampling.config import load_config
    from sampling.masks import make_mask
    assert not (args.inputs/'protocol.json').exists(), 'Frozen protocol exists'
    args.inputs.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.sampling_inputs/'protocol.json').read_text())
    archive = json.load(gzip.open(args.paper/'source_data/main_configs_archive.json.gz', 'rt'))
    rows = list(csv.DictReader((args.paper/'source_data/baseline_effective_counts_verified.csv').open()))
    protocol = dict(version=1, task='sparse_inverse', distribution='ID', batch_size=1,
                    physical_examples=32, seeds=[0,1,2], seed=20260907,
                    pilot_ids=previous['pilot_ids'], evaluation_ids=previous['evaluation_ids'],
                    prior_sampling_protocol_sha256=digest(args.sampling_inputs/'protocol.json'),
                    fm_steps=[25,50,100,200], pde_opt_steps=[50,100,500],
                    sensor_count=500, noise_level=0., dtype='float32', cpu_threads=2,
                    timing='CUDA-synchronized perf_counter, CPU sparse observations to GPU physical prediction; '
                           'includes preprocessing, transfers, sampling/optimization and decoding; '
                           'excludes loading, warmup, scoring and writes',
                    repetitions='FM: three inference seeds; deterministic baselines: three timing repeats, '
                                'not three independent error observations',
                    scope='Existing checkpoints and inverse hyperparameters; no retuning or retraining. '
                          'IDs preselected for the sampling study; historical training/calibration disjointness unverified.',
                    baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.baseline_root,text=True).strip(),
                    fm_configs={}, artifacts=[], baselines={}, geometry={})

    def freeze(source, relative):
        dest=args.inputs/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        protocol['artifacts'].append(dict(path=str(relative),source=str(source),sha256=digest(dest)))
        return str(relative)

    for pde in ['poisson','darcy']:
        conf=load_config(ROOT/f'configs/main/inverse/{pde}.yaml').asdict()
        archived=[r for r in archive['records'] if r['config']['pde']==pde and
                  r['config']['task']=='inverse' and r['config']['test_type']=='id' and r['config']['num_obs']==500]
        assert archived
        # Verify every archived sparse-ID row has the same operative weights.
        keys=['zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold','checkpoint_path',
              'stochastic_guidance_coeff','loss_state','gradient_target','pde_guidance_start_ratio']
        assert all(all(r['config'][k]==conf[k] for k in keys) for r in archived)
        protocol['fm_configs'][pde]=conf
        freeze(args.sampling_inputs/'weights'/f'{pde}.pth',Path('weights')/f'fm_{pde}.pth')
        freeze(args.sampling_inputs/'cache'/f'{pde}_ground_truth.pt',Path('cache')/f'{pde}_ground_truth.pt')
        truths=torch.load(args.inputs/'cache'/f'{pde}_ground_truth.pt',map_location='cpu',weights_only=False)['truths']
        masks={i:make_mask(truths[i].sol.shape,500,'random',20260907+i) for i in previous['pilot_ids']+previous['evaluation_ids']}
        mp=args.inputs/'cache'/f'{pde}_masks.pt';torch.save(masks,mp)
        protocol['artifacts'].append(dict(path=str(mp.relative_to(args.inputs)),source='deterministic CPU mask generator',sha256=digest(mp)))
        protocol['baselines'][pde]={}
        for label,method in METHODS.items():
            selected=[r for r in rows if r['PDE'].lower()==pde and r['TASK']=='inverse' and r['DIST']=='Smooth' and r['Method']==label]
            assert len(selected)==1
            run=Path(selected[0]['raw_path']).parent
            if method=='pde_opt':
                configs=list(run.glob('*_config.json'));assert len(configs)==1
                rel=freeze(configs[0],Path('weights')/f'{method}_{pde}.json')
            else:
                checkpoints=list(run.glob('*.pt'))
                assert len(checkpoints)==1,(run,checkpoints)
                rel=freeze(checkpoints[0],Path('weights')/f'{method}_{pde}.pt')
            protocol['baselines'][pde][method]=rel
            if method=='recfno':
                sample=next(run.glob('*_samples/sample_000000.pt'))
                d=torch.load(sample,map_location='cpu',weights_only=False)
                # Freeze only public geometry / PDE conventions, never template targets.
                public_keys=['canonical_layout','elliptic_operator_sign','elliptic_operator_convention',
                             'coordinate_layout','observation_source_channel_names','task_channel_names']
                geom={k:d[k] for k in ['coords','channel_names','input_channel_names','target_channel_names','pde_params']}
                geom['metadata']={k:d['metadata'][k] for k in public_keys if k in d['metadata']}
                gp=args.inputs/'cache'/f'{pde}_geometry.pt';torch.save(geom,gp)
                protocol['artifacts'].append(dict(path=str(gp.relative_to(args.inputs)),source=str(sample),sha256=digest(gp)))
                protocol['geometry'][pde]=str(gp.relative_to(args.inputs))
        print('FROZEN',pde,flush=True)
    write_json(args.inputs/'protocol.json',protocol)


def fm_predict(config, bundle, gt, masks):
    """Original sampler update path, omitting scoring, progress and file I/O.

    Pilot validation compares this directly with run_single_ablation. No
    replacement denoiser, residual, derivative, schedule or update is used.
    """
    import torch
    import sampling.runner as r
    cfg=r.finalize_ground_truth_config(config);cfg.validate();r._disable_unreliable_pde_guidance(cfg)
    r._set_seed(cfg.sample_seed);device=torch.device(cfg.device)
    net,normalizer,payload=bundle
    a=r.add_observation_noise(gt.coef,masks.coef,0.,seed=cfg.noise_seed)
    u=r.add_observation_noise(gt.sol,masks.sol,0.,seed=cfg.noise_seed+1)
    obs=r.ObservationTargets(a.clean,u.clean,a.noisy,u.noisy)
    scalar,_=r._scalar_conditioning_for_sampling(checkpoint_payload=payload,gt=gt,config=cfg,device=device)
    classes,_=r._class_conditioning_for_sampling(checkpoint_payload=payload,pde=cfg.pde,batch_size=1,device=device,cfg_scale=cfg.cfg_scale)
    extra={**classes,**(scalar or {})} or None
    r._check_sampling_channels(gt,normalizer,payload)
    grid=r.make_time_grid(cfg.time_grid,cfg.num_steps,device=device,eta=cfg.time_grid_eta)
    x=r._sample_initial_noise(cfg,gt,device)
    for k in range(cfg.num_steps):
        phase=r.phase_for_step(cfg.sampler_phase,cfg.switch_ratio,k,cfg.num_steps)
        cur=x.detach().clone().requires_grad_(True)
        t,tn=grid[k],grid[k+1]
        out=r.sampler_step(net=net,x_cur=cur,t=t,t_next=tn,phase=phase,
                          step_method=cfg.step_method,loss_state=cfg.loss_state,device=device,model_extra=extra,
                          stochastic_noise_source_batch_size=cfg.initial_noise_source_batch_size,
                          stochastic_noise_source_indices=cfg.initial_noise_source_indices or None,
                          deterministic_endpoint_mode=cfg.deterministic_endpoint_mode,
                          deterministic_endpoint_time_grid=grid[k:],
                          deterministic_rollout_checkpoint=cfg.deterministic_rollout_checkpoint)
        physical=r._physical_from_model_state(out.x_loss_state,cfg,normalizer)
        losses=r.compute_guidance_losses(physical,gt,masks,cfg,obs)
        assert not r._calibrate_l2_observation_zeta(cfg,losses,step=k)
        affine=r.affine_coefficients(r.scheduler_coefficients(t,scheduler='CondOT'),training='velocity')
        schedule=r.make_zeta_schedule(cfg,t,tn,affine.b_t)
        gradient=r.compute_guidance_gradient(losses,r._gradient_target_tensor(cfg,cur,out),schedule,cfg)
        x=r.apply_guidance_update(out.x_raw_next,gradient,out,schedule,cfg).detach()
    final=r._physical_from_model_state(x,cfg,normalizer)
    return final.coef.detach(),final.sol.detach()


def baseline_batch(pde, observed, mask, geometry, sample_id, method, device):
    import torch
    from baselines.common.data_adapter import PDEBatch
    from baselines.common.voronoi import voronoi_fill
    # All caller-supplied fields are physical CPU observations. No hidden field
    # or complete truth is available in this constructor.
    metadata=copy.deepcopy(geometry['metadata'])
    metadata.update(task='sparse_inverse',num_sensors=500,noise_level=0.)
    metadata['masked_grid']=observed.to(device)
    if method in {'recfno','voronoicnn'}:
        metadata['voronoi_grid']=voronoi_fill(observed,mask).to(device)
    indices=mask.reshape(-1).bool()
    coords=geometry['coords'].unsqueeze(0)
    values=observed.flatten(2).transpose(1,2)[:,indices]
    obs_coords=coords[:,indices]
    x=observed.to(device);zero=torch.zeros_like(x)
    return PDEBatch(pde_name=pde,task='sparse_inverse',full_tensor=torch.cat([zero,x],dim=1),
                    input_fields=x,target_fields=zero,coords=coords.to(device),mask=mask.to(device),
                    obs_values=values.to(device),obs_coords=obs_coords.to(device),
                    channel_names=geometry['channel_names'],input_channel_names=geometry['input_channel_names'],
                    target_channel_names=geometry['target_channel_names'],metadata=metadata,
                    pde_params=copy.deepcopy(geometry['pde_params']),split='test',
                    sample_indices=torch.tensor([sample_id],device=device),
                    global_sample_ids=[str(sample_id)],file_paths=[])


def run(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    import sampling.runner as runner
    protocol=json.loads((args.inputs/'protocol.json').read_text());ph=digest(args.inputs/'protocol.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.baseline_root,text=True).strip()==protocol['baseline_commit']
    for artifact in protocol['artifacts']:
        assert digest(args.inputs/artifact['path'])==artifact['sha256'],artifact['path']
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=True)
    environment=dict(host=socket.gethostname(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                     cpu_threads=torch.get_num_threads(),tf32=False,batch_size=1,
                     deterministic_algorithms=True,cudnn_deterministic=True,cublas_workspace_config=':4096:8',
                     fm_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                     baseline_commit=protocol['baseline_commit'],protocol_sha256=ph)
    write_json(args.output/'environment.json',environment)
    for pde in args.pdes:
        target=args.output/pde;target.mkdir(parents=True,exist_ok=True)
        with (target/'worker.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            truths=torch.load(args.inputs/'cache'/f'{pde}_ground_truth.pt',weights_only=False,map_location='cpu')['truths']
            masks=torch.load(args.inputs/'cache'/f'{pde}_masks.pt',weights_only=False,map_location='cpu')
            observations={i:truths[i].sol*masks[i] for i in masks}
            geometry=torch.load(args.inputs/protocol['geometry'][pde],weights_only=False,map_location='cpu')
            variants=[('fm',n) for n in protocol['fm_steps']]+[(m,1) for m in ['recfno','senseiver','voronoicnn']]+[('pde_opt',n) for n in protocol['pde_opt_steps']]
            for method in ['fm','recfno','senseiver','voronoicnn','pde_opt']:
                if method=='fm':
                    model=load_fm4pde_checkpoint_bundle(str(args.inputs/'weights'/f'fm_{pde}.pth'),pde,args.device,model_profile='recommended')
                    count={'calls':0}
                    def hook(*_):count['calls']+=1
                    handle=model[0].model.register_forward_hook(hook)
                else:
                    path=args.inputs/protocol['baselines'][pde][method]
                    payload=json.loads(path.read_text()) if method=='pde_opt' else torch.load(path,weights_only=False,map_location='cpu')
                    cls=getattr(importlib.import_module('baselines.methods.'+method),CLASSES[method])
                    effective=payload['config']
                    if method=='pde_opt':
                        from baselines.configuration import resolve_method_config
                        from scripts.experiments.provenance import baseline_config_sha256
                        original=payload['args']
                        assert not original['method_override']
                        keys=['epochs','lr','steps','refine_steps','particles','implementation_mode',
                              'official_backend','device','seed','dry_run']
                        effective=resolve_method_config(effective,baseline=method,pde=pde,
                                                        **{k:original[k] for k in keys})
                        # Same resolver and semantic hash as the original CLI.
                        assert baseline_config_sha256(effective)==original['baseline_config_sha256']
                        assert effective['early_stopping'] and effective['lr']==.01
                        write_json(target/'pde_opt_effective_config.json',effective)
                        effective['device']=args.device
                    model=cls().build(effective,payload['data_spec'])
                    if method!='pde_opt':
                        model.load_state_dict({k:v for k,v in payload['state_dict'].items() if torch.is_tensor(v)},strict=True)
                        model.load_payload(payload)
                    model.to(args.device).eval()
                    if method=='pde_opt':assert not model.uses_normalization
                    del payload

                def predict(i,seed,n,scale=1.,hidden=False):
                    observed=observations[i]*scale
                    if method=='fm':
                        conf=copy.deepcopy(protocol['fm_configs'][pde])
                        conf.update(device=args.device,batch_size=1,num_steps=n,offset=i,
                                    sample_seed=protocol['seed']+10000*seed+i,
                                    checkpoint_path=str(args.inputs/'weights'/f'fm_{pde}.pth'),
                                    save_plots=False,save_intermediate=False,save_per_sample_curves=False,
                                    output_dir=str(target/'reference'),initial_noise_source_indices=[],
                                    initial_noise_source_batch_size=None)
                        cfg=AblationConfig(**conf)
                        a=torch.zeros_like(observed).to(args.device);u=observed.to(args.device)
                        if hidden:
                            a=a+7.;u=u+3.*(1-masks[i].to(args.device))
                        gt=PDEGroundTruth(pde,a,u,torch.cat([a,u],1),copy.deepcopy(truths[i].pde_params),
                                          truths[i].channel_names_coef,truths[i].channel_names_sol,{'sample_ids':[i],'offsets':[i]})
                        pm=PairMasks(torch.zeros_like(a),masks[i].to(args.device),{'scope':'matched inverse; u only'})
                        count['calls']=0
                        prediction,solution=fm_predict(cfg,model,gt,pm)
                        return prediction,{'nfe':count['calls']},(cfg,gt,pm,solution)
                    batch=baseline_batch(pde,observed,masks[i],geometry,i,method,args.device)
                    if hidden:
                        batch.target_fields=batch.target_fields+7.
                        batch.full_tensor[:,0:1]+=7.
                        batch.full_tensor[:,1:2]+=3.*(1-batch.mask)
                    if method=='pde_opt':
                        model.config['steps']=n
                        prediction=model.predict_physical(batch)
                    else:
                        with torch.no_grad():prediction=model.predict_physical(batch)
                    status={k:v for k,v in batch.metadata.items() if k.startswith('optimization_')}
                    return prediction.detach(),status,None

                if args.mode=='pilot':
                    checks=[]
                    # Each method is validated on all four preselected pilot IDs.
                    for i in protocol['pilot_ids']:
                        n=25 if method=='fm' else 50 if method=='pde_opt' else 1
                        pred,status,extra=predict(i,0,n)
                        repeat,_,_=predict(i,0,n);hidden,_,_=predict(i,0,n,hidden=True)
                        positive,_,_=predict(i,0,n,scale=.83)
                        row=dict(sample_id=i,repeat_max_abs=float((pred-repeat).abs().max()),
                                 hidden_max_abs=float((pred-hidden).abs().max()),
                                 observation_change_max_abs=float((pred-positive).abs().max()),**status)
                        assert torch.isfinite(pred).all()
                        assert row['repeat_max_abs']==0 and row['hidden_max_abs']==0,row
                        if method!='pde_opt':
                            assert row['observation_change_max_abs']>0,row
                        else:
                            # A best-state restore can return the zero initial
                            # unknown for both observations. Keep this failure
                            # mode; do not tune stopping or omit those cases.
                            row['zero_prediction']=bool(pred.count_nonzero()==0)
                        if method=='fm':
                            cfg,gt,pm,solution=extra
                            result=runner.run_single_ablation(cfg,model,ground_truth=gt,observation_masks=pm)
                            saved=torch.load(Path(result['run_dir'])/'result.pt',weights_only=False,map_location=args.device)
                            row['reference_max_abs']=max(float((pred-saved['coef_final']).abs().max()),float((solution-saved['sol_final']).abs().max()))
                            assert row['reference_max_abs']==0,row
                        checks.append(row)
                    write_json(target/f'pilot_{method}.json',dict(protocol_sha256=ph,checks=checks,status='pass'))
                    print('PILOT PASS',pde,method,flush=True)
                else:
                    pilot=json.loads((target/f'pilot_{method}.json').read_text())
                    assert pilot['status']=='pass' and pilot['protocol_sha256']==ph
                    # Unreported warm-up for each method and budget after loading.
                    budgets=[n for m,n in variants if m==method]
                    for n in budgets:predict(protocol['pilot_ids'][0],0,n)
                    schedule=[(i,seed,n) for i in protocol['evaluation_ids'] for seed in protocol['seeds'] for n in budgets]
                    random.Random(protocol['seed']).shuffle(schedule)
                    for i,seed,n in schedule:
                        folder=target/method/f'n{n}'/f'seed{seed}'/f'sample{i}';folder.mkdir(parents=True,exist_ok=True)
                        if (folder/'receipt.json').exists():
                            old=json.loads((folder/'receipt.json').read_text());assert old['protocol_sha256']==ph
                            continue
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        start=time.perf_counter();pred,status,_=predict(i,seed,n);torch.cuda.synchronize()
                        elapsed=time.perf_counter()-start
                        peak=torch.cuda.max_memory_allocated()
                        pred=pred.detach().cpu();truth=truths[i].coef
                        assert pred.shape==truth.shape and torch.isfinite(pred).all()
                        error=float((pred.double()-truth.double()).norm()/truth.double().norm())
                        torch.save({'prediction':pred,'truth':truth,'observations':truths[i].sol*masks[i],'mask':masks[i]},folder/'prediction.pt')
                        write_json(folder/'receipt.json',dict(protocol_sha256=ph,pde=pde,method=method,budget=n,
                                   sample_id=i,seed=seed,seconds=elapsed,rel_l2_a=error,peak_allocated_bytes=peak,
                                   mask_sha256=hashlib.sha256(masks[i].numpy().tobytes()).hexdigest(),
                                   tensor_sha256=digest(folder/'prediction.pt'),status='ok',**status))
                        print('DONE',pde,method,n,i,seed,f'{elapsed:.4f}s',flush=True)
                if method=='fm':handle.remove()
                del model;torch.cuda.empty_cache()
            write_json(target/f'{args.mode}_complete.json',dict(protocol_sha256=ph,time=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','pilot','run'])
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--sampling-inputs',type=Path)
    parser.add_argument('--paper',type=Path)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--pdes',nargs='+',default=['poisson','darcy'])
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();sys.path.insert(0,str(args.baseline_root))
    prepare(args) if args.mode=='prepare' else run(args)


if __name__=='__main__':main()

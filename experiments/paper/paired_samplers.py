"""Matched FM4PDE/DiffusionPDE timing and endpoint-error trajectories."""
import copy
import csv
import dataclasses
import os
from pathlib import Path
import pickle
import sys
import time
import yaml

from experiments.paper.run import ROOT, digest, write_json


def distribution_for(spec, pde):
    return spec.get('distributions_by_pde', {}).get(pde, spec['distribution'])


def selected_groups(spec, args):
    """Resolve the same experiment selection for planning and execution."""
    if args.job_ids:
        raise ValueError('Timing and traces use --pdes/--tasks/--steps, not --job-ids')
    methods = [m for m in ['FM4PDE', 'DiffusionPDE'] if not args.methods or m in args.methods]
    if not methods:
        raise ValueError('Timing/trace study requires FM4PDE or DiffusionPDE')
    settings = ([('FM4PDE', spec['fm_steps']), ('DiffusionPDE', spec['diffusion_steps'])]
                if spec['engine'] == 'traces' else
                [(m, n) for n in spec['steps'] for m in ['FM4PDE', 'DiffusionPDE']])
    settings = [(m, n) for m, n in settings if m in methods and (not args.steps or n in args.steps)]
    tasks = spec.get('tasks', [spec.get('task', 'both')])
    groups = [(pde, task, settings) for pde in spec['pdes'] for task in tasks
              if (not args.pdes or pde in args.pdes) and (not args.tasks or task in args.tasks)
              and (not args.test_types or distribution_for(spec, pde) in args.test_types)]
    count = min(spec['count'], args.limit or spec['count'])
    positions = list(range(count))[args.shard_index::args.num_shards]
    if not groups or not settings or not positions:
        raise ValueError('No timing/trace cases match the requested selection')
    return groups, positions


def run(spec,args):
    import json
    from sampling.config import load_config, parse_cli_overrides
    groups, positions = selected_groups(spec, args)
    extra = parse_cli_overrides(args.override)
    # These values identify the paired cohort and are selected by the manifest.
    reserved = {'pde', 'task', 'device', 'batch_size', 'offset', 'num_steps', 'test_type',
                'sample_seed', 'mask_seed', 'num_obs', 'sensor_mode', 'shared_mask',
                'noise_level', 'noise_level_coef', 'noise_level_sol'}
    if spec['engine'] == 'timing':
        reserved |= {'data_path', 'data_paths', 'checkpoint_path'}
    if reserved.intersection(extra):
        raise ValueError(f'Paired cohort controls cannot be overridden: {sorted(reserved.intersection(extra))}; use the selection flags or edit the manifest')
    plans = []
    for pde, task, settings in groups:
        timing={}
        if spec['engine']=='timing':
            from experiments.paper.timing_inputs import plan_overrides
            timing=plan_overrides(pde)
        cfg = load_config(ROOT/f'configs/main/{task}/{pde}.yaml',
                          {**timing,**extra,'device':args.device,'test_type':distribution_for(spec,pde)})
        if args.phases and cfg.sampler_phase not in args.phases:
            continue
        plans.append(dict(pde=pde, task=task, settings=settings, positions=positions,
                          config=cfg.asdict(), frozen_inputs=spec['engine']=='timing'))
    if not plans:
        raise ValueError('No timing/trace cases match --phases')
    if args.plan_only:
        print(json.dumps(dict(paragraph=spec['paragraph'], groups=plans,
                              measurements=sum(len(p['settings'])*len(positions) for p in plans)), indent=2))
        return
    import numpy as np
    import torch
    import sampling.runner as runner
    from sampling.data import load_ground_truth
    from sampling.masks import make_pair_masks
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from experiments.paper.fast_sampling import fm_predict
    from experiments.trajectories.collect import Trace, diffusion_functions
    from plot.diffusion_timing_adapter import build, MODULES

    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('The manuscript timing experiments require CUDA')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from experiments.paper.provenance import runtime_identity
    runtime = runtime_identity(args.device)
    diffusion=Path(os.environ.get('DIFFUSION_ROOT',ROOT.parent/'DiffusionPDE')).resolve()
    sys.path.append(str(diffusion))
    weights=Path(os.environ.get('DIFFUSION_CHECKPOINT_ROOT',diffusion/'output/pretrained'))
    traces=spec['engine']=='traces'
    selected_methods = [m for m in ['FM4PDE','DiffusionPDE'] if not args.methods or m in args.methods]
    if not selected_methods:
        raise ValueError('Timing/trace study requires FM4PDE or DiffusionPDE')
    all_records=[]
    asset_hashes={}
    def asset_digest(path):
        path=Path(path).resolve()
        stat=path.stat()
        key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in asset_hashes:asset_hashes[key]=digest(path)
        return asset_hashes[key]
    for pde in dict.fromkeys(p['pde'] for p in plans):
        from experiments.paper.timing_inputs import TimingInputs
        frozen=TimingInputs(pde) if not traces else None
        stem=MODULES[pde].replace('_','-')
        config_file=diffusion/'configs'/f'{stem}.yaml'
        dc0=(copy.deepcopy(frozen.protocol['diffusion_configs'][pde]) if frozen else yaml.safe_load(config_file.read_text())) if 'DiffusionPDE' in selected_methods else {}
        checkpoint=Path(os.environ.get('DIFFUSION_CHECKPOINT_'+pde.upper(),weights/f'pretrained-{stem}.pkl'))
        if frozen and 'DiffusionPDE' in selected_methods:
            checkpoint=frozen.diffusion_checkpoint(checkpoint)
        dm=None; source=''; instrumented=None; plain=None; traced=None
        if 'DiffusionPDE' in selected_methods:
            with checkpoint.open('rb') as f:
                dm=pickle.load(f)['ema'].to(args.device).eval().requires_grad_(False)
            fast,source=build(diffusion,pde,native_precision=traces)
            plain,traced,instrumented=diffusion_functions(source,torch,np) if traces else (fast,None,None)
        target=args.output/pde;target.mkdir(parents=True,exist_ok=True)
        if source:(target/'diffusion_effective_sampler.py').write_text(source)
        if instrumented:(target/'diffusion_trace_sampler.py').write_text(instrumented)
        tasks=[p['task'] for p in plans if p['pde']==pde]
        cfg0=load_config(ROOT/f'configs/main/{tasks[0]}/{pde}.yaml',{
            **(frozen.overrides() if frozen else {}), **extra, 'device':args.device})
        fm=load_fm4pde_checkpoint_bundle(cfg0.checkpoint_path,pde,args.device,wrap=True,model_profile='auto')
        for task in tasks:
            count=min(spec['count'],args.limit or spec['count'])
            # A separate realization warms up every method and budget.
            ids=(frozen.ids if frozen else list(range(spec['count'])))[:count]
            pilot=frozen.pilot if frozen else spec['count']
            seed=frozen.protocol['seed'] if frozen else spec['seed']
            for index in [pilot,*ids[args.shard_index::args.num_shards]]:
                cfg=load_config(ROOT/f'configs/main/{task}/{pde}.yaml',{
                    **(frozen.overrides() if frozen else {}), **extra,
                    'device':args.device, 'test_type':distribution_for(spec,pde), 'offset':index,
                    'mask_seed':spec.get('mask_seed',0), 'sample_seed':seed+index, 'batch_size':1,
                    'save_plots':False, 'save_intermediate':False, 'save_per_sample_curves':False})
                if frozen:
                    truth,masks=frozen.case(index,args.device)
                else:
                    truth=load_ground_truth(cfg)
                    masks=make_pair_masks(truth.coef.shape,truth.sol.shape,500,'random',False,spec['mask_seed'],args.device)
                # Preserve both archived diagnostic point sets. The task profile
                # sets the inactive field's guidance weight to zero; its points
                # are still needed for the observed-location error curves.
                if pde=='burger' and not frozen:masks.coef.zero_()
                observed=dataclasses.replace(truth,coef=truth.coef*masks.coef,sol=truth.sol*masks.sol)
                observed.pair=observed.sol if pde=='burger' else torch.cat([observed.coef,observed.sol],1)
                settings=[('FM4PDE',spec['fm_steps']),('DiffusionPDE',spec['diffusion_steps'])] if traces else [(method,n) for n in spec['steps'] for method in ['FM4PDE','DiffusionPDE']]
                settings=[(method,n) for method,n in settings if method in selected_methods and (not args.steps or n in args.steps)]
                for method,n in settings:
                    c=copy.deepcopy(cfg);c.num_steps=n
                    dc=copy.deepcopy(dc0)
                    if method=='DiffusionPDE':
                        dc['test']['iterations']=n
                        dc['generate'].update(device=args.device,seed=seed+index,batch_size=1)
                        if task=='forward':dc['generate']['zeta_obs_u']=0
                        if task=='inverse':dc['generate']['zeta_obs_a']=0
                    ma,mu=masks.coef[0,0],masks.sol[0,0]
                    def predict(fm_bundle=fm, diffusion_model=dm):
                        return fm_predict(c,fm_bundle,observed,masks) if method=='FM4PDE' else plain(dc,diffusion_model,observed.coef,observed.sol,ma,mu)
                    if index==pilot:
                        prediction=predict()
                        if not all(torch.isfinite(x).all() for x in prediction):raise ValueError('Non-finite warmup prediction')
                        if method=='FM4PDE' and n==100:
                            reference_config=copy.deepcopy(c)
                            reference_config.output_dir=str(target/'validation'/task)
                            result=runner.run_single_ablation(reference_config,checkpoint_bundle=fm,
                                ground_truth=truth,observation_masks=masks)
                            saved=torch.load(Path(result['run_dir'])/'result.pt',map_location=args.device,weights_only=False)
                            differences=[float((x-saved[key]).abs().max()) for x,key in
                                         zip(prediction,['coef_final','sol_final'])]
                            if max(differences)!=0:
                                raise ValueError(f'Timing sampler disagrees with standard sampling: {differences}')
                            write_json(target/f'validation_{task}.json',dict(
                                sample_id=pilot,steps=100,maximum_absolute_differences=differences,
                                identical_to_standard_sampler=True))
                        continue
                    output=target/task/f'{method}_{n}_{index:04d}';output.parent.mkdir(parents=True,exist_ok=True)
                    tracer=Trace(c,truth,masks,runner,fm[1],torch);tracer.method=method
                    torch.cuda.synchronize();started=time.perf_counter()
                    if traces:
                        tracer.begin()
                        if method=='FM4PDE':
                            original=runner.sampler_step;step=[0]
                            def hook(*aa, fm_normalizer=fm[1], **kw):
                                result=original(*aa,**kw)
                                ep=runner._physical_from_model_state(result.x_endpoint.detach(),c,fm_normalizer)
                                native=runner._physical_from_model_state(result.x_raw_current.detach(),c,fm_normalizer)
                                tracer(step[0],(ep.coef,ep.sol),(native.coef,native.sol),float(result.t));step[0]+=1
                                return result
                            runner.sampler_step=hook
                            try:prediction=predict()
                            finally:runner.sampler_step=original
                        else:prediction=traced(dc,dm,observed.coef,observed.sol,ma,mu,tracer)
                        tracer.final(prediction)
                    else:prediction=predict()
                    torch.cuda.synchronize();seconds=time.perf_counter()-started-tracer.paused
                    if not all(torch.isfinite(x).all() for x in prediction):raise ValueError(f'Non-finite result: {output}')
                    torch.save(dict(coef=prediction[0].cpu(),sol=prediction[1].cpu(),
                                    truth_coef=truth.coef.cpu(),truth_sol=truth.sol.cpu(),
                                    mask_coef=masks.coef.cpu(),mask_sol=masks.sol.cpu()),output.with_suffix('.pt'))
                    record=dict(pde=pde,task=task,index=index,method=method,steps=n,nfe=n if method=='FM4PDE' else 2*n-1,
                                runtime=runtime,
                                seconds=seconds,diagnostic_seconds=tracer.paused,gpu=torch.cuda.get_device_name(),
                                config=c.asdict() if method=='FM4PDE' else dc,
                                checkpoint_sha256=asset_digest(cfg.checkpoint_path if method=='FM4PDE' else checkpoint),
                                input_sha256=asset_digest(cfg.data_path),prediction_sha256=digest(output.with_suffix('.pt')),
                                input_protocol=frozen.hashes if frozen else dict(seed=seed,mask_seed=spec['mask_seed']),
                                precision='native' if traces else 'float32',errors=tracer.metrics(prediction))
                    write_json(output.with_suffix('.json'),record);all_records.append(record)
                    if traces:
                        with output.with_suffix('.csv').open('w') as f:
                            writer=csv.DictWriter(f,fieldnames=list(tracer.rows[0]));writer.writeheader();writer.writerows(tracer.rows)
                    write_json(args.output/'measurements.json',all_records)
                    print(pde,task,method,n,index,seconds,flush=True)
        del fm,dm
        torch.cuda.empty_cache()

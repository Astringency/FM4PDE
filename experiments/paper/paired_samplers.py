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


def run(spec,args):
    import numpy as np
    import torch
    import sampling.runner as runner
    from sampling.config import load_config
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
    diffusion=Path(os.environ.get('DIFFUSION_ROOT',ROOT.parent/'DiffusionPDE')).resolve()
    sys.path.append(str(diffusion))
    weights=Path(os.environ.get('DIFFUSION_CHECKPOINT_ROOT',diffusion/'output/pretrained'))
    traces=spec['engine']=='traces'
    all_records=[]
    for pde in spec['pdes']:
        if args.pdes and pde not in args.pdes:continue
        stem=MODULES[pde].replace('_','-')
        config_file=diffusion/'configs'/f'{stem}.yaml'
        dc0=yaml.safe_load(config_file.read_text())
        checkpoint=Path(os.environ.get('DIFFUSION_CHECKPOINT_'+pde.upper(),weights/f'pretrained-{stem}.pkl'))
        with checkpoint.open('rb') as f:
            dm=pickle.load(f)['ema'].to(args.device).eval().requires_grad_(False)
        fast,source=build(diffusion,pde,native_precision=traces)
        plain,traced,instrumented=diffusion_functions(source,torch,np) if traces else (fast,None,None)
        target=args.output/pde;target.mkdir(parents=True,exist_ok=True)
        (target/'diffusion_effective_sampler.py').write_text(source)
        if instrumented:(target/'diffusion_trace_sampler.py').write_text(instrumented)
        tasks=spec.get('tasks',[spec.get('task','both')])
        cfg0=load_config(ROOT/f'configs/main/{tasks[0]}/{pde}.yaml',{'device':args.device})
        fm=load_fm4pde_checkpoint_bundle(cfg0.checkpoint_path,pde,args.device,wrap=True,model_profile='auto')
        for task in tasks:
            count=min(spec['count'],args.limit or spec['count'])
            # A separate realization warms up every method and budget.
            for index in [spec['count'],*range(count)]:
                cfg=load_config(ROOT/f'configs/main/{task}/{pde}.yaml',dict(
                    device=args.device,test_type=spec['distribution'],offset=index,
                    mask_seed=index,sample_seed=index,batch_size=1,save_plots=False,
                    save_intermediate=False,save_per_sample_curves=False))
                truth=load_ground_truth(cfg)
                masks=make_pair_masks(truth.coef.shape,truth.sol.shape,500,'per_sample_random',False,index,args.device)
                if task=='forward':masks.sol.zero_()
                if task=='inverse' or pde=='burger':masks.coef.zero_()
                observed=dataclasses.replace(truth,coef=truth.coef*masks.coef,sol=truth.sol*masks.sol)
                observed.pair=observed.sol if pde=='burger' else torch.cat([observed.coef,observed.sol],1)
                settings=[('FM4PDE',spec['fm_steps']),('DiffusionPDE',spec['diffusion_steps'])] if traces else [(method,n) for n in spec['steps'] for method in ['FM4PDE','DiffusionPDE']]
                for method,n in settings:
                    c=copy.deepcopy(cfg);c.num_steps=n
                    dc=copy.deepcopy(dc0);dc['test']['iterations']=n
                    dc['generate'].update(device=args.device,seed=index,batch_size=1)
                    if task=='forward':dc['generate']['zeta_obs_u']=0
                    if task=='inverse':dc['generate']['zeta_obs_a']=0
                    ma,mu=masks.coef[0,0],masks.sol[0,0]
                    def predict(fm_bundle=fm, diffusion_model=dm):
                        return fm_predict(c,fm_bundle,observed,masks) if method=='FM4PDE' else plain(dc,diffusion_model,observed.coef,observed.sol,ma,mu)
                    if index==spec['count']:
                        prediction=predict()
                        if not all(torch.isfinite(x).all() for x in prediction):raise ValueError('Non-finite warmup prediction')
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
                                seconds=seconds,diagnostic_seconds=tracer.paused,gpu=torch.cuda.get_device_name(),
                                config=c.asdict() if method=='FM4PDE' else dc,
                                checkpoint_sha256=digest(cfg.checkpoint_path if method=='FM4PDE' else checkpoint),
                                input_sha256=digest(cfg.data_path),prediction_sha256=digest(output.with_suffix('.pt')),
                                precision='native' if traces else 'float32',errors=tracer.metrics(prediction))
                    write_json(output.with_suffix('.json'),record);all_records.append(record)
                    if traces:
                        with output.with_suffix('.csv').open('w') as f:
                            writer=csv.DictWriter(f,fieldnames=list(tracer.rows[0]));writer.writeheader();writer.writerows(tracer.rows)
                    write_json(args.output/'measurements.json',all_records)
                    print(pde,task,method,n,index,seconds,flush=True)
        del fm,dm
        torch.cuda.empty_cache()

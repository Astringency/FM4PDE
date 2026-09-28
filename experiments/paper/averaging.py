"""Generate one 1,000-draw pool per fixed input and evaluate its prefix means."""
import copy
import json
from pathlib import Path
from experiments.paper.run import ROOT, digest, write_json


def run(spec, args):
    import torch
    from sampling.config import load_config, parse_cli_overrides
    from sampling.data import load_ground_truth
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from experiments.paper.conditional_sampler import fixed_observations, fast_sample

    tasks=[task for task in spec['tasks'] if not args.tasks or task in args.tasks]
    offsets=spec['offsets'][:args.limit][args.shard_index::args.num_shards]
    if args.pdes and spec['pde'] not in args.pdes:
        raise ValueError('No conditional-averaging cases match --pdes')
    if args.test_types and 'id' not in args.test_types:
        raise ValueError('Conditional averaging uses ID inputs only')
    if args.phases and 'stochastic' not in args.phases:
        raise ValueError('Conditional averaging uses stochastic sampling only')
    if args.steps and 100 not in args.steps:
        raise ValueError('Conditional averaging uses 100 sampling steps')
    if args.job_ids:
        raise ValueError('Conditional averaging selects inputs with --limit and --num-shards, not --job-ids')
    if args.methods and 'FM4PDE' not in args.methods:
        raise ValueError('Conditional averaging requires FM4PDE')
    if not tasks or not offsets:
        raise ValueError('No conditional-averaging pools match the selection')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    extra=parse_cli_overrides(args.override)
    def configuration(task,offset):
        seed=spec['seed']+offset
        overrides=dict(device=args.device,offset=offset,mask_seed=seed,sample_seed=seed,
            batch_size=1,shared_mask=task=='both',initial_noise_source_batch_size=spec['draws'],
            save_plots=False,save_intermediate=False)
        overrides.update(extra)
        cfg=load_config(ROOT/f'configs/main/{task}/poisson.yaml',overrides)
        if (cfg.num_steps != 100 or cfg.sampler_phase != 'stochastic'
                or cfg.time_grid != 'uniform' or cfg.num_obs != 500
                or cfg.noise_level or cfg.noise_level_coef or cfg.noise_level_sol):
            raise ValueError('The conditional pool requires 100 uniform stochastic steps and 500 noise-free observations')
        cfg.runtime_metadata['fused_guidance']=True
        return cfg
    if args.plan_only:
        print(json.dumps(dict(tasks=tasks,offsets=offsets,draws_per_pool=spec['draws'],
            example=configuration(tasks[0],offsets[0]).asdict()),indent=2))
        return
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Conditional averaging requires a CUDA device')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    source_hashes={str(p.relative_to(ROOT)):digest(p)
                   for directory in ['sampling','experiments/paper']
                   for p in (ROOT/directory).rglob('*.py')}
    bundle = None
    rows=[]
    for task in tasks:
        for offset in offsets:
            cfg=configuration(task,offset)
            folder=args.output/task/f'{offset:04d}';folder.mkdir(parents=True,exist_ok=True)
            identity=dict(config=cfg.asdict(),checkpoint_sha256=digest(cfg.checkpoint_path),
                          data_sha256=digest(cfg.data_path),sampler_sha256=digest(Path(__file__).with_name('conditional_sampler.py')),
                          source_hashes=source_hashes,tf32=True,fused_guidance=True)
            identity_path=folder/'identity.json'
            if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
                raise ValueError(f'{folder}: changed inputs or settings; use a new output directory')
            write_json(identity_path,identity)
            if bundle is None:
                bundle=load_fm4pde_checkpoint_bundle(cfg.checkpoint_path,'poisson',args.device,wrap=True,model_profile=cfg.model_profile)
            truth=load_ground_truth(cfg)
            cpu_truth=copy.deepcopy(truth)
            cpu_truth.coef=truth.coef.cpu();cpu_truth.sol=truth.sol.cpu()
            fixed=fixed_observations(cfg,cpu_truth)
            target=torch.cat([truth.coef,truth.sol],1).cpu().double()
            total=torch.zeros_like(target)
            individual=[]
            start=0
            for count in spec['prefix_counts']:
                while start<count:
                    stop=min(start+args.batch_size,count)
                    path=folder/f'draws_{start:04d}_{stop:04d}.pt'
                    if args.resume and path.exists():
                        payload=torch.load(path,map_location='cpu',weights_only=False)
                        if payload['identity']!=identity or payload['indices']!=list(range(start,stop)):
                            raise ValueError(f'Unmatched draw shard: {path}')
                        predictions=payload['predictions']
                    else:
                        predictions,_,stats=fast_sample(cfg,truth,bundle,list(range(start,stop)),fixed)
                        torch.save(dict(predictions=predictions,indices=list(range(start,stop)),
                                        identity=identity,statistics=stats),path)
                    print(task,offset,'draws',start,stop,flush=True)
                    if not torch.isfinite(predictions).all():raise ValueError(f'Non-finite draws: {path}')
                    total+=predictions.double().sum(0,keepdim=True)
                    for sample in predictions.double():
                        individual.append([float((sample[c]-target[0,c]).norm()/target[0,c].norm().clamp_min(1e-12)) for c in range(2)])
                    start=stop
                mean=total/count
                errors=[float((mean[0,c]-target[0,c]).norm()/target[0,c].norm().clamp_min(1e-12)) for c in range(2)]
                rows.append(dict(task=task,offset=offset,draws=count,relative_l2_a=errors[0],relative_l2_u=errors[1]))
                torch.save(dict(mean=mean.float(),truth=target.float(),masks=fixed['masks']),folder/f'mean_{count:04d}.pt')
                write_json(args.output/'prefix_errors.json',rows)
            write_json(folder/'individual_errors.json',individual)
            print(task,offset,'complete',flush=True)
    write_json(args.output/'completion.json',dict(tasks=tasks,offsets=offsets,
        expected_pools=len(tasks)*len(offsets),completed_pools=len(rows)//len(spec['prefix_counts']),
        draws_per_pool=spec['draws']))

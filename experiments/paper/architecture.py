"""Compare U-Net and official OFM priors with the appendix OFM guidance."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import yaml
from experiments.paper.run import ROOT, digest, write_json


def native_jobs(spec):
    """Retain the archived OFM-matched batch partitions and random seeds."""
    parameters=yaml.safe_load((ROOT/'configs/ofm_guidance.yaml').read_text())
    jobs=[]
    for pde in spec['pdes']:
        for task,values in parameters[pde].items():
            size=spec.get('native_batches',{}).get(pde,{}).get(task,1)
            for dist in spec['distributions']:
                singles=set(spec.get('native_single_cases',{}).get(f'{pde}_{task}_{dist}',[]))
                i=0
                while i < spec['count']:
                    n=size if i+size <= spec['count'] and not any(j in singles for j in range(i,i+size)) else 1
                    overrides=dict(zip(['zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold'],values))
                    overrides.update(test_type=dist,offset=i,batch_size=n,sample_seed=i,mask_seed=0,
                                     num_steps=100,model_profile='auto')
                    jobs.append(dict(id=f'{pde}_{task}_{dist}_{i:04d}',pde=pde,task=task,
                                     observation_protocol='common_comparison',overrides=overrides))
                    i+=n
    return jobs


def run(spec,args):
    import torch
    from sampling.config import load_config
    from sampling.data import load_ground_truth
    from sampling.runner import run_single_ablation
    priors = [p for p, name in [('fm4pde', 'FM4PDE'), ('ofm', 'FM4PDE-OFM')]
              if not args.methods or name in args.methods]
    if not priors:
        raise ValueError('Architecture study requires FM4PDE or FM4PDE-OFM')
    if 'fm4pde' in priors:
        import copy
        from experiments.paper.run import run_sampling
        native_args=copy.copy(args)
        native_args.output=args.output/'fm4pde'
        run_sampling(dict(paragraph=spec['paragraph'],jobs=native_jobs(spec)),native_args)
        priors.remove('fm4pde')
    if not priors:
        return
    import copy
    import json
    from experiments.paper.run import resolved_jobs
    from experiments.paper.observations import for_job
    from experiments.paper.summarize import summarize
    baseline=Path(os.environ.get('GENERATIVE_BASELINE_ROOT',ROOT.parent/'FunDPS_DDIS_ECI_OFM'))
    compact_root=Path(os.environ.get('OFM_DATA_ROOT', baseline/'datasets/compact'))
    checkpoint_root=Path(os.environ.get('OFM_CHECKPOINT_ROOT', baseline/'outputs/training/flow'))
    ofm_args=copy.copy(args)
    ofm_args.output=args.output/'ofm'
    jobs=list(resolved_jobs(dict(jobs=native_jobs(spec)),ofm_args))
    for job,config in jobs:
        config.checkpoint_path=str(Path(os.environ.get('OFM_CHECKPOINT_'+config.pde.upper(),
                                                       checkpoint_root/config.pde/spec['ofm_checkpoints'][config.pde])))
    if args.plan_only:
        print(json.dumps(dict(paragraph=spec['paragraph'],prior='ofm',jobs=len(jobs),
                              realizations=sum(c.batch_size for _,c in jobs),
                              example=jobs[0][1].asdict()),indent=2))
        return
    sys.path.insert(0,str(baseline/'adapters'))
    from shared_prior_runtime import load_prior, native_noise_provider
    source_hashes={str(p.relative_to(ROOT)):digest(p)
                   for directory in ['sampling','models','flow_matching','torchdiffeq','experiments/paper','data']
                   for p in (ROOT/directory).rglob('*.py')}
    source_hashes.update({str(p.relative_to(baseline)):digest(p)
                          for directory in ['adapters','official/OFM']
                          for p in (baseline/directory).rglob('*.py')})
    records=[]
    asset_hashes={}
    previous=None
    bundle=None
    torch.set_num_threads(args.threads)
    from experiments.paper.provenance import runtime_identity
    runtime = runtime_identity(args.device)
    for job,config in jobs:
        manifest=compact_root/config.pde/'manifest.json'
        for path in [config.checkpoint_path,config.data_path,str(manifest)]:
            if path not in asset_hashes:asset_hashes[path]=digest(path)
        identity=dict(config=config.asdict(),sources=source_hashes,prior='ofm',
                      runtime=runtime,
                      observation_protocol=job.get('observation_protocol'),
                      data_sha256=asset_hashes[config.data_path],
                      checkpoint_sha256=asset_hashes[config.checkpoint_path],
                      normalizer_manifest_sha256=asset_hashes[str(manifest)])
        folder=Path(config.output_dir)
        receipt=folder/'receipt.json'
        if receipt.exists() and args.resume:
            saved=json.loads(receipt.read_text())
            if saved['identity']!=identity:
                raise ValueError(f'{receipt}: settings or source changed; choose a new output directory')
            if digest(saved['result'])!=saved['result_sha256']:
                raise ValueError(f'{receipt}: saved prediction checksum mismatch')
            records.append(saved)
            continue
        key=(config.pde,config.checkpoint_path,config.device)
        if previous!=key:
            bundle=None
            if torch.cuda.is_available():torch.cuda.empty_cache()
            options=SimpleNamespace(fm4pde=ROOT,ofm=baseline/'official/OFM',source=compact_root/config.pde,
                                    checkpoint=Path(config.checkpoint_path),device=config.device,pde=config.pde,prior='ofm')
            net,normalizer,payload,noise,_=load_prior(options,1 if config.pde=='burger' else 2)
            bundle=(net,normalizer,payload)
            previous=key
        truth=load_ground_truth(config)
        masks=for_job(job,truth,config)
        with native_noise_provider(noise,enabled=True):
            result=run_single_ablation(config,checkpoint_bundle=bundle,ground_truth=truth,observation_masks=masks)
        if result.get('status')!='ok' or result.get('pde_eval_error_count',0):
            raise RuntimeError(f"{job['id']}: sampling or PDE evaluation failed")
        path=Path(result['run_dir'])/'result.pt'
        saved=dict(id=job['id'],pde=config.pde,task=config.task,prior='ofm',
                   identity=identity,result=str(path),result_sha256=digest(path))
        write_json(receipt,saved)
        records.append(saved)
        write_json(ofm_args.output/'index.json',records)
        print(f"{job['id']}: complete",flush=True)
    write_json(ofm_args.output/'index.json',records)
    if records:summarize(ofm_args.output)
    write_json(ofm_args.output/'completion.json',dict(expected=len(jobs),completed=len(records),failed=0))

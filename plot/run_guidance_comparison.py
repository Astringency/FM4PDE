"""Compare guidance and sampling variants on fixed inputs and observations.

Prepare on the local mounted archive, then synchronize code with Git and copy
the new input directory to an isolated remote checkout. No historical output
is changed. Each process handles separate PDEs and uses a lock to reject a
second worker for the same PDE. All variants retain the existing sampler.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PDES = ['poisson', 'darcy', 'nsnonbounded', 'burger']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def configs():
    variants = []
    for guidance in ['noguide', 'pde_only', 'obs_only', 'obs_pde']:
        variants.append(dict(name='guidance_' + guidance, family='guidance',
                             guidance_components=guidance, num_steps=100,
                             sampler_phase='stochastic', coefficient_rule='fixed'))
    for n in [25, 50, 200]:
        for rule in ['fixed', 'normalized']:
            variants.append(dict(name=f'steps_{n}_{rule}', family='steps',
                                 guidance_components='obs_pde', num_steps=n,
                                 sampler_phase='stochastic', coefficient_rule=rule))
    for phase in ['deterministic', 'hybrid_d2s', 'hybrid_s2d']:
        variants.append(dict(name='phase_' + phase, family='phase',
                             guidance_components='obs_pde', num_steps=100,
                             sampler_phase=phase, switch_ratio=.2, coefficient_rule='fixed'))
    return variants


def prepare(args):
    import torch
    from sampling.config import load_config
    from sampling.cached_inputs import prepare_samples
    from scripts.training.export_checkpoint import make_inference_checkpoint
    target = args.inputs
    target.mkdir(parents=True, exist_ok=True)
    protocol_path = target/'protocol.json'
    if protocol_path.exists():
        raise FileExistsError('Frozen inputs already exist; inspect them rather than overwrite.')
    # These prior screening/holdout IDs are excluded before any new evaluation.
    excluded = [909,383,313,466,598,100,40,602,514,260,615,604]
    ids = random.Random(20260907).sample([i for i in range(1000) if i not in excluded], 36)
    pilot_ids, evaluation_ids = ids[:4], ids[4:]
    bases = {p:load_config(ROOT/f'configs/main/both/{p}.yaml').asdict() for p in PDES}
    protocol = dict(version=1, seed=20260907, physical_examples=32, inference_seeds=[0,1,2],
                    pilot_ids=pilot_ids, evaluation_ids=evaluation_ids, excluded_prior_ids=excluded,
                    task='both', distribution='ID', batch_size=4, dtype='float32',
                    guidance_clip=50.0, masks='per-example random, frozen across variants and inference seeds',
                    primary_score='max(rel_l2_a,rel_l2_u); Burgers rel_l2_u over full trajectory',
                    selection='IDs sampled before outcomes; no new hyperparameter selection',
                    timing_scope='synchronized full run_single_ablation call, including diagnostics and artifact writes; '
                                 'excluding model/data loading and ground-truth batching; one unreported 10-step warm-up per PDE',
                    variants=configs(), base_configs=bases,
                    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(target/'design.json', protocol)
    prepare_samples(SimpleNamespace(source_root=str(args.source_root),root=target),PDES,ids)
    weights = []
    for pde,base in bases.items():
        source = ROOT/base['checkpoint_path']; dest=target/'weights'/f'{pde}.pth'
        make_inference_checkpoint(source,dest)
        payload=torch.load(dest,map_location='cpu',weights_only=False,mmap=True)
        # Digest selected weights and normalizer, independently of serialization.
        state_hash=hashlib.sha256()
        for name,value in sorted(payload['model'].items()):
            state_hash.update(name.encode()); state_hash.update(value.detach().contiguous().numpy().tobytes())
        weights.append(dict(pde=pde,source=str(source),source_size=source.stat().st_size,
                            selected_weights_sha256=state_hash.hexdigest(),file_sha256=digest(dest),
                            copied_normalizer=payload['normalizer']))
        del payload
    # Normalizer can contain tensors; keep it in the checkpoint, not JSON.
    for row in weights:
        row.pop('copied_normalizer')
    protocol['weights']=weights
    protocol['cache_sha256']={p:digest(target/'cache'/f'{p}_ground_truth.pt') for p in PDES}
    write_json(protocol_path,protocol)
    print('PREPARED',target,flush=True)


def finite_number(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else None
    except (ValueError,TypeError):
        return None


def run_pde(args, pde, protocol):
    import torch
    from sampling.config import AblationConfig
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    import sampling.runner as runner
    run_single_ablation = runner.run_single_ablation
    from sampling.batching import combine_truths
    target=args.output/pde; target.mkdir(parents=True,exist_ok=True)
    with (target/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol_hash=digest(args.inputs/'protocol.json')
        write_json(target/'worker.json',dict(pid=os.getpid(),host=socket.gethostname(),
                                            protocol_sha256=protocol_hash,command=sys.argv))
        from sampling.study_models import model_config, bind_model
        selected_model=model_config(args,pde)
        bind_model(target,selected_model)
        checkpoint=Path(selected_model.checkpoint_path)
        cache=args.inputs/'cache'/f'{pde}_ground_truth.pt'
        assert digest(cache)==protocol['cache_sha256'][pde],cache
        truths=torch.load(cache,map_location='cpu',weights_only=False)['truths']
        bundle=load_fm4pde_checkpoint_bundle(str(checkpoint),pde,args.device,model_profile=selected_model.model_profile)
        # Count actual model forward calls; this includes endpoint prediction.
        counter={'calls':0}
        def count(*_): counter['calls']+=1
        handle=bundle[0].model.register_forward_hook(count)
        environment=dict(host=socket.gethostname(),python=sys.version,torch=torch.__version__,
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),device=args.device,
                         visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                         tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                         gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True))
        write_json(target/'environment.json',environment)
        base=copy.deepcopy(protocol['base_configs'][pde])
        base.update(checkpoint_path=str(checkpoint),model_profile=selected_model.model_profile,device=args.device,clip_threshold=50.,
                    clip_mode='global_norm',save_plots=False,save_intermediate=False,
                    save_per_sample_curves=True,sensor_mode='per_sample_random',
                    num_obs=500,noise_level=0.,time_grid='uniform',step_method='euler',
                    loss_state='endpoint',pde_guidance_reduction='mse',ablation_name='')
        initial_noise={}
        original_noise=runner._sample_initial_noise
        def recorded_noise(*pos,**kw):
            value=original_noise(*pos,**kw)
            initial_noise['sha256']=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            return value
        runner._sample_initial_noise=recorded_noise

        def one(variant,seed,batch_ids,stage):
            label=f'{stage}/{variant["name"]}/seed{seed}/batch{batch_ids[0]}'
            folder=target/label; folder.mkdir(parents=True,exist_ok=True)
            receipt=folder/'receipt.json'
            if receipt.exists():
                previous=json.loads(receipt.read_text())
                assert previous['protocol_sha256']==protocol_hash and previous['sample_ids']==batch_ids,receipt
                print('EXISTS',pde,label,flush=True); return previous
            conf=copy.deepcopy(base)
            for k in ['guidance_components','num_steps','sampler_phase','switch_ratio']:
                if k in variant: conf[k]=variant[k]
            if variant['coefficient_rule']=='normalized':
                conf['stochastic_guidance_coeff']=base['stochastic_guidance_coeff']*101/(variant['num_steps']+1)
            conf.update(output_dir=str(folder),batch_size=len(batch_ids),offset=batch_ids[0],
                        sample_seed=protocol['seed']+10000*seed+batch_ids[0],
                        mask_seed=protocol['seed']+batch_ids[0],ablation_group='jmlr_sampling_confirmation')
            cfg=AblationConfig(**conf)
            cfg.validate()
            gt=combine_truths(truths,batch_ids,args.device)
            counter['calls']=0
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            result=None; error=None
            with (folder/'run.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
                try:
                    result=run_single_ablation(cfg,checkpoint_bundle=bundle,ground_truth=gt)
                except torch.OutOfMemoryError:
                    raise
                except Exception as exc:
                    import traceback
                    traceback.print_exc(); error=repr(exc)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            rows=[]
            if result:
                with (Path(result['run_dir'])/'metrics_per_sample.csv').open() as stream:
                    rows=list(csv.DictReader(stream))
                assert [int(r['sample_id']) for r in rows]==batch_ids,rows
                masks=torch.load(Path(result['run_dir'])/'masks.pt',map_location='cpu',weights_only=False)
                mask_hash=hashlib.sha256(masks['coef'].contiguous().numpy().tobytes()+masks['sol'].contiguous().numpy().tobytes()).hexdigest()
            else:
                mask_hash=None
            summary=dict(protocol_sha256=protocol_hash,pde=pde,variant=variant,stage=stage,seed=seed,
                         sample_ids=batch_ids,elapsed_seconds=elapsed,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),model_evaluations=counter['calls'],
                         status=result['status'] if result else 'error',error=error,
                         initial_noise_sha256=initial_noise.get('sha256'),mask_tensor_sha256=mask_hash,
                         run_dir=result['run_dir'] if result else None,rows=rows)
            write_json(receipt,summary)
            print('DONE',pde,label,f'{elapsed:.3f}s',summary['status'],flush=True)
            return summary

        warm=dict(name='warmup',family='warmup',guidance_components='obs_pde',num_steps=10,
                  sampler_phase='stochastic',coefficient_rule='fixed')
        # Warm-up is repeated after process restarts and is never a timing result.
        warm['name']=f'warmup_pid{os.getpid()}'
        warm_ids=protocol['pilot_ids'][:1] if args.mode=='probe' else protocol['pilot_ids']
        if args.mode!='probe':
            probes=list((target/'probe').glob('*/seed0/*/receipt.json'))
            if not probes:
                raise RuntimeError('Run a batch-one memory probe before batching.')
            peak=max(json.loads(p.read_text())['peak_allocated_bytes'] for p in probes)
            parameter_bytes=sum(p.numel()*p.element_size() for p in bundle[0].model.parameters())
            predicted=parameter_bytes+4*max(0,peak-parameter_bytes)+(2<<30)
            total=torch.cuda.get_device_properties(0).total_memory
            if predicted>.8*total:
                raise RuntimeError(f'Conservative batch-four memory estimate {predicted} exceeds 80% of {total}')
        one(warm,0,warm_ids,'warmup')
        if args.mode=='probe':
            one(next(v for v in protocol['variants'] if v['name']=='guidance_obs_pde'),0,warm_ids,'probe')
        elif args.mode=='smoke':
            for variant in protocol['variants']:
                one(variant,0,protocol['pilot_ids'],'pilot')
        else:
            for seed in protocol['inference_seeds']:
                variants=copy.deepcopy(protocol['variants'])
                random.Random(protocol['seed']+seed).shuffle(variants)
                for start in range(0,32,protocol['batch_size']):
                    batch_ids=protocol['evaluation_ids'][start:start+protocol['batch_size']]
                    for variant in variants:
                        one(variant,seed,batch_ids,'evaluation')
        handle.remove()
        runner._sample_initial_noise=original_noise
        write_json(target/f'{args.mode}_complete.json',dict(protocol_sha256=protocol_hash,finished_unix=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','probe','smoke','run'])
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--source-root',type=Path,default=ROOT/'outputs/main/MAIN1000_100_TEST_id')
    parser.add_argument('--pdes',nargs='+',default=PDES,choices=PDES)
    parser.add_argument('--device',default='cuda:0')
    from sampling.study_models import add_model_arguments, print_model_plan
    add_model_arguments(parser)
    args=parser.parse_args(); args.inputs=args.inputs.resolve()
    if print_model_plan(args): return
    if args.mode=='prepare': prepare(args); return
    if args.output is None: parser.error('--output is required for sampling')
    args.output=args.output.resolve()
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    protocol=json.loads((args.inputs/'protocol.json').read_text())
    for pde in args.pdes:
        run_pde(args,pde,protocol)


if __name__=='__main__':
    main()

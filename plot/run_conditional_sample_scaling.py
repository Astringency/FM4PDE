#!/usr/bin/env python3
"""Nested conditional-sample means with measured batched 100-step sampling.

Read-only archived inputs/configurations. Results go only to --output. The
canonical 1000-row Gaussian source makes every draw's path invariant to batch
partition and K. Stock sampling functions implement every model/guidance step.
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
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'plot'))
from run_paper_ablation_revision import digest, write
KS = [1, 3, 10, 100, 1000]


def configuration(protocol, selection, task, source):
    from sampling.config import AblationConfig
    rows = [x['config'] for x in protocol['archived']
            if x['config']['task'] == task
            and x['config']['ablation_group'] == 'guidance_components'
            and x['config']['guidance_components'] == 'obs_pde']
    assert len(rows) == 1
    c = copy.deepcopy(rows[0])
    c.update(selection['task_updates'][task])
    c.update(checkpoint_path=str(source/'weights.pth'), device='cuda:0',
             save_plots=False, save_intermediate=False, save_per_sample_curves=False,
             empty_cache_each_step=False, allow_synthetic_data=False,
             initial_noise_source_batch_size=1000)
    cfg = AblationConfig(**c)
    cfg.validate()
    assert cfg.pde == 'poisson' and cfg.num_steps == 100
    assert cfg.sampler_phase == 'stochastic' and cfg.step_method == 'euler'
    assert cfg.noise_level == 0 and cfg.gradient_target == 'current_state_chain_rule'
    return cfg


def fast_sample(cfg, truth, bundle, indices, steps=100):
    import torch
    from scripts.tuning.compare_pde_guidance_schedules import combine_truths
    import sampling.runner as r
    c = copy.deepcopy(cfg)
    c.batch_size = len(indices)
    c.initial_noise_source_indices = list(indices)
    gt = combine_truths({c.offset: truth}, [c.offset]*len(indices), 'cuda:0')
    masks = r.make_pair_masks(gt.coef.shape, gt.sol.shape, c.num_obs,
                            c.sensor_mode, c.shared_mask, c.mask_seed, device='cuda:0')
    net, normalizer, _ = bundle
    assert c.obs_l2_reference_mse_zeta_a is None and c.obs_l2_reference_mse_zeta_u is None
    r._set_seed(c.sample_seed)
    grid = r.make_time_grid(c.time_grid, c.num_steps, device='cuda:0', eta=c.time_grid_eta)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    x = r._sample_initial_noise(c, gt, 'cuda:0')
    for step in range(steps):
        x_cur = x.detach().clone().requires_grad_(True)
        t, t_next = grid[step], grid[step+1]
        out = r.sampler_step(net, x_cur, t, t_next, 'stochastic', c.step_method,
                             c.loss_state, device='cuda:0',
                             stochastic_noise_source_batch_size=1000,
                             stochastic_noise_source_indices=list(indices))
        phys = r._physical_from_model_state(out.x_loss_state, c, normalizer)
        losses = r.compute_guidance_losses(phys, gt, masks, c)
        assert losses.pde_residual_status != 'error'
        coeffs = r.scheduler_coefficients(t, scheduler='CondOT')
        affine = r.affine_coefficients(coeffs, training='velocity')
        schedule = r.make_zeta_schedule(c, t, t_next, affine.b_t)
        target = r._gradient_target_tensor(c, x_cur, out)
        if c.runtime_metadata.get('fused_guidance'):
            # Global clipping applies after the weighted gradient sum; linearity
            # therefore permits one reverse pass through the velocity network.
            from types import SimpleNamespace
            from sampling.guidance import _clip_per_sample
            assert c.clip_mode in {'global_norm', 'none'}
            total_loss = (schedule.zeta_obs_a_t * losses.guidance_L_obs_a
                          + schedule.zeta_obs_u_t * losses.guidance_L_obs_u
                          + schedule.zeta_pde_t * losses.guidance_L_pde)
            total = torch.autograd.grad(total_loss * len(indices), target)[0]
            total, _ = _clip_per_sample(total, c.clip_threshold, c.clip_mode == 'global_norm')
            gradient = SimpleNamespace(grad_total=total, metadata={})
        else:
            gradient = r.compute_guidance_gradient(losses, target, schedule, c)
        x = r.apply_guidance_update(out.x_raw_next, gradient, out, schedule, c).detach()
    with torch.no_grad():
        phys = r._physical_from_model_state(x, c, normalizer)
        # Averaging is included in the synchronized sampling time.
        mean = torch.cat([phys.coef, phys.sol], dim=1).double().mean(0).float()
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    pred = torch.cat([phys.coef, phys.sol], dim=1).detach().cpu()
    assert torch.isfinite(pred).all()
    return pred, mean.cpu(), dict(seconds=seconds, peak_bytes=peak, batch_size=len(indices),
                                 seed_indices=list(indices), num_steps=steps)


def main():
    import torch
    import numpy as np
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pilot', 'run'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--tf32', action='store_true')
    parser.add_argument('--fused-guidance', action='store_true')
    parser.add_argument('--tasks', nargs='+', default=['forward','inverse','both'])
    parser.add_argument('--offsets', nargs='+', type=int, default=list(range(1500,1532)))
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    assert 1 <= args.batch_size <= 1000
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.inputs / 'poisson'
    protocol = json.loads((source/'protocol.json').read_text())
    selection = json.loads(args.selection.read_text())
    assert digest(source/'protocol.json') == selection['protocol_sha256']
    assert digest(source/'truths.pt') == protocol['truth_sha256']
    assert digest(source/'weights.pth') == protocol['weights_sha256']
    assert set(args.offsets) <= set(protocol['evaluation_ids'])
    truths = torch.load(source/'truths.pt', map_location='cpu', weights_only=False)
    bundle = load_fm4pde_checkpoint_bundle(str(source/'weights.pth'), 'poisson', 'cuda:0', model_profile='recommended')
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in bundle[0].model.modules())
    env = dict(host=socket.gethostname(), pid=os.getpid(), python=sys.version,
               torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               script_sha256=digest(__file__), inputs=str(source), protocol_sha256=digest(source/'protocol.json'),
               selection_sha256=digest(args.selection), truth_sha256=protocol['truth_sha256'],
               weights_sha256=protocol['weights_sha256'], tf32=args.tf32, K=KS,
               random_source='1000-row IID Gaussian pool, selected row retained at every step',
               random_seed_formula='20260912 + physical_offset; each draw is a distinct canonical row',
               timing='CUDA-synchronized wall time; initial/bridge draws, 100 sampler steps, physical transform and within-batch average; excludes model/data loading and disk I/O',
               args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    write(args.output/f'environment_{args.mode}_{args.shard_index}.json',env)
    if args.mode == 'pilot':
        cfg = configuration(protocol, selection, 'both', source)
        cfg.runtime_metadata['fused_guidance'] = args.fused_guidance
        cfg.offset = 1500
        cfg.mask_seed = 20260912 + cfg.offset
        cfg.sample_seed = 20260912 + cfg.offset
        pilot=[]
        base = None
        for b in [1, 4, 8, 16, 32, 64, 96, 128]:
            if b > args.batch_size: break
            free,total = torch.cuda.mem_get_info()
            if pilot and pilot[-1]['peak_bytes']*b/pilot[-1]['batch_size'] > 0.70*total:
                print('MEMORY_GUARD', b, flush=True)
                break
            pred,mean,receipt = fast_sample(cfg,truths[1500],bundle,range(b),steps=5)
            pilot.append(receipt)
            print('PILOT',json.dumps(receipt),flush=True)
            del pred, mean
            torch.cuda.empty_cache()
        chosen = pilot[-1]['batch_size']
        for b in [1,chosen]:
            pred,mean,receipt = fast_sample(cfg,truths[1500],bundle,range(b))
            if base is None: base=pred[0].clone()
            else:
                receipt['batch_vs_single_relative_difference'] = float(torch.linalg.vector_norm(pred[0]-base)/torch.linalg.vector_norm(base))
                assert receipt['batch_vs_single_relative_difference'] < (5e-3 if args.tf32 else 2e-4),receipt
            torch.save(dict(predictions=pred,receipt=receipt),args.output/f'full_pilot_batch{b}.pt')
            pilot.append(receipt)
            print('FULL_PILOT',json.dumps(receipt),flush=True)
        # Compare the fast loop against the unmodified production runner.
        import sampling.runner as r
        from contextlib import redirect_stdout, redirect_stderr
        from scripts.tuning.compare_pde_guidance_schedules import combine_truths
        cfg.initial_noise_source_indices=[0]
        cfg.batch_size=1
        cfg.output_dir=str(args.output/'stock_reference')
        with (args.output/'stock_reference.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
            result=r.run_single_ablation(cfg,checkpoint_bundle=bundle,
                ground_truth=combine_truths(truths,[1500],'cuda:0'))
        saved=torch.load(Path(result['run_dir'])/'result.pt',map_location='cpu',weights_only=False)
        reference=torch.cat([saved['coef_final'],saved['sol_final']],1)[0]
        difference=float(torch.linalg.vector_norm(base-reference)/torch.linalg.vector_norm(reference))
        assert difference < (5e-4 if args.fused_guidance else 1e-7),difference
        write(args.output/'pilot_complete.json',dict(pilot=pilot,stock_relative_difference=difference,selected_batch_size=chosen,
              estimated_full_seconds=96*1000/chosen*pilot[-1]['seconds']*1.15))
        return
    jobs=[(t,i) for t in args.tasks for i in args.offsets]
    assigned=[v for j,v in enumerate(jobs) if j%args.num_shards==args.shard_index]
    with (args.output/f'run_{args.shard_index}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for task,offset in assigned:
            folder=args.output/task/f'offset{offset}'
            folder.mkdir(parents=True,exist_ok=True)
            cfg=configuration(protocol,selection,task,source)
            cfg.runtime_metadata['fused_guidance'] = args.fused_guidance
            cfg.offset=offset
            cfg.mask_seed=20260912+offset
            cfg.sample_seed=20260912+offset
            write(folder/'config.json',cfg.asdict())
            # Every K is independently timed, with the same nested noise paths.
            for k in KS:
                dest=folder/f'K{k}.pt'
                receipt_path=folder/f'K{k}.json'
                if receipt_path.exists():
                    old=json.loads(receipt_path.read_text())
                    assert digest(dest)==old['result_sha256']
                    continue
                predictions=[]; batches=[]
                for start in range(0,k,args.batch_size):
                    pred,_,receipt=fast_sample(cfg,truths[offset],bundle,range(start,min(k,start+args.batch_size)))
                    predictions.append(pred); batches.append(receipt)
                pred=torch.cat(predictions)
                average=pred.double().mean(0).float()
                truth=torch.cat([truths[offset].coef,truths[offset].sol],1)[0]
                errors={f:float(torch.linalg.vector_norm((average[j]-truth[j]).double())/torch.linalg.vector_norm(truth[j].double()))
                        for j,f in enumerate(['a','u'])}
                payload=dict(predictions=pred,mean=average,truth=truth,task=task,offset=offset,K=k,
                             mask_seed=cfg.mask_seed,sample_seed=cfg.sample_seed,config=cfg.asdict())
                torch.save(payload,dest)
                row=dict(task=task,offset=offset,K=k,errors=errors,
                         seconds=sum(x['seconds'] for x in batches),
                         peak_bytes=max(x['peak_bytes'] for x in batches),batches=batches,
                         num_steps=100,nfe_per_draw=100,result_sha256=digest(dest),script_sha256=digest(__file__))
                write(receipt_path,row)
                print('DONE',task,offset,k,f"{row['seconds']:.3f}s",errors,flush=True)
        write(args.output/f'complete_{args.shard_index}.json',dict(jobs=assigned,completed_unix=time.time()))


if __name__ == '__main__': main()

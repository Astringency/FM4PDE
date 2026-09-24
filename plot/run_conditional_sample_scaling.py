#!/usr/bin/env python3
"""Fixed-observation conditional means from one canonical 1000-draw pool.

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
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'plot'))
from run_ablation_study import digest, write
KS = [1, 3, 10, 100, 1000]
TASKS = ['forward', 'inverse', 'both']
OFFSETS = list(range(1500, 1532))
HISTORICAL_COMMIT = '1f1573bfbc246b83e48a3b46402c8f2467488e0d'


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode() + json.dumps(list(value.shape), separators=(',', ':')).encode()
                          + value.numpy().tobytes()).hexdigest()


def source_identity():
    paths = sorted({*ROOT.glob('sampling/*.py'), *ROOT.glob('flow_matching/path/*.py'),
                    ROOT / 'sampling/batching.py'})
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def family_identity(cfg):
    c = cfg.asdict()
    for key in ['offset', 'mask_seed', 'sample_seed', 'checkpoint_path', 'output_dir', 'device']:
        c.pop(key, None)
    return json_digest(c)


def batch_ranges(batch_size):
    """Stop exactly when a requested prefix becomes available."""
    start = 0
    for k in KS:
        while start < k:
            stop = min(start + batch_size, k)
            yield start, stop
            start = stop


def fixed_observations(cfg, truth):
    """Generate one mask pair per physical input, before expanding any draws."""
    import torch
    from sampling.masks import make_pair_masks
    pair = torch.cat([truth.coef, truth.sol], 1).detach().cpu()
    assert pair.shape == (1, 2, 128, 128)
    masks = make_pair_masks(truth.coef.shape, truth.sol.shape, cfg.num_obs,
                            cfg.sensor_mode, cfg.shared_mask, cfg.mask_seed, device='cpu')
    mask = torch.cat([masks.coef, masks.sol], 1).cpu()
    assert set(mask.unique().tolist()) <= {0., 1.}
    assert torch.equal(mask.sum((-2, -1)), torch.full((1, 2), 500.))
    values = pair * mask
    fixed = dict(truth=pair, masks=mask, observations=values,
                 observed_fields=['coef'] if cfg.task == 'forward' else ['sol'] if cfg.task == 'inverse' else ['coef', 'sol'],
                 metadata={**masks.metadata, 'source': 'single_mask_pair_broadcast_to_every_draw'})
    fixed['hashes'] = {key: tensor_digest(fixed[key]) for key in ['truth', 'masks', 'observations']}
    return fixed


def expand_fixed(fixed, gt, batch_size, device):
    import torch
    from sampling.masks import PairMasks
    from sampling.losses import ObservationTargets
    mask = fixed['masks'].to(device).expand(batch_size, -1, -1, -1)
    values = fixed['observations'].to(device).expand(batch_size, -1, -1, -1)
    pair = torch.cat([gt.coef, gt.sol], 1)
    # Check the tensors actually handed to the loss, not merely metadata.
    assert torch.equal(pair, fixed['truth'].to(device).expand_as(pair))
    assert torch.equal(pair * mask, values)
    assert torch.equal(mask, mask[:1].expand_as(mask))
    assert torch.equal(values, values[:1].expand_as(values))
    proof = dict(hashes={key: tensor_digest(value[:1]) for key, value in
                        [('truth', pair), ('masks', mask), ('observations', values)]},
                 all_rows_identical=True, checked_rows=batch_size,
                 observed_fields=fixed['observed_fields'])
    assert proof['hashes'] == fixed['hashes']
    masks = PairMasks(mask[:, :1], mask[:, 1:], copy.deepcopy(fixed['metadata']))
    observations = ObservationTargets(values[:, :1], values[:, 1:], values[:, :1], values[:, 1:])
    return masks, observations, proof


def atomic_torch(path, payload):
    import torch
    path = Path(path)
    temp = path.with_name('.partial_' + uuid.uuid4().hex + '_' + path.name)
    torch.save(payload, temp)
    os.replace(temp, path)


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


def fast_sample(cfg, truth, bundle, indices, fixed, steps=100):
    import torch
    from sampling.batching import combine_truths
    import sampling.runner as r
    c = copy.deepcopy(cfg)
    c.batch_size = len(indices)
    c.initial_noise_source_indices = list(indices)
    gt = combine_truths({c.offset: truth}, [c.offset]*len(indices), 'cuda:0')
    masks, observations, observation_proof = expand_fixed(fixed, gt, len(indices), 'cuda:0')
    net, normalizer, _ = bundle
    assert c.obs_l2_reference_mse_zeta_a is None and c.obs_l2_reference_mse_zeta_u is None
    r._set_seed(c.sample_seed)
    grid = r.make_time_grid(c.time_grid, c.num_steps, device='cuda:0', eta=c.time_grid_eta)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    calls = {'count':0}
    def count_call(*_): calls['count'] += 1
    hook = net.model.register_forward_hook(count_call)
    start = time.perf_counter()
    x = r._sample_initial_noise(c, gt, 'cuda:0')
    initial_noise_hashes = [tensor_digest(row) for row in x.detach().cpu()]
    for step in range(steps):
        x_cur = x.detach().clone().requires_grad_(True)
        t, t_next = grid[step], grid[step+1]
        out = r.sampler_step(net, x_cur, t, t_next, 'stochastic', c.step_method,
                             c.loss_state, device='cuda:0',
                             stochastic_noise_source_batch_size=1000,
                             stochastic_noise_source_indices=list(indices))
        phys = r._physical_from_model_state(out.x_loss_state, c, normalizer)
        losses = r.compute_guidance_losses(phys, gt, masks, c, observations=observations)
        assert losses.pde_residual_status != 'error'
        coeffs = r.scheduler_coefficients(t, scheduler='CondOT')
        affine = r.affine_coefficients(coeffs, training='velocity')
        schedule = r.make_zeta_schedule(c, t, t_next, affine.b_t, step=step)
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
    hook.remove()
    assert calls['count'] == steps
    peak = torch.cuda.max_memory_allocated()
    pred = torch.cat([phys.coef, phys.sol], dim=1).detach().cpu()
    assert torch.isfinite(pred).all()
    return pred, mean.cpu(), dict(seconds=seconds, peak_bytes=peak, batch_size=len(indices),
                                 seed_indices=list(indices), num_steps=steps, nfe=calls['count'],
                                 observations=observation_proof, initial_noise_hashes=initial_noise_hashes)


def configured_case(protocol, selection, source, task, offset, output):
    cfg = configuration(protocol, selection, task, source)
    cfg.offset = offset
    cfg.mask_seed = cfg.sample_seed = 20260912 + offset
    cfg.output_dir = str(output.resolve())
    cfg.runtime_metadata.update(fused_guidance=True, observation_source='single_frozen_pair',
                                conditional_pool_size=1000, timing_mode='cumulative_prefix')
    return cfg


def compare_predictions(left, right, truth):
    import torch
    rows = []
    for j, field in enumerate(['a', 'u']):
        x, ref, y = left[:, j].double(), right[:, j].double(), truth[:, j].double()
        diff = float(torch.linalg.vector_norm(x - ref) / torch.linalg.vector_norm(ref))
        left_error = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y))
        ref_error = float(torch.linalg.vector_norm(ref - y) / torch.linalg.vector_norm(y))
        rows.append(dict(field=field, prediction_relative_difference=diff,
                         error_delta_percentage_points=100 * (left_error - ref_error)))
        assert diff < .005, rows[-1]  # Original TF32 validation threshold; not widened.
    return rows


def run_pilot(args, protocol, selection, source, truths, bundle, identity):
    import torch
    rows = []
    for task in TASKS:
        cfg = configured_case(protocol, selection, source, task, 1100, args.output / 'pilot' / task)
        fixed = fixed_observations(cfg, truths[1100])
        warmups = []
        for b in sorted({1, 8, 32, args.batch_size}):
            if b > args.batch_size:
                continue
            free, total = torch.cuda.mem_get_info()
            if warmups:
                projected = warmups[-1]['peak_bytes'] * b / warmups[-1]['batch_size']
                assert projected < .70 * total, ('pilot memory guard', b, projected, total)
            pred, mean, receipt = fast_sample(cfg, truths[1100], bundle, range(b), fixed, steps=5)
            warmups.append(receipt)
            print('PILOT_MEMORY', task, b, receipt['peak_bytes'], flush=True)
            del pred, mean
            torch.cuda.empty_cache()
        single, _, single_receipt = fast_sample(cfg, truths[1100], bundle, [0], fixed)
        batch, _, batch_receipt = fast_sample(cfg, truths[1100], bundle, range(args.batch_size), fixed)
        assert batch_receipt['initial_noise_hashes'][0] == single_receipt['initial_noise_hashes'][0]
        batch_checks = compare_predictions(batch[:1], single, fixed['truth'])
        stock_cfg = copy.deepcopy(cfg)
        stock_cfg.runtime_metadata['fused_guidance'] = False
        stock, _, stock_receipt = fast_sample(stock_cfg, truths[1100], bundle, [0], fixed)
        stock_checks = compare_predictions(single, stock, fixed['truth'])
        # Additional canonical rows exercise a split prefix, not just row zero.
        split, _, split_receipt = fast_sample(cfg, truths[1100], bundle, [1, 2], fixed)
        assert split_receipt['initial_noise_hashes'] == batch_receipt['initial_noise_hashes'][1:3]
        prefix_checks = compare_predictions(split, batch[1:3], fixed['truth'].expand(2, -1, -1, -1))
        assert batch_receipt['peak_bytes'] < .70 * torch.cuda.get_device_properties(0).total_memory
        row = dict(task=task, family_sha256=family_identity(cfg), warmups=warmups,
                   single=single_receipt, batch=batch_receipt, stock=stock_receipt, split=split_receipt,
                   batch_vs_single=batch_checks, fused_vs_stock=stock_checks, prefix_vs_batch=prefix_checks)
        rows.append(row)
        atomic_torch(args.output / f'pilot_{task}.pt', dict(fixed=fixed, single=single, batch=batch,
                     stock=stock, split=split, checks=row, config=cfg.asdict()))
        write(args.output / 'pilot_progress.json', dict(status='running', tasks=rows))
        print('PILOT_TASK_PASS', task, json.dumps({k: row[k] for k in
              ['batch_vs_single', 'fused_vs_stock', 'prefix_vs_batch']}), flush=True)
        del single, batch, stock, split
        torch.cuda.empty_cache()
    write(args.output / 'pilot_complete.json', dict(status='pass', batch_size=args.batch_size,
          identity=identity, gpu=torch.cuda.get_device_name(), checks=rows,
          completed_unix=time.time(), timing_mode='cumulative_prefix'))


def run_case(args, cfg, truth, bundle, identity, folder):
    import torch
    fixed = fixed_observations(cfg, truth)
    config = cfg.asdict()
    binding = dict(identity=identity, config_sha256=json_digest(config), observation_hashes=fixed['hashes'],
                   task=cfg.task, offset=cfg.offset, batch_size=args.batch_size)
    marker = folder / 'complete.json'
    if marker.exists():
        done = json.loads(marker.read_text())
        assert done['binding'] == binding and done['status'] == 'complete'
        assert digest(folder / 'pool.pt') == done['pool_sha256']
        return done
    chunks = folder / 'batches'
    chunks.mkdir(parents=True, exist_ok=True)
    predictions, batches, prefix_records = [], [], {}
    for start, stop in batch_ranges(args.batch_size):
        target = chunks / f'{start:04d}_{stop:04d}'
        if target.exists():
            receipt = json.loads((target / 'receipt.json').read_text())
            assert receipt['binding'] == binding and receipt['seed_indices'] == list(range(start, stop))
            assert digest(target / 'prediction.pt') == receipt['prediction_file_sha256']
            pred = torch.load(target / 'prediction.pt', map_location='cpu', weights_only=False)
            assert tensor_digest(pred) == receipt['prediction_tensor_sha256']
        else:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            pred, _, receipt = fast_sample(cfg, truth, bundle, range(start, stop), fixed)
            torch.cuda.synchronize()
            receipt['active_seconds'] = time.perf_counter() - start_time
            receipt.update(binding=binding, prediction_tensor_sha256=tensor_digest(pred))
            temp = chunks / ('.partial_' + uuid.uuid4().hex)
            temp.mkdir()
            torch.save(pred, temp / 'prediction.pt')
            receipt['prediction_file_sha256'] = digest(temp / 'prediction.pt')
            write(temp / 'receipt.json', receipt)
            os.rename(temp, target)
            print('BATCH', cfg.task, cfg.offset, start, stop, f"{receipt['active_seconds']:.3f}s", flush=True)
        assert receipt['observations']['hashes'] == fixed['hashes']
        assert receipt['observations']['all_rows_identical']
        assert receipt['observations']['checked_rows'] == stop - start
        assert receipt['nfe'] == receipt['num_steps'] == 100
        predictions.append(pred)
        batches.append(receipt)
        if stop in KS:
            prefix_path = folder / f'prefix_{stop}.json'
            mean_start = time.perf_counter()
            mean = torch.cat(predictions).double().mean(0)
            mean_seconds = time.perf_counter() - mean_start
            errors = {field: float(torch.linalg.vector_norm(mean[j] - fixed['truth'][0, j].double()) /
                                  torch.linalg.vector_norm(fixed['truth'][0, j].double()))
                      for j, field in enumerate(['a', 'u'])}
            if prefix_path.exists():
                prefix = json.loads(prefix_path.read_text())
                assert prefix['binding'] == binding and prefix['mean_sha256'] == tensor_digest(mean)
            else:
                prefix = dict(K=stop, binding=binding, mean_sha256=tensor_digest(mean), errors=errors,
                     mean_seconds=mean_seconds,
                     seconds=sum(b['active_seconds'] for b in batches)
                             + sum(p['mean_seconds'] for p in prefix_records.values()) + mean_seconds,
                     compute_seconds=sum(b['seconds'] for b in batches),
                     peak_bytes=max(b['peak_bytes'] for b in batches),
                     timing_mode='cumulative_active_generation_and_prefix_means_excluding_checkpoint_io')
                write(prefix_path, prefix)
            prefix_records[stop] = prefix
    prediction = torch.cat(predictions)
    assert prediction.shape == (1000, 2, 128, 128) and torch.isfinite(prediction).all()
    assert [i for batch in batches for i in batch['seed_indices']] == list(range(1000))
    payload = dict(predictions=prediction, means={k: prediction[:k].double().mean(0) for k in KS},
                   fixed=fixed, config=config, binding=binding, batches=batches, prefixes=prefix_records)
    atomic_torch(folder / 'pool.pt', payload)
    done = dict(status='complete', binding=binding, pool_sha256=digest(folder / 'pool.pt'),
                predictions=1000, prefix_counts=KS, completed_unix=time.time(),
                seconds=prefix_records[1000]['seconds'])
    write(marker, done)
    print('CASE_COMPLETE', cfg.task, cfg.offset, done['seconds'], flush=True)
    return done


def main():
    import torch
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pilot', 'run'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--pilot-certificate', type=Path)
    args = parser.parse_args()
    assert args.output.is_absolute() and args.inputs.is_absolute()
    assert 3 <= args.batch_size <= 64
    assert args.worker.replace('_', '').replace('-', '').isalnum()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info()
    assert free > .75 * total, ('GPU is not sufficiently free', free, total)
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.inputs / 'poisson'
    protocol = json.loads((source / 'protocol.json').read_text())
    selection = json.loads(args.selection.read_text())
    assert digest(source / 'protocol.json') == selection['protocol_sha256']
    assert digest(source / 'truths.pt') == protocol['truth_sha256']
    assert digest(source / 'weights.pth') == protocol['weights_sha256']
    assert protocol['evaluation_ids'] == OFFSETS
    identity = dict(protocol_sha256=digest(source / 'protocol.json'), truth_sha256=protocol['truth_sha256'],
                     weights_sha256=protocol['weights_sha256'], selection_sha256=digest(args.selection),
                     runner_sha256=digest(__file__), source_hashes=source_identity(), historical_commit=HISTORICAL_COMMIT,
                     torch=torch.__version__, tf32=True, fused_guidance=True)
    if args.mode == 'run':
        assert args.pilot_certificate is not None
        pilot = json.loads(args.pilot_certificate.read_text())
        assert pilot['status'] == 'pass' and pilot['identity'] == identity
        assert pilot['batch_size'] == args.batch_size and pilot['gpu'] == torch.cuda.get_device_name()
        assert {row['task'] for row in pilot['checks']} == set(TASKS)
        for row in pilot['checks']:
            cfg = configured_case(protocol, selection, source, row['task'], 1500, args.output)
            assert row['family_sha256'] == family_identity(cfg)
    invocation = dict(identity=identity, host=socket.gethostname(), pid=os.getpid(), gpu=torch.cuda.get_device_name(),
             visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), args={k: str(v) if isinstance(v, Path) else v
               for k, v in vars(args).items()}, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             started_unix=time.time(), case_count=96, pool_size=1000, tasks=TASKS, offsets=OFFSETS,
             timing='Cumulative active preparation, generation, physical conversion, transfer and prefix means; excludes input/model loading, checkpoint writing, resume downtime and final pool serialization.')
    invocation_path = args.output / 'invocations' / f'{args.mode}_{args.worker}_{uuid.uuid4().hex}.json'
    write(invocation_path, invocation)
    truths = torch.load(source / 'truths.pt', map_location='cpu', weights_only=False)
    bundle = load_fm4pde_checkpoint_bundle(str(source / 'weights.pth'), 'poisson', 'cuda:0', model_profile='recommended')
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in bundle[0].model.modules())
    if args.mode == 'pilot':
        run_pilot(args, protocol, selection, source, truths, bundle, identity)
        return
    state_path = args.output / 'workers' / f'{args.worker}.json'
    state = dict(status='running', worker=args.worker, invocation=str(invocation_path), completed_cases=[], pid=os.getpid())
    write(state_path, state)
    with (args.output / f'worker_{args.worker}.lock').open('a+') as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for offset in OFFSETS:
                for task in TASKS:
                    folder = args.output / 'cases' / task / f'offset{offset}'
                    folder.mkdir(parents=True, exist_ok=True)
                    with (folder / 'case.lock').open('a+') as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        cfg = configured_case(protocol, selection, source, task, offset, folder)
                        state.update(active_case=[task, offset], updated_unix=time.time())
                        write(state_path, state)
                        run_case(args, cfg, truths[offset], bundle, identity, folder)
                        state['completed_cases'].append([task, offset])
                        state.update(active_case=None, updated_unix=time.time())
                        write(state_path, state)
            state.update(status='complete', completed_unix=time.time())
            write(state_path, state)
        except BaseException as exc:
            state.update(status='error', error=repr(exc), updated_unix=time.time())
            write(state_path, state)
            raise


if __name__ == '__main__':
    main()

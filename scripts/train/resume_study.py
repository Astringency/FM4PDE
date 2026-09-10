"""Bounded full-data continuation of a saved formal FM4PDE training checkpoint.

Keeps the saved architecture, normalizer, parameters and Adam moments. Compares
three one-epoch continuations with matched random draws, then continues the best
trained candidate with a fresh cosine schedule. Validation never advances the
training RNG. All outputs, including recoverable optimizer state, stay in the
explicit pretrained output directory.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from data.specs import get_pde_spec
from data.training_manifest import load_training_file_manifest
from data.transform import PDEStandardizer
from models.legacy_checkpoint import read_checkpoint
from models.model_configs import instantiate_model
from train import _load_training_and_validation_data, _train_val_split_indices
from training.load_and_save import _load_optimizer_state_preserving_runtime_options

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [cpu_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(cpu_tree(v) for v in x)
    return x


def optimizer_steps(optimizer):
    return [int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]


def restore(model, optimizer, checkpoint, lr):
    model.train()
    model.load_state_dict(checkpoint['model_for_resume'], strict=True)
    _load_optimizer_state_preserving_runtime_options(optimizer, checkpoint['optimizer'])
    for group in optimizer.param_groups:
        group['lr'] = float(lr)
        group['initial_lr'] = float(lr)
    optimizer.zero_grad(set_to_none=True)


def flow_loss(model, xt, t, target, coarse_weight=0.0, amp=True):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        velocity = model(xt, t, extra={})
    error = velocity.float() - target
    loss = error.square().mean()
    if coarse_weight:
        loss = loss + coarse_weight * F.avg_pool2d(error, 8).square().mean()
    return loss


def train_group(model, optimizer, samples, batch_size, coarse_weight, clip_grad):
    optimizer.zero_grad(set_to_none=True)
    samples = samples.cuda(non_blocking=True)
    noise = torch.randn_like(samples)
    t = torch.rand(len(samples), device='cuda').clamp(1e-5, 1 - 1e-5)
    xt = noise.lerp(samples, t[:, None, None, None])
    target = samples - noise
    total = 0.0
    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        loss = flow_loss(model, xt[start:stop], t[start:stop], target[start:stop], coarse_weight)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite training loss')
        weight = (stop - start) / len(samples)
        (loss * weight).backward()
        total += float(loss.detach()) * weight
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=True)
    optimizer.step()
    return total, float(norm)


def probe_batch(model, optimizer, checkpoint, shape, out):
    measurements = []
    max_budget = min(torch.cuda.get_device_properties(0).total_memory * 0.70, 56 * 2**30)
    for batch_size in (1, 2, 4, 8, 16, 32, 64):
        if measurements:
            previous = measurements[-1]
            projected = previous['base_bytes'] + 2.6 * (previous['peak_bytes'] - previous['base_bytes']) + 2 * 2**30
            if projected > max_budget:
                break
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn((batch_size, *shape), device='cuda')
        torch.cuda.synchronize()
        begin = time.monotonic()
        for _ in range(2):
            train_group(model, optimizer, x, batch_size, 0.0, 1.0)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - begin
        peak = torch.cuda.max_memory_allocated()
        record = dict(batch_size=batch_size, base_bytes=base, peak_bytes=peak,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      samples_per_second=2 * batch_size / elapsed, seconds=elapsed)
        measurements.append(record)
        write(out / 'batch_probe.json', measurements)
        print('BATCH_PROBE', json.dumps(record), flush=True)
        del x
        if peak > max_budget:
            break
    eligible = [r for r in measurements if r['peak_bytes'] < max_budget]
    if not eligible:
        raise RuntimeError('No batch passed the conservative memory limit')
    best = max(eligible, key=lambda r: r['samples_per_second'])
    restore(model, optimizer, checkpoint, 1e-5)
    return int(best['batch_size'])


@torch.no_grad()
def evaluate(model, data, positions, seed, repeats=1, batch_size=4):
    was_training = model.training
    model.eval()
    values = []
    coarse_values = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        # CPU generator makes draws independent of training and GPU kernel choice.
        gen = torch.Generator().manual_seed(seed)
        for repeat in range(repeats):
            losses, coarse = [], []
            t_all = torch.rand(len(positions), generator=gen).clamp(1e-5, 1 - 1e-5)
            for start in range(0, len(positions), batch_size):
                pos = positions[start:start + batch_size]
                sample = data[pos].cuda()
                noise = torch.randn(sample.shape, generator=gen).cuda()
                t = t_all[start:start + len(pos)].cuda()
                xt = noise.lerp(sample, t[:, None, None, None])
                # FP32 forward for model selection and reporting, identical for all candidates.
                velocity = model(xt, t, extra={})
                error = velocity.float() - (sample - noise)
                losses.extend(error.square().flatten(1).mean(1).cpu().tolist())
                coarse.extend(F.avg_pool2d(error, 8).square().flatten(1).mean(1).cpu().tolist())
            values.append(losses)
            coarse_values.append(coarse)
    model.train(was_training)
    per_sample = np.asarray(values).mean(axis=0)
    coarse_sample = np.asarray(coarse_values).mean(axis=0)
    assert np.isfinite(per_sample).all()
    return dict(mean=float(per_sample.mean()), coarse_mean=float(coarse_sample.mean()),
                per_sample=per_sample.tolist(), coarse_per_sample=coarse_sample.tolist(),
                positions=positions.tolist(), seed=seed, repeats=repeats,
                inference_dtype='float32', tf32=True)


def save_checkpoint(path, source, model, optimizer, scheduler, epoch, metadata):
    payload = {k: v for k, v in source.items()
               if k not in {'model', 'model_for_resume', 'model_ema', 'optimizer', 'lr_schedule', 'scaler'}}
    state = cpu_tree(model.state_dict())
    args = copy.copy(source['args'])
    args = dict(args) if isinstance(args, dict) else dict(vars(args))
    scheduler_name = 'constant' if isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR) else 'warmup_cosine'
    args.update(lr=metadata['learning_rate'], sampling_dtype='bfloat16', batch_size=metadata['batch_size'],
                accum_iter=64 // metadata['batch_size'], world_size=1, resume=metadata['source_checkpoint'],
                epochs=epoch + 1, start_epoch=source['epoch'] + 1, lr_scheduler=scheduler_name, warmup_epochs=0,
                min_lr=1e-6, use_ema=False, output_dir=str(Path(path).parent), clip_grad=1.0)
    payload.update(model=state, model_for_resume=state, model_ema=None, optimizer=cpu_tree(optimizer.state_dict()),
                   lr_schedule=cpu_tree(scheduler.state_dict()), scaler=None, epoch=epoch,
                   args=argparse.Namespace(**args), lr_scheduler=scheduler_name,
                   resolved_lr_scheduler=scheduler_name, warmup_epochs=0, min_lr=1e-6,
                   resume_study=metadata, use_ema=False, has_ema=False, inference_weight='raw')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    torch.save(payload, temporary)
    temporary.replace(path)


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    begin = time.monotonic()
    deadline = begin + args.minutes * 60
    out = Path(args.output).resolve()
    assert '/outputs/pretrained/' in str(out), 'Explicit pretrained output directory required'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists(), 'This run is already complete'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    free, total = torch.cuda.mem_get_info()
    assert free > 70 * 2**30, 'Expected an idle 80GB GPU before the model is loaded'
    source = read_checkpoint(args.checkpoint, args.pde)
    assert source['checkpoint_schema_version'] == 3
    assert not source.get('use_ema') and source.get('optimizer')
    assert source['args'].dataset == args.pde if not isinstance(source['args'], dict) else source['args']['dataset'] == args.pde
    saved_args = vars(source['args']) if not isinstance(source['args'], dict) else source['args']
    normalizer = PDEStandardizer.from_state_dict(source['normalizer'])
    checkpoint_sha = file_sha(args.checkpoint)
    manifest = load_training_file_manifest(ROOT / 'configs/training_data.yaml',
                                          data_root=args.data_root, pde_names=[args.pde])
    assert all('test' not in str(p).lower() for p in manifest[args.pde])
    source_files = [dict(path=str(p), bytes=Path(p).stat().st_size,
                         mtime_ns=Path(p).stat().st_mtime_ns) for p in manifest[args.pde]]
    write(out / 'protocol.json', dict(args=vars(args), checkpoint_sha256=checkpoint_sha,
          source_epoch=source['epoch'], model_config=source['model_config'],
          normalizer_source='unchanged checkpoint normalizer', training_files=source_files,
          validation_policy='same original deterministic 45000/5000 split; 512 development, 4488 confirmation',
          effective_batch_size=64, training_dtype='bfloat16 autocast with FP32 parameters and Adam moments',
          scaler_policy='BF16 continuation without loss scaling; original scaler retained in source checkpoint',
          lr_policy='restore all Adam moments and step counters, intentionally restart LR schedule',
          trials=[{'lr': 1e-5, 'coarse_weight': 0.0}, {'lr': 3e-5, 'coarse_weight': 0.0}, {'lr': 1e-5, 'coarse_weight': 1.0}],
          git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          python=os.sys.version, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
          host=socket.gethostname(), pid=os.getpid(), wall_start_unix=time.time()))
    print('LOAD_DATA', args.pde, flush=True)
    raw, _, _, raw_val, _, _, split = _load_training_and_validation_data(
        [args.pde], args.data_root, data_size=5, seed=int(saved_args.get('seed', 0)),
        train_files_by_pde=manifest)
    assert len(raw) == 45000 and len(raw_val) == 5000
    train_idx, val_idx = _train_val_split_indices(50000,
        seed=int(saved_args.get('seed', 0)) + int(get_pde_spec(args.pde).label_id) * 1009, val_ratio=0.1)
    assert not set(train_idx.tolist()) & set(val_idx.tolist())
    train = normalizer.transform(raw).contiguous()
    val = normalizer.transform(raw_val).contiguous()
    fitted = PDEStandardizer.fit(raw)
    write(out / 'data_check.json', dict(split=split, normalizer_unchanged=True,
          checkpoint_mean=normalizer.mean.flatten().tolist(), checkpoint_std=normalizer.std.flatten().tolist(),
          current_train_mean=fitted.mean.flatten().tolist(), current_train_std=fitted.std.flatten().tolist(),
          std_relative_difference=((fitted.std - normalizer.std).abs() / normalizer.std).flatten().tolist(),
          train_ids_sha256=hashlib.sha256(train_idx.numpy().tobytes()).hexdigest(),
          val_ids_sha256=hashlib.sha256(val_idx.numpy().tobytes()).hexdigest()))
    perm = torch.randperm(len(val), generator=torch.Generator().manual_seed(args.seed + 81))
    development, confirmation = perm[:512], perm[512:]
    torch.save(dict(train_indices=train_idx, val_indices=val_idx, development_positions=development,
                    confirmation_positions=confirmation), out / 'split_indices.pt')
    del raw, raw_val
    gc.collect()
    model = instantiate_model(args.pde, use_ema=False, model_config=source['model_config']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    restore(model, optimizer, source, 1e-5)
    old_steps = optimizer_steps(optimizer)
    assert len(set(old_steps)) == 1
    source_step = old_steps[0]
    microbatch = probe_batch(model, optimizer, source, tuple(train.shape[1:]), out)
    assert optimizer_steps(optimizer) == old_steps
    write(out / 'resume_verification.json', dict(source_epoch=source['epoch'],
          optimizer_states_restored=len(old_steps), optimizer_step=source_step,
          optimizer_moments_preserved=True, model_for_resume_loaded_strictly=True,
          selected_batch_size=microbatch, effective_batch_size=64,
          accumulation=64 // microbatch, source_checkpoint_sha256=checkpoint_sha))
    baseline = evaluate(model, val, development, args.seed + 9001, repeats=2)
    write(out / 'baseline_development.json', baseline)
    # Confirmation is measured before training and is never read by model selection.
    baseline_confirm = evaluate(model, val, confirmation, args.seed + 19001)
    write(out / 'baseline_confirmation.json', baseline_confirm)
    if args.probe_only:
        write(out / 'probe_complete.json', dict(batch_size=microbatch, seconds=time.monotonic()-begin))
        return
    best_score = math.inf
    best_path = out / 'best_resume.pth'
    current_updates = 0
    epoch_times = []
    trials = [(1e-5, 0.0), (3e-5, 0.0), (1e-5, 1.0)]
    results = []

    def epoch_run(epoch, lr, coarse_weight, warmup=False):
        nonlocal current_updates
        started = time.monotonic()
        model.train()
        torch.manual_seed(args.seed + epoch)
        permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch))
        sampler = permutation.tolist()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train), batch_size=64,
                 sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
        count, total_loss, grad_total = 0, 0.0, 0.0
        for i, (samples,) in enumerate(loader):
            if warmup:
                scale = 0.2 + 0.8 * min((i + 1) / 128, 1.0)
                for group in optimizer.param_groups:
                    group['lr'] = lr * scale
            loss, grad = train_group(model, optimizer, samples, microbatch, coarse_weight, 1.0)
            total_loss += loss * len(samples)
            grad_total += grad
            count += len(samples)
            current_updates += 1
            if i % 100 == 0:
                print('TRAIN', args.pde, epoch, i, len(loader), 'loss', loss,
                      'lr', optimizer.param_groups[0]['lr'], flush=True)
        assert count == len(train)
        assert set(optimizer_steps(optimizer)) == {source_step + current_updates}
        elapsed = time.monotonic() - started
        epoch_times.append(elapsed)
        return dict(epoch=epoch, train_objective=total_loss / count,
                    mean_gradient_norm=grad_total / len(loader), samples=count,
                    updates=current_updates, seconds=elapsed)

    def record_candidate(phase, epoch, lr, coarse_weight, scheduler, train_stats):
        nonlocal best_score
        score = evaluate(model, val, development, args.seed + 9001, repeats=2)
        metadata = dict(pde=args.pde, phase=phase, source_checkpoint=str(Path(args.checkpoint).resolve()),
                        source_sha256=checkpoint_sha, source_epoch=source['epoch'], source_optimizer_step=source_step,
                        updates=current_updates, learning_rate=lr, coarse_weight=coarse_weight,
                        batch_size=microbatch, effective_batch_size=64, validation_score=score['mean'],
                        train_stats=train_stats, scheduler_step_unit='epoch')
        row = dict(**metadata, validation_coarse=score['coarse_mean'], elapsed_seconds=time.monotonic()-begin)
        results.append(row)
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        write(out/'validation'/f'{phase}_epoch{epoch}.json', score)
        if score['mean'] < best_score:
            best_score = score['mean']
            save_checkpoint(best_path, source, model, optimizer, scheduler, epoch, metadata)
            write(out/'best_selection.json', row)
        print('VALIDATION', args.pde, phase, epoch, score['mean'], 'best', best_score, flush=True)
        return row

    for index, (lr, coarse_weight) in enumerate(trials):
        restore(model, optimizer, source, lr)
        current_updates = 0
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
        epoch = int(source['epoch']) + 1
        stats = epoch_run(epoch, lr, coarse_weight, warmup=True)
        scheduler.step()
        record_candidate(f'pilot{index}', epoch, lr, coarse_weight, scheduler, stats)
    selected = read_checkpoint(best_path, args.pde)
    winner = selected['resume_study']
    lr, coarse_weight = winner['learning_rate'], winner['coarse_weight']
    restore(model, optimizer, selected, lr)
    current_updates = int(winner['updates'])
    start_epoch = int(selected['epoch']) + 1
    epoch = int(selected['epoch'])
    last_meta = dict(winner)
    del selected
    estimated_epoch = max(epoch_times[-3:])
    # Reserve validation and an atomic final checkpoint; never start an epoch that
    # is expected to consume the remaining budget.
    remaining = deadline - time.monotonic()
    epochs = max(0, int((remaining - max(300, estimated_epoch * 0.6)) / (estimated_epoch * 1.25)))
    epochs = min(epochs, args.max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    write(out/'continuation_plan.json', dict(selected=winner, extra_epochs=epochs,
          estimated_epoch_seconds=estimated_epoch, remaining_seconds=remaining,
          minutes_budget=args.minutes, selection_uses_confirmation=False))
    for offset in range(epochs):
        if offset and deadline - time.monotonic() < estimated_epoch * 1.25 + 240:
            break
        epoch = start_epoch + offset
        stats = epoch_run(epoch, lr, coarse_weight)
        scheduler.step()
        record_candidate('continuation', epoch, lr, coarse_weight, scheduler, stats)
        last_meta = dict(results[-1])
    # Always retain the last resumed optimizer state, even if earlier weights won.
    save_checkpoint(out/'last_resume.pth', source, model, optimizer, scheduler, epoch, last_meta)
    final = read_checkpoint(best_path, args.pde)
    restore(model, optimizer, final, final['resume_study']['learning_rate'])
    confirmation_score = evaluate(model, val, confirmation, args.seed + 19001)
    write(out/'selected_confirmation.json', confirmation_score)
    before = np.asarray(baseline_confirm['per_sample'])
    after = np.asarray(confirmation_score['per_sample'])
    diff = after - before
    status = dict(status='complete', pde=args.pde, source_checkpoint=str(Path(args.checkpoint).resolve()),
          best_checkpoint=str(best_path), best_sha256=file_sha(best_path),
          best_epoch=final['epoch'], best_optimizer_step=final['resume_study']['source_optimizer_step']+final['resume_study']['updates'],
          last_epoch=epoch, baseline_confirmation_mse=float(before.mean()),
          selected_confirmation_mse=float(after.mean()), relative_change=float(after.mean()/before.mean()-1),
          paired_mean_change=float(diff.mean()), paired_mean_change_95ci=[float(diff.mean()-1.96*diff.std(ddof=1)/math.sqrt(len(diff))),float(diff.mean()+1.96*diff.std(ddof=1)/math.sqrt(len(diff)))],
          improved_confirmation=bool(after.mean()<before.mean()), confirmation_samples=len(confirmation),
          checkpoint_sha256_unchanged=file_sha(args.checkpoint)==checkpoint_sha,
          seconds=time.monotonic()-begin, trials=results[:3],
          selected_batch_size=microbatch, effective_batch_size=64,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          downstream_sampling_evaluation='pending separate paired sampler evaluation')
    assert status['checkpoint_sha256_unchanged']
    write(out/'complete.json',status)
    print('COMPLETE',json.dumps(status),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pde', required=True, choices=['poisson','nsnonbounded','darcy','helmholtz','burger'])
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--data-root',default='/large_storage/zhangxf/PDEdata')
    p.add_argument('--minutes',type=float,default=90)
    p.add_argument('--max-epochs',type=int,default=80)
    p.add_argument('--seed',type=int,default=20260910)
    p.add_argument('--probe-only',action='store_true')
    main(p.parse_args())

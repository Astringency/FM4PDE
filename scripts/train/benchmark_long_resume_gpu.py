"""Measure real validation sampling on an additional GPU without touching formal selection."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import torch

from sampling.model_io import load_fm4pde_checkpoint_bundle
from scripts.train.main_resume_sampling import paired_identity, sample_once, validation_inputs
from scripts.train.resume_study import ROOT, file_sha, write


def main(args):
    study = args.study.resolve()
    out = args.output.resolve()
    assert study.is_absolute() and out.is_relative_to(study / 'remote_execution')
    assert 0 < args.memory_limit_gib < args.minimum_free_gib
    assert args.maximum_batch in [1, 2, 4, 8, 16]
    assert not (out / 'complete.json').exists(), 'This benchmark already completed'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    free, total = torch.cuda.mem_get_info()
    assert free > args.minimum_free_gib * 2**30, 'Insufficient free GPU memory'
    limit = args.memory_limit_gib * 2**30
    plan = json.loads((study / 'study_plan.json').read_text())
    job = next(row for row in plan['jobs'] if row['pde'] == args.pde)
    bindings = json.loads((study / 'main_checkpoint_binding.json').read_text())
    binding = next(row for row in bindings['models'] if row['pde'] == args.pde)
    source = Path(job['source'])
    source_sha = file_sha(source)
    assert source_sha == binding['resume_sha256']
    cache = torch.load(study / args.pde / 'sampling_validation_inputs.pt',
                       map_location='cpu', weights_only=False)
    split = torch.load(study / args.pde / 'split_indices.pt', map_location='cpu', weights_only=False)
    ids = cache['original_training_file_pool_ids'].tolist()
    assert cache['pde'] == args.pde and len(ids) == len(set(ids)) == 32
    assert set(ids).isdisjoint(split['train_indices'].tolist())
    assert torch.equal(cache['original_training_file_pool_ids'],
                       split['val_indices'][cache['validation_positions']])
    records = [json.loads(path.read_text()) for path in sorted(
        (study / 'evaluation_inputs' / args.pde / 'id').glob('*.json'))]
    assert len(records) == (2 if args.pde == 'burger' else 5)
    protocol = dict(pde=args.pde,source_checkpoint=str(source),source_sha256=source_sha,
        validation_ids=ids,steps=100,seed=0,precision='FP32 parameters and tensors; TF32 enabled',
        memory_limit_bytes=limit,minimum_free_bytes=args.minimum_free_gib*2**30,
        maximum_batch=args.maximum_batch,paired_candidate_epoch=5,
        buffer_step_metrics=args.buffer_step_metrics,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='Resource and paired functional benchmark on training-validation inputs only; no formal model selection or test sampling')
    if (out / 'protocol.json').exists():
        assert json.loads((out / 'protocol.json').read_text()) == protocol
    else:
        write(out / 'protocol.json', protocol)
    write(out / 'process.json', dict(pid=os.getpid(),host=socket.gethostname(),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        initial_free_bytes=free,total_bytes=total,started_unix=time.time()))
    bundle = load_fm4pde_checkpoint_bundle(str(source), args.pde, 'cuda:0', model_profile='auto')

    def run(record, batch, model, checkpoint, digest, folder):
        positions = list(range(batch))
        gt, masks, sample_ids = validation_inputs(cache, record['config'], positions, 'cuda:0')
        row = sample_once(record['config'], model, str(checkpoint), digest, folder,
                          gt, masks, sample_ids, 32, positions,
                          buffer_step_metrics=args.buffer_step_metrics)
        assert row['peak_bytes'] < limit
        for metrics in row['fields'].values():
            for metric, values in metrics.items():
                if metric != 'basis':
                    assert all(math.isfinite(value) for value in values), metric
        return row

    measurements = []
    baseline_receipts = {}
    for record in records:
        setting = record['setting']
        row = run(record, 1, bundle, source, source_sha, out / 'baseline' / setting / 'batch1')
        baseline_receipts[(setting, 1)] = row
        measurements.append(dict(setting=setting,batch=1,peak_bytes=row['peak_bytes'],seconds=row['seconds']))
        write(out / 'progress.json', dict(state='probing', measurements=measurements))
        print('PROBE_COMPLETE',args.pde,setting,1,row['peak_bytes'],row['seconds'],flush=True)
    previous = max(measurements, key=lambda row: row['peak_bytes'])
    worst = next(record for record in records if record['setting'] == previous['setting'])
    batch = 1
    for candidate in [2, 4, 8, 16]:
        if candidate > args.maximum_batch:
            break
        projected = 2.6 * previous['peak_bytes'] + 2 * 2**30
        if projected >= limit:
            break
        # Reclaim obsolete cached allocations before checking capacity for the next probe.
        torch.cuda.empty_cache()
        available, _ = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        if projected - allocated + 2 * 2**30 >= available:
            break
        row = run(worst, candidate, bundle, source, source_sha,
                  out / 'baseline' / worst['setting'] / f'batch{candidate}')
        baseline_receipts[(worst['setting'], candidate)] = row
        previous = dict(setting=worst['setting'],batch=candidate,
                        peak_bytes=row['peak_bytes'],seconds=row['seconds'])
        measurements.append(previous)
        batch = candidate
        write(out / 'progress.json', dict(state='probing',measurements=measurements))
        print('PROBE_COMPLETE',args.pde,worst['setting'],batch,row['peak_bytes'],row['seconds'],flush=True)
    bundle[0].model.cpu()
    del bundle
    gc.collect()
    torch.cuda.empty_cache()
    resumed = study / args.pde / 'checkpoints' / 'resume_epoch_005.pth'
    resumed_sha = file_sha(resumed)
    bundle = load_fm4pde_checkpoint_bundle(str(resumed), args.pde, 'cuda:0', model_profile='auto')
    assert bundle[2]['resume_study']['source_sha256'] == source_sha
    assert bundle[2]['resume_study']['completed_epochs'] == 5
    row = run(worst, batch, bundle, resumed, resumed_sha,
              out / 'resumed_epoch005' / worst['setting'] / f'batch{batch}')
    paired_identity(baseline_receipts[(worst['setting'], batch)], row)
    write(out / 'complete.json', dict(status='complete',batch_size=batch,
        measurements=measurements,source_sha256=source_sha,resumed_sha256=resumed_sha,
        paired_identity_verified=True,paired_setting=worst['setting'],paired_inputs=batch,
        resumed_peak_bytes=row['peak_bytes'],resumed_seconds=row['seconds'],
        protocol_sha256=file_sha(out/'protocol.json'),ended_unix=time.time(),scope=protocol['scope']))
    print('BENCHMARK_COMPLETE',args.pde,batch,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pde', choices=['poisson','nsnonbounded','darcy','helmholtz','burger'], required=True)
    parser.add_argument('--memory-limit-gib', type=float, default=20.)
    parser.add_argument('--minimum-free-gib', type=float, default=22.)
    parser.add_argument('--maximum-batch', type=int, default=16)
    parser.add_argument('--buffer-step-metrics', action='store_true')
    main(parser.parse_args())

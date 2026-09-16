#!/usr/bin/env python3
"""Compare the fixed-mask B1 pilot with the historical production runner."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'plot'))
from run_paper_ablation_revision import digest, write
from run_conditional_sample_scaling import TASKS, fixed_observations
from sampling.batching import combine_truths
from sampling.config import AblationConfig
from sampling.masks import PairMasks
import sampling.runner as runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    cert = json.loads((args.pilot / 'pilot_complete.json').read_text())
    assert cert['status'] == 'pass'
    source = args.inputs / 'poisson'
    assert digest(source / 'truths.pt') == cert['identity']['truth_sha256']
    assert digest(source / 'weights.pth') == cert['identity']['weights_sha256']
    truths = torch.load(source / 'truths.pt', map_location='cpu', weights_only=False)
    bundle = runner.load_fm4pde_checkpoint_bundle(str(source / 'weights.pth'), 'poisson', 'cuda:0', model_profile='recommended')
    checks = []
    for task in TASKS:
        obj = torch.load(args.pilot / f'pilot_{task}.pt', map_location='cpu', weights_only=False)
        cfg = AblationConfig(**copy.deepcopy(obj['config']))
        cfg.batch_size = 1; cfg.initial_noise_source_indices = [0]
        cfg.output_dir = str((args.output / task).resolve())
        fixed = fixed_observations(cfg, truths[1100])
        assert fixed['hashes'] == obj['fixed']['hashes']
        # Exact B1 masks used by both paths, without resampling.
        mask = fixed['masks'].to('cuda:0')
        masks = PairMasks(mask[:, :1], mask[:, 1:], fixed['metadata'])
        gt = combine_truths(truths, [1100], 'cuda:0')
        with (args.output / f'{task}_stock.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            result = runner.run_single_ablation(cfg, checkpoint_bundle=bundle, ground_truth=gt, observation_masks=masks)
        raw = torch.load(Path(result['run_dir']) / 'result.pt', weights_only=False, map_location='cpu')
        reference = torch.cat([raw['coef_final'], raw['sol_final']], 1).double()
        actual = obj['single'].double()
        assert torch.equal(raw['masks']['coef'], masks.coef.cpu()) if 'masks' in raw else True
        rows = []
        for j, field in enumerate(['a', 'u']):
            relative = float(torch.linalg.vector_norm(actual[:, j] - reference[:, j]) / torch.linalg.vector_norm(reference[:, j]))
            assert relative < 5e-4, (task, field, relative)
            rows.append(dict(field=field, prediction_relative_difference=relative))
        checks.append(dict(task=task, comparisons=rows, stock_result=str(Path(result['run_dir']) / 'result.pt'),
                           stock_sha256=digest(Path(result['run_dir']) / 'result.pt'), pilot_sha256=digest(args.pilot / f'pilot_{task}.pt')))
        print('ACTUAL_STOCK_PASS', task, rows, flush=True)
    write(args.output / 'stock_complete.json', dict(status='pass', checks=checks,
           pilot_certificate_sha256=digest(args.pilot / 'pilot_complete.json'), identity=cert['identity'],
           threshold=5e-4, completed_unix=time.time(), validator_sha256=digest(__file__)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Independently audit the fixed-observation pools and export per-input results."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'plot'))
from run_ablation_study import digest, write
from run_conditional_sample_scaling import configuration, KS, TASKS, OFFSETS


def tensor_hash(x):
    x = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(json.dumps(list(x.shape), separators=(',', ':')).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def json_hash(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit_case(folder, source, protocol, selection, truths, reference_identity=None):
    from sampling.masks import make_pair_masks, PairMasks
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.state import SplitState
    from sampling.losses import compute_guidance_losses, ObservationTargets
    marker = json.loads((folder / 'complete.json').read_text())
    assert marker['status'] == 'complete' and marker['predictions'] == 1000
    assert marker['prefix_counts'] == KS and digest(folder / 'pool.pt') == marker['pool_sha256']
    obj = torch.load(folder / 'pool.pt', map_location='cpu', weights_only=False)
    binding = obj['binding']; task = binding['task']; offset = binding['offset']
    assert marker['binding'] == binding and task in TASKS and offset in OFFSETS
    assert folder.name == f'offset{offset}' and folder.parent.name == task
    identity = binding['identity']
    if reference_identity is not None:
        assert reference_identity == identity
    assert identity['protocol_sha256'] == digest(source / 'protocol.json')
    assert identity['truth_sha256'] == protocol['truth_sha256']
    assert identity['weights_sha256'] == protocol['weights_sha256']
    assert identity['historical_commit'] == '1f1573bfbc246b83e48a3b46402c8f2467488e0d'
    assert identity['tf32'] is True and identity['fused_guidance'] is True
    assert identity['runner_sha256'] == digest(ROOT / 'plot/run_conditional_sample_scaling.py')
    for relative, expected_hash in identity['source_hashes'].items():
        assert digest(ROOT / relative) == expected_hash
    cfg = obj['config']
    expected = configuration(protocol, selection, task, Path(cfg['checkpoint_path']).parent)
    expected.offset = offset; expected.mask_seed = expected.sample_seed = 20260912 + offset
    expected.output_dir = cfg['output_dir']
    expected.runtime_metadata.update(fused_guidance=True, observation_source='single_frozen_pair',
                                    conditional_pool_size=1000, timing_mode='cumulative_prefix')
    assert cfg == expected.asdict(), (task, offset, 'configuration changed')
    assert binding['config_sha256'] == json_hash(cfg)
    fixed = obj['fixed']
    truth = torch.cat([truths[offset].coef, truths[offset].sol], 1).cpu()
    assert torch.equal(fixed['truth'], truth)
    expected_masks = make_pair_masks((1, 1, 128, 128), (1, 1, 128, 128), 500,
                                     cfg['sensor_mode'], cfg['shared_mask'], cfg['mask_seed'])
    mask = torch.cat([expected_masks.coef, expected_masks.sol], 1)
    assert torch.equal(fixed['masks'], mask)
    assert torch.equal(mask.sum((-2, -1)), torch.full((1, 2), 500.))
    assert torch.equal(fixed['observations'], truth * mask)
    expected_fields = ['coef'] if task == 'forward' else ['sol'] if task == 'inverse' else ['coef', 'sol']
    assert fixed['observed_fields'] == expected_fields
    actual_hashes = {key: tensor_hash(fixed[key]) for key in ['truth', 'masks', 'observations']}
    assert actual_hashes == fixed['hashes'] == binding['observation_hashes']
    prediction = obj['predictions']
    assert prediction.shape == (1000, 2, 128, 128) and prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    assert len({tensor_hash(x) for x in prediction}) == 1000
    seen = []; noise_hashes = []; stop = 0; computed_seconds = 0.; active_seconds = 0.
    prefix_batches = {}
    for batch in obj['batches']:
        indices = batch['seed_indices']
        assert indices == list(range(stop, stop + len(indices)))
        assert len(indices) == batch['batch_size'] <= binding['batch_size']
        assert batch['num_steps'] == batch['nfe'] == 100
        assert batch['binding'] == binding
        assert batch['observations']['all_rows_identical'] is True
        assert batch['observations']['checked_rows'] == len(indices)
        assert batch['observations']['hashes'] == actual_hashes
        assert batch['observations']['observed_fields'] == expected_fields
        assert tensor_hash(prediction[indices]) == batch['prediction_tensor_sha256']
        assert batch['active_seconds'] >= batch['seconds'] > 0
        assert len(batch['initial_noise_hashes']) == len(indices)
        seen += indices; noise_hashes += batch['initial_noise_hashes']; stop += len(indices)
        computed_seconds += batch['seconds']; active_seconds += batch['active_seconds']
        if stop in KS:
            prefix_batches[stop] = (computed_seconds, active_seconds)
    assert seen == list(range(1000)) and len(set(noise_hashes)) == 1000
    assert set(prefix_batches) == set(KS)
    gt = PDEGroundTruth('poisson', truth[:, :1].double(), truth[:, 1:].double(), truth.double(), {}, ['a'], ['u'], {'synthetic': False})
    masks = PairMasks(mask[:, :1].double(), mask[:, 1:].double(), fixed['metadata'])
    obs = fixed['observations'].double()
    observations = ObservationTargets(obs[:, :1], obs[:, 1:], obs[:, :1], obs[:, 1:])
    pred = prediction.double(); y = truth[0].double()
    rows = []; arrays = {}; mean_seconds = 0.; previous_time = 0.
    key = f'{task}_{offset}'
    arrays[key + '_truth'] = y.numpy(); arrays[key + '_mask'] = mask[0].numpy()
    for k in KS:
        prefix = obj['prefixes'][k]
        assert prefix['K'] == k and prefix['binding'] == binding
        mean = pred[:k].mean(0)
        assert torch.equal(mean, obj['means'][k])
        assert tensor_hash(mean) == prefix['mean_sha256']
        assert prefix == json.loads((folder / f'prefix_{k}.json').read_text())
        mean_seconds += prefix['mean_seconds']
        assert np.isclose(prefix['seconds'], prefix_batches[k][1] + mean_seconds, atol=1e-8, rtol=1e-12)
        assert np.isclose(prefix['compute_seconds'], prefix_batches[k][0], atol=1e-8, rtol=1e-12)
        assert prefix['seconds'] > previous_time
        previous_time = prefix['seconds']
        loss = compute_guidance_losses(SplitState(mean[None, :1], mean[None, 1:]), gt, masks,
                                       AblationConfig(**cfg), observations=observations)
        assert loss.pde_residual_status != 'error'
        row = dict(task=task, offset=offset, K=k, seconds=prefix['seconds'],
                   compute_seconds=prefix['compute_seconds'], peak_bytes=prefix['peak_bytes'],
                   L_obs_a=float(loss.L_obs_a), L_obs_u=float(loss.L_obs_u), L_pde=float(loss.L_pde))
        for j, field in enumerate(['a', 'u']):
            error = float(torch.linalg.vector_norm(mean[j] - y[j]) / torch.linalg.vector_norm(y[j]))
            assert np.isclose(error, prefix['errors'][field], rtol=1e-12, atol=1e-14)
            individual_squared = ((pred[:k, j] - y[j]) ** 2).flatten(1).sum(1) / (y[j] ** 2).sum()
            spread = ((pred[:k, j] - mean[j]) ** 2).flatten(1).sum(1).mean() / (y[j] ** 2).sum()
            assert np.isclose(float(individual_squared.mean()), error ** 2 + float(spread), rtol=1e-10, atol=1e-12)
            row[f'rel_l2_{field}'] = error
            row[f'mean_individual_error_{field}'] = float(individual_squared.sqrt().mean())
            row[f'normalized_spread_{field}'] = float(spread)
        arrays[key + f'_K{k}'] = mean.numpy(); rows.append(row)
    proof = dict(task=task, offset=offset, pool_sha256=marker['pool_sha256'], path=str(folder / 'pool.pt'),
                 observation_hashes=actual_hashes, all_1000_draw_observations_identical=True,
                 config_sha256=binding['config_sha256'], initial_noise_identities_sha256=json_hash(noise_hashes))
    return rows, arrays, proof, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='Poisson directory with three frozen files')
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    protocol = json.loads((args.inputs / 'protocol.json').read_text())
    selection = json.loads(args.selection.read_text())
    assert digest(args.inputs / 'protocol.json') == selection['protocol_sha256']
    assert digest(args.inputs / 'truths.pt') == protocol['truth_sha256']
    assert digest(args.inputs / 'weights.pth') == protocol['weights_sha256']
    truths = torch.load(args.inputs / 'truths.pt', weights_only=False, map_location='cpu')
    rows = []; arrays = {}; proofs = []; identity = None
    expected = {(task, offset) for task in TASKS for offset in OFFSETS}
    discovered = {(path.parent.parent.name, int(path.parent.name.removeprefix('offset')))
                  for path in args.results.glob('cases/*/offset*/complete.json')}
    assert discovered <= expected
    if not args.allow_partial:
        assert discovered == expected, (len(discovered), len(expected), sorted(expected - discovered))
    for task, offset in sorted(discovered):
        rs, fields, proof, this_identity = audit_case(args.results / 'cases' / task / f'offset{offset}',
                                                   args.inputs, protocol, selection, truths, identity)
        identity = this_identity
        assert identity['selection_sha256'] == digest(args.selection)
        rows.extend(rs); arrays.update(fields); proofs.append(proof)
        print('AUDITED', task, offset, flush=True)
    assert rows
    stock_path = args.results / 'actual_stock_check/stock_complete.json'
    pilot_path = args.results / 'pilot/pilot_complete.json'
    stock = json.loads(stock_path.read_text()); pilot = json.loads(pilot_path.read_text())
    assert stock['status'] == pilot['status'] == 'pass'
    assert stock['identity'] == pilot['identity'] == identity
    assert stock['pilot_certificate_sha256'] == digest(pilot_path)
    assert stock['threshold'] == 5e-4
    assert {r['task'] for r in stock['checks']} == set(TASKS)
    assert all(c['prediction_relative_difference'] < 5e-4 for r in stock['checks'] for c in r['comparisons'])
    # All three tasks retain the same initial Gaussian identities per input.
    # Later stochastic innovations are governed by the frozen sampler code;
    # this audit does not reconstruct and hash every step of those paths.
    for offset in OFFSETS:
        hashes = {p['initial_noise_identities_sha256'] for p in proofs if p['offset'] == offset}
        assert len(hashes) <= 1
    write_csv(args.output / 'conditional_scaling_per_input.csv', rows)
    np.savez_compressed(args.output / 'conditional_scaling_fields.npz', **arrays)
    complete = discovered == expected
    manifest = dict(status='pass' if complete else 'partial', complete=complete,
         expected_jobs=sorted(expected), completed_jobs=sorted(discovered), conditional_trajectories=1000 * len(discovered),
         unique_sampling_trajectories=1000 * len(discovered), prefix_timing_records=len(rows),
         independent_timing_trajectories=0, original_input_truths_verified=True,
         fixed_observations_verified_all_draws=True, frozen_guidance_configurations_verified=True,
         exact_nested_prefix_means_verified=True, distinct_predictions_per_pool_verified=1000,
         all_fields_finite=True, variance_identity_verified=True, source_identity=identity,
         timing_source='Cumulative active generation and prefix means from one 1000-draw pool; no independent K timing.',
         accuracy_source='Means of the first K physical predictions from the same fixed-observation 1000-draw pool.',
         pilot_certificate_sha256=digest(pilot_path), actual_stock_certificate_sha256=digest(stock_path),
         results=proofs, exporter_sha256=digest(__file__), audit_seconds=time.time() - started,
         resolved_inputs=str(args.inputs.resolve()), resolved_selection=str(args.selection.resolve()))
    manifest['export_files'] = {name: digest(args.output / name) for name in
                              ['conditional_scaling_per_input.csv', 'conditional_scaling_fields.npz']}
    write(args.output / 'conditional_scaling_manifest.json', manifest)
    print('AUDIT_COMPLETE', complete, len(proofs), 1000 * len(proofs), flush=True)


if __name__ == '__main__':
    main()

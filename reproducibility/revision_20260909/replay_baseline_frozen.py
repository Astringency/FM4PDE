#!/usr/bin/env python3
"""Audit or replay one original baseline cell using a verified raw-data cache.

The baseline checkout supplies every task, normalization and inference method.
This wrapper does not invoke its training-data-dependent eval-only CLI.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_checkout(path):
    """Compare actual file bytes with Git blobs, without EOL clean filters.

    Historical VIVID blobs contain CRLF although their tracked attributes
    request LF. Filtered git-diff reports those unmodified checkouts as dirty.
    Raw blob identity checks the source that Python actually reads.
    """
    tree = subprocess.check_output(['git', '-C', str(path), 'ls-tree', '-r', '-z', 'HEAD'])
    count = 0
    for row in tree.split(b'\0'):
        if not row:
            continue
        info, name = row.split(b'\t', 1)
        mode, kind, expected = info.split()
        assert kind == b'blob', 'Nested repositories need an explicit source dependency gate'
        file = path / os.fsdecode(name)
        content = os.fsencode(os.readlink(file)) if mode == b'120000' else file.read_bytes()
        digest = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert digest == expected.decode(), 'Tracked source bytes differ: '+str(file)
        count += 1
    return count


def tensors(value, prefix=''):
    import numpy as np
    import torch
    result = {}
    if isinstance(value, (torch.Tensor, np.ndarray)):
        array = value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.ascontiguousarray(value)
        if np.issubdtype(array.dtype, np.inexact):
            assert np.isfinite(array).all(), 'Nonfinite cache array: '+prefix
        result[prefix] = dict(shape=list(value.shape), dtype=str(value.dtype),
                              sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())
    elif isinstance(value, dict):
        for key, item in value.items():
            result.update(tensors(item, f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(tensors(item, f'{prefix}.{index}'))
    return result


def resolve_args(snapshot, summary):
    """Evaluation fields override the recorded source-training arguments."""
    result = copy.deepcopy(snapshot['args'])
    for key in ('baseline', 'pde', 'task', 'seed', 'sensor_seed', 'batch_size',
                'num_sensors', 'sensor_mode', 'sensor_budget_mode', 'noise_level',
                'experiment_mode', 'load_full_trajectory', 'condition_mode',
                'condition_probabilities', 'scalar_param_mode', 'source_train_task'):
        if key in summary:
            result[key] = summary[key]
    if 'evaluation_condition_mode' in summary:
        result['condition_mode'] = summary['evaluation_condition_mode']
    if isinstance(result.get('condition_probabilities'), str):
        result['condition_probabilities'] = json.loads(result['condition_probabilities'])
    result['load_full_trajectory'] = bool(result['load_full_trajectory'] or
                                          result['baseline'] in {'var4d', 'vivid'})
    assert not summary['synthetic_data'] and int(summary['test_size']) == 1000
    assert summary['split'] == 'test'
    return argparse.Namespace(**result)


def assert_equal(left, right, path='value'):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), path
        assert left.shape == right.shape and left.dtype == right.dtype, path
        assert torch.equal(left.cpu(), right.cpu()), path
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype, path
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f'{path}.{index}')
    else:
        assert left == right, (path, left, right)


def validate_cache(payload, entry, summary, args):
    import torch
    assert payload['schema_version'] == 'baseline-frozen-raw-v1'
    assert payload['pde'] == entry['pde'] == summary['pde']
    assert bool(payload['load_full_trajectory']) == bool(entry['load_full_trajectory'])
    # Static loaders ignore this flag; temporal caches retain both branches.
    if summary['pde'] in {'nsnonbounded', 'burger'}:
        assert bool(entry['load_full_trajectory']) == args.load_full_trajectory
    raw, provenance = payload['raw'], payload['provenance']
    assert provenance['source_mat_sha256'] == entry['source_mat_sha256']
    assert len(provenance['source_mat_sha256']) == 64
    assert provenance['source_indices'] == list(range(1000))
    assert raw['split'] == 'test' and raw['full_tensor'].shape[0] == 1000
    assert torch.equal(raw['sample_indices'], torch.arange(1000))
    originals = json.loads(summary['data_files_json'])['test']
    assert len(originals) == 1
    name = Path(originals[0]).name
    assert Path(provenance['source_mat_path']).name == name
    assert raw['global_sample_ids'] == [f'{name}:{i}' for i in range(1000)]
    assert [Path(p).name for p in raw['file_paths']] == [name]
    assert tensors(raw) == entry['tensors'], 'Cached array hashes/shapes/dtypes differ'
    return raw


def main(cli):
    import torch
    torch.set_num_threads(2)
    summary = json.loads(cli.summary.read_text())
    snapshot = json.loads(cli.config.read_text())
    assert sha256(cli.summary) == cli.summary_sha256
    assert sha256(cli.config) == cli.config_sha256
    revision = subprocess.check_output(['git', '-C', str(cli.baseline_code),
                                        'rev-parse', 'HEAD'], text=True).strip()
    assert revision == summary['commit_hash'], 'Use the original evaluation commit'
    source_files = verify_checkout(cli.baseline_code)
    sys.path.insert(0, str(cli.baseline_code.resolve()))
    from baselines import run as native
    from baselines.common.data_adapter import PDEBatchDataset, build_default_registry, pde_collate
    from baselines.common.sample_artifacts import _sample_payload
    args = resolve_args(snapshot, summary)
    manifest = json.loads(cli.cache_manifest.read_text())
    assert manifest['schema_version'] == 'baseline-frozen-input-manifest-v1'
    assert manifest['status'] == 'pass' and manifest['complete']
    assert len(manifest['entries']) == 18 and manifest['selected_run_count'] == 186
    candidates = [e for e in manifest['entries'] if Path(e['cache_path']).name == cli.cache.name]
    assert len(candidates) == 1, 'Cache selection must be unambiguous'
    entry = candidates[0]
    assert sha256(cli.cache) == entry['cache_sha256']
    payload = torch.load(cli.cache, map_location='cpu', weights_only=False)
    raw = validate_cache(payload, entry, summary, args)
    bindings = [b for b in payload['provenance']['selected_run_bindings']
                if b['summary_sha256'] == cli.summary_sha256]
    assert len(bindings) == 1 and bindings[0]['config_sha256'] == cli.config_sha256
    assert args.load_full_trajectory in entry['supported_load_full_trajectory']
    registry = build_default_registry()
    task = registry.make_task(registry.to_canonical(raw, args.pde), args.pde, args.task,
        num_sensors=args.num_sensors if args.task.startswith('sparse') else None,
        sensor_mode=args.sensor_mode, sensor_budget_mode=args.sensor_budget_mode,
        noise_level=args.noise_level, seed=args.sensor_seed, experiment_mode=args.experiment_mode,
        build_voronoi_grid=args.baseline in {'recfno', 'voronoicnn', 'var4d', 'vivid'},
        condition_mode=getattr(args, 'condition_mode', 'mixed'),
        condition_probabilities=getattr(args, 'condition_probabilities', None))
    dataset = PDEBatchDataset(task)
    expected_masks = json.loads(summary['split_mask_manifest'])['test']
    actual_masks = native._split_mask_manifest(dataset, None, dataset)['test']
    assert actual_masks == expected_masks, 'Original complete test mask contract differs'
    indices = (list(range(1000)) if cli.indices == 'all' else [] if cli.indices == 'none'
               else [int(i) for i in cli.indices.split(',')])
    assert indices or not cli.predict, 'Prediction requires at least one selected index'
    assert len(set(indices)) == len(indices) and all(0 <= i < 1000 for i in indices)
    assert not cli.output.exists(), 'Use a fresh verification output directory'
    cli.output.mkdir(parents=True)
    refs = None
    if cli.reference_manifest:
        assert cli.reference_root is not None
        refs = {int(r['sample_ordinal']): r for r in
                map(json.loads, cli.reference_manifest.read_text().splitlines())}
        assert len(refs) == 1000
    observation_bundle = None
    if cli.observation_bundle:
        assert args.pde == 'burger' and cli.observation_bundle_sha256
        assert sha256(cli.observation_bundle) == cli.observation_bundle_sha256
        observation_bundle = torch.load(cli.observation_bundle, map_location='cpu', weights_only=False)
        assert torch.equal(torch.as_tensor(observation_bundle['sample_ids']), torch.arange(1000))
    report = dict(status='running', original_commit=revision, original_source_files_verified=source_files,
        source_check='Exact working-file bytes versus original Git blobs; no EOL clean filters', original_summary=summary,
        original_summary_sha256=cli.summary_sha256, original_config_sha256=cli.config_sha256,
        cache_file_sha256=entry['cache_sha256'], original_mat_provenance=payload['provenance'],
        cache_manifest_sha256=sha256(cli.cache_manifest), arrays=entry['tensors'],
        original_mask_contract=actual_masks, selected_indices=indices,
        original_batch_size=args.batch_size, device=cli.device, torch_version=torch.__version__,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        original_runtime_recreated=False,
        wrapper_sha256=sha256(__file__), predicts=cli.predict, comparisons=[])
    model = None
    if cli.predict:
        torch.manual_seed(args.seed)
        if cli.checkpoint:
            expected_sha = summary.get('checkpoint_sha256') or cli.checkpoint_sha256
            assert expected_sha and sha256(cli.checkpoint) == expected_sha
            if summary.get('checkpoint_sha256') and cli.checkpoint_sha256:
                assert expected_sha == cli.checkpoint_sha256
            model = native.load_baseline_checkpoint(cli.checkpoint, map_location=cli.device,
                expected={'baseline': args.baseline, 'pde': args.pde,
                          'source_train_run_id': summary['source_train_run_id'],
                          'run_fingerprint': summary['source_train_run_fingerprint']},
                require_provenance=True)
            report['checkpoint_sha256'] = expected_sha
            report['checkpoint_recovered_from_run_directory'] = not bool(summary.get('checkpoint_sha256'))
            report['normalization_tensors'] = tensors(model.normalization_stats.state_dict()
                if model.normalization_stats is not None else {})
            report['uses_normalization'] = model.uses_normalization
        else:
            assert args.baseline == 'var4d' and int(summary['train_size']) == 0, \
                'A trained baseline, including VIVID, requires its original checkpoint'
            model = native.BASELINES[args.baseline]().build(snapshot['method'], snapshot['data_spec']).to(cli.device)
        model.eval()
    starts = sorted({i // args.batch_size * args.batch_size for i in indices})
    for start in starts:
        stop = min(start + args.batch_size, 1000)
        batch = pde_collate([dataset[i] for i in range(start, stop)])
        inference = native._make_inference_batch(batch)
        assert torch.count_nonzero(inference.target_fields) == 0
        assert torch.count_nonzero(inference.full_tensor) == 0
        pred = None
        if model is not None:
            device_batch = native._to_device_batch_for_eval(inference, cli.device)
            with torch.set_grad_enabled(args.baseline in {'var4d', 'vivid'}):
                pred = model.predict_physical(device_batch).detach().cpu()
            assert pred.shape == batch.target_fields.shape and torch.isfinite(pred).all()
        for index in indices:
            if not start <= index < stop:
                continue
            item = index - start
            current = _sample_payload(batch, torch.zeros_like(batch.target_fields) if pred is None else pred,
                item=item, ordinal=index, batch_index=start // args.batch_size, run_metadata={},
                predictive_std=None, posterior_samples=None, metrics={})
            comparison = dict(index=index, global_sample_id=current['global_sample_id'], batch_size=stop-start)
            if observation_bundle is not None:
                assert_equal(current['target_fields'], observation_bundle['truth'][index], 'bundle.truth')
                assert_equal(current['mask'].bool(), observation_bundle['mask'][index].bool(), 'bundle.mask')
                comparison['archived_burgers_truth_and_mask_bitwise_equal'] = True
                comparison['archived_burgers_bundle_sha256'] = cli.observation_bundle_sha256
            if refs is not None:
                ref = refs[index]
                path = cli.reference_root / Path(ref['artifact_path']).name
                assert sha256(path) == ref['artifact_sha256']
                saved = torch.load(path, map_location='cpu', weights_only=False)
                for key in ('global_sample_id', 'sample_index', 'pde_name', 'task', 'split',
                            'channel_names', 'input_channel_names', 'target_channel_names',
                            'input_fields', 'target_fields', 'full_tensor', 'coords', 'mask',
                            'obs_values', 'obs_coords', 'pde_params'):
                    assert_equal(current[key], saved[key], key)
                for key in ('canonical_layout', 'grid_layout', 'domain_length', 'final_time',
                            'time_values', 'nu', 'k', 'elliptic_operator_sign',
                            'joint_reconstruction', 'joint_input_channels',
                            'joint_solution_channels', 'mask_id', 'mask_tensor_sha1'):
                    if key in current['metadata'] or key in saved['metadata']:
                        assert_equal(current['metadata'].get(key), saved['metadata'].get(key), 'metadata.'+key)
                comparison['original_fields_and_masks_bitwise_equal'] = True
                comparison['original_artifact_sha256'] = ref['artifact_sha256']
                if pred is not None:
                    delta = current['prediction'].double() - saved['prediction'].double()
                    comparison['prediction_max_abs_difference'] = float(delta.abs().max())
                    comparison['prediction_relative_difference'] = float(delta.norm() / saved['prediction'].double().norm().clamp_min(1e-30))
            if pred is not None:
                truth = current['target_fields'].double()
                comparison['whole_target_relative_l2'] = float((current['prediction'].double()-truth).norm()/truth.norm().clamp_min(1e-30))
                # Explicitly a replay diagnostic, not a replacement for the
                # native split solution/coefficient metrics in the paper.
                torch.save(current, cli.output / f'sample_{index:06d}.pt')
            report['comparisons'].append(comparison)
    report.update(status='pass', complete_test_mask_contract_checked=True,
                  full_inference_completed=bool(cli.predict and len(indices) == 1000),
                  reference_tensors_checked=refs is not None)
    (cli.output/'replay_receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='pass', checked=len(indices), predicts=cli.predict)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-code', 'cache', 'cache-manifest', 'summary', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--summary-sha256', required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--reference-manifest', type=Path)
    p.add_argument('--reference-root', type=Path)
    p.add_argument('--observation-bundle', type=Path)
    p.add_argument('--observation-bundle-sha256')
    p.add_argument('--indices', default='0,17,999')
    p.add_argument('--predict', action='store_true')
    p.add_argument('--device', default='cpu')
    main(p.parse_args())

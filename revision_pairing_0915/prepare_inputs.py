"""Freeze exact existing baseline inputs; CPU only, no model inference.

Workflow: ``catalog`` locally, Git-deploy code, ``prepare --cohort supervised``
on the machine with baseline tensors, ``prepare --cohort diffusion`` beside
the original Diffusion data, synchronize both outputs, then ``merge``.
Each PDE/distribution cache has a resumable, hash-checked receipt. An existing
completed cache is never replaced. No source data or original output is edited.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from input_sources import (COUNT, GRID, TASKS, diffusion_mask, file_sha256,
                           load_cell, supervised_mask, tensor_sha256)

PDES = ['poisson', 'helmholtz', 'darcy', 'nsnonbounded']
DISTS = ['id', 'smooth', 'rough']
TASK_NAME = 'baseline_pairing_20260915_57k'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.partial')
    torch.save(value, tmp)
    os.replace(tmp, path)


def params(pde, metadata):
    if pde == 'nsnonbounded':
        result = dict(nu=float(metadata['nu']), T=float(metadata['T']),
                      solver_dt=float(metadata.get('dt', metadata.get('solver_dt', .0001))))
        assert result == dict(nu=.001, T=1., solver_dt=.0001), result
        return result
    if pde == 'burger':
        assert float(metadata.get('nu', .01)) == .01
        return dict(nu=.01)
    return dict(k=1.) if pde == 'helmholtz' else {}


def catalog(args):
    if args.output.exists():
        raise FileExistsError(f'Catalog is immutable: {args.output}')
    candidates = read(args.evidence / 'baseline_source_candidates.json')
    prov = {x['summary']: x for x in read(args.evidence / 'baseline_metric_provenance.json')}
    selected = [x for x in candidates if x['pde'] in PDES + ['burger']
                and x['method'] in ['recfno', 'senseiver', 'voronoicnn']
                and x['task'] in ['sparse_forward', 'sparse_inverse', 'sparse_solution']
                and not x['ablation'] and x['sensor_mode'] == 'random_per_sample']
    assert len(selected) == 117
    datasets = []
    total_anchors = 0
    for pde in PDES + ['burger']:
        for dist in DISTS:
            group = [x for x in selected if x['pde'] == pde and x['distribution'] == dist]
            ref = next(x for x in group if x['method'] == 'recfno' and x['task'] == 'sparse_solution')
            record = prov[ref['summary_path']]
            assert record['sample_files_found'] == COUNT
            first = record['samples'][0]
            prefix = first['id'].rsplit(':', 1)[0]
            metadata = first['metadata']
            anchors = []
            for c in group:
                assert c['matrix']['sensor_seed'] == 1 and c['matrix']['num_sensors'] == 500
                p = prov[c['summary_path']]
                assert p['sample_files_found'] == COUNT
                for sample in p['samples']:
                    assert sample['id'] == f'{prefix}:{sample["sample_index"]}'
                    mask = supervised_mask(sample['id']).expand(*sample['mask_shape'])
                    assert tensor_sha256(mask) == sample['mask_sha256'], sample['path']
                    anchors.append(dict(method=c['method'], task=c['task'], **{
                        k: sample[k] for k in ['id', 'sample_index', 'path', 'sha256',
                                                'truth_sha256', 'mask_sha256', 'mask_shape']}))
            total_anchors += len(anchors)
            sample_dir = str(Path(first['path']).parent)
            relative = sample_dir.split('/FM4PDEbaseline/', 1)[1]
            datasets.append(dict(pde=pde, distribution=dist, sample_id_prefix=prefix,
                                 reference_method='RecFNO', reference_summary=ref['summary_path'],
                                 reference_samples_dir_relative=relative,
                                 source_data_paths=metadata['files'], pde_params=params(pde, metadata),
                                 axes=['time', 'x'] if pde == 'burger' else ['x', 'y'],
                                 channel_names=metadata['input_channel_names'],
                                 coordinate_layout=metadata.get('coordinate_layout'),
                                 reference_anchors=[a for a in anchors if a['method'] == 'recfno'
                                                    and a['task'] == 'sparse_solution'],
                                 all_method_anchors=anchors))
    bp = read(args.burgers_inputs / 'protocol.json')
    assert len(bp['baselines']) == 30
    assert all(x['same_truth'] and x['same_masks'] and x['samples'] == COUNT for x in bp['baselines'])
    inv = read(args.diffusion_inventory)
    with np.load(args.diffusion_truths) as npz:
        diffusion = {}
        ids = inv['evaluation_ids'] + [inv['pilot_id']]
        assert len(ids) == 21 and len(set(ids)) == 21
        for pde in PDES:
            module = 'ns_nonbounded' if pde == 'nsnonbounded' else pde
            rel = f'scripts/generate_{module}.py'
            source = subprocess.check_output(['git', 'show', f'151e721b9991:{rel}'],
                                             cwd=args.diffusion_root, text=True)
            tree = ast.parse(source)
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'random_index')
            env = dict(np=np, torch=torch)
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<historic mask>', 'exec'), env)
            for seed in [0, 1]:
                actual = env['random_index'](500, GRID, seed=seed, device=torch.device('cpu'))
                assert torch.equal(actual, diffusion_mask(seed)[0, 0].to(actual.dtype))
            pair = np.concatenate([npz[f'{pde}_a'], npz[f'{pde}_u']], axis=1)
            diffusion[pde] = dict(source=inv['data_sources'][pde],
                source_config=inv['diffusion_configs'][pde],
                mask_source_revision='151e721b9991', mask_source_file=rel,
                mask_source_sha256=__import__('hashlib').sha256(source.encode()).hexdigest(),
                mask_sha256={field: tensor_sha256(diffusion_mask(seed))
                             for field, seed in [('coef', 1), ('sol', 0)]},
                physical_source_anchor_hashes=[dict(index=i, truth_sha256=tensor_sha256(pair[j:j+1]))
                                              for j, i in enumerate(ids)])
    evidence_paths = [args.evidence/'baseline_source_candidates.json',
                      args.evidence/'baseline_metric_provenance.json',
                      args.burgers_inputs/'protocol.json', args.diffusion_inventory,
                      args.diffusion_truths]
    value = dict(version=1, task_name=TASK_NAME, count=COUNT, datasets=datasets,
                 baseline_tensor_mask_hash_anchors_verified=total_anchors,
                 burgers_protocol=bp, burgers_protocol_sha256=file_sha256(args.burgers_inputs/'protocol.json'),
                 diffusion=diffusion, mask_contract='mask|1|test|filename:index|0; SHA256 first8 big-endian modulo2**63-1; torch CPU randperm; shared channels',
                 evidence=[dict(path=str(p), sha256=file_sha256(p)) for p in evidence_paths])
    write(args.output, value)
    print(json.dumps(dict(status='catalog_frozen', datasets=len(datasets),
                          verified_mask_anchors=total_anchors, output=str(args.output))), flush=True)


def baseline_fields(dataset, args):
    folder = args.baseline_output_root / dataset['reference_samples_dir_relative']
    prefix = dataset['sample_id_prefix']
    anchors = {x['sample_index']: x for x in dataset['reference_anchors']}
    expected_files = [folder / f'sample_{i:06d}.pt' for i in range(COUNT)]
    assert all(p.is_file() for p in expected_files), f'Incomplete reference samples: {folder}'
    def one(item):
        i, path = item
        raw = path.read_bytes()
        raw_sha = __import__('hashlib').sha256(raw).hexdigest()
        d = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        assert int(d['sample_index']) == i and d['global_sample_id'] == f'{prefix}:{i}', path
        truth = d['target_fields'].to(torch.float32)
        assert torch.all((d['mask']==0)|(d['mask']==1)), (path, 'nonbinary raw mask')
        mask = d['mask'].to(torch.uint8)
        assert truth.shape == mask.shape == (2, GRID, GRID), path
        assert torch.isfinite(truth).all(), path
        expected = supervised_mask(d['global_sample_id']).expand(2, -1, -1)
        assert torch.equal(mask, expected), (path, 'historical per-example mask rule mismatch')
        th = tensor_sha256(truth[None]); mh = tensor_sha256(mask)
        if i in anchors:
            a = anchors[i]
            assert raw_sha == a['sha256'] and th == a['truth_sha256'] and mh == a['mask_sha256'], path
        return truth, mask[:1], dict(index=i, sample_id=d['global_sample_id'],
                    source=str(path), source_sha256=raw_sha, truth_sha256=th, mask_sha256=mh)
    truths, masks, rows = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for truth, mask, receipt in pool.map(one, enumerate(expected_files)):
            truths.append(truth); masks.append(mask); rows.append(receipt)
    fields = torch.stack(truths)
    mask = torch.stack(masks)
    # All available archived other-method/task anchors must match these exact
    # exported physical fields, rather than merely the reconstructed masks.
    for a in dataset['all_method_anchors']:
        i = a['sample_index']
        target = fields[i:i+1]
        if a['task'] == 'sparse_forward': target = target[:, 1:2]
        if a['task'] == 'sparse_inverse': target = target[:, 0:1]
        assert tensor_sha256(target) == a['truth_sha256'], (dataset['pde'], a['path'], 'truth anchor')
        assert tensor_sha256(mask[i].expand(*a['mask_shape'])) == a['mask_sha256']
    metadata = dict(source_kind='all 1000 actual saved RecFNO joint target/mask tensors',
                    reference_samples_dir=str(folder), all_reference_masks_equal_rebuilt_contract=True,
                    actual_reference_tensors_read=COUNT,
                    cross_method_task_archived_truth_mask_anchors_checked=len(dataset['all_method_anchors']))
    return fields, dict(coef=mask, sol=mask), rows, metadata


def burgers_fields(dataset, args, cat):
    file = args.burgers_inputs / f'{dataset["distribution"]}_random.pt'
    expected = cat['burgers_protocol']['artifacts'][file.name]
    assert file_sha256(file) == expected, file
    d = torch.load(file, map_location='cpu', weights_only=False)
    assert d['sample_ids'] == list(range(COUNT))
    fields, mask = d['truth'].float(), d['mask'].to(torch.uint8)
    assert fields.shape == mask.shape == (COUNT, 1, GRID, GRID)
    assert torch.isfinite(fields).all()
    rows = []
    for i in range(COUNT):
        sid = f'{dataset["sample_id_prefix"]}:{i}'
        assert torch.equal(mask[i], supervised_mask(sid)), (file, i)
        rows.append(dict(index=i, sample_id=sid, source=str(file), source_sha256=expected,
                         truth_sha256=tensor_sha256(fields[i:i+1]), mask_sha256=tensor_sha256(mask[i])))
    for a in dataset['all_method_anchors']:
        i = a['sample_index']
        assert tensor_sha256(fields[i:i+1]) == a['truth_sha256']
        assert tensor_sha256(mask[i]) == a['mask_sha256']
    return fields, dict(coef=mask, sol=mask), rows, dict(source_kind='frozen actual baseline Burgers tensors',
        frozen_source=str(file), frozen_source_sha256=expected,
        baseline_protocol_sha256=cat['burgers_protocol_sha256'],
        actual_frozen_masks_checked=COUNT, all_masks_equal_rebuilt_contract=True,
        inherited_full_equality='Five baseline methods, truth and masks, all1000 examples per cell')


def source_fields(pde, path):
    """Read exactly the historical physical axes, before FM float32 casting."""
    if pde in ['poisson', 'helmholtz']:
        import scipy.io
        keys = ('f_data', 'phi_data' if pde == 'poisson' else 'psi_data')
        raw = scipy.io.loadmat(path, variable_names=list(keys))
        pair = np.stack([raw[k][:COUNT] for k in keys], axis=1).astype(np.float32)
    else:
        import h5py
        with h5py.File(path, 'r') as f:
            if pde == 'darcy':
                # The historical Diffusion generator indexes [:,:,offset].
                assert f['thresh_a_data'].shape[:2] == (GRID, GRID)
                pair = np.stack([np.moveaxis(f[k][:, :, :COUNT], -1, 0)
                                 for k in ['thresh_a_data', 'thresh_p_data']], axis=1).astype(np.float32)
            else:
                assert f['w0'].shape[1:] == (GRID, GRID)
                pair = np.stack([f['w0'][:COUNT], f['w'][:COUNT, :, :, -1]], axis=1).astype(np.float32)
    assert pair.shape == (COUNT, 2, GRID, GRID) and np.isfinite(pair).all(), path
    return torch.from_numpy(pair)


def diffusion_fields(dataset, args, cat):
    pde = dataset['pde']; spec = cat['diffusion'][pde]
    original = Path(spec['source']['path'])
    path = args.diffusion_data_root / pde / original.name
    stat = path.stat()
    assert stat.st_size == spec['source']['size'], (path, 'source file size changed')
    fields = source_fields(pde, path)
    for anchor in spec['physical_source_anchor_hashes']:
        i = anchor['index']
        assert tensor_sha256(fields[i:i+1]) == anchor['truth_sha256'], (path, i, 'Diffusion source anchor')
    masks = dict(coef=diffusion_mask(1), sol=diffusion_mask(0))
    for key, value in masks.items():
        assert tensor_sha256(value) == spec['mask_sha256'][key]
    rows = [dict(index=i, sample_id=f'{original.name}:{i}', source=str(path),
                 truth_sha256=tensor_sha256(fields[i:i+1]),
                 mask_coef_sha256=tensor_sha256(masks['coef']),
                 mask_sol_sha256=tensor_sha256(masks['sol'])) for i in range(COUNT)]
    meta = dict(source_kind='original Diffusion PDE source, indices0..999, original axis convention',
                source=str(path), historical_source=str(original),
                source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                historical_source_mtime_ns=spec['source']['mtime_ns'],
                extraction_sha256=tensor_sha256(fields),
                physical_source_anchor_hashes_verified=len(spec['physical_source_anchor_hashes']),
                arithmetic='Source physical fields cast to float32 for FM; source float64 arithmetic is not redefined',
                mask_source_sha256=spec['mask_source_sha256'], mask_source_revision=spec['mask_source_revision'])
    if args.hash_raw_source:
        meta['source_file_sha256'] = file_sha256(path)
    return fields, masks, rows, meta


def cells_for(dataset, cohort, truth_file, truth_sha, masks_file, masks_sha):
    pde, dist = dataset['pde'], dataset['distribution']
    settings = ['sparse_forward', 'sparse_inverse', 'sparse_joint']
    if pde == 'burger': settings = ['sparse_joint']
    if pde == 'nsnonbounded' and cohort == 'supervised':
        settings += ['full_forward', 'full_inverse']
    rows = []
    for setting in settings:
        task = TASKS[setting]
        observed = (['sol'] if pde == 'burger' else ['coef'] if task == 'forward'
                    else ['sol'] if task == 'inverse' else ['coef', 'sol'])
        rows.append(dict(cell_id=f'{cohort}/{pde}/{dist}/{setting}', cohort=cohort, pde=pde,
            distribution=dist, setting=setting, task=task, count=COUNT,
            truth_file=truth_file, truth_sha256=truth_sha, masks_file=masks_file, masks_sha256=masks_sha,
            observed_fields=observed, num_obs=16384 if setting.startswith('full_') else 500))
    return rows


def prepare(args):
    assert args.output.is_absolute(), 'Explicit absolute output path required'
    args.output.mkdir(parents=True, exist_ok=True)
    cat = read(args.catalog); ch = file_sha256(args.catalog)
    assert cat['task_name'] == TASK_NAME and cat['count'] == COUNT
    desired = [x for x in cat['datasets'] if x['pde'] in args.pdes and x['distribution'] in args.distributions
               and (args.cohort == 'supervised' or x['pde'] in PDES and x['distribution'] == 'smooth')]
    assert desired, 'No input datasets selected'
    all_cells, receipts = [], []
    for dataset in desired:
        pde, dist, cohort = dataset['pde'], dataset['distribution'], args.cohort
        ident = f'{cohort}_{pde}_{dist}'
        rp = args.output / 'receipts' / f'{ident}.json'
        if rp.exists():
            old = read(rp)
            assert old['catalog_sha256'] == ch, 'Cannot replace input cache from a different catalog'
            if old['status'] == 'complete':
                for c in old['cells']: load_cell(args.output, c)
                assert file_sha256(args.output/old['per_input_file']) == old['per_input_sha256']
                all_cells += old['cells']; receipts.append(old)
                print('REUSED_VERIFIED', ident, flush=True)
                continue
            assert old['status'] == 'preparing', old
        else:
            for name in [f'truths/{ident}.pt', f'masks/{ident}.pt', f'per_input/{ident}.json']:
                assert not (args.output/name).exists(), f'Refuse unowned existing file: {name}'
            write(rp, dict(status='preparing', catalog_sha256=ch, dataset_id=ident))
        print('READING', ident, flush=True)
        if cohort == 'diffusion': fields, masks, per_input, meta = diffusion_fields(dataset, args, cat)
        elif pde == 'burger': fields, masks, per_input, meta = burgers_fields(dataset, args, cat)
        else: fields, masks, per_input, meta = baseline_fields(dataset, args)
        sample_ids = [x['sample_id'] for x in per_input]
        assert len(set(sample_ids)) == COUNT
        common = dict(cohort=cohort, pde=pde, distribution=dist, sample_ids=sample_ids,
                      source_indices=torch.arange(COUNT, dtype=torch.int64))
        truth = dict(**common, coef_ground_truth=fields[:, :1],
                     sol_ground_truth=fields[:, :1] if pde == 'burger' else fields[:, 1:2],
                     pde_params=dataset['pde_params'],
                     metadata=dict(meta, axes=dataset['axes'], channel_names=dataset['channel_names'],
                                   coordinate_layout=dataset['coordinate_layout'], units='physical',
                                   source_index_range=[0, COUNT-1], endpoint_pair=pde != 'burger',
                                   catalog_sha256=ch))
        saved_mask = dict(**common, **masks, metadata=dict(cohort=cohort,
            source_kind=meta['source_kind'], shared_fields=cohort == 'supervised',
            locations_per_field=500, exact_masks_saved=True, catalog_sha256=ch))
        tf, mf, pf = f'truths/{ident}.pt', f'masks/{ident}.pt', f'per_input/{ident}.json'
        save(args.output/tf, truth); save(args.output/mf, saved_mask); write(args.output/pf, per_input)
        cells = cells_for(dataset, cohort, tf, file_sha256(args.output/tf), mf, file_sha256(args.output/mf))
        for c in cells: load_cell(args.output, c)
        receipt = dict(status='complete', catalog_sha256=ch, dataset_id=ident, cohort=cohort,
                       pde=pde, distribution=dist, count=COUNT, cells=cells, metadata=meta,
                       per_input_file=pf, per_input_sha256=file_sha256(args.output/pf),
                       producer_sha256=file_sha256(Path(__file__)),
                       input_interface_sha256=file_sha256(Path(__file__).with_name('input_sources.py')))
        write(rp, receipt); all_cells += cells; receipts.append(receipt)
        print('FROZEN', ident, 'cells', len(cells), 'examples', COUNT, flush=True)
    expected = 45 if args.cohort == 'supervised' else 12
    manifest = dict(version=1, task_name=TASK_NAME, status='complete' if len(all_cells)==expected else 'partial',
                    cohort=args.cohort, catalog_sha256=ch, cells=all_cells,
                    datasets=[dict(dataset_id=r['dataset_id'], receipt=f'receipts/{r["dataset_id"]}.json',
                                   receipt_sha256=file_sha256(args.output/'receipts'/f'{r["dataset_id"]}.json')) for r in receipts],
                    validation=dict(examples_per_cell=COUNT, cells=len(all_cells),
                                    intended_prediction_count=sum(c['count'] for c in all_cells),
                                    no_inference_performed=True))
    suffix = '' if manifest['status']=='complete' else '_'+ '_'.join(args.pdes)+'_'+ '_'.join(args.distributions)
    target = args.output/f'inputs_manifest_{args.cohort}{suffix}.json'
    if target.exists():
        assert read(target) == manifest, 'Frozen cohort input manifest cannot change'
    else: write(target, manifest)
    print('INPUTS_READY', args.cohort, len(all_cells), 'cells', flush=True)


def merge(args):
    """Collect completed per-dataset receipts, allowing independent preparation."""
    root = args.output
    receipts = [read(p) for p in sorted((root/'receipts').glob('*.json'))]
    assert len(receipts) == 19 and all(r['status'] == 'complete' for r in receipts)
    catalogs = {r['catalog_sha256'] for r in receipts}; assert len(catalogs) == 1
    cells = [c for r in receipts for c in r['cells']]
    assert len(cells) == len({c['cell_id'] for c in cells}) == 57
    assert sum(c['count'] for c in cells) == 57000
    for c in cells: load_cell(root, c)
    for r in receipts:
        assert file_sha256(root/r['per_input_file']) == r['per_input_sha256']
    truth_comparisons = []
    for pde in PDES:
        keys = [f'{cohort}/{pde}/smooth/sparse_joint' for cohort in ['supervised','diffusion']]
        values = [load_cell(root, next(c for c in cells if c['cell_id']==key)) for key in keys]
        a, b = values
        matches = ((a['coef_ground_truth'] == b['coef_ground_truth']).flatten(1).all(1)
                   & (a['sol_ground_truth'] == b['sol_ground_truth']).flatten(1).all(1))
        truth_comparisons.append(dict(pde=pde, distribution='smooth', examples=COUNT,
                                      exactly_equal_float32_physical_inputs=int(matches.sum()),
                                      separate_protocol_truth_files_retained=True))
    result = dict(version=1, task_name=TASK_NAME, status='complete', cells=cells,
                  catalog_sha256=next(iter(catalogs)),
                  datasets=[dict(dataset_id=r['dataset_id'], receipt=f'receipts/{r["dataset_id"]}.json',
                                 receipt_sha256=file_sha256(root/'receipts'/f'{r["dataset_id"]}.json')) for r in receipts],
                  validation=dict(cells=57, intended_prediction_count=57000,
                                  supervised_prediction_count=45000,diffusion_prediction_count=12000,
                                  separately_sourced_smooth_physical_comparisons=truth_comparisons,
                                  all_frozen_input_and_mask_files_hash_checked=True,
                                  no_inference_performed=True))
    target = root/'inputs_manifest.json'
    if target.exists():
        assert read(target) == result, 'Frozen complete input manifest cannot be changed'
    else: write(target, result)
    print('MERGED_VERIFIED', len(cells), 'cells', flush=True)


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    c = sub.add_parser('catalog')
    c.add_argument('--evidence', type=Path, required=True)
    c.add_argument('--burgers-inputs', type=Path, required=True)
    c.add_argument('--diffusion-inventory', type=Path, required=True)
    c.add_argument('--diffusion-truths', type=Path, required=True)
    c.add_argument('--diffusion-root', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--catalog', type=Path, required=True)
    q.add_argument('--cohort', choices=['supervised','diffusion'], required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--pdes', nargs='+', choices=PDES+['burger'], default=PDES+['burger'])
    q.add_argument('--distributions', nargs='+', choices=DISTS, default=DISTS)
    q.add_argument('--baseline-output-root', type=Path,
                   default=Path('/large_storage/zhangxf/outputs/FM4PDEbaseline'))
    q.add_argument('--burgers-inputs', type=Path)
    q.add_argument('--diffusion-data-root', type=Path, default=Path('/data0/zhangxf/PDEdata'))
    q.add_argument('--workers', type=int, default=4)
    q.add_argument('--hash-raw-source', action='store_true')
    m = sub.add_parser('merge'); m.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'prepare':
        assert 1 <= args.workers <= 16
        if args.cohort == 'supervised' and 'burger' in args.pdes:
            assert args.burgers_inputs, '--burgers-inputs is required for Burgers'
    {'catalog':catalog, 'prepare':prepare, 'merge':merge}[args.mode](args)


if __name__ == '__main__': main()

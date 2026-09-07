"""Independently rescore the frozen NS guidance grid and paired evaluation.

Selection is checked only against calibration predictions. Numerical failures
and very large finite errors are retained. Partial audits are engineering data.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ns_loss_exchange import fm_pde_loss, diffusion_pde_loss
from run_ns_loss_study import sha, write
from spectral_diagnostics import spectral_record, self_check


def csvwrite(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def independent_selection(protocol, rows):
    """Input averages and stable fixed-order tie breaking, without evaluation."""
    selected, summary = {}, []
    groups = defaultdict(list)
    for row in rows:
        assert row['stage'] == 'calibration'
        groups[row['task'], row['variant']].append(row)
    for task in protocol['tasks']:
        eligible = []
        for order, candidate in enumerate(protocol['candidates']):
            rr = groups[task, candidate['name']]
            good = [r for r in rr if r['status'] == 'complete']
            assert len(rr) <= 12
            values = [np.mean([r['primary_error'] for r in good if r['sample_id'] == i])
                      for i in protocol['calibration_ids']] if len(good) == 12 else None
            row = dict(task=task, **candidate, calls=len(rr), finite_calls=len(good), eligible=len(good) == 12,
                       mean_primary=float(np.mean(values)) if values is not None else None,
                       sd_input_primary=float(np.std(values, ddof=1)) if values is not None else None)
            summary.append(row)
            if row['eligible']:
                eligible.append((row['mean_primary'], order, candidate))
        if eligible:
            selected[task] = min(eligible, key=lambda x: (x[0], x[1]))[2]
    return selected, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['inputs', 'results', 'baseline-audit', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    import torch
    from sampling.config import AblationConfig
    torch.set_num_threads(2)
    protocol = json.loads((args.results / 'protocol.json').read_text())
    ph = sha(args.results / 'protocol.json')
    source = json.loads((args.inputs / 'source.json').read_text())
    assert sha(args.inputs / 'source.json') == protocol['source_sha256']
    assert sha(args.inputs / 'fields_masks.npz') == protocol['fields_sha256'] == source['fields_sha256']
    for name, h in protocol['code_sha256'].items():
        assert sha(ROOT / name) == h, name
    ids, seeds = protocol['evaluation_ids'], protocol['seeds']
    assert ids == source['evaluation_ids'] and seeds == [0, 1, 2]
    assert protocol['calibration_ids'] == source['pilot_ids'] == [251, 878, 197, 364]
    assert not set(ids) & set(protocol['calibration_ids'])
    assert len(ids) == 32 and protocol['steps'] == 100
    assert protocol['calibration_calls'] == 540 and protocol['evaluation_calls'] == 576
    assert protocol['tasks'] == ['forward', 'inverse', 'both']
    assert {(c['observation_multiplier'], c['pde_multiplier']) for c in protocol['candidates']} == {
        (o, p) for o in [.1, .3, 1., 3., 10.] for p in [0., 1., 1000.]}
    assert len(protocol['candidates']) == 15
    assert protocol['candidates'][0] == dict(name='obs1_pde1', observation_multiplier=1., pde_multiplier=1.)
    data = np.load(args.inputs / 'fields_masks.npz')
    params = {k: torch.tensor([v], dtype=torch.float64) for k, v in source['pde_params'].items()}
    configs = {t: AblationConfig(**source['fm_configs'][t]) for t in protocol['tasks']}
    environment = []
    for shard in [0, 1]:
        pilot = json.loads((args.results / f'implementation_check_{shard}.json').read_text())
        assert pilot['status'] == 'pass' and pilot['protocol_sha256'] == ph
        assert len(pilot['checks']) == 2
        assert all(r['repeat'] == r['hidden'] == 0 and r['sensitivity'] > 0 for r in pilot['checks'])
        assert pilot['checks'][0]['original_runner'] == 0
        env = json.loads((args.results / f'environment_{shard}.json').read_text())
        assert env['protocol_sha256'] == ph and env['tf32'] is False and env['steps'] == 100 and env['batch_size'] == 1
        environment.append(env)
    assert environment[0]['uuid'] != environment[1]['uuid']
    selection_path = args.results / 'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    sh = sha(selection_path) if selection else None
    if selection:
        assert selection['protocol_sha256'] == ph and selection['calibration_calls_verified'] == 540
    expected = {
        ('calibration', t, c['name'], i, s)
        for t in protocol['tasks'] for c in protocol['candidates'] for i in protocol['calibration_ids'] for s in seeds
    } | {('evaluation', t, v, i, s) for t in protocol['tasks'] for v in ['reference', 'selected'] for i in ids for s in seeds}
    seen, hashes, rows, spectra, examples = set(), {}, [], [], {}
    outcomes = {stage: Counter() for stage in ['calibration', 'evaluation']}
    ensemble = defaultdict(list)

    def truth_masks(task, i):
        gt = [torch.from_numpy(data[f'{f}_{i}']) for f in ['a', 'u']]
        ma, mu = [torch.from_numpy(data[f'mask_{f}_{i}']) for f in ['a', 'u']]
        if task == 'both':
            mu = ma.clone()
        if task == 'forward':
            mu = torch.zeros_like(mu)
        if task == 'inverse':
            ma = torch.zeros_like(ma)
        return gt, [ma, mu]

    def score(pred, truth, masks, task):
        a, u = [x.double() for x in pred]
        with torch.no_grad():
            values = dict(secant_rms=float(fm_pde_loss(a, u, configs[task], params)) ** .5,
                          spatial_loss=float(diffusion_pde_loss(a, u)))
        for field, x, y, mask in zip(['a', 'u'], [a, u], truth, masks):
            y = y.double()
            values['rel_l2_' + field] = float((x - y).norm() / y.norm())
            values['obs_rel_l2_' + field] = float(((x - y) * mask).norm() / (y * mask).norm()) if mask.any() else None
        values['primary_error'] = (values['rel_l2_u'] if task == 'forward' else values['rel_l2_a'] if task == 'inverse'
                                   else max(values['rel_l2_a'], values['rel_l2_u']))
        assert np.isfinite([v for v in values.values() if v is not None]).all()
        return values

    def add_spectra(pred, truth, metadata):
        for field, x, y in zip(['a', 'u'], pred, truth):
            x, y = x[0, 0].double().numpy(), y[0, 0].double().numpy()
            result = spectral_record(x, y, 'periodic_fft')
            result['sensitivity_bands'] = {
                f'{lo}/{hi}': spectral_record(x, y, 'periodic_fft', (lo, hi))['bands'] for lo, hi in [(4, 16), (16, 48)]}
            spectra.append(dict(**metadata, field=field, **result))

    for stage in outcomes:
        for path in sorted((args.results / stage).rglob('*.json')):
            receipt = json.loads(path.read_text())
            key = tuple(receipt[k] for k in ['stage', 'task', 'variant', 'sample_id', 'seed'])
            assert key in expected and key not in seen and key[0] == stage
            _, task, variant, i, seed = key
            expected_path = args.results / stage / task / variant / f'sample{i}_seed{seed}.json'
            assert path == expected_path and receipt['protocol_sha256'] == ph
            assert receipt['selection_sha256'] == (sh if stage == 'evaluation' else None)
            if stage == 'evaluation':
                assert selection, 'Evaluation cannot precede frozen calibration selection'
                candidate = protocol['candidates'][0] if variant == 'reference' else selection['selected'][task]
            else:
                candidate = next(c for c in protocol['candidates'] if c['name'] == variant)
            assert receipt['candidate'] == candidate
            seen.add(key)
            status = receipt['status']
            assert status in ['complete', 'nonfinite', 'unstable']
            outcomes[stage][status] += 1
            assert np.isfinite(receipt['seconds']) and receipt['seconds'] > 0
            row = dict(stage=stage, task=task, variant=variant, sample_id=i, seed=seed, status=status,
                       candidate=candidate['name'], observation_multiplier=candidate['observation_multiplier'],
                       pde_multiplier=candidate['pde_multiplier'], seconds=receipt['seconds'], nfe=receipt['nfe'])
            rows.append(row)
            hashes[str(path)] = sha(path)
            if status == 'unstable':
                assert receipt.get('error')
                row['error'] = receipt['error']
                continue
            tensor = path.with_suffix('.pt')
            assert sha(tensor) == receipt['prediction_sha256']
            hashes[str(tensor)] = receipt['prediction_sha256']
            d = torch.load(tensor, map_location='cpu', weights_only=False)
            assert all(d['receipt'][k] == v for k, v in receipt.items() if k != 'prediction_sha256')
            gt, masks = truth_masks(task, i)
            assert receipt['nfe'] == 100
            for pred, truth, mask, expected_truth, expected_mask in zip(d['prediction'], d['truth'], d['masks'], gt, masks):
                assert pred.dtype == truth.dtype == torch.float32 and tuple(pred.shape) == (1, 1, 128, 128)
                assert torch.equal(truth, expected_truth.float()) and torch.equal(mask, expected_mask)
                assert int(mask.sum()) in [0, 500]
            finite = all(bool(torch.isfinite(x).all()) for x in d['prediction'])
            assert finite == (status == 'complete')
            if not finite:
                assert receipt['primary_error'] is None
                continue
            assert len(d['trace']) == 100 and [r['step'] for r in d['trace']] == list(range(100))
            values = score(d['prediction'], gt, masks, task)
            for name in ['rel_l2_a', 'rel_l2_u', 'primary_error']:
                assert np.isclose(values[name], receipt[name], rtol=1e-10, atol=1e-12), (key, name)
            row.update(values)
            if stage == 'evaluation':
                meta = dict(task=task, variant=variant, sample_id=i, seed=seed, estimator='single')
                add_spectra(d['prediction'], gt, meta)
                ensemble[task, variant, i].append((seed, [x.double() for x in d['prediction']]))
                if i == ids[0] and seed == 0:
                    for field, pred, truth, mask in zip(['a', 'u'], d['prediction'], gt, masks):
                        examples[f'{task}_{variant}_{field}'] = pred[0, 0].numpy()
                        examples[f'{task}_truth_{field}'] = truth[0, 0].numpy()
                        examples[f'{task}_mask_{field}'] = mask[0, 0].numpy()

    calibration = [r for r in rows if r['stage'] == 'calibration']
    independently_selected, grid = independent_selection(protocol, calibration)
    if selection:
        assert len(calibration) == 540 and independently_selected == selection['selected']
        assert len(selection['calibration_summary']) == len(grid) == 45
        for actual, frozen in zip(grid, selection['calibration_summary']):
            assert actual.keys() == frozen.keys()
            for k, v in actual.items():
                assert np.isclose(v, frozen[k], rtol=1e-10, atol=1e-12) if isinstance(v, float) else v == frozen[k], (k, actual, frozen)
        for shard in [0, 1]:
            assert json.loads((args.results / f'calibration_complete_{shard}.json').read_text()) == dict(protocol_sha256=ph, calls=270)
    identical_control_pairs = 0
    if selection:
        for task in protocol['tasks']:
            if selection['selected'][task] != protocol['candidates'][0]:
                continue
            for i in ids:
                reference = dict(ensemble.get((task, 'reference', i), []))
                selected = dict(ensemble.get((task, 'selected', i), []))
                for seed in reference.keys() & selected.keys():
                    assert all(torch.equal(x, y) for x, y in zip(reference[seed], selected[seed])), (task, i, seed)
                    identical_control_pairs += 1
    ensemble_rows = []
    for (task, variant, i), values in ensemble.items():
        if len(values) != 3:
            continue  # Full report separately retains failed/missing seed counts.
        assert {s for s, _ in values} == {0, 1, 2}
        pred = [torch.stack([v[j] for _, v in values]).mean(0) for j in [0, 1]]
        gt, masks = truth_masks(task, i)
        ensemble_rows.append(dict(task=task, variant=variant, sample_id=i, calls=3, **score(pred, gt, masks, task)))
        add_spectra(pred, gt, dict(task=task, variant=variant, sample_id=i, seed=-1, estimator='mean3'))

    # The baseline part of this separate audit is already complete, even while
    # its formal generative-loss portion is partial. Require all 288 baseline
    # prediction hashes and all 384 field metrics, independently of that status.
    bm = json.loads((args.baseline_audit / 'ns_audit_manifest.json').read_text())
    assert bm['baseline_predictions_verified'] == 288 and bm['baseline_field_metrics'] == 384
    assert bm['source_sha256'] == protocol['source_sha256']
    baseline_hashes = {p: h for p, h in bm['source_hashes'].items() if 'baseline_results_gpu_v2' in Path(p).parts}
    assert len(baseline_hashes) == 288
    for path, h in baseline_hashes.items():
        assert sha(Path(path)) == h
    baseline_csv = args.baseline_audit / 'ns_baseline_per_field.csv'
    assert sha(baseline_csv) == bm['outputs'][baseline_csv.name]
    baselines = list(csv.DictReader(baseline_csv.open()))
    assert len(baselines) == 384
    baseline_keys = {(r['task'], r['method'], r['field'], int(r['sample_id'])) for r in baselines}
    assert len(baseline_keys) == 384
    assert baseline_keys == {(t, m, f, i) for t in protocol['tasks'] for m in ['RecFNO', 'Senseiver', 'VoronoiCNN']
                             for f in (['u'] if t == 'forward' else ['a'] if t == 'inverse' else ['a', 'u']) for i in ids}
    baseline_spectra_path = args.baseline_audit / 'ns_frequency_records.json.gz'
    assert sha(baseline_spectra_path) == bm['outputs'][baseline_spectra_path.name]
    with gzip.open(baseline_spectra_path, 'rt') as f:
        baseline_spectra = [r for r in json.load(f) if r['method'] in ['RecFNO', 'Senseiver', 'VoronoiCNN']]
    assert len(baseline_spectra) == 384
    for r in baseline_spectra:
        key = (r['task'], r['method'], r['field'], r['sample_id'])
        assert key in baseline_keys and r['seed'] == 0
        spectra.append(dict(task=r['task'], variant=r['method'], field=r['field'], sample_id=r['sample_id'], seed=0,
                            estimator='deterministic', **{k: r[k] for k in [
                                'reference_power', 'prediction_power', 'error_power', 'cross_power',
                                'reference_total', 'rel_l2', 'bands', 'sensitivity_bands']}))
    complete = seen == expected
    if args.require_complete:
        assert complete, f'Incomplete: {len(calibration)}/540 calibration, {len(rows)-len(calibration)}/576 evaluation'
        assert selection
        for shard in [0, 1]:
            assert json.loads((args.results / f'evaluation_complete_{shard}.json').read_text()) == dict(protocol_sha256=ph, calls=288)
    # Write only after assertions; unsuccessful production validation leaves no export.
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in [('calibration_per_call.csv', rows), ('calibration_grid.csv', grid),
                         ('calibration_mean3.csv', ensemble_rows), ('calibration_baselines.csv', baselines)]:
        csvwrite(args.output / name, values)
    with gzip.open(args.output / 'calibration_spectra.json.gz', 'wt') as f:
        json.dump(spectra, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(), allow_nan=False)
    np.savez_compressed(args.output / 'calibration_examples.npz', **examples)
    manifest = dict(status='complete' if complete else 'partial',
                    outcomes={k: dict(v) for k, v in outcomes.items()},
                    protocol_sha256=ph, selection_sha256=sh, selected=selection['selected'] if selection else None,
                    selection_independently_verified=bool(selection), source_sha256=protocol['source_sha256'],
                    identical_setting_prediction_pairs_verified=identical_control_pairs,
                    source_hashes=hashes, baseline_prediction_hashes=baseline_hashes,
                    baseline_audit_sha256=sha(args.baseline_audit / 'ns_audit_manifest.json'),
                    environment=environment, spectral_checks=self_check(), script_sha256=sha(Path(__file__)),
                    scope='Four reserved development inputs from the same test source, 32 disjoint evaluation inputs. Three paired seeds. No evaluation result used for selection. Failed and large finite outcomes retained.',
                    outputs={p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'calibration_audit_manifest.json'})
    write(args.output / 'calibration_audit_manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['status', 'outcomes', 'selected', 'selection_independently_verified']}))


if __name__ == '__main__':
    main()

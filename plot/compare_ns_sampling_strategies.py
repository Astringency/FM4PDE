"""Compare single draws and three-field averages after both NS studies finish.

No sampling or source mutation. All loss settings, calibrated controls and
matched baselines remain visible; computational counts are not latency.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from export_ns_loss_spectra import summarize
from run_ns_loss_study import sha, write, TASKS, VARIANTS

METRICS = ['primary_error', 'rel_l2_a', 'rel_l2_u']
BASELINES = ['RecFNO', 'Senseiver', 'VoronoiCNN']


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    assert rows
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimators(predictions, truth, task):
    """Score three draws and their field average; check the variance identity."""
    x, y = np.asarray(predictions, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    assert x.shape == (3, 2, 128, 128) and y.shape == (2, 128, 128)
    assert np.isfinite(x).all() and np.isfinite(y).all() and task in TASKS
    denominator = np.sum(y*y, axis=(1, 2))
    assert (denominator > 0).all()
    mean = x.mean(axis=0)
    individual_squared = np.sum((x-y[None])**2, axis=(2, 3))/denominator
    mean_squared = np.sum((mean-y)**2, axis=(1, 2))/denominator
    spread = np.mean(np.sum((x-mean[None])**2, axis=(2, 3)), axis=0)/denominator
    assert np.allclose(individual_squared.mean(0)-mean_squared, spread, rtol=2e-11, atol=2e-13)
    e, em = np.sqrt(individual_squared), np.sqrt(mean_squared)
    primary = lambda v: v[..., 1] if task == 'forward' else v[..., 0] if task == 'inverse' else v.max(axis=-1)
    single = dict(primary_error=float(primary(e).mean()), rel_l2_a=float(e[:, 0].mean()), rel_l2_u=float(e[:, 1].mean()))
    mean3 = dict(primary_error=float(primary(em)), rel_l2_a=float(em[0]), rel_l2_u=float(em[1]))
    assert all(mean3[k] <= single[k]+2e-12*max(1., single[k]) for k in METRICS)
    identities = [dict(field=field, mean_individual_squared_error=float(individual_squared[:, j].mean()),
                       ensemble_squared_error=float(mean_squared[j]), normalized_spread=float(spread[j]),
                       identity_gap=float(individual_squared[:, j].mean()-mean_squared[j]-spread[j]))
                  for j, field in enumerate(['a', 'u'])]
    return dict(single=single, mean3=mean3), mean, identities


def load_prediction(path, manifest, fields, task, sample_id, seed, expected_nfe):
    import torch
    assert sha(path) == manifest['source_hashes'][str(path)]
    data = torch.load(path, map_location='cpu', weights_only=False)
    receipt = data['receipt']
    assert receipt['status'] == 'complete' and receipt['sample_id'] == sample_id and receipt['seed'] == seed
    assert receipt['task'] == task and receipt['nfe'] == expected_nfe
    assert receipt['protocol_sha256'] == manifest['protocol_sha256']
    truth = np.stack([fields[f'{f}_{sample_id}'][0, 0] for f in ['a', 'u']]).astype(np.float64)
    for j, f in enumerate(['a', 'u']):
        assert np.array_equal(data['truth'][j][0, 0].double().numpy(), truth[j])
        mask = fields[f'mask_{"a" if task == "both" else f}_{sample_id}'][0, 0].copy()
        if (task == 'forward' and f == 'u') or (task == 'inverse' and f == 'a'):
            mask[:] = 0
        assert np.array_equal(data['masks'][j][0, 0].numpy(), mask)
    pred = np.stack([v[0, 0].double().numpy() for v in data['prediction']])
    return pred, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); study = args.study
    formal_audit = study/'ns_complete_audit'; calibration_audit = study/'calibration_complete_audit'
    fp, cp = formal_audit/'ns_audit_manifest.json', calibration_audit/'calibration_audit_manifest.json'
    fm, cm = json.loads(fp.read_text()), json.loads(cp.read_text())
    assert fm['status'] == 'complete' and fm['calls_verified'] == 1728
    assert fm['outcome_counts'] == {'complete': 1728} and fm['recipient_guidance_traces_verified'] == 1728
    assert fm['trace_weighting_schema'] == 'recipient_update_v2'
    assert cm['status'] == 'complete' and cm['selection_independently_verified']
    assert cm['outcomes'] == {'calibration': {'complete': 540}, 'evaluation': {'complete': 576}}
    source_path = study/'inputs_v2/source.json'; source = json.loads(source_path.read_text())
    assert fm['source_sha256'] == cm['source_sha256'] == sha(source_path)
    fields_path = study/'inputs_v2/fields_masks.npz'; assert sha(fields_path) == source['fields_sha256']
    fields = np.load(fields_path); ids = source['evaluation_ids']; seeds = source['inference_seeds']
    assert len(ids) == len(set(ids)) == 32 and seeds == [0, 1, 2]
    for audit, manifest, names in [
        (formal_audit, fm, ['ns_loss_per_run.csv', 'ns_baseline_per_field.csv']),
        (calibration_audit, cm, ['calibration_per_call.csv', 'calibration_mean3.csv', 'calibration_baselines.csv'])]:
        for name in names: assert sha(audit/name) == manifest['outputs'][name]
    protocol_path = study/'guidance_calibration_v1/protocol.json'
    assert sha(protocol_path) == cm['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    assert protocol['evaluation_ids'] == ids and protocol['seeds'] == seeds
    assert protocol['fields_sha256'] == source['fields_sha256'] and protocol['steps'] == 100
    formal_protocol_path = study/'ns_results_v3/protocol.json'
    assert sha(formal_protocol_path) == fm['protocol_sha256']
    formal_protocol = json.loads(formal_protocol_path.read_text())
    assert formal_protocol['fm_checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert formal_protocol['evaluation_ids'] == ids and formal_protocol['seeds'] == seeds
    assert formal_protocol['tasks'] == protocol['tasks'] == TASKS and formal_protocol['variants'] == VARIANTS
    assert not set(protocol['calibration_ids']) & set(ids)
    assert sha(study/'guidance_calibration_v1/selection.json') == cm['selection_sha256']

    import torch
    torch.set_num_threads(2)
    hashes = {}; settings = []; per_input = []; identities = []; arrays = {}; comparisons = []
    values = {}; individual = {}

    def add_setting(key, label, family, nfe, path_for, manifest, check_rows=None):
        settings.append(dict(setting=key, label=label, family=family, nfe=nfe))
        for task in TASKS:
            for sample_id in ids:
                paths = [path_for(task, sample_id, seed) for seed in seeds]
                predictions = [load_prediction(path, manifest, fields, task, sample_id, seed, nfe)[0]
                               for seed, path in zip(seeds, paths)]
                hashes.update({str(path): manifest['source_hashes'][str(path)] for path in paths})
                truth = np.stack([fields[f'{f}_{sample_id}'][0, 0] for f in ['a', 'u']])
                scored, mean, checks = estimators(predictions, truth, task)
                arrays[f'{key}_{task}_sample{sample_id}'] = mean
                for mode, metrics in scored.items():
                    if check_rows is not None:
                        expected = check_rows[task, sample_id, mode]
                        assert all(np.isclose(metrics[k], expected[k], rtol=2e-11, atol=2e-12) for k in METRICS)
                    values[key, mode, task, sample_id] = metrics
                    per_input.append(dict(setting=key, family=family, estimator=mode, task=task, sample_id=sample_id,
                                          sampling_calls=1 if mode == 'single' else 3,
                                          nfe_per_prediction=nfe*(1 if mode == 'single' else 3), **metrics))
                identities.extend(dict(setting=key, task=task, sample_id=sample_id, **r) for r in checks)
                if key in ['FM4PDE_100_original', 'FM_cal_reference']:
                    for seed, pred in zip(seeds, predictions):individual[key, task, sample_id, seed] = pred

    raw = read_csv(formal_audit/'ns_loss_per_run.csv')
    for method, n, exchange in VARIANTS:
        key = f'{method}_{n}_{"exchanged" if exchange else "original"}'
        loss = 'F' if (method == 'FM4PDE') != exchange else 'D'
        checks = {}; grouped = defaultdict(list)
        for r in raw:
            if (r['method'], int(r['steps']), r['exchange'] == 'True') == (method, n, exchange):
                grouped[r['task'], int(r['sample_id'])].append(r)
        for k, rows in grouped.items():
            assert {int(r['seed']) for r in rows} == set(seeds) and len(rows) == 3
            for row in rows:
                ea, eu = float(row['rel_l2_a']), float(row['rel_l2_u'])
                row['primary_error'] = eu if k[0] == 'forward' else ea if k[0] == 'inverse' else max(ea, eu)
            checks[k] = {metric: float(np.mean([float(r[metric]) for r in rows])) for metric in METRICS}
        add_setting(key, f'{method} {n}, $L_{loss}$', 'loss_exchange', n if method == 'FM4PDE' else 2*n-1,
                    lambda t, i, seed, method=method, n=n, exchange=exchange: study/'ns_results_v3/results'/t/f'{method}_{n}'/
                    ('exchanged' if exchange else 'original')/f'sample{i}_seed{seed}.pt', fm)
        assert len(checks) == 96
        for (task, sample_id), expected in checks.items():
            assert all(np.isclose(values[key, 'single', task, sample_id][k], expected[k], rtol=2e-11, atol=2e-12) for k in METRICS)

    cal_calls = [r for r in read_csv(calibration_audit/'calibration_per_call.csv') if r['stage'] == 'evaluation']
    cal_means = read_csv(calibration_audit/'calibration_mean3.csv')
    for variant in ['reference', 'selected']:
        checks = {}
        for task in TASKS:
            for sample_id in ids:
                rows = [r for r in cal_calls if r['variant'] == variant and r['task'] == task and int(r['sample_id']) == sample_id]
                assert len(rows) == 3 and {int(r['seed']) for r in rows} == set(seeds)
                checks[task, sample_id, 'single'] = {k: float(np.mean([float(r[k]) for r in rows])) for k in METRICS}
                row = next(r for r in cal_means if r['variant'] == variant and r['task'] == task and int(r['sample_id']) == sample_id)
                checks[task, sample_id, 'mean3'] = {k: float(row[k]) for k in METRICS}
        add_setting('FM_cal_'+variant, 'FM4PDE 100, cal. '+variant, 'calibration', 100,
                    lambda t, i, seed, variant=variant: study/'guidance_calibration_v1/evaluation'/t/variant/f'sample{i}_seed{seed}.pt', cm, checks)
    assert len(identities) == 1536 and len(arrays) == 768 and len(hashes) == 2304
    reference_differences = []
    for task in TASKS:
        for sample_id in ids:
            for seed in seeds:
                a = individual['FM4PDE_100_original', task, sample_id, seed]
                b = individual['FM_cal_reference', task, sample_id, seed]
                reference_differences.append(dict(task=task, sample_id=sample_id, seed=seed,
                    max_abs_difference=float(np.max(np.abs(a-b))), relative_prediction_difference=float(np.linalg.norm(a-b)/np.linalg.norm(b))))
    del individual

    fb = read_csv(formal_audit/'ns_baseline_per_field.csv'); cb = read_csv(calibration_audit/'calibration_baselines.csv')
    key_for = lambda r: (r['task'], r['method'], int(r['sample_id']), r['field'])
    baseline = {key_for(r): float(r['rel_l2']) for r in fb}
    assert len(baseline) == len(cb) == 384 and set(baseline) == {key_for(r) for r in cb}
    assert all(np.isclose(baseline[key_for(r)], float(r['rel_l2']), rtol=2e-12) for r in cb)
    for method in BASELINES:
        settings.append(dict(setting=method, label=method, family='baseline', nfe=None))
        for task in TASKS:
            for sample_id in ids:
                ea, eu = (baseline.get((task, method, sample_id, f)) for f in ['a', 'u'])
                primary = eu if task == 'forward' else ea if task == 'inverse' else max(ea, eu)
                metrics = dict(primary_error=primary, rel_l2_a=ea, rel_l2_u=eu)
                values[method, 'deterministic', task, sample_id] = metrics
                per_input.append(dict(setting=method, family='baseline', estimator='deterministic', task=task,
                    sample_id=sample_id, sampling_calls=0, nfe_per_prediction=None, **metrics))

    summary = []
    for setting in settings:
        key = setting['setting']; modes = ['deterministic'] if setting['family'] == 'baseline' else ['single', 'mean3']
        for mode in modes:
            for task in TASKS:
                for metric in METRICS:
                    vv = [values[key, mode, task, i][metric] for i in ids]
                    assert all(v is not None for v in vv) or all(v is None for v in vv)
                    summary.append(dict(setting=key, estimator=mode, task=task, metric=metric,
                                        **summarize([v for v in vv if v is not None])))
        if setting['family'] != 'baseline':
            comparisons.append((key, 'mean3', key, 'single', 'averaging'))
            if key != 'FM_cal_selected':
                comparisons += [('FM_cal_selected', mode, key, mode, 'selected_vs_same_estimator') for mode in ['single', 'mean3']]
                comparisons.append(('FM_cal_selected', 'mean3', key, 'single', 'selected_mean3_vs_single'))
        else:
            comparisons += [('FM_cal_selected', mode, key, 'deterministic', 'selected_vs_baseline') for mode in ['single', 'mean3']]
    effects = []
    for method, mode, other, other_mode, contrast in comparisons:
        for task in TASKS:
            for metric in METRICS:
                a, b = ([values[k, e, task, i][metric] for i in ids] for k, e in [(method, mode), (other, other_mode)])
                if all(v is None for v in b):continue
                assert all(v is not None for v in a+b)
                effects.append(dict(setting=method, estimator=mode, comparator=other, comparator_estimator=other_mode,
                    contrast=contrast, task=task, metric=metric, **summarize(np.asarray(a)-np.asarray(b))))
    assert len(per_input) == 1824

    tex = [r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\caption{NS sampling strategies on the same 32 inputs (\%, mean $\pm$ sample SD). Single averages three individual seed errors within each input; mean3 scores the average of the three physical predictions. The joint score is the per-prediction maximum of the two field errors. NFE counts velocity/denoiser evaluations per returned prediction; mean3 triples calls. It is not a FLOP or latency comparison. Both original and exchanged losses, both calibration controls, and all matched baselines are retained. The calibration reference separately records original FM settings under the calibration environment. Only FM guidance coefficients were calibrated; DiffusionPDE uses the archived coefficients.}',
        r'\label{tab:ns-sampling-strategies}', r'\begin{tabular}{@{}lrrrr@{}}\toprule',
        r'Setting / estimator & NFE & Forward & Inverse & Joint \\\midrule']
    for setting in settings:
        for mode in (['deterministic'] if setting['family'] == 'baseline' else ['single', 'mean3']):
            count = '---' if setting['nfe'] is None else str(setting['nfe']*(3 if mode == 'mean3' else 1))
            cells = []
            for task in TASKS:
                r = next(r for r in summary if (r['setting'], r['estimator'], r['task'], r['metric']) ==
                         (setting['setting'], mode, task, 'primary_error'))
                assert r['n'] == 32
                cells.append(f"${100*r['mean']:.2f}\\pm{100*r['sd']:.2f}$")
            tex.append(' & '.join([setting['label']+(' / '+mode if mode != 'deterministic' else ''), count, *cells])+r' \\')
    tex += [r'\bottomrule\end{tabular}\end{table}']
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('ns_strategy_per_input.csv', per_input), ('ns_strategy_summary.csv', summary),
                       ('ns_strategy_paired_effects.csv', effects), ('ns_ensemble_identity.csv', identities),
                       ('ns_reference_environment_differences.csv', reference_differences)]:write_csv(args.output/name, rows)
    np.savez_compressed(args.output/'ns_strategy_ensemble_fields.npz', **arrays)
    (args.output/'ns_strategy_table.tex').write_text('\n'.join(tex)+'\n')
    write(args.output/'ns_strategy_manifest.json', dict(status='complete', calls_verified=1728,
        calibration_evaluation_calls_verified=576, ensemble_groups=768, variance_identity_field_checks=len(identities),
        source_sha256=sha(source_path), formal_audit_sha256=sha(fp), calibration_audit_sha256=sha(cp),
        calibration_selection_sha256=cm['selection_sha256'], source_prediction_sha256=hashes,
        script_sha256=sha(Path(__file__)),
        statistics='Seed averages within input, sample SD and pointwise paired-input bootstrap intervals across 32 physical inputs. Mean3 is a field average, not an average error or certified posterior mean.',
        scope='Mixed hardware/native arithmetic retained; separate original-FM calibration control and cross-environment prediction differences retained. NFE is not a latency or architecture-normalized compute metric. Fixed-grid calibration was applied to FM only; Diff weights remain archived settings.',
        outputs={p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'ns_strategy_manifest.json'}))


if __name__ == '__main__':main()

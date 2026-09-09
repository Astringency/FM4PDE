#!/usr/bin/env python3
"""Recompute completed Poisson field errors and reported summary statistics.

This reads the compact physical fields after the independent raw-pool audits.
It does not sample models or substitute for visual inspection of the figures.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

KS = (1, 3, 10, 100, 1000)
TASKS = ('forward', 'inverse', 'both')
FIELDS = (('forward', 'u'), ('inverse', 'a'), ('both', 'a'), ('both', 'u'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def index_rows(rows, keys):
    index = {tuple(row[key] for key in keys): row for row in rows}
    assert len(index) == len(rows), 'Duplicate summary keys'
    return index


def main(args):
    assert not args.output.exists(), 'Use a new verification output'
    source = args.paper / 'source_data/conditional_scaling_0909'
    ready_path = args.audit / 'READY_FOR_PAPER_REVIEW.json'
    final_path = source / 'conditional_scaling_final_manifest.json'
    ready = json.loads(ready_path.read_text())
    assert Path(ready['manifest']).name == final_path.name
    final = json.loads(final_path.read_text())
    assert final['complete'] and final['physical_inputs'] == 32
    assert final['canonical_trajectories'] == 96000
    assert final['timed_trajectories'] == 106944 and final['rows'] == 480
    assert final['K'] == list(KS) and final['bootstrap_resamples'] == 100000
    assert final['simultaneous_interval_comparisons'] == 16
    inputs = [ready_path, final_path]
    for name in ('per_input.csv', 'fields.npz', 'summary.csv', 'paired_effects.csv', 'timing.csv'):
        inputs.append(source / ('conditional_scaling_' + name))
    before = {str(path): sha(path) for path in inputs}
    rows = read_rows(source / 'conditional_scaling_per_input.csv')
    index = index_rows(rows, ('task', 'offset', 'K'))
    expected = {(task, str(offset), str(k)) for task in TASKS
                for offset in range(1500, 1532) for k in KS}
    assert len(rows) == 480 and set(index) == expected
    arrays = np.load(source / 'conditional_scaling_fields.npz', allow_pickle=False)
    expected_arrays = {f'{task}_{offset}_{suffix}' for task in TASKS
                       for offset in range(1500, 1532)
                       for suffix in ('truth', 'mask', *(f'K{k}' for k in KS))}
    assert set(arrays.files) == expected_arrays
    max_errors = {'field_error': 0., 'observation_mse': 0., 'statistics': 0.}
    checks = 0

    def close(actual, reference, category):
        nonlocal checks
        actual, reference = float(actual), float(reference)
        assert np.isfinite(actual) and np.isfinite(reference)
        difference = abs(actual - reference)
        assert difference <= 1e-12 + 2e-10 * abs(reference), (category, actual, reference)
        max_errors[category] = max(max_errors[category], difference)
        checks += 1

    for task in TASKS:
        for offset in range(1500, 1532):
            prefix = f'{task}_{offset}'
            truth, mask = arrays[prefix + '_truth'], arrays[prefix + '_mask']
            assert truth.shape == mask.shape == (2, 128, 128)
            assert np.isfinite(truth).all() and np.isin(mask, (0, 1)).all()
            assert np.array_equal(mask.sum(axis=(1, 2)), [500, 500])
            for k in KS:
                row = index[task, str(offset), str(k)]
                prediction = arrays[prefix + f'_K{k}']
                assert prediction.shape == truth.shape and np.isfinite(prediction).all()
                for channel, field in enumerate(('a', 'u')):
                    difference = prediction[channel] - truth[channel]
                    error = np.sqrt(np.sum(difference ** 2) / np.sum(truth[channel] ** 2))
                    obs = np.mean(difference[mask[channel].astype(bool)] ** 2)
                    close(error, row['rel_l2_' + field], 'field_error')
                    close(obs, row['L_obs_' + field], 'observation_mse')
                    assert float(row['normalized_spread_' + field]) >= 0
                    assert error <= float(row['mean_individual_error_' + field]) + 1e-10
                    if k == 1:
                        close(row['normalized_spread_' + field], 0, 'statistics')
                assert 0 < float(row['compute_seconds']) <= float(row['seconds'])
                assert 0 < int(row['peak_bytes']) < 80 * 2**30
    summary = index_rows(read_rows(source / 'conditional_scaling_summary.csv'), ('task', 'field', 'K'))
    effects = index_rows(read_rows(source / 'conditional_scaling_paired_effects.csv'), ('task', 'field', 'K'))
    timings = index_rows(read_rows(source / 'conditional_scaling_timing.csv'), ('task', 'K'))
    assert set(summary) == {(t, f, str(k)) for t, f in FIELDS for k in KS}
    assert set(effects) == {(t, f, str(k)) for t, f in FIELDS for k in KS[1:]}
    assert set(timings) == {(t, str(k)) for t in TASKS for k in KS}
    # The recorded bootstrap seed permits exact interval reconstruction; the
    # errors themselves above are recomputed without importing the exporter.
    bootstrap = np.random.default_rng(20260909).integers(0, 32, size=(100000, 32))
    findings = []
    for task, field in FIELDS:
        series = np.array([[float(index[task, str(i), str(k)]['rel_l2_' + field]) * 100
                            for i in range(1500, 1532)] for k in KS])
        for j, k in enumerate(KS):
            row = summary[task, field, str(k)]
            assert int(row['n']) == 32
            boot_means = np.sum(series[j][bootstrap], axis=1) / 32
            for key, value in dict(mean_percent=np.sum(series[j]) / 32,
                                   sd_percent=np.sqrt(np.sum((series[j] - series[j].mean()) ** 2) / 31),
                                   mean_ci_low=np.quantile(boot_means, .025),
                                   mean_ci_high=np.quantile(boot_means, .975)).items():
                close(value, row[key], 'statistics')
            if k == 1:
                continue
            delta = series[j] - series[0]
            paired_boot = np.sum(delta[bootstrap], axis=1) / 32
            paired = effects[task, field, str(k)]
            assert int(paired['n']) == 32 and int(paired['improved_inputs']) == np.count_nonzero(delta < 0)
            for key, value in dict(mean_delta_pp=delta.mean(),
                                   simultaneous_ci_low=np.quantile(paired_boot, .05 / 32),
                                   simultaneous_ci_high=np.quantile(paired_boot, 1 - .05 / 32),
                                   relative_mean_error_change_percent=100 * (series[j].mean() / series[0].mean() - 1)).items():
                close(value, paired[key], 'statistics')
        findings.append(dict(task=task, field=field, means_percent=series.mean(axis=1).tolist(),
                             lowest_mean_K=KS[int(series.mean(axis=1).argmin())],
                             means_nondecreasing_with_K=bool(np.all(np.diff(series.mean(axis=1)) >= 0)),
                             means_nonincreasing_with_K=bool(np.all(np.diff(series.mean(axis=1)) <= 0))))
    for task in TASKS:
        singles = np.array([float(index[task, str(i), '1']['seconds']) for i in range(1500, 1532)])
        for k in KS:
            values = np.array([float(index[task, str(i), str(k)]['seconds']) for i in range(1500, 1532)])
            row = timings[task, str(k)]
            assert int(row['n']) == 32
            for key, value in dict(median_seconds=np.median(values), mean_seconds=values.mean(),
                                   sd_seconds=values.std(ddof=1), q25_seconds=np.quantile(values, .25),
                                   q75_seconds=np.quantile(values, .75),
                                   median_ratio_to_K1=np.median(values / singles),
                                   median_speedup_vs_serial=np.median(k * singles / values)).items():
                close(value, row[key], 'statistics')
    arrays.close()
    assert all(sha(Path(path)) == expected_sha for path, expected_sha in before.items())
    output = dict(status='pass', numerical_review_complete=True,
                  completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  ready_for_paper_review_sha256=before[str(ready_path)],
                  final_manifest_sha256=before[str(final_path)], script_sha256=sha(Path(__file__)),
                  rows=480, inputs=32, task_fields=4, scalar_comparisons=checks,
                  maximum_absolute_differences=max_errors, findings=findings, input_sha256=before,
                  scope='Recomputed both field errors and both observation MSEs from all 480 compact means; checked 20 summaries, 16 paired intervals, and 15 measured-latency summaries. Raw-pool identity, PDE residuals, and stochastic-path audits remain in the two host export manifests. Figure inspection and manuscript interpretation require a separate final review.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: value for key, value in output.items() if key not in ('input_sha256', 'scope')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

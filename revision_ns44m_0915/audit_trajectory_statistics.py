"""Independently reconstruct the published NS trajectory statistics from CSVs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--visuals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.task / 'protocol.json').read_text())
    manifest_path = args.visuals / 'trajectory_sources.json'
    manifest = json.loads(manifest_path.read_text())
    historical = json.loads(Path(manifest['original_protocol_source']).read_text())['protocol']
    ids = historical['evaluation_ids']
    assert len(ids) == len(set(ids)) == 32 and historical['inference_seeds'] == [0, 1, 2]
    assert manifest['sample_ids_in_bootstrap_order'] == ids
    assert manifest['protocol_sha256'] == digest(args.task / 'protocol.json')
    id_position = {value: i for i, value in enumerate(ids)}
    variants = ['guidance_obs_only', 'guidance_obs_pde']
    errors = {v: np.full((3, 32, 100, 2), np.nan) for v in variants}
    curves = {v: [] for v in variants}
    sources = {}
    for job in protocol['jobs']:
        parts = job['job_id'].split('/')
        if parts[0] != 'guidance_controls' or parts[1] not in variants:
            continue
        variant, seed = parts[1], int(parts[2][4:])
        folder = args.task / 'jobs' / job['job_id']
        receipt_path = folder / 'receipt.json'
        r = json.loads(receipt_path.read_text())
        assert r['sample_ids'] == job['sample_ids'] and len(r['sample_ids']) == 4
        assert r['parameter_count'] == 44121218 and r['model_sha256'] == protocol['model_sha256']
        assert r['config']['batch_size'] == 4 and r['config']['num_steps'] == 100
        paths = {name: folder / r['artifacts'][name]['path'] for name in ['curves', 'per_sample_curves']}
        for name, path in paths.items():
            assert digest(path) == r['artifacts'][name]['sha256']
        source = manifest['sources'][job['job_id']]
        assert digest(receipt_path) == source['receipt_sha256']
        for name, path in paths.items():
            assert source['artifacts'][name]['sha256'] == digest(path)
        rows = read_csv(paths['per_sample_curves'])
        assert len(rows) == 400
        for row in rows:
            sample, step = int(row['sample_id']), int(row['step'])
            assert sample in r['sample_ids'] and step in range(100)
            slot = errors[variant][seed, id_position[sample], step]
            assert np.isnan(slot).all(), (variant, seed, sample, step)
            slot[:] = float(row['rel_l2_a']), float(row['rel_l2_u'])
        curve = read_csv(paths['curves'])
        assert [int(row['step']) for row in curve] == list(range(100))
        curves[variant].append(curve)
        sources[job['job_id']] = dict(receipt_sha256=digest(receipt_path),
            **{name+'_sha256': digest(path) for name, path in paths.items()})
    assert len(sources) == 48 and all(len(v) == 24 for v in curves.values())
    draws = np.random.default_rng(20260907).integers(0, 32, (10000, 32))[:3000]
    expected = {}
    input_means = {}
    direct = {'t':'t_next', 'guidance_t':'t', 'state_residual':'eval_pde_residual_norm',
        'endpoint_residual':'guidance_pde_residual_norm',
        'state_observation_mse_a':'eval_L_obs_a', 'state_observation_mse_u':'eval_L_obs_u',
        'endpoint_observation_mse_a':'guidance_L_obs_a', 'endpoint_observation_mse_u':'guidance_L_obs_u',
        'mean_clip_scale':'clip_scale'}
    for variant in variants:
        assert np.isfinite(errors[variant]).all()
        means = errors[variant].max(axis=-1).mean(axis=0)
        for step in range(100):
            values = means[:, step]
            low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
            row = dict(mean_error=float(values.mean()), ci95_low=float(low), ci95_high=float(high))
            batch_values = {key: np.array([float(curve[step][key]) for curve in curves[variant]])
                for key in set(direct.values()) | {'zeta_pde_t', 'grad_norm_pde', 'zeta_obs_a_t',
                    'grad_norm_obs_a', 'zeta_obs_u_t', 'grad_norm_obs_u'}}
            row.update({name: float(batch_values[source].mean()) for name, source in direct.items()})
            row['weighted_physical_gradient'] = float((batch_values['zeta_pde_t'] * batch_values['grad_norm_pde']).mean())
            row['weighted_observation_norm_sum'] = float((batch_values['zeta_obs_a_t'] * batch_values['grad_norm_obs_a'] +
                batch_values['zeta_obs_u_t'] * batch_values['grad_norm_obs_u']).mean())
            expected[variant, step] = row
            for position, sample in enumerate(ids):
                input_means[variant, sample, step] = float(values[position])
    output_path = args.visuals / 'guidance_trajectories.csv'
    actual = read_csv(output_path)
    old = read_csv(Path(manifest['old_source']))
    assert [x for x in actual if x['pde'] != 'nsnonbounded'] == [x for x in old if x['pde'] != 'nsnonbounded']
    assert len(actual) == 800
    actual_ns = [x for x in actual if x['pde'] == 'nsnonbounded']
    assert len(actual_ns) == len(expected) == 200
    diffs = {}
    seen = set()
    for row in actual_ns:
        key = row['variant'], int(row['step'])
        assert key not in seen
        seen.add(key)
        assert set(row) - {'pde', 'variant', 'step'} == set(expected[key])
        for column, value in expected[key].items():
            observed = float(row[column])
            diffs[column] = max(diffs.get(column, 0.), abs(observed-value))
            np.testing.assert_allclose(observed, value, rtol=1e-12, atol=1e-12)
    actual_means = read_csv(args.visuals / 'guidance_trajectory_input_means.csv')
    assert len(actual_means) == 6400
    seen = set(); maximum_mean_difference = 0.
    for row in actual_means:
        key = row['variant'], int(row['sample_id']), int(row['step'])
        assert key not in seen
        seen.add(key)
        difference = abs(float(row['mean_primary_error_over_seeds']) - input_means[key])
        maximum_mean_difference = max(maximum_mean_difference, difference)
        assert difference < 1e-12
    assert seen == set(input_means)
    report = dict(status='pass', complete=True, independently_recomputed_ns_rows=200,
        independently_recomputed_input_mean_rows=6400, unchanged_non_ns_rows=600,
        verified_source_batches=48, max_absolute_differences=diffs,
        max_input_mean_difference=maximum_mean_difference,
        statistical_unit='32 physical inputs after averaging three seeds; pointwise input-bootstrap intervals',
        gradient_unit='Means of 24 norms of four-input batches; sum of weighted observation norms',
        protocol_sha256=digest(args.task / 'protocol.json'),
        source_manifest_sha256=digest(manifest_path), output_csv_sha256=digest(output_path),
        auditor_sha256=digest(Path(__file__)), sources=sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), args.output
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()

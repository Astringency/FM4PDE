"""Independently verify saved mains2d batches, on the server or a local mirror."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import yaml


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, require_complete=False, check_inputs=False):
    manifest = read(root / 'experiment.json')
    original_root = Path(manifest['output_root'])
    distributions = ('id', 'smooth', 'rough')
    static_cells = [(p, t, 'random') for p in ('poisson', 'helmholtz', 'darcy', 'nsnonbounded')
                    for t in ('forward', 'inverse', 'both')]
    expected_cells = {(d, *c) for d in distributions for c in
                      static_cells + [('burger', 'both', s) for s in ('random', 'sensor_column')]}
    keys = ('test_type', 'pde', 'task', 'sensor_mode')
    plans = {tuple(r[k] for k in keys): r for r in read(root / 'execution/resolved_matrix.json')['rows']}
    assert set(plans) == expected_cells and len(expected_cells) == 42
    for relative, hashes in manifest['configs'].items():
        assert sha256(root / 'configs' / relative) == hashes['experiment_sha256'], relative

    samples = defaultdict(dict)
    runs = defaultdict(int)
    folders = set()
    commits = set()
    guarded_steps = 0
    for distribution in distributions:
        for marker_path in sorted((root / distribution).glob('.sample_sweeps/*/completed/**/*.json')):
            marker = read(marker_path)
            folder = root / Path(marker['run_dir']).relative_to(original_root)
            assert folder not in folders, folder
            folders.add(folder)
            config = yaml.safe_load((folder / 'resolved_config.yaml').read_text())
            cell = tuple(config[k] for k in keys)
            assert cell in expected_cells and cell[0] == distribution, folder
            plan = plans[cell]
            assert marker['signature'] == plan['signature'], folder
            assert marker['experiment'] == '/'.join([cell[1], cell[2], 'hybrid_s2d', cell[3]])
            differences = {k for k in config.keys() | plan['resolved_config'].keys()
                           if config.get(k) != plan['resolved_config'].get(k)}
            assert differences <= {'batch_size', 'offset', 'device'}, (folder, differences)
            assert config['batch_size'] == marker['batch_size'] == 10
            assert config['offset'] == marker['offset'] and config['device'] in ('cuda:0', 'cuda:1')
            assert config['sampler_phase'] == 'hybrid_s2d' and config['switch_ratio'] == .2
            assert config['num_steps'] == 100 and config['sample_seed'] == 0
            assert config['num_obs'] == 500 and config['allow_synthetic_data'] is False
            if cell[3] == 'sensor_column':
                assert config['num_sensor_columns'] == 16
            meta = read(folder / 'run_metadata.json')
            commits.add(meta['git_commit'])
            for key in ('checkpoint_path', 'data_path', 'batch_size', 'offset', 'sample_seed'):
                assert meta[key] == config[key], (folder, key)
            metrics = read(folder / 'metrics_final.json')
            assert metrics['status'] == 'ok' and metrics['synthetic_data'] is False, folder
            assert metrics['pde_eval_error_count'] == 0, folder
            assert metrics['num_samples'] == 10 and metrics['num_recorded_steps'] == 100
            steps = [json.loads(line) for line in (folder / 'metrics_step.jsonl').read_text().splitlines()]
            assert [step['phase'] for step in steps] == ['stochastic'] * 20 + ['deterministic'] * 80, folder
            for index, step in enumerate(steps):
                assert step['step'] == index and abs(step['t'] - index / 100) < 1e-6, folder
                assert abs(step['t_next'] - (index + 1) / 100) < 1e-6, folder
                assert step.get('pde_eval_error_count', 0) == 0, folder
                guarded_steps += int(step.get('nonfinite_correction_samples', 0) > 0)
            with (folder / 'metrics_per_sample.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 10
            assert {int(r['sample_id']) for r in rows} == set(range(config['offset'], config['offset'] + 10))
            for row in rows:
                index = int(row['sample_id'])
                assert 0 <= index < 1000 and index not in samples[cell], (cell, index)
                values = [float(row['rel_l2_a']), float(row['rel_l2_u'])]
                assert all(math.isfinite(v) and v >= 0 for v in values), (folder, index)
                samples[cell][index] = values
            for field in ('rel_l2_a', 'rel_l2_u'):
                assert math.isclose(metrics[field], statistics.mean(float(r[field]) for r in rows),
                                    rel_tol=1e-6, abs_tol=1e-9), (folder, field)
            assert (folder / 'result.pt').stat().st_size > 0
            runs[cell] += 1

    total = sum(len(s) for s in samples.values())
    complete = all(set(samples[cell]) == set(range(1000)) for cell in expected_cells)
    summaries = [dict(zip(keys, cell), samples=len(samples[cell]), runs=runs[cell],
                      a_relative_error_pct=100 * statistics.mean(v[0] for v in samples[cell].values()),
                      u_relative_error_pct=100 * statistics.mean(v[1] for v in samples[cell].values()))
                 for cell in sorted(expected_cells) if samples[cell]]
    if require_complete:
        assert complete and total == 42000, ('Incomplete experiment', total)
        assert (root / 'execution/driver.exit').read_text().strip() == '0'
        for distribution in distributions:
            for stage in ('main', 'burger'):
                assert read(root / f'execution/{distribution}_{stage}.json')['exit_code'] == 0
        completion = read(root / 'completion.json')
        assert completion['status'] == 'complete' and completion['verified_samples'] == 42000
        assert completion['verified_cells'] == 42 and completion['verified_runs'] == len(folders)
        with (root / 'summary.csv').open() as stream:
            summary_rows = list(csv.DictReader(stream))
        for reported in (completion['rows'], summary_rows):
            indexed = {tuple(row[k] for k in keys): row for row in reported}
            assert len(reported) == 42 and set(indexed) == expected_cells
            for row in summaries:
                other = indexed[tuple(row[k] for k in keys)]
                assert int(other['samples']) == 1000 and int(other['runs']) == row['runs']
                for field in ('a_relative_error_pct', 'u_relative_error_pct'):
                    assert math.isclose(float(other[field]), row[field], rel_tol=1e-10, abs_tol=1e-10)
        successful = {p.parent for d in distributions for p in (root / d).rglob('metrics_final.json')
                      if read(p)['status'] == 'ok'}
        assert successful == folders, 'Unaccounted successful run directories'

    if check_inputs:
        for name, identity in manifest['weights'].items():
            path = Path(name)
            assert path.stat().st_size == identity['size'] and sha256(path) == identity['sha256'], path
        for name, identity in manifest['data'].items():
            stat = Path(name).stat()
            assert stat.st_size == identity['size'] and stat.st_mtime_ns == identity['mtime_ns'], name

    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                  verified_samples=total, expected_samples=42000, verified_runs=len(folders),
                  complete_cells=sum(len(samples[c]) == 1000 for c in expected_cells),
                  expected_cells=42, whole_experiment_complete=complete,
                  completion_receipts_checked=require_complete, input_identities_checked=check_inputs,
                  code_commits=sorted(commits), steps_with_nonfinite_correction_guard=guarded_steps,
                  error_source='Saved per-sample relative L2 metrics; tensor errors are not recomputed.',
                  rows=summaries)
    target = root / 'execution/verified_results.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    verify(args.root.resolve(), args.require_complete, args.check_inputs)

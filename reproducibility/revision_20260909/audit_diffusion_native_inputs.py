#!/usr/bin/env python3
"""Recompute all 26,000 saved native Diffusion predictions from frozen truth.

Checks ratios (not percentages); sample SD uses ddof=1. Original MATs are not
read by this audit, and original predictions, masks and metrics are read only.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen-inputs', type=Path, required=True)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--result-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    import torch
    from freeze_diffusion_inputs import sha, array_info, check_saved_results
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    manifest_path = args.frozen_inputs / 'manifest.json'
    assert sha(manifest_path) == args.manifest_sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'pass' and manifest['complete'] and len(manifest['entries']) == 5
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.time()
    records, summaries = [], []
    for entry in manifest['entries']:
        path = args.frozen_inputs / entry['cache_path']
        assert sha(path) == entry['cache_sha256']
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        assert {key: array_info(value) for key, value in arrays.items()} == entry['arrays']
        for cell in entry['selected_cells']:
            print('AUDIT', entry['pde'], cell['study'], cell['task'], flush=True)
            selected = dict(entry, selected_cells=[cell])
            cell_records = check_saved_results(selected, arrays, args.result_root, indices=range(1000))
            assert len(cell_records) == 1000 and [x['index'] for x in cell_records] == list(range(1000))
            summary = dict(pde=entry['pde'], study=cell['study'], task=cell['task'], n=1000,
                           source_truth_cache_sha256=entry['cache_sha256'],
                           metric_json_examples=sum(x['metrics_path'] is not None for x in cell_records),
                           final_loss_examples=sum(x['metrics_path'] is None for x in cell_records))
            for field in ['a', 'u']:
                if field not in cell_records[0]['field_errors']:
                    continue
                for quantity in ['recomputed', 'recorded', 'observation_relative_l2', 'observation_mse']:
                    values = np.array([x['field_errors'][field][quantity] for x in cell_records], dtype=np.float64)
                    assert np.isfinite(values).all()
                    summary[f'{field}_{quantity}_mean'] = float(values.mean())
                    summary[f'{field}_{quantity}_sd'] = float(values.std(ddof=1))
                deltas = [x['field_errors'][field]['absolute_difference'] for x in cell_records]
                summary[field + '_max_original_error_absolute_difference'] = float(max(deltas))
                obs_deltas = [abs(x['field_errors'][field]['observed_recomputed'] - x['field_errors'][field]['observed_recorded'])
                              for x in cell_records if 'observed_recorded' in x['field_errors'][field]]
                summary[field + '_observation_reference_examples'] = len(obs_deltas)
                summary[field + '_max_observation_relative_l2_absolute_difference'] = float(max(obs_deltas)) if obs_deltas else None
            records.extend(cell_records)
            summaries.append(summary)
            print('PASS', entry['pde'], cell['study'], cell['task'], flush=True)
        del arrays
    assert len(summaries) == 26 and len(records) == 26000
    raw_json = (json.dumps(records, separators=(',', ':')) + '\n').encode()
    raw_path = args.output / 'per_example_checks.json.gz'
    raw_path.write_bytes(gzip.compress(raw_json, mtime=0))
    table_path = args.output / 'cell_summary.csv'
    columns = ['pde', 'study', 'task', 'n'] + sorted(set().union(*(x.keys() for x in summaries)) - {'pde', 'study', 'task', 'n'})
    with table_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    assert not torch.cuda.is_initialized()
    report = dict(status='pass', complete=True, cells=26, examples=26000, n_per_cell=1000,
                  independent_truth_source='frozen NPZ, exact native dtype and source slices; no original MAT reread in this audit',
                  relative_l2_tolerance=dict(rtol=2e-7, atol=1e-10), sd_ddof=1, units='ratios; multiply relative errors by100 for percent',
                  observation_mse_definition='mean squared physical-field error over saved binary observation mask',
                  observations='All masks come from original saved pickle records; observed-relative-L2 is compared where metric JSON exists.',
                  manifest_sha256=args.manifest_sha256,
                  input_caches={x['cache_path']:x['cache_sha256'] for x in manifest['entries']},
                  audit_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip(),
                  audit_script_sha256=sha(__file__), helper_sha256=sha(Path(__file__).with_name('freeze_diffusion_inputs.py')),
                  per_example_sha256=sha(raw_path), cell_summary_sha256=sha(table_path),
                  elapsed_seconds=time.time() - started, cuda_initialized=False, summaries=summaries)
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key:report[key] for key in ['status', 'cells', 'examples', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Recompute the completed NS main study from a verified, read-only archive."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'plot'), str(Path(__file__).parent)]
from archive_results import tree, sha, RECEIPT

ORIGINAL = Path('/home/tat512/C01Python/audit/ns_main_revision_0909')
PINNED = {
    'FINAL_HANDOFF.json': '3787c15af2ad3b97c722f74c51fc62c3bdb0f368a5d8d137074e9e49f447fb20',
    'ARCHIVE_INVENTORY.json': '1069710f114f694b61b307a5b334b97b471961b4bc29bdf20d3fb5167acb4639',
    'collection_complete.json': '097dfca793f6c3b3867f174838a9a768297332614a4f6a86099f55e215183ac1',
    'tables/ns_main_summary.csv': '9ddd893d7bbe7e88b7a7c42ad0f4a42775815931c8ce4a566bf1e363412a52eb',
    'tables/ns_main_per_sample.csv': 'e451a8171837feb846682e9521c80f7f81f1607c9f2298af8703084d0648154b',
    'tables/ns_main_validation.json': '124540d4415a4aceac55b955b8e655a86617a068b5d5e04d9de20621f984965b',
    'tables/ns_main_residual_audit.json': '236181acd0b97c5ee25ff65d8ee13a3e613d67ee555e5b2fcb216dba60e5197b',
    'inputs/protocol.json': '0490e326f30cc8425031b4d855058878d278fc149c360ffaf88275bdde48a446',
    'selection.json': '59cb9f345d76304da7632675b71414686e9329ad89005bc96dd3908612cdabb5',
    'weights.pth': '481a0db0f95db6a44197a16a8c0d45bd5075bfe34922bf85ab681275a5b697db',
}
METRICS = ['error_a', 'error_u', 'obs_a', 'obs_u', 'pde_mse']


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def differences(a, b, *, rtol, atol):
    import numpy as np
    old, new = np.asarray(a, dtype='float64'), np.asarray(b, dtype='float64')
    assert old.shape == new.shape and np.isfinite(old).all() and np.isfinite(new).all()
    delta = np.abs(new - old)
    return dict(n=int(old.size), exact=bool(np.array_equal(old, new)),
        max_absolute=float(delta.max(initial=0)),
        max_relative=float((delta / np.maximum(np.abs(old), 1e-30)).max(initial=0)),
        rtol=rtol, atol=atol, pass_tolerance=bool(np.allclose(old, new, rtol=rtol, atol=atol)))


def compare_csv(old_path, new_path, keys, metrics, study, *, pde_rtol=1e-12, pde_atol=1e-14):
    old, new = csv_rows(old_path), csv_rows(new_path)
    assert len(old) == len(new)
    oi = {tuple(r[k] for k in keys): r for r in old}
    ni = {tuple(r[k] for k in keys): r for r in new}
    assert len(oi) == len(ni) == len(old) and oi.keys() == ni.keys()
    result = {}
    for metric in metrics:
        is_pde = metric.startswith('pde_mse')
        result[metric] = differences([oi[k][metric] for k in oi], [ni[k][metric] for k in oi],
            rtol=pde_rtol if is_pde else 1e-12, atol=pde_atol if is_pde else 1e-14)
        assert result[metric]['pass_tolerance'], (metric, result[metric])
    path_changes = 0
    for key in oi:
        assert set(oi[key]) == set(ni[key])
        for field in oi[key]:
            if field in metrics:
                continue
            if field == 'result':
                old_relative = Path(oi[key][field]).relative_to(ORIGINAL / 'main_results')
                new_relative = Path(ni[key][field]).relative_to(study / 'main_results')
                assert old_relative == new_relative
                path_changes += oi[key][field] != ni[key][field]
            else:
                assert oi[key][field] == ni[key][field], (key, field)
    return dict(rows=len(old), columns=list(old[0]), numeric_differences=result,
        permitted_result_path_changes=path_changes, all_other_fields_equal=True)


def run_child(script, arguments, output):
    command = [sys.executable, str(ROOT / 'plot' / script), *map(str, arguments)]
    start = time.monotonic()
    with (output / (Path(script).stem + '.log')).open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    return dict(command=command, seconds=time.monotonic() - start)


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'CUDA must be explicitly hidden'
    for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        assert os.environ.get(variable) == '2', (variable, 'must be 2')
    import numpy as np
    import torch
    from audit_ns_main_residual_0909 import endpoint_mse
    from run_ns_main_revision_0909 import EVAL, SETTINGS
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert not torch.cuda.is_available()
    study, output = args.study.resolve(), args.output.resolve()
    assert study.is_dir() and output != study and study not in output.parents
    assert 'verification_runs' in output.parts and not output.exists()
    output.mkdir(parents=True)
    started = time.monotonic()
    receipt_path = study / RECEIPT
    assert sha(receipt_path) == args.receipt_sha256
    receipt = read(receipt_path)
    assert receipt['source']['id'] == 'ns_main_complete_local_ns_study'
    assert receipt['source']['source_path'] == str(ORIGINAL)
    assert receipt['source_preserved']
    print('VERIFY_FULL_ARCHIVE_HASHES', flush=True)
    before = tree(study, ignore_receipt=True)
    assert before == receipt['files'], 'Full archive tree differs from its receipt'
    assert all(row['kind'] in ('file', 'directory') for row in before.values())
    for relative, expected in PINNED.items():
        assert before[relative]['sha256'] == expected, relative
    assert before['inputs/weights.pth']['sha256'] == PINNED['weights.pth']
    protocol = read(study / 'inputs/protocol.json')
    for dist in ['id', 'smooth', 'rough']:
        assert before[f'inputs/{dist}.npz']['sha256'] == protocol['sources'][dist]['cache_sha256']
    assert read(study / 'collection_complete.json')['status'] == 'complete'
    assert read(study / 'FINAL_HANDOFF.json')['main_samples'] == 15000
    assert sum(p.startswith('main_results/') and p.endswith('.pt') for p in before) == 450
    gate = dict(status='pass', archive=str(study), receipt_sha256=args.receipt_sha256,
        source_preserved=True, all_archive_hashes_match=True, archive_entries=len(before),
        archive_files=sum(r['kind'] == 'file' for r in before.values()),
        archive_bytes=sum(r.get('bytes', 0) for r in before.values()), pinned=PINNED,
        required_raw_batches=450, required_examples=15000)
    write(output / 'dependency_gate.json', gate)
    environment = dict(python=sys.executable, python_version=sys.version, numpy=np.__version__,
        torch=torch.__version__, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        cuda_available=torch.cuda.is_available(), torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        thread_environment={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'plot/export_ns_main_revision_0909.py', ROOT/'plot/audit_ns_main_residual_0909.py',
            ROOT/'plot/run_ns_main_revision_0909.py', ROOT/'sampling/config.py']},
        packages=sorted(({'name': d.metadata['Name'], 'version': d.version}
            for d in importlib.metadata.distributions()), key=lambda r:r['name'].lower()))
    write(output / 'environment.json', environment)
    print('RECOMPUTE_ALL_MAIN_ERRORS_AND_OBSERVATION_MSE', flush=True)
    stages = [run_child('export_ns_main_revision_0909.py',
        ['main','--results',study/'main_results','--output',output/'tables'], output)]
    print('RECOMPUTE_ALL_NUMPY_RESIDUALS_AND_RUNTIME_AUDIT', flush=True)
    stages.append(run_child('audit_ns_main_residual_0909.py',
        ['--audit',study,'--results',study/'main_results','--output',output/'tables/ns_main_residual_audit.json'], output))
    summary_metrics = [metric + suffix for metric in METRICS for suffix in ('_mean', '_sd')]
    comparisons = dict(
        frozen_reductions_summary=compare_csv(study/'tables/ns_main_summary.csv', output/'tables/ns_main_summary.csv',
            ['dist','setting'], summary_metrics, study),
        frozen_reductions_per_sample=compare_csv(study/'tables/ns_main_per_sample.csv', output/'tables/ns_main_per_sample.csv',
            ['dist','setting','offset'], METRICS, study))
    old_audit = read(study/'tables/ns_main_residual_audit.json')
    new_audit = read(output/'tables/ns_main_residual_audit.json')
    assert new_audit['complete'] and new_audit['status'] == 'pass'
    for key in old_audit:
        if key != 'max_pde_mse_difference':
            assert old_audit[key] == new_audit[key], key
    comparisons['independent_residual_maxima'] = {key:differences([old_audit['max_pde_mse_difference'][key]],
        [new_audit['max_pde_mse_difference'][key]], rtol=1e-10, atol=1e-14)
        for key in old_audit['max_pde_mse_difference']}
    assert all(r['pass_tolerance'] for r in comparisons['independent_residual_maxima'].values())
    # The published reduction preserves the producer's float32 PDE MSEs. In
    # addition, export a complete alternative reduction using independent
    # NumPy float64 PDE MSEs, making the expected arithmetic difference explicit.
    print('EXPORT_ALL_15000_INDEPENDENT_FLOAT64_RESIDUALS', flush=True)
    residual_rows = {}
    counts = defaultdict(int)
    for path in sorted((study/'main_results').rglob('offset*.pt')):
        r = read(path.with_suffix('.json'))
        values = endpoint_mse(torch.load(path, map_location='cpu', weights_only=False)['predictions'].numpy())
        assert len(values) == len(r['ids'])
        for offset, value in zip(r['ids'], values):
            key = r['dist'], r['setting'], str(offset)
            assert key not in residual_rows
            residual_rows[key] = float(value)
        counts[r['dist'], r['setting']] += 1
    assert len(residual_rows) == 15000 and set(counts.values()) == {30}
    independent = csv_rows(output/'tables/ns_main_per_sample.csv')
    for row in independent:
        row['pde_mse'] = residual_rows[row['dist'], row['setting'], row['offset']]
    write_csv(output/'tables/ns_main_all_numpy_per_sample.csv', independent)
    alternative = []
    for dist in ['id', 'smooth', 'rough']:
        for setting in SETTINGS:
            group = [r for r in independent if r['dist'] == dist and r['setting'] == setting]
            assert sorted(int(r['offset']) for r in group) == EVAL
            row = dict(dist=dist, setting=setting, n=len(group))
            for metric in METRICS:
                values = np.array([float(r[metric]) for r in group])
                row[metric+'_mean'], row[metric+'_sd'] = float(values.mean()), float(values.std(ddof=1))
            alternative.append(row)
    write_csv(output/'tables/ns_main_all_numpy_summary.csv', alternative)
    # A per-example maximum perturbation d changes the mean by at most d,
    # and the sample SD by at most sqrt(n/(n-1))*d. A relative tolerance on
    # a nearly zero SD would not express the original residual audit bound.
    pde_summary_bound = new_audit['max_pde_mse_difference']['absolute'] * np.sqrt(1000/999) + 1e-14
    comparisons['all_numpy_summary'] = compare_csv(study/'tables/ns_main_summary.csv',
        output/'tables/ns_main_all_numpy_summary.csv', ['dist','setting'], summary_metrics, study,
        pde_rtol=0, pde_atol=pde_summary_bound)
    comparisons['all_numpy_summary']['pde_statistic_absolute_bound'] = float(pde_summary_bound)
    comparisons['all_numpy_summary']['bound_derivation'] = 'sqrt(1000/999) times maximum independently checked per-example PDE-MSE difference, plus1e-14 for reduction roundoff; bounds both the mean and sample-SD perturbation.'
    comparisons['all_numpy_per_sample'] = compare_csv(study/'tables/ns_main_per_sample.csv',
        output/'tables/ns_main_all_numpy_per_sample.csv', ['dist','setting','offset'], METRICS, study,
        pde_rtol=2e-5, pde_atol=1e-9)
    print('VERIFY_ARCHIVE_UNCHANGED_AFTER_RECOMPUTATION', flush=True)
    assert tree(study, ignore_receipt=True) == before
    assert sha(receipt_path) == args.receipt_sha256
    result = dict(status='pass', completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        seconds=time.monotonic()-started, archive=str(study), output=str(output),
        archive_receipt_sha256=args.receipt_sha256, complete=True, examples=15000, cells=15,
        raw_batches=450, cpu_only=True, source_full_hashes_unchanged=True,
        all_protocol_config_receipt_bytes_preserved=True, stages=stages, comparisons=comparisons,
        difference_interpretation='Only result-source paths are relocated. Frozen-style PDE summaries retain the stored CUDA float32 PDE MSE; the all-NumPy exports instead reduce independently recomputed float64 FFT residuals. Any differences are explicitly reported with the original audit tolerances.',
        output_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write(output/'verification_complete.json', result)
    print(json.dumps(dict(status='pass', examples=15000, cells=15, batches=450,
        result=str(output/'verification_complete.json'), seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    main(parser.parse_args())

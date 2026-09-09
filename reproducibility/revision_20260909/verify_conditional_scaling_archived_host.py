#!/usr/bin/env python3
"""Re-export one fully archived K host and compare its numerical artifacts.

This CPU-only replay reads the archived predictions and frozen inputs. It does
not sample, overwrite the archive, or mark the entire result archive complete.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def inventory(root, receipt):
    actual = {}
    for relative, expected in receipt['files'].items():
        p = root / relative
        assert p.resolve().is_relative_to(root), relative
        assert not p.is_symlink(), relative
        if expected['kind'] == 'directory':
            assert p.is_dir(), relative
            continue
        assert expected['kind'] == 'file' and p.is_file(), relative
        found = {'bytes': p.stat().st_size, 'sha256': sha(p)}
        assert found == {k: expected[k] for k in found}, relative
        actual[relative] = found
    all_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert all_files == set(actual) | {'.revision_archive_receipt.json'}
    assert sum(v['bytes'] for v in actual.values()) == receipt['content_bytes']
    return actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True, help='Verified host/results directory')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--receipt-sha256', required=True)
    p.add_argument('--expected-jobs', type=int, choices=[24, 72], required=True)
    p.add_argument('--output', type=Path, required=True, help='New evidence directory; compact export goes in export/')
    args = p.parse_args()
    for field in ('repo', 'archive', 'inputs', 'selection', 'output'):
        setattr(args, field, getattr(args, field).resolve())
    assert not args.output.exists(), 'Use a new output directory'
    assert not args.output.is_relative_to(args.archive)
    assert not args.output.is_relative_to(args.inputs)
    args.output.mkdir(parents=True)
    start = time.monotonic()
    resources = {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'load_average': list(os.getloadavg()),
        'meminfo': Path('/proc/meminfo').read_text(),
        'python': sys.version, 'python_executable': sys.executable,
        'numpy': np.__version__, 'platform': platform.platform(),
        'gpu_inventory': run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv']) if shutil.which('nvidia-smi') else None,
    }
    mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in resources['meminfo'].splitlines()}
    assert mem['MemAvailable'] >= 3 * (1 << 30)
    write(args.output / 'resource_preflight.json', resources)
    head = run(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'])
    assert not run(['git', '-C', str(args.repo), 'diff', 'HEAD', '--name-only']), 'Canonical tracked files must be clean'
    entry = args.repo / 'plot/export_conditional_sample_scaling.py'
    source = {'git_head': head, 'exporter_sha256': sha(entry),
              'verifier_sha256': sha(__file__),
              'verifier_git_head': run(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'])}
    receipt_path = args.archive / '.revision_archive_receipt.json'
    assert sha(receipt_path) == args.receipt_sha256
    receipt = json.loads(receipt_path.read_text())
    shutil.copyfile(receipt_path, args.output / 'source_archive_receipt.json')
    before = inventory(args.archive, receipt)
    write(args.output / 'source_hashes_before.json', before)
    envs = [json.loads(q.read_text()) for q in sorted((args.archive / 'production').glob('environment_run_*.json'))]
    assert envs
    inputs = {name: {'path': str(path), 'sha256': sha(path)} for name, path in {
        'truth': args.inputs / 'truths.pt', 'weights': args.inputs / 'weights.pth',
        'protocol': args.inputs / 'protocol.json', 'selection': args.selection}.items()}
    for e in envs:
        for name, data in inputs.items():
            assert data['sha256'] == e[name + '_sha256'], name
    write(args.output / 'input_bindings.json', inputs)
    command = [sys.executable, str(entry), '--results', str(args.archive / 'production'),
               '--inputs', str(args.inputs), '--selection', str(args.selection),
               '--output', str(args.output / 'export')]
    environ = os.environ.copy()
    environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    write(args.output / 'invocation.json', {'command': command, 'shell_command': shlex.join(command),
          'cwd': str(args.repo), 'environment': {k: environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}, **source})
    with (args.output / 'export.log').open('w') as log:
        subprocess.run(command, cwd=args.repo, env=environ, stdout=log, stderr=subprocess.STDOUT, check=True)
    old, new = args.archive / 'export', args.output / 'export'
    csv_name = 'conditional_scaling_per_input.csv'
    old_rows = list(csv.DictReader((old / csv_name).open()))
    new_rows = list(csv.DictReader((new / csv_name).open()))
    assert len(old_rows) == len(new_rows) == args.expected_jobs * 5
    comparisons = 0
    for a, b in zip(old_rows, new_rows):
        assert a.keys() == b.keys()
        for k in a:
            if k == 'task':
                assert a[k] == b[k]
            else:
                assert math.isfinite(float(a[k])) and float(a[k]) == float(b[k]), (a['task'], a['offset'], a['K'], k, a[k], b[k])
                comparisons += 1
    array_evidence = {}
    with np.load(old / 'conditional_scaling_fields.npz') as a, np.load(new / 'conditional_scaling_fields.npz') as b:
        assert set(a.files) == set(b.files) and len(a.files) == args.expected_jobs * 7
        for k in sorted(a.files):
            x, y = a[k], b[k]
            assert x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y), k
            array_evidence[k] = {'shape': list(x.shape), 'dtype': str(x.dtype), 'array_sha256': hashlib.sha256(x.tobytes(order='C')).hexdigest(), 'exact_equal': True}
    write(args.output / 'array_comparison.json', array_evidence)
    om = json.loads((old / 'conditional_scaling_manifest.json').read_text())
    nm = json.loads((new / 'conditional_scaling_manifest.json').read_text())
    assert om['complete'] and nm['complete']
    assert len(nm['completed_jobs']) == args.expected_jobs
    assert nm['conditional_trajectories'] == args.expected_jobs * 1000
    stripped = [copy.deepcopy(x) for x in (om, nm)]
    metadata_differences = []
    for key in ('export_resources', 'exporter_sha256', 'resolved_inputs', 'resolved_selection'):
        values = [x.pop(key, None) for x in stripped]
        if values[0] != values[1]:
            metadata_differences.append({'key': key, 'original': values[0], 'replayed': values[1]})
    for a, b in zip(stripped[0]['results'], stripped[1]['results']):
        pa, pb = a.pop('path'), b.pop('path')
        if pa != pb:
            metadata_differences.append({'key': f"results/{a['task']}/{a['offset']}/{a['K']}/path", 'original': pa, 'replayed': pb})
    assert stripped[0] == stripped[1], 'Scientific manifest content differs'
    write(args.output / 'metadata_differences.json', metadata_differences)
    after = inventory(args.archive, receipt)
    assert before == after and sha(receipt_path) == args.receipt_sha256
    write(args.output / 'source_hashes_after.json', after)
    for data in inputs.values():
        assert sha(data['path']) == data['sha256']
    assert run(['git', '-C', str(args.repo), 'rev-parse', 'HEAD']) == head
    assert not run(['git', '-C', str(args.repo), 'diff', 'HEAD', '--name-only'])
    assert sha(entry) == source['exporter_sha256']
    report = {'status': 'pass', 'scope': 'CPU re-export of saved physical draws; no inference or training',
        'canonical_jobs_replayed': args.expected_jobs, 'canonical_trajectories_replayed': args.expected_jobs * 1000,
        'complete_study_canonical_trajectories': 96000, 'csv_rows': len(new_rows),
        'numeric_csv_comparisons': comparisons, 'all_csv_numbers_exact': True,
        'csv_bytes_identical': (old / csv_name).read_bytes() == (new / csv_name).read_bytes(),
        'arrays': len(array_evidence), 'all_arrays_exact': True, 'scientific_manifest_identical': True,
        'metadata_differences': len(metadata_differences),
        'source_archive': str(args.archive), 'source_receipt_sha256': args.receipt_sha256,
        'source_files': len(before), 'source_bytes': receipt['content_bytes'], 'source_hashes_unchanged': True,
        'inputs_unchanged': True, 'source_code_unchanged': True,
        'elapsed_seconds': time.monotonic() - start, **source,
        'outputs': {q.name: {'bytes': q.stat().st_size, 'sha256': sha(q)} for q in sorted(new.iterdir()) if q.is_file()}}
    write(args.output / 'verification_complete.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

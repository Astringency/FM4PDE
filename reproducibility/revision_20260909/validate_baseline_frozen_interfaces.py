#!/usr/bin/env python3
"""Run a bounded CPU check of frozen-input interfaces, without a full-cell replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replay_baseline_frozen import sha256, resolve_args


def actual_path(original):
    prefix = '/home/zhangxf/share/zhangxfA100/large_storage/'
    assert original.startswith(prefix) or original.startswith('/large_storage/zhangxf/'), original
    return Path(original.replace(prefix, '/large_storage/zhangxf/', 1))


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1')
    assert all(os.environ.get(k) == '2' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    manifest = json.loads(args.cache_manifest.read_text())
    assert sha256(args.cache_manifest) == args.cache_manifest_sha256
    assert manifest['status'] == 'pass' and manifest['complete'] and len(manifest['entries']) == 18
    plan = json.loads(args.plan.read_text())
    configs = {c['raw']:c for c in plan['effective_configs']}
    entries = {e['id']:e for e in manifest['entries']}
    prepared = {}
    for revision in sorted({r['summary']['commit_hash'] for r in plan['all_mask_contracts']}):
        path = args.output/'baseline_sources'/revision
        path.parent.mkdir(exist_ok=True)
        subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', str(args.baseline_code), str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--quiet', '--detach', revision], check=True)
        prepared[revision] = path
    receipt = dict(status='running', cache_manifest_sha256=args.cache_manifest_sha256,
                   plan_sha256=sha256(args.plan), checks=[], cpu_only=True, threads=2,
                   code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip())
    started = time.monotonic()
    def run(record, kind, index):
        summary = record['summary']
        config = configs[record['relative_raw']]
        effective = resolve_args(config['config'], summary)
        dist = record['relative_raw'].split('/')[1]
        branch = 'trajectory' if effective.load_full_trajectory and summary['pde'] in {'nsnonbounded', 'burger'} else 'endpoint'
        entry = entries[f"{summary['pde']}_{dist}_{branch}"]
        source = Path(record['source_path'])
        config_path = Path(config['path'])
        command = [sys.executable, str(HERE/'replay_baseline_frozen.py'),
                   '--baseline-code', str(prepared[summary['commit_hash']]),
                   '--cache', str(args.cache_manifest.parent/entry['cache_path']),
                   '--cache-manifest', str(args.cache_manifest),
                   '--summary', str(source), '--summary-sha256', record['sha256'],
                   '--config', str(config_path), '--config-sha256', config['sha256'],
                   '--indices', 'none' if kind=='mask_contract' else '0,17,999',
                   '--output', str(args.output/f'{kind}_{index:03d}'), '--device', 'cpu']
        if kind != 'mask_contract':
            command += ['--reference-manifest', str(actual_path(summary['sample_manifest_path'])),
                        '--reference-root', str(actual_path(summary['sample_artifact_dir']))]
            if record['predict']:
                assert summary['baseline'] != 'vivid'
                command += ['--predict', '--checkpoint', str(actual_path(summary['checkpoint_path']))]
        log = args.output/f'{kind}_{index:03d}.log'
        start = time.monotonic()
        with log.open('w') as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        row = dict(kind=kind, run=record['relative_raw'], command=command,
                   returncode=proc.returncode, seconds=time.monotonic()-start, log_sha256=sha256(log))
        receipt['checks'].append(row)
        (args.output/'progress.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(kind, index, summary['pde'], summary['baseline'], proc.returncode, flush=True)
        assert proc.returncode == 0, f'Inspect {log}; no failed case is skipped or substituted'
    # Data-only contracts cover every original layout. Actual inference is
    # confined to four representative original batch groups in the plan.
    for index, record in enumerate(plan['all_mask_contracts']):
        run(record, 'mask_contract', index)
    for index, record in enumerate(plan['representative_cases']):
        run(record, 'representative', index)
    assert len(receipt['checks']) == 196
    receipt.update(status='pass', complete=True, elapsed_seconds=time.monotonic()-started,
                   data_contract_cells=186, representative_field_cells=10,
                   representative_inference_cells=4, full_inference_cells=0)
    (args.output/'validation_complete.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('BASELINE_FROZEN_INTERFACE_PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--baseline-code', type=Path, required=True)
    p.add_argument('--cache-manifest', type=Path, required=True)
    p.add_argument('--cache-manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())

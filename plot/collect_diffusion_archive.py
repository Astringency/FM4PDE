"""Read-only SSH snapshot of completed DiffusionPDE evaluation JSON records.

No remote files are written. Run before plot_jmlr_revision.py when refreshing
the paper. An existing snapshot is never overwritten.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import subprocess


REMOTE_READER = r'''
from pathlib import Path
import datetime, hashlib, json, re, subprocess
root = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE')
out = {'host': 'server216', 'root': str(root),
       'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'distribution': 'Smooth', 'distribution_basis': 'author instruction; legacy test files, not an ID/Smooth/Rough sweep',
       'records': [], 'inventory': [], 'configs': [], 'code': [], 'read_errors': []}
for steps in (100, 1000, 2000):
    base = root / 'outputs' / ('MAIN1000_' + str(steps))
    for pde in ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger'):
        for task in (('both',) if pde == 'burger' else ('forward', 'inverse', 'both')):
            folder = base / 'metrics' / pde / task
            files = sorted(folder.glob('*_metrics_final.json'))
            out['inventory'].append({'steps': steps, 'pde': pde, 'task': task, 'metric_files': len(files),
                'sample_files': len(list((base / 'samples' / pde / task).glob('*__results.pkl')))})
            for path in files:
                try:
                    raw = path.read_bytes()
                    row = json.loads(raw)
                    match = re.search(r'_step\((\d+)\)', row.get('result_path', ''))
                    assert match and int(match[1]) == steps, str(path)
                    assert row['pde'] == pde and row['problem'] == task, str(path)
                    out['records'].append(dict(row, steps=steps, source_path=str(path),
                        source_sha256=hashlib.sha256(raw).hexdigest()))
                except Exception as exc:
                    out['read_errors'].append({'path': str(path), 'error': repr(exc)})
    for path in sorted((base / '.sample_sweep' / 'configs').glob('*.yaml')):
        raw = path.read_bytes()
        out['configs'].append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'text': raw.decode()})
for name in ('evaluate_results.py', 'generate_pde.py', 'scripts/sample/run_sample_sweep.sh'):
    path = root / name
    raw = path.read_bytes()
    out['code'].append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'text': raw.decode()})
out['processes'] = subprocess.check_output(['ps', '-u', 'zjinzxf2025', '-o', 'pid,etime,args'], text=True)
out['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(out))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite snapshot: {args.output}')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'server216', 'python3', '-'], input=REMOTE_READER,
                            text=True, capture_output=True, check=True, timeout=180)
    snapshot = json.loads(result.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'xt', encoding='utf-8') as stream:
        json.dump(snapshot, stream, ensure_ascii=False)
    print(json.dumps({'output': str(args.output), 'records': len(snapshot['records']),
                      'inventory': snapshot['inventory'], 'read_errors': snapshot['read_errors']}, indent=2))


if __name__ == '__main__':
    main()

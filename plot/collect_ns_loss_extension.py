"""Collect immutable eight-worker NS results, retaining four-worker lineage."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ns_acceleration import digest, stem
from ns_acceleration_extension import SHARDS, validate_plan
from run_ns_loss_study import write

HOSTS = [dict(name='server193', ssh=['ssh','-p','9088'], rsync_ssh='ssh -p 9088',
              login='zhangxf@192.168.191.193', shards=[0,1],
              root='/home/zhangxf/C01Python/FM4PDE_ns_loss_spectrum_20260907/ns_results_accelerated_v2'),
         dict(name='server197', ssh=['ssh'], rsync_ssh='ssh',
              login='zhangxifeng@192.168.191.197', shards=[2,3],
              root='/research_data/users/zhangxifeng/C01Python/ns_loss_spectrum_20260907/ns_results_accelerated_v2'),
         dict(name='server216', ssh=['ssh'], rsync_ssh='ssh',
              login='zjinzxf2025@175.102.135.216', shards=[4,5,6,7],
              root='/data1/zjinzxf2025/C01Python/ns_loss_extension_20260907/ns_results_accelerated_v2')]


def preserve_copy(src, dst):
    if dst.exists():
        assert digest(src) == digest(dst), f'Conflicting committed artifact: {dst}'
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp = dst.with_name(dst.name+f'.collect-{os.getpid()}.tmp')
        shutil.copy2(src, temp)
        temp.replace(dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    study = args.study
    canonical, work = study/'ns_results_v3', study/'acceleration_v2'
    work.mkdir(exist_ok=True)
    lock = (work/'collector.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = json.loads(args.plan.read_text())
    parent_path = canonical/'acceleration_plan.json'
    parent = json.loads(parent_path.read_text())
    assert digest(parent_path) == plan['parent_plan_sha256']
    owner = validate_plan(plan, json.loads((study/'inputs_v2/source.json').read_text()), parent)
    ph = plan['protocol_sha256']
    for key, initial in plan['initial_completed'].items():
        path = canonical/'results'/key
        assert digest(path.with_suffix('.json')) == initial['receipt_sha256']
        assert digest(path.with_suffix('.pt')) == initial['prediction_sha256']
    preserve_copy(args.plan, canonical/'acceleration_extension_plan.json')
    while True:
        try:
            origins = {}
            for host in HOSTS:
                script = f'''from pathlib import Path
import json
p=Path({host['root']!r});shards={host['shards']!r};names=[];rows=[]
for i in shards:
 w=p/f'worker_{{i}}'
 for name in ['protocol.json','acceleration_plan.json','acceleration_extension_plan.json','environment_0.json','pilot_0.json','complete_0.json',f'extension_worker_{{i}}.json']:
  if (w/name).exists():names.append(str((w/name).relative_to(p)))
 for path in sorted((w/'results').rglob('*.json')):
  row=json.loads(path.read_text());rows.append(dict(worker=i,receipt=row));names.append(str(path.relative_to(p)))
  if row.get('prediction_sha256'):names.append(str(path.with_suffix('.pt').relative_to(p)))
print(json.dumps(dict(files=names,receipts=rows)))
'''
                report = json.loads(subprocess.check_output(host['ssh']+['-o','ConnectTimeout=20',host['login'],'python3','-'],
                                    input=script, text=True, timeout=60))
                for entry in report['receipts']:
                    row, worker = entry['receipt'], entry['worker']
                    key = stem(tuple(row[k] for k in ['task','method','steps','exchange','sample_id','seed']))
                    assert key in owner and row['protocol_sha256'] == ph
                    assert key not in plan['initial_completed'] and owner[key] == worker and worker in host['shards'], (host['name'],key)
                    origins.setdefault(key, []).append(host['name'])
                staging = work/'collected'/host['name']
                staging.mkdir(parents=True, exist_ok=True)
                listing = work/f'files_{host["name"]}_{os.getpid()}.txt'
                listing.write_text('\n'.join(report['files'])+'\n')
                subprocess.run(['rsync','-a','--timeout=60','--files-from='+str(listing),'-e',host['rsync_ssh'],
                                host['login']+':'+host['root']+'/',str(staging)+'/'], check=True)
                for worker in host['shards']:
                    folder = staging/f'worker_{worker}'
                    assert digest(folder/'protocol.json') == ph
                    assert digest(folder/'acceleration_plan.json') == plan['parent_plan_sha256']
                    assert digest(folder/'acceleration_extension_plan.json') == digest(args.plan)
                for entry in report['receipts']:
                    row, worker = entry['receipt'], entry['worker']
                    key = stem(tuple(row[k] for k in ['task','method','steps','exchange','sample_id','seed']))
                    path = Path('results')/key
                    folder = staging/f'worker_{worker}'
                    if row.get('prediction_sha256'):
                        tensor = (folder/path).with_suffix('.pt')
                        assert digest(tensor) == row['prediction_sha256']
                        preserve_copy(tensor, (canonical/path).with_suffix('.pt'))
                    preserve_copy((folder/path).with_suffix('.json'), (canonical/path).with_suffix('.json'))
                for name in report['files']:
                    parts = Path(name).parts
                    if len(parts) != 2:
                        continue
                    worker = int(parts[0].removeprefix('worker_'))
                    leaf = parts[1]
                    target = next((f'extension_{prefix}_{worker}.json' for prefix in ['environment','pilot','complete']
                                   if leaf == f'{prefix}_0.json'), leaf)
                    preserve_copy(staging/name, canonical/target)
            rows = [json.loads(p.read_text()) for p in (canonical/'results').rglob('*.json')]
            keys = [stem(tuple(r[k] for k in ['task','method','steps','exchange','sample_id','seed'])) for r in rows]
            assert len(keys) == len(set(keys)) and set(keys) <= set(owner)
            complete = len(keys) == 1728 and all((canonical/f'extension_complete_{i}.json').exists() for i in range(SHARDS))
            summary = dict(checked_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), calls=len(rows), expected=1728,
                           outcomes=dict(Counter(r['status'] for r in rows)), sampling_complete=complete,
                           worker_pool='2 RTX4090 + 2 A100 + 4 A800', plan_sha256=digest(args.plan))
            write(study/'sampling_progress.json', summary)
            write(work/'collected_origins.json', origins)
            print(json.dumps(summary), flush=True)
            if complete:
                subprocess.run([sys.executable,str(Path(__file__).with_name('audit_ns_loss_results.py')),
                    '--inputs',str(study/'inputs_v2'),'--results',str(canonical),'--baselines',str(study/'baseline_results_gpu_v2'),
                    '--output',str(study/'ns_complete_audit'),'--require-complete'], check=True)
                print('COMPLETE: all eight-worker continuation calls collected and audited.', flush=True)
                return
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            print('Collection retry; samplers unchanged:',repr(exc),flush=True)
            if not args.watch:
                raise
        if not args.watch:
            return
        time.sleep(120)


if __name__ == '__main__':
    main()

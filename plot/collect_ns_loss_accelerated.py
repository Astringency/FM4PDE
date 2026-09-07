"""Collect immutable NS continuation results from the two authorized hosts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import Counter

from ns_acceleration import digest, stem, validate_plan
from run_ns_loss_study import write

HOSTS=[dict(name='server193',ssh=['ssh','-p','9088'],rsync_ssh='ssh -p 9088',
            login='zhangxf@192.168.191.193',shards=[0,1],
            root='/home/zhangxf/C01Python/FM4PDE_ns_loss_spectrum_20260907/ns_results_accelerated_v1'),
       dict(name='server197',ssh=['ssh'],rsync_ssh='ssh',
            login='zhangxifeng@192.168.191.197',shards=[2,3],
            root='/research_data/users/zhangxifeng/C01Python/ns_loss_spectrum_20260907/ns_results_accelerated_v1')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();s=args.study
    plan=json.loads(args.plan.read_text());ph=plan['protocol_sha256']
    source=json.loads((s/'inputs_v2/source.json').read_text());owner=validate_plan(plan,source)
    canonical=s/'ns_results_v3';work=s/'acceleration_v1';work.mkdir(exist_ok=True)
    lock=(work/'collector.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def preserve_copy(src,dst):
        if dst.exists():assert digest(src)==digest(dst),f'Conflicting committed artifact: {dst}'
        else:
            dst.parent.mkdir(parents=True,exist_ok=True)
            temp=dst.with_name(dst.name+f'.collect-{os.getpid()}.tmp')
            shutil.copy2(src,temp);temp.replace(dst)

    preserve_copy(args.plan,canonical/'acceleration_plan.json')
    while True:
        try:
            origins={}
            for host in HOSTS:
                script=f'''from pathlib import Path
import json
p=Path({host['root']!r});shards={host['shards']!r};names=[];rows=[]
for name in ['protocol.json','acceleration_plan.json']:
 if (p/name).exists():names.append(name)
for i in shards:
 for prefix in ['environment','pilot','complete','acceleration_worker']:
  name=f'{{prefix}}_{{i}}.json'
  if (p/name).exists():names.append(name)
for path in sorted((p/'results').rglob('*.json')):
 row=json.loads(path.read_text());rows.append(row);names.append(str(path.relative_to(p)))
 if row.get('prediction_sha256'):names.append(str(path.with_suffix('.pt').relative_to(p)))
print(json.dumps(dict(files=names,receipts=rows)))
'''
                report=json.loads(subprocess.check_output(host['ssh']+['-o','ConnectTimeout=20',host['login'],'python3','-'],input=script,text=True,timeout=60))
                for row in report['receipts']:
                    job=tuple(row[k] for k in ['task','method','steps','exchange','sample_id','seed'])
                    key=stem(job);assert key in owner and row['protocol_sha256']==ph
                    assert key in plan['initial_completed'] or owner[key] in host['shards'],(host['name'],key)
                    origins.setdefault(key,[]).append(host['name'])
                staging=work/'collected'/host['name'];staging.mkdir(parents=True,exist_ok=True)
                listing=work/f'files_{host["name"]}_{os.getpid()}.txt'
                listing.write_text('\n'.join(report['files'])+'\n')
                subprocess.run(['rsync','-a','--timeout=60','--files-from='+str(listing),'-e',host['rsync_ssh'],
                                host['login']+':'+host['root']+'/',str(staging)+'/'],check=True)
                assert digest(staging/'protocol.json')==ph
                assert digest(staging/'acceleration_plan.json')==digest(args.plan)
                # Commit prediction before receipt; only stable JSONs were listed.
                for row in report['receipts']:
                    key=stem(tuple(row[k] for k in ['task','method','steps','exchange','sample_id','seed']))
                    path=Path('results')/key
                    if row.get('prediction_sha256'):
                        tensor=(staging/path).with_suffix('.pt');assert digest(tensor)==row['prediction_sha256']
                        preserve_copy(tensor,(canonical/path).with_suffix('.pt'))
                    preserve_copy((staging/path).with_suffix('.json'),(canonical/path).with_suffix('.json'))
                for name in report['files']:
                    if '/' in name:continue
                    target='acceleration_'+name if name.startswith('environment_') else name
                    preserve_copy(staging/name,canonical/target)
            rows=[json.loads(p.read_text()) for p in (canonical/'results').rglob('*.json')]
            keys=[stem(tuple(r[k] for k in ['task','method','steps','exchange','sample_id','seed'])) for r in rows]
            assert len(keys)==len(set(keys)) and set(keys)<=set(owner)
            complete=len(keys)==1728 and all((canonical/f'complete_{i}.json').exists() for i in range(4))
            summary=dict(checked_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
                         calls=len(rows),expected=1728,outcomes=dict(Counter(r['status'] for r in rows)),
                         sampling_complete=complete,worker_pool='2 RTX4090 + 2 A100',plan_sha256=digest(args.plan))
            write(s/'sampling_progress.json',summary);write(work/'collected_origins.json',origins)
            print(json.dumps(summary),flush=True)
            if complete:
                subprocess.run([sys.executable,str(Path(__file__).with_name('audit_ns_loss_results.py')),
                    '--inputs',str(s/'inputs_v2'),'--results',str(canonical),'--baselines',str(s/'baseline_results_gpu_v2'),
                    '--output',str(s/'ns_complete_audit'),'--require-complete'],check=True)
                print('COMPLETE: all continuation calls collected and audited.',flush=True)
                return
        except (subprocess.SubprocessError,ValueError,OSError) as exc:
            print('Collection retry; samplers unchanged:',repr(exc),flush=True)
            if not args.watch:raise
        if not args.watch:return
        time.sleep(120)


if __name__=='__main__':main()

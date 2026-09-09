#!/usr/bin/env python3
"""Observe the four registered samplers, audit terminal outputs, and plot.

Transient SSH failures never restart sampling. A missing process is treated
as terminal only after checking the recorded PID and command line; incomplete
terminal runs cause an explicit failure instead of an automatic restart.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'plot'))
from run_paper_ablation_revision import write
HOSTS={
 'server197':dict(base='/research_data/users/zhangxifeng/C01Python',
                  python='/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python',shards=[0]),
 'server216':dict(base='/data1/zjinzxf2025/C01Python',
                  python='/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python',shards=[1,2,3]),
}


def remote_status(host,info):
    source="""
import json,os,time
from pathlib import Path
base=Path(BASE)
rows=[]
for shard in SHARDS:
    env=json.loads((base/'conditional_scaling_results_20260909/production'/f'environment_run_{shard}.json').read_text())
    pid=env['pid']
    proc=Path('/proc')/str(pid)/'cmdline'
    try:command=proc.read_bytes().decode().replace('\\0',' ')
    except FileNotFoundError:command=''
    live='run_conditional_sample_scaling.py run' in command
    result=base/'conditional_scaling_results_20260909/production'
    complete=(result/f'complete_{shard}.json').exists()
    rows.append(dict(shard=shard,pid=pid,live=live,command=command,complete=complete,
                     receipts=len(list(result.glob('*/*/K*.json')))))
print(json.dumps(rows))
""".replace('BASE',repr(info['base'])).replace('SHARDS',repr(info['shards']))
    command=shlex.join([info['python'],'-c',source])
    result=subprocess.run(['ssh',host,command],check=True,text=True,capture_output=True,timeout=45)
    return json.loads(result.stdout)


def export_host(host,info,dest):
    dest.mkdir(parents=True,exist_ok=True)
    base=info['base']
    command=shlex.join([info['python'],base+'/conditional_scaling_export_20260909/plot/export_conditional_sample_scaling.py',
      '--results',base+'/conditional_scaling_results_20260909/production',
      '--output',base+'/conditional_scaling_results_20260909/export'])
    with (dest/'export.log').open('w') as log:
        subprocess.run(['ssh',host,command],stdout=log,stderr=subprocess.STDOUT,check=True)
    for name in ['conditional_scaling_per_input.csv','conditional_scaling_fields.npz','conditional_scaling_manifest.json']:
        subprocess.run(['scp',host+':'+base+'/conditional_scaling_results_20260909/export/'+name,str(dest/name)],check=True)
    manifest=json.loads((dest/'conditional_scaling_manifest.json').read_text())
    assert manifest['complete']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--paper',type=Path,required=True)
    p.add_argument('--once',action='store_true')
    args=p.parse_args();args.audit.mkdir(parents=True,exist_ok=True)
    while True:
        snapshot=dict(observed_unix=time.time(),hosts={})
        known=True;all_terminal=True
        for host,info in HOSTS.items():
            try:rows=remote_status(host,info)
            except (subprocess.SubprocessError,OSError,ValueError) as exc:
                known=False;all_terminal=False
                snapshot['hosts'][host]=dict(observation_error=str(exc))
                continue
            snapshot['hosts'][host]=rows
            if any(r['live'] for r in rows):all_terminal=False
            for row in rows:
                if not row['live'] and not row['complete']:
                    snapshot['status']='terminal_incomplete';write(args.audit/'STATUS.json',snapshot)
                    raise RuntimeError(f'Terminal incomplete sampler: {host} shard {row["shard"]}')
        snapshot['status']='verified_running' if known and not all_terminal else 'terminal_complete' if all_terminal else 'observation_retry'
        write(args.audit/'STATUS.json',snapshot)
        print(snapshot['status'],snapshot['observed_unix'],flush=True)
        if all_terminal:break
        if args.once:return
        time.sleep(60)
    exports=[]
    for host,info in HOSTS.items():
        dest=args.audit/host
        marker=dest/'conditional_scaling_manifest.json'
        if not marker.exists() or not json.loads(marker.read_text())['complete']:
            export_host(host,info,dest)
        exports.append(dest)
    output=args.paper/'source_data/conditional_scaling_0909'
    with (args.audit/'plot.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'plot/plot_conditional_sample_scaling.py'),
            '--exports',*[str(x) for x in exports],'--output',str(output),'--figures',str(args.paper/'figures')],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest=json.loads((output/'conditional_scaling_final_manifest.json').read_text())
    assert manifest['complete']
    write(args.audit/'READY_FOR_PAPER_REVIEW.json',dict(completed_unix=time.time(),manifest=str(output/'conditional_scaling_final_manifest.json'),
        remaining='Inspect exported figures, write results interpretation, integrate TeX, compile paper and verify final layout.'))
    print('AUDITED_AND_PLOTTED',flush=True)


if __name__=='__main__':main()

"""Read-only collection of committed receipts from the separate NS study."""
import argparse,json,shlex,subprocess,sys,time
from collections import Counter,defaultdict
from pathlib import Path
from run_ns_loss_study import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--watch',action='store_true')
    args=p.parse_args();s=args.study;s.mkdir(parents=True,exist_ok=True)
    remote='zhangxf@192.168.191.193'
    root='/home/zhangxf/C01Python/FM4PDE_ns_loss_spectrum_20260907/ns_results_v3'
    script=f'''import pathlib,json
p=pathlib.Path({root!r});names=[];receipts=[]
for f in p.glob('*.json'):names.append(f.name)
for f in sorted((p/'results').rglob('*.json')):
 r=json.loads(f.read_text());names.append(str(f.relative_to(p)));receipts.append(r)
 if r.get('prediction_sha256'):names.append(str(f.with_suffix('.pt').relative_to(p)))
print(json.dumps(dict(files=names,receipts=receipts)))
'''
    while True:
        try:
            report=json.loads(subprocess.check_output(['ssh','-p','9088','-o','ConnectTimeout=20',remote,'python3','-'],input=script,text=True,timeout=60))
            listing=s/'ns_collect_files.txt';listing.write_text('\n'.join(report['files'])+'\n')
            dest=s/'ns_results_v3';dest.mkdir(exist_ok=True)
            subprocess.run(['rsync','-a','--timeout=60','--files-from='+str(listing),'-e','ssh -p 9088',remote+':'+root+'/',str(dest)+'/'],check=True)
            counts=Counter(r['status'] for r in report['receipts'])
            summary=dict(checked_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),calls=len(report['receipts']),expected=1728,outcomes=dict(counts),
                sampling_complete=len(report['receipts'])==1728 and all(f'complete_{i}.json' in report['files'] for i in [0,1]))
            write(s/'sampling_progress.json',summary);print(json.dumps(summary),flush=True)
            if summary['sampling_complete']:
                cmd=[sys.executable,str(Path(__file__).with_name('audit_ns_loss_results.py')),'--inputs',str(s/'inputs_v2'),'--results',str(dest),
                     '--baselines',str(s/'baseline_results_gpu_v2'),'--output',str(s/'ns_complete_audit'),'--require-complete']
                subprocess.run(cmd,check=True)
                print('COMPLETE: all formal source predictions collected and audited; NS paper integration remains required.',flush=True)
                return
        except (subprocess.SubprocessError,ValueError,OSError) as exc:
            print('Collection check failed; source jobs are unchanged:',repr(exc),flush=True)
            if not args.watch:raise
        if not args.watch:return
        # This runs in its own tmux job; it never blocks the assistant tool
        # call. Read-only polling needs no repeated GPU work or user action.
        time.sleep(600)


if __name__=='__main__':main()

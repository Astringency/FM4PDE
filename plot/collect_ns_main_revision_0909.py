"""Read-only remote progress monitoring and complete-artifact collection."""
from pathlib import Path
import argparse
import datetime
import json
import shlex
import subprocess
import sys
import time

SOURCES={
 'a100':('server197','/research_data/users/zhangxifeng/C01Python/NSMainRevision0909/main_results'),
 'rtx4090':('server193','/home/zhangxf/C01Python/NSMainWorker0909/main_results'),
 'timing':('server193','/home/zhangxf/C01Python/NSMainRevision0909/timing_results')}


def progress(host,path,kind):
    code='''from pathlib import Path
import json
p=Path(PATH)
rows=[]
for f in p.rglob('*.json'):
 try:
  d=json.loads(f.read_text())
  if isinstance(d,dict) and ('ids' in d if KIND!='timing' else 'sample_id' in d and 'seconds' in d and '_contended_' not in f.name):rows.append(d)
 except (ValueError,OSError):pass
cells={}
for d in rows:
 k=(d['dist']+'/'+d['setting']) if KIND!='timing' else (d['method']+'/'+str(d['steps']))
 cells[k]=cells.get(k,0)+(len(d['ids']) if KIND!='timing' else 1)
print(json.dumps(dict(cells=cells,completed=sum(cells.values()),complete=bool(list(p.rglob('complete*.json'))))))
'''.replace('PATH',repr(path)).replace('KIND',repr(kind))
    return json.loads(subprocess.check_output(['ssh',host,'python3 -c '+shlex.quote(code)],text=True,timeout=30))


def main(a):
    a.output.mkdir(parents=True,exist_ok=True);collected=set()
    exporter=Path(__file__).with_name('export_ns_main_revision_0909.py')
    while True:
        status={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':{}}
        for label,(host,path) in SOURCES.items():
            try:status['workers'][label]=progress(host,path,label)
            except Exception as e:status['workers'][label]={'error':str(e)}
        (a.output/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        for label,row in status['workers'].items():
            expected={'a100':10500,'rtx4090':4500,'timing':80}[label]
            if row.get('complete') and row.get('completed')==expected and label not in collected:
                host,path=SOURCES[label]
                target=a.output/('timing_results' if label=='timing' else 'main_results')
                target.mkdir(parents=True,exist_ok=True)
                subprocess.run(['rsync','-a',host+':'+path+'/',str(target)+'/'],check=True)
                collected.add(label)
                if label=='timing':subprocess.run([sys.executable,str(exporter),'timing','--results',str(target),'--output',str(a.output/'tables')],check=True)
                print('COLLECTED',label,flush=True)
        if collected==set(SOURCES):
            subprocess.run([sys.executable,str(exporter),'main','--results',str(a.output/'main_results'),'--output',str(a.output/'tables')],check=True)
            (a.output/'collection_complete.json').write_text(json.dumps(dict(status='complete',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
            return
        time.sleep(45)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

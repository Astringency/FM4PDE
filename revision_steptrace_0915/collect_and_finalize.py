"""Local relay collector; preserves all remote outputs and verifies final assets."""
import argparse,json,pathlib,subprocess,sys,time

def run(argv):
 return subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--paper-output',type=pathlib.Path,required=True);p.add_argument('--interval',type=int,default=45);p.add_argument('--source',action='append',help='Explicit host:absolute-directory; repeat for multiple source machines.');p.add_argument('--canonical-root',default='server197:/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/fm_diffusion_steptrace_cocogen_20260915/');a=p.parse_args()
 sources=[tuple(s.split(':',1)) for s in a.source] if a.source else [('server193','/home/zhangxf/C01Python/FM4PDE/outputs/fm_diffusion_steptrace_cocogen_20260915'),('server216','/data1/zjinzxf2025/C01Python/FM4PDE/outputs/fm_diffusion_steptrace_cocogen_20260915')]
 assert all(len(s)==2 and s[0] and s[1].startswith('/') for s in sources)
 canonical=a.canonical_root.rstrip('/')+'/'
 code=pathlib.Path(__file__).parent;status=a.root/'collector_status.json'
 while True:
  failures=[]
  for host,source in sources:
   for name in ['traces','pilots']:
    dest=a.root/name;dest.mkdir(parents=True,exist_ok=True)
    result=run(['rsync','-a','--exclude=*.partial*',f'{host}:{source}/{name}/',str(dest)+'/'])
    if result.returncode:failures.append({'host':host,'area':name,'returncode':result.returncode,'stderr':result.stderr[-500:]})
  receipts=list((a.root/'traces').glob('*/*_1000_*.json'))
  state={'updated_unix':time.time(),'complete_calls':len(receipts),'expected_calls':200,'sync_errors':failures}
  status.write_text(json.dumps(state,indent=2)+'\n');print('COLLECTED',len(receipts),'/200','errors',len(failures),flush=True)
  # New task only; no --delete and no remote source mutations.
  for name in ['traces','pilots']:
   r=run(['rsync','-a',str(a.root/name)+'/',canonical+name+'/'])
   if r.returncode:print('CANONICAL_SYNC_PENDING',name,r.stderr[-300:],flush=True)
  if len(receipts)==200 and not failures:
   subprocess.run([sys.executable,str(code/'audit_traces.py'),'--root',str(a.root)],check=True)
   subprocess.run([sys.executable,str(code/'export_figures.py'),'--root',str(a.root),'--output',str(a.paper_output),'--traces'],check=True)
   for name in ['audit']:
    subprocess.run(['rsync','-a',str(a.root/name)+'/',canonical+name+'/'],check=True)
   subprocess.run(['rsync','-a',str(a.paper_output)+'/',canonical+'trace_figures/'],check=True)
   state['status']='finalized';status.write_text(json.dumps(state,indent=2)+'\n');subprocess.run(['rsync','-a',str(status),canonical],check=True);print('FINALIZED',flush=True);return
  time.sleep(a.interval)
if __name__=='__main__':main()

"""Wait for all frozen GPU calls, collect, audit, integrate and rebuild locally.

The GPU workers run separately in their own remote tmux sessions. This
collector only reads their results and writes the local revision artifacts.
A completion marker means ready for final visual review, not goal completion.
"""
import argparse,json,shlex,subprocess,sys,time
from pathlib import Path

PDES=['poisson','helmholtz','darcy','nsnonbounded','burger']

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--paper',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];remote='/home/zhangxf/C01Python/FM4PDE_diffusion_timing_20260907/timing_results_v2'
 cmd="from pathlib import Path; import json; p=Path("+repr(remote)+"); print(json.dumps({x:{'complete':(p/x/'complete.json').exists(),'calls':sum(1 for f in (p/x).glob('*.json') if f.name.startswith(('FM4PDE_','DiffusionPDE_')) and '_contended_' not in f.name)} for x in "+repr(PDES)+"}))"
 ssh=['ssh','-p','9088','-o','ConnectTimeout=15','zhangxf@192.168.191.193']
 a.study.mkdir(parents=True,exist_ok=True)
 while True:
  try:
   out=subprocess.check_output(ssh+['python3 -c '+shlex.quote(cmd)],text=True,timeout=30)
   status=json.loads(out);status_file=a.study/'timing_progress.json'
   status_file.write_text(json.dumps(dict(checked_at=time.time(),pdes=status),indent=2)+'\n')
   print(time.strftime('%Y-%m-%d %H:%M:%S'),json.dumps(status),flush=True)
   if all(r['complete'] for r in status.values()):break
  except (subprocess.SubprocessError,json.JSONDecodeError) as e:print('COLLECTION RETRY',repr(e),flush=True)
  time.sleep(60)
 dest=a.study/'timing_results_v2';dest.mkdir(exist_ok=True)
 subprocess.run(['rsync','-az','-e','ssh -p 9088','zhangxf@192.168.191.193:'+remote+'/',str(dest)+'/'],check=True)
 commands=[
  [sys.executable,'plot/export_diffusion_fm_timing.py','--inputs',str(a.study/'timing_inputs_v2'),'--results',str(dest),'--paper',str(a.paper)],
  [sys.executable,'plot/integrate_diffusion_timing.py','--paper',str(a.paper)],
  ['bash',str(a.paper/'build_all.sh')],
  [sys.executable,str(a.paper/'audit/verify_presentation_0907.py'),'--require-timing'],
  [sys.executable,str(a.paper/'audit/verify_revision_artifacts.py')]]
 for command in commands:
  print('RUN',shlex.join(command),flush=True);subprocess.run(command,cwd=root,check=True)
 (a.study/'timing_ready_for_visual_review.json').write_text(json.dumps(dict(status='audited_integrated_built_ready_for_visual_review',finished_at=time.time(),calls=400),indent=2)+'\n')
 print('ALL 400 CALLS AUDITED; PDFs rebuilt; final visual review required',flush=True)

if __name__=='__main__':main()

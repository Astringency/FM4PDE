"""Sequential GPU worker: validate each frozen family, then collect its 20 traces."""
import argparse,json,pathlib,subprocess,sys,time

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--diffusion-root',type=pathlib.Path,required=True);p.add_argument('--pdes',nargs='+',default=['poisson','darcy','helmholtz','burger','nsnonbounded']);a=p.parse_args()
 logs=a.root/'logs';logs.mkdir(parents=True,exist_ok=True)
 for pde in a.pdes:
  for mode,steps in [('pilot',100),('pilot',1000),('run',1000)]:
   name=f'{pde}_{mode}_{steps}';cmd=[sys.executable,str(pathlib.Path(__file__).with_name('collect_trace.py')),'--root',str(a.root),'--diffusion-root',str(a.diffusion_root),'--pde',pde,'--mode',mode,'--steps',str(steps)]
   before=time.time();print('START',name,flush=True)
   with (logs/(name+'.log')).open('a') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
   (logs/(name+'.stage.json')).write_text(json.dumps({'argv':cmd,'started_unix':before,'finished_unix':time.time(),'exit_code':result.returncode},indent=2)+'\n')
   print('EXIT',name,result.returncode,flush=True)
   if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()

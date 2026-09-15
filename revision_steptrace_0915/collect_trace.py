"""Non-mutating diagnostics around the frozen controlled-timing samplers.

Steps 0..N-1: clean endpoint estimate at the current native state, using the
network evaluation already required by the original update. Step N: final
sample. Additional native-state metrics are distinguished explicitly.
Metric callbacks never consume random numbers or change sampler tensors.
"""
from __future__ import annotations
import argparse, ast, copy, csv, dataclasses, hashlib, json, os, pathlib, pickle, socket, sys, time


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def write_json(p,x):
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def diffusion_functions(source, torch, np):
 """Add one callback after the first denoiser call; all original AST stays."""
 tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
 loop=next(n for n in fn.body if isinstance(n,ast.For))
 pos=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='x_N' for t in n.targets))
 footer_start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='x_final' for t in n.targets))
 decoder=ast.parse('def decode(x_next):\n pass').body[0];decoder.body=copy.deepcopy(fn.body[footer_start:])
 # The decoder is exactly the frozen final physical-unit conversion.
 scope={'torch':torch,'np':np,'F':torch.nn.functional}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[decoder],type_ignores=[])),'[physical_decode]','exec'),scope)
 original=copy.deepcopy(tree)
 exec(compile(original,'[frozen_diffusion]','exec'),scope);plain=scope['predict']
 fn.args.args.append(ast.arg(arg='trace'))
 loop.body.insert(pos+1,ast.parse('trace(i, decode(x_N), decode(x_cur), float(sigma_t))').body[0])
 ast.fix_missing_locations(tree);instrumented=ast.unparse(tree)+'\n'
 exec(compile(tree,'[traced_diffusion]','exec'),scope)
 return plain,scope['predict'],instrumented


class Trace:
 def __init__(self,cfg,truth,masks,r,normalizer,torch):
  self.cfg,self.truth,self.masks,self.r,self.normalizer,self.torch=cfg,truth,masks,r,normalizer,torch
  self.rows=[];self.paused=0.;self.start=None
  self.metric_cfg=copy.deepcopy(cfg)
  self.metric_cfg.guidance_components='obs_pde';self.metric_cfg.zeta_pde=1.
  self.metric_cfg.pde_guidance_reduction='mse';self.metric_cfg.obs_guidance_reduction='mse'
  self.metric_cfg.pde_residual_region='full';self.residual_meta=None
 def begin(self):
  self.torch.cuda.synchronize();self.start=time.perf_counter()
 def metrics(self,pair):
  torch=self.torch;a,u=(v.detach() for v in pair)
  # Relative errors are independently accumulated in float64 physical units.
  def rel(p,q,m=None):
   p,q=p.double(),q.double()
   if m is not None:p=p*m;q=q*m
   den=q.norm()
   return float((p-q).norm()/den) if float(den)>0 else None
  values={'relative_l2_a':None if self.cfg.pde=='burger' else rel(a,self.truth.coef),
          'relative_l2_u':rel(u,self.truth.sol),
          'observed_relative_l2_a':None if self.cfg.pde=='burger' else rel(a,self.truth.coef,self.masks.coef),
          'observed_relative_l2_u':rel(u,self.truth.sol,self.masks.sol)}
  state=self.r.SplitState(a,u) if hasattr(self.r,'SplitState') else None
  if state is None:
   from sampling.state import SplitState
   state=SplitState(a,u)
  loss=self.r.compute_guidance_losses(state,self.truth,self.masks,self.metric_cfg)
  values['L_pde']=None if loss.L_pde is None else float(loss.L_pde)
  if self.residual_meta is None:self.residual_meta=loss.metadata
  for k,v in values.items():
   if v is not None and not __import__('math').isfinite(v):
    raise FloatingPointError(f'nonfinite {self.cfg.pde} {k}: {v}')
  return values
 def __call__(self,step,estimate,native,native_time):
  torch=self.torch;torch.cuda.synchronize();t0=time.perf_counter()
  active=t0-self.start-self.paused
  with torch.no_grad():
   row={'step':step,'state':'final_sample' if step==self.cfg.num_steps else 'clean_endpoint_estimate','native_time':native_time,'sampling_seconds_excluding_diagnostics':active,'elapsed_seconds_including_diagnostics':t0-self.start,**self.metrics(estimate)}
   if native is not None:row.update({'native_'+k:v for k,v in self.metrics(native).items()})
   else:row.update({'native_'+k:v for k,v in self.metrics(estimate).items()})
  torch.cuda.synchronize();self.paused+=time.perf_counter()-t0
  row['cumulative_diagnostic_seconds']=self.paused;self.rows.append(row)
 def final(self,pair):self(self.cfg.num_steps,pair,None,0. if self.method=='DiffusionPDE' else 1.)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--pde',required=True);ap.add_argument('--mode',choices=['pilot','run'],required=True);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--sample-id',type=int,action='append');ap.add_argument('--methods',nargs='+',default=['FM4PDE','DiffusionPDE']);ap.add_argument('--diffusion-root',type=pathlib.Path,required=True);args=ap.parse_args()
 root=args.root.resolve();manifest=json.load(open(root/'manifest.json'));info=manifest['cells'][args.pde];inputs=root/info['input_root'];protocol=json.load(open(inputs/'protocol.json'))
 assert args.steps in [100,1000]
 # Sources are immutable archived producer snapshots, not the current sampler.
 sys.path[:0]=[str(root/info['fm_code']),str(root/info['fm_code']/'plot'),str(args.diffusion_root)]
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 import numpy as np,torch
 from sampling.config import AblationConfig
 from sampling.data import PDEGroundTruth
 from sampling.masks import PairMasks
 from sampling.model_io import load_fm4pde_checkpoint_bundle
 from data.specs import get_pde_spec
 import sampling.runner as r
 from run_matched_timing import fm_predict
 torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 device='cuda:0';out=root/('pilots' if args.mode=='pilot' else 'traces')/args.pde;out.mkdir(parents=True,exist_ok=True)
 # Verify only this PDE's explicit inputs; shared source archives are hashed too.
 for item in manifest['artifacts']:
  if item['path'].startswith(info['input_root']+'/') and '/original_timing/' not in item['path']:
   assert sha(root/item['path'])==item['sha256'],item
 fm=load_fm4pde_checkpoint_bundle(str(root/info['fm_weights']),args.pde,device,model_profile='recommended')
 with open(root/info['diffusion_weights'],'rb') as f:dm=pickle.load(f)['ema'].to(device).eval()
 dm.requires_grad_(False);source=(root/info['diffusion_source']).read_text();dm_plain,dm_trace,traced_source=diffusion_functions(source,torch,np)
 (out/'instrumented_diffusion.py').write_text(traced_source)
 ids=args.sample_id or ([info['pilot_id']] if args.mode=='pilot' else info['evaluation_ids'])
 allowed=[info['pilot_id']] if args.mode=='pilot' else info['evaluation_ids'];assert all(i in allowed for i in ids)
 data=np.load(inputs/'truths.npz');mdata=np.load(inputs/'masks.npz');original_ids=protocol['evaluation_ids']+[protocol['pilot_id']]
 env={'host':socket.gethostname(),'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'tf32':False,'batch_size':1,'manifest_sha256':sha(root/'manifest.json'),'collector_sha256':sha(pathlib.Path(__file__)),'fm_commit':info['fm_commit'],'diffusion_commit':info['diffusion_commit']}
 pilot_path=root/'pilots'/args.pde/f'certificate_{args.steps}.json'
 if args.mode=='run':
  certificate=json.load(open(pilot_path));assert certificate['status']=='pass'
  assert certificate['environment']['collector_sha256']==env['collector_sha256'];assert certificate['environment']['manifest_sha256']==env['manifest_sha256'];assert certificate['environment']['gpu']==env['gpu']
 spec=get_pde_spec(args.pde);checks=[]
 for i in ids:
  j=original_ids.index(i);a=torch.from_numpy(data[args.pde+'_a'][j:j+1]).to(device);u=torch.from_numpy(data[args.pde+'_u'][j:j+1]).to(device)
  ma=torch.from_numpy(mdata[f'{args.pde}_{i}_a'])[None,None].to(device);mu=torch.from_numpy(mdata[f'{args.pde}_{i}_u'])[None,None].to(device);masks=PairMasks(ma,mu,{'source':'original frozen controlled timing masks'})
  def gt(aa,uu):return PDEGroundTruth(args.pde,aa,uu,aa if args.pde=='burger' else torch.cat([aa,uu],1),{},list(spec.coef_channel_names),list(spec.sol_channel_names),{'sample_ids':[str(i)],'sample_offsets':[i],'offset':i,'batch_size':1,'synthetic':False,'endpoint_pair':args.pde!='burger'})
  truth=gt(a,u);observed=gt(a*ma,u*mu)
  c=copy.deepcopy(protocol['fm_configs'][args.pde]);c.update(device=device,num_steps=args.steps,batch_size=1,offset=i,sample_seed=protocol['seed']+i,checkpoint_path=str(root/info['fm_weights']),output_dir=str(out),initial_noise_source_indices=[],initial_noise_source_batch_size=None,save_plots=False,save_intermediate=False,save_per_sample_curves=False)
  if args.pde=='burger':c.update(sensor_mode='sensor_column',num_sensor_columns=5,num_obs=640)
  cfg=AblationConfig(**c);cfg=r.finalize_ground_truth_config(cfg);cfg.validate()
  dc=copy.deepcopy(protocol['diffusion_configs'][args.pde]);dc['generate'].update(device=device,seed=protocol['seed']+i,batch_size=1);dc['test']['iterations']=args.steps
  for method in args.methods:
   dest=out/f'{method}_{args.steps}_{i}';receipt=dest.with_suffix('.json')
   if receipt.exists():
    old=json.load(open(receipt));assert old['environment']==env;assert sha(dest.with_suffix('.csv'))==old['trace_sha256'];print('SKIP',dest.name,flush=True);continue
   tracer=Trace(cfg,truth,masks,r,fm[1],torch);tracer.method=method
   def plain():return fm_predict(cfg,fm,observed,masks) if method=='FM4PDE' else dm_plain(dc,dm,observed.coef,observed.sol,ma[0,0],mu[0,0])
   baseline=plain() if args.mode=='pilot' else None
   torch.cuda.reset_peak_memory_stats();tracer.begin()
   if method=='FM4PDE':
    original=r.sampler_step;k=[0]
    def hook(*aa,**kk):
     result=original(*aa,**kk)
     ep=r._physical_from_model_state(result.x_endpoint.detach(),cfg,fm[1]);nat=r._physical_from_model_state(result.x_raw_current.detach(),cfg,fm[1])
     tracer(k[0],(ep.coef,ep.sol),(nat.coef,nat.sol),float(result.t));k[0]+=1;return result
    r.sampler_step=hook
    try:prediction=plain()
    finally:r.sampler_step=original
   else:prediction=dm_trace(dc,dm,observed.coef,observed.sol,ma[0,0],mu[0,0],tracer)
   tracer.final(prediction);torch.cuda.synchronize();wall=time.perf_counter()-tracer.start
   assert len(tracer.rows)==args.steps+1;assert [x['step'] for x in tracer.rows]==list(range(args.steps+1))
   row={'status':'complete','pde':args.pde,'method':method,'sample_id':i,'steps':args.steps,'nfe':args.steps if method=='FM4PDE' else 2*args.steps-1,'environment':env,'effective_fm_config':dataclasses.asdict(cfg),'effective_diffusion_config':dc,'metric_residual_metadata':tracer.residual_meta,'wall_seconds_including_diagnostics':wall,'diagnostic_seconds':tracer.paused,'sampling_seconds_excluding_diagnostics':wall-tracer.paused,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
   if baseline is not None:
    delta=max(float((x-y).abs().max()) for x,y in zip(prediction,baseline));row['uninstrumented_max_abs']=delta;assert delta==0.,(args.pde,method,delta);checks.append({'method':method,'sample_id':i,'uninstrumented_max_abs':delta,'finite':True})
   oldpath=inputs/'original_timing'/f'{method}_{args.steps}_{i}.pt'
   if oldpath.exists():
    old=torch.load(oldpath,map_location=device,weights_only=False);row['historical_final_max_abs']=max(float((prediction[0]-old['coef']).abs().max()),float((prediction[1]-old['sol']).abs().max()))
    row['historical_final_relative_l2_difference']=[float((x-y).norm()/y.norm()) for x,y in zip(prediction,(old['coef'],old['sol']))]
   temp=dest.with_suffix('.partial.csv')
   with temp.open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(tracer.rows[0]));writer.writeheader();writer.writerows(tracer.rows)
   temp.rename(dest.with_suffix('.csv'));row['trace_sha256']=sha(dest.with_suffix('.csv'))
   torch.save({'coef':prediction[0].detach().cpu(),'sol':prediction[1].detach().cpu(),'sample_id':i,'input_manifest_sha256':env['manifest_sha256']},dest.with_suffix('.pt'));row['prediction_sha256']=sha(dest.with_suffix('.pt'))
   write_json(receipt,row);print('TRACE_COMPLETE',args.pde,method,args.steps,i,round(wall,3),flush=True)
 if args.mode=='pilot':
  assert set(x['method'] for x in checks)==set(args.methods) or all((out/f'{m}_{args.steps}_{ids[0]}.json').exists() for m in args.methods)
  write_json(pilot_path,{'status':'pass','environment':env,'checks':checks,'steps':args.steps,'sample_ids':ids,'metric_semantics':'clean endpoint estimate; final index is final sample','timing_note':manifest['timing_note']})
 print('COMPLETE',args.pde,args.mode,args.steps,flush=True)
if __name__=='__main__':main()

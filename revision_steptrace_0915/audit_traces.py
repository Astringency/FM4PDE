"""Independent final-step physical error checks and frozen source-tree audit."""
import argparse,csv,hashlib,json,math,pathlib,tarfile,subprocess,sys
import numpy as np
import torch

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def source_tree(root,manifest):
 result=[]
 for rel in sorted({i['fm_code'] for i in manifest['cells'].values()}):
  d=root/rel;archive=d.with_suffix('.tar');spec=next(x for x in manifest['artifacts'] if x['path']==str(archive.relative_to(root)));assert sha(archive)==spec['sha256']
  n=0
  with tarfile.open(archive) as tf:
   for member in tf:
    if not member.isfile():continue
    expected=hashlib.sha256(tf.extractfile(member).read()).hexdigest();actual=sha(d/member.name);assert expected==actual,(rel,member.name);n+=1
  result.append({'tree':rel,'archive_sha256':spec['sha256'],'files_exact':n})
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--allow-partial',action='store_true');p.add_argument('--sources-only',action='store_true');p.add_argument('--diffusion-root',type=pathlib.Path);a=p.parse_args();root=a.root
 manifest=json.load(open(root/'manifest.json'));report={'status':'running','manifest_sha256':sha(root/'manifest.json'),'source_trees':source_tree(root,manifest),'calls':[],'missing':[]}
 decision_path=root/'provenance/environment_decision.json'
 decision=json.load(open(decision_path)) if decision_path.exists() else None
 if decision:report['environment_decision_sha256']=sha(decision_path)
 source_proof=root/'audit/source_audit.json'
 if source_proof.exists() and not a.sources_only:
  proof=json.load(open(source_proof));assert proof['status']=='pass' and proof['manifest_sha256']==report['manifest_sha256']
  assert proof['source_trees']==report['source_trees']
  if proof.get('diffusion_source'):
   assert proof['diffusion_source']['tracked_tree_matches_commit']
   assert proof['diffusion_source']['commit']==next(iter(manifest['cells'].values()))['diffusion_commit']
   report['executed_diffusion_source_proof']={'path':str(source_proof),'sha256':sha(source_proof),**proof['diffusion_source']}
 if a.diffusion_root:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.diffusion_root,text=True).strip();assert commit==next(iter(manifest['cells'].values()))['diffusion_commit']
  changed=subprocess.check_output(['git','diff','--name-only','HEAD','--'],cwd=a.diffusion_root,text=True);assert not changed,changed
  report['diffusion_source']={'root':str(a.diffusion_root),'commit':commit,'tracked_tree_matches_commit':True}
 if not a.sources_only:
  report['verified_artifacts']=len(manifest['artifacts'])
  for item in manifest['artifacts']:assert sha(root/item['path'])==item['sha256'],item['path']
  for pde,info in manifest['cells'].items():
   inp=root/info['input_root'];protocol=json.load(open(inp/'protocol.json'));data=np.load(inp/'truths.npz');masks=np.load(inp/'masks.npz')
   for method in ['FM4PDE','DiffusionPDE']:
    pilot_path=root/'pilots'/pde/'certificate_1000.json'
    if not pilot_path.exists() and a.allow_partial:
     report['missing'].extend(f'traces/{pde}/{method}_1000_{i}' for i in info['evaluation_ids']);continue
    pilot=json.load(open(pilot_path));assert pilot['status']=='pass'
    for i in info['evaluation_ids']:
     stem=root/'traces'/pde/f'{method}_1000_{i}';receipt=stem.with_suffix('.json')
     if not receipt.exists():report['missing'].append(str(stem.relative_to(root)));continue
     rec=json.load(open(receipt));assert rec['status']=='complete';assert (rec['pde'],rec['method'],rec['sample_id'])==(pde,method,i);assert rec['environment']['manifest_sha256']==report['manifest_sha256'];assert rec['environment']['collector_sha256']==pilot['environment']['collector_sha256'];assert rec['environment']['gpu']==pilot['environment']['gpu']
     for key in ['python','torch','cuda','gpu','tf32','batch_size','fm_commit','diffusion_commit']:
      assert rec['environment'][key]==pilot['environment'][key],(pde,method,i,key)
     if decision:
      assert rec['environment']['torch']==decision['torch'] and rec['environment']['cuda']==decision['cuda']
      assert rec['environment']['python'].startswith(decision['python_version'])
     assert sha(stem.with_suffix('.csv'))==rec['trace_sha256'];assert sha(stem.with_suffix('.pt'))==rec['prediction_sha256']
     rows=list(csv.DictReader(open(stem.with_suffix('.csv'))));assert [int(r['step']) for r in rows]==list(range(1001));assert all(float(r['sampling_seconds_excluding_diagnostics'])>=0 for r in rows)
     times=np.array([float(r['sampling_seconds_excluding_diagnostics']) for r in rows]);assert np.all(np.diff(times)>=0)
     for key in ['relative_l2_a','relative_l2_u','observed_relative_l2_a','observed_relative_l2_u','L_pde']:
      if pde=='burger' and key.endswith('_a'):continue
      assert all(r[key]!='' and math.isfinite(float(r[key])) for r in rows),(pde,method,i,key)
     assert all(r['state']=='clean_endpoint_estimate' for r in rows[:-1]) and rows[-1]['state']=='final_sample'
     pred=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=False);assert pred['sample_id']==i;idx=protocol['evaluation_ids'].index(i);checks={}
     for name,key in [('a','coef'),('u','sol')]:
      if pde=='burger' and name=='a':
       assert all(r['relative_l2_a']=='' and r['observed_relative_l2_a']=='' for r in rows);continue
      x=pred[key].numpy().astype(np.float64);y=data[pde+'_'+name][idx:idx+1].astype(np.float64);m=masks[f'{pde}_{i}_{name}'][None,None].astype(np.float64)
      assert np.isfinite(x).all()
      for field,mask in [(f'relative_l2_{name}',1.),(f'observed_relative_l2_{name}',m)]:
       value=float(np.linalg.norm((x-y)*mask)/np.linalg.norm(y*mask));saved=float(rows[-1][field]);assert math.isclose(value,saved,rel_tol=2e-12,abs_tol=2e-12),(pde,method,i,field,value,saved);checks[field]=value
     historical_path=inp/'original_timing'/f'{method}_1000_{i}.pt'
     historical_check=None
     if historical_path.exists():
      historical=torch.load(historical_path,map_location='cpu',weights_only=False)
      delta=max(float((pred[k]-historical[k]).abs().max()) for k in ['coef','sol'])
      assert delta==rec['historical_final_max_abs'],(pde,method,i,'historical maxabs')
      historical_check={'prediction_sha256':sha(historical_path),'max_abs':delta,'exact':all(torch.equal(pred[k],historical[k]) for k in ['coef','sol'])}
     elif decision:
      raise ValueError(f'Missing original timing prediction: {historical_path}')
     report['calls'].append({'pde':pde,'method':method,'sample_id':i,'rows':1001,'receipt_sha256':sha(receipt),'final_errors_recomputed':checks,'historical_final_recomputed':historical_check,'gpu':rec['environment']['gpu'],'torch':rec['environment']['torch']})
  if report['missing'] and not a.allow_partial:raise ValueError(f"Missing {len(report['missing'])}/200 traces")
  report['final_residual_audits']=[]
  for pde in manifest['cells']:
   if not any(x['pde']==pde for x in report['calls']):continue
   command=[sys.executable,str(pathlib.Path(__file__).with_name('audit_residuals.py')),'--root',str(root),'--pde',pde]
   if a.allow_partial:command.append('--allow-partial')
   subprocess.run(command,check=True)
   path=root/'audit'/f'{pde}_final_residual_audit.json';result=json.load(open(path))
   report['final_residual_audits'].append({'pde':pde,'path':str(path),'sha256':sha(path),'calls':len(result['records']),'failed_calls':result['failed_calls']})
 report['status']='pass' if not report['missing'] else 'partial';report['complete_calls']=len(report['calls']);report['metric_rows']=1001*len(report['calls'])
 historical=[x['historical_final_recomputed'] for x in report['calls'] if x.get('historical_final_recomputed')]
 report['historical_final_comparison']={'verified_calls':len(historical),'exact_calls':sum(x['exact'] for x in historical),'max_abs':max((x['max_abs'] for x in historical),default=None)}
 out=root/'audit';out.mkdir(parents=True,exist_ok=True);(out/('source_audit.json' if a.sources_only else 'final_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['complete_calls'],len(report['missing']))
if __name__=='__main__':main()

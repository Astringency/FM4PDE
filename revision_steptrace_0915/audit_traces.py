"""Independent final-step physical error checks and frozen source-tree audit."""
import argparse,csv,hashlib,json,math,pathlib,tarfile,subprocess
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
 if a.diffusion_root:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.diffusion_root,text=True).strip();assert commit==next(iter(manifest['cells'].values()))['diffusion_commit']
  changed=subprocess.check_output(['git','diff','--name-only','HEAD','--'],cwd=a.diffusion_root,text=True);assert not changed,changed
  report['diffusion_source']={'root':str(a.diffusion_root),'commit':commit,'tracked_tree_matches_commit':True}
 if not a.sources_only:
  for pde,info in manifest['cells'].items():
   inp=root/info['input_root'];protocol=json.load(open(inp/'protocol.json'));data=np.load(inp/'truths.npz');masks=np.load(inp/'masks.npz')
   for method in ['FM4PDE','DiffusionPDE']:
    pilot=json.load(open(root/'pilots'/pde/'certificate_1000.json'));assert pilot['status']=='pass'
    for i in info['evaluation_ids']:
     stem=root/'traces'/pde/f'{method}_1000_{i}';receipt=stem.with_suffix('.json')
     if not receipt.exists():report['missing'].append(str(stem.relative_to(root)));continue
     rec=json.load(open(receipt));assert rec['status']=='complete';assert (rec['pde'],rec['method'],rec['sample_id'])==(pde,method,i);assert rec['environment']['manifest_sha256']==report['manifest_sha256'];assert rec['environment']['collector_sha256']==pilot['environment']['collector_sha256'];assert rec['environment']['gpu']==pilot['environment']['gpu']
     assert sha(stem.with_suffix('.csv'))==rec['trace_sha256'];assert sha(stem.with_suffix('.pt'))==rec['prediction_sha256']
     rows=list(csv.DictReader(open(stem.with_suffix('.csv'))));assert [int(r['step']) for r in rows]==list(range(1001));assert all(float(r['sampling_seconds_excluding_diagnostics'])>=0 for r in rows)
     times=np.array([float(r['sampling_seconds_excluding_diagnostics']) for r in rows]);assert np.all(np.diff(times)>=0)
     pred=torch.load(stem.with_suffix('.pt'),map_location='cpu',weights_only=False);assert pred['sample_id']==i;idx=protocol['evaluation_ids'].index(i);checks={}
     for name,key in [('a','coef'),('u','sol')]:
      if pde=='burger' and name=='a':
       assert all(r['relative_l2_a']=='' and r['observed_relative_l2_a']=='' for r in rows);continue
      x=pred[key].numpy().astype(np.float64);y=data[pde+'_'+name][idx:idx+1].astype(np.float64);m=masks[f'{pde}_{i}_{name}'][None,None].astype(np.float64)
      assert np.isfinite(x).all()
      for field,mask in [(f'relative_l2_{name}',1.),(f'observed_relative_l2_{name}',m)]:
       value=float(np.linalg.norm((x-y)*mask)/np.linalg.norm(y*mask));saved=float(rows[-1][field]);assert math.isclose(value,saved,rel_tol=2e-12,abs_tol=2e-12),(pde,method,i,field,value,saved);checks[field]=value
     report['calls'].append({'pde':pde,'method':method,'sample_id':i,'rows':1001,'receipt_sha256':sha(receipt),'final_errors_recomputed':checks,'gpu':rec['environment']['gpu'],'torch':rec['environment']['torch']})
  if report['missing'] and not a.allow_partial:raise ValueError(f"Missing {len(report['missing'])}/200 traces")
 report['status']='pass' if not report['missing'] else 'partial';report['complete_calls']=len(report['calls']);report['metric_rows']=1001*len(report['calls'])
 out=root/'audit';out.mkdir(parents=True,exist_ok=True);(out/('source_audit.json' if a.sources_only else 'final_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['complete_calls'],len(report['missing']))
if __name__=='__main__':main()

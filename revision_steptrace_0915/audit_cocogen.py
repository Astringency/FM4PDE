"""Independently bind all nine Darcy sparse comparisons to actual saved inputs."""
import argparse,hashlib,json,pathlib,sys

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--cocogen-code',type=pathlib.Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.cocogen_code));import torch,numpy as np
 from cocogen_eval.main_inputs import historical_input
 from cocogen_eval.inputs import identity,scores
 torch.set_num_threads(2)
 out=a.output/'cocogen';out.mkdir(parents=True,exist_ok=True);report={'status':'running','source':str(a.source),'cells':[]}
 cat=json.load(open(a.source/'protocol/catalog.json'))
 for cell in cat['cells']:
  if cell['pde']!='darcy' or not cell['setting'].startswith('sparse'):continue
  rec=json.load(open(cell['frozen_record']));assert sha(cell['frozen_record'])==cell['frozen_sha256']
  truth,masks,ids,params=historical_input(rec);ident=identity(truth,masks,ids)
  assert ids==list(range(1000));summary_path=a.source/'main'/cell['cell']/'summary.json';summary=json.load(open(summary_path));assert summary['status']=='complete' and summary['n']==1000;assert summary['input']==ident
  seen=[];errors={'a':[],'u':[]};batches=[]
  for receipt_path in sorted((a.source/'main'/cell['cell']/'batches').glob('*/receipt.json')):
   receipt=json.load(open(receipt_path));req=receipt['request'];subset=req['input']['ids'];idx=[ids.index(i) for i in subset]
   assert req['input']==identity(truth[idx],masks[idx],subset)
   assert sha(receipt['prediction_path'])==receipt['prediction_sha256']
   pred=torch.load(receipt['prediction_path'],map_location='cpu',weights_only=False)
   assert pred['request']==req and pred['ids']==subset and torch.isfinite(pred['prediction']).all()
   metric=scores(pred['prediction'],truth[idx],'darcy')
   for field in errors:
    assert np.array_equal(metric[field],receipt['errors'][field])
    errors[field].extend(metric[field])
   seen.extend(subset);batches.append({'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),'prediction_sha256':receipt['prediction_sha256'],'n':len(subset)})
  assert seen==ids
  for field in errors:assert np.array_equal(errors[field],summary['errors'][field])
  fmconfigs=[b['config'] for b in rec['batches']]
  gates=sorted({str(c.get('pde_guidance_start_ratio','historical_missing')) for c in fmconfigs})
  record={'cell':cell['cell'],'n':1000,'identity':ident,'all_historical_result_hashes_verified':len(rec['batches']),'all_cocogen_prediction_hashes_and_input_hashes_verified':len(batches),'all_errors_independently_recomputed':True,'summary_path':str(summary_path),'summary_sha256':sha(summary_path),'record_path':cell['frozen_record'],'record_sha256':cell['frozen_sha256'],'historical_fm_gates':gates,'historical_fm_configs':fmconfigs,'batches':batches,'summary':summary}
  report['cells'].append(record);print('AUDITED',cell['cell'],1000,gates,flush=True)
  if cell['cell']=='darcy/smooth/sparse_joint':
   # Each original FM batch is preserved for seed/noise coupling; only the
   # physics-guidance start will change in the supplemental FM execution.
   torch.save({'truth':truth,'masks':masks,'ids':ids,'record':rec,'identity':ident},out/'darcy_smooth_joint_inputs.pt')
 report['status']='pass';report['samples_verified']=9000
 (out/'reuse_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE',out/'reuse_audit.json',flush=True)
if __name__=='__main__':main()

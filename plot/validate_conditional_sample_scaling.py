#!/usr/bin/env python3
"""Full-step stock/fast/precision checks on the four earlier calibration inputs."""
from __future__ import annotations
import argparse
import json
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'plot'))
from run_conditional_sample_scaling import configuration,fast_sample
from run_paper_ablation_revision import digest,write
from scripts.tuning.compare_pde_guidance_schedules import combine_truths
import sampling.runner as r


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--inputs',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cudnn.benchmark=False
 source=args.inputs/'poisson';protocol=json.loads((source/'protocol.json').read_text());selection=json.loads(args.selection.read_text())
 assert digest(source/'protocol.json')==selection['protocol_sha256']
 truths=torch.load(source/'truths.pt',map_location='cpu',weights_only=False)
 bundle=r.load_fm4pde_checkpoint_bundle(str(source/'weights.pth'),'poisson','cuda:0',model_profile='recommended')
 rows=[]
 for task in ['forward','inverse','both']:
  for offset in [1100,1101,1102,1103]:
   cfg=configuration(protocol,selection,task,source)
   cfg.offset=offset;cfg.mask_seed=20260912+offset;cfg.sample_seed=20260912+offset
   cfg.batch_size=1;cfg.initial_noise_source_indices=[0]
   refs={}
   for precision in ['strict','tf32']:
    torch.backends.cuda.matmul.allow_tf32=precision=='tf32';torch.backends.cudnn.allow_tf32=precision=='tf32'
    cfg.output_dir=str(args.output/task/f'offset{offset}'/precision)
    with (args.output/f'{task}_{offset}_{precision}.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
     result=r.run_single_ablation(cfg,checkpoint_bundle=bundle,ground_truth=combine_truths(truths,[offset],'cuda:0'))
    data=torch.load(Path(result['run_dir'])/'result.pt',map_location='cpu',weights_only=False)
    refs[precision]=torch.cat([data['coef_final'],data['sol_final']],1).double()
   cfg.runtime_metadata['fused_guidance']=True
   for b in [1,3]:
    pred,_,receipt=fast_sample(cfg,truths[offset],bundle,range(b));pred=pred[:1].double()
    for j,field in enumerate(['a','u']):
     gt=truths[offset].coef.double() if j==0 else truths[offset].sol.double()
     z=pred[:,j:j+1]
     row=dict(task=task,offset=offset,batch_size=b,field=field,seconds=receipt['seconds'])
     for precision,ref in refs.items():
      ref=ref[:,j:j+1]
      row[precision+'_prediction_relative_difference']=float(torch.linalg.vector_norm(z-ref)/torch.linalg.vector_norm(ref))
      row[precision+'_rel_l2']=float(torch.linalg.vector_norm(ref-gt)/torch.linalg.vector_norm(gt))
     row['fast_rel_l2']=float(torch.linalg.vector_norm(z-gt)/torch.linalg.vector_norm(gt))
     row['strict_error_delta_pp']=100*(row['fast_rel_l2']-row['strict_rel_l2'])
     assert row['tf32_prediction_relative_difference']<.005,row
     assert row['strict_prediction_relative_difference']<.005,row
     rows.append(row)
   print('VALIDATED',task,offset,flush=True)
   write(args.output/'checks.json',rows)
 write(args.output/'complete.json',dict(complete=True,inputs=[1100,1101,1102,1103],tasks=['forward','inverse','both'],rows=len(rows),
  max_fast_vs_stock_tf32=max(x['tf32_prediction_relative_difference'] for x in rows),
  max_fast_vs_stock_strict=max(x['strict_prediction_relative_difference'] for x in rows),
  max_abs_error_delta_pp=max(abs(x['strict_error_delta_pp']) for x in rows),
  script_sha256=digest(__file__)))

if __name__=='__main__':main()

"""Independently recompute NS errors/observation MSE and export all fixed cells."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parent))
from run_ns_main_revision_0909 import EVAL, SETTINGS, digest, write


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(a):
    torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='timing':
        rows=[]
        for method in ['FM4PDE','DiffusionPDE']:
            for steps in [100,1000]:
                files=sorted((a.results/'nsnonbounded').glob(f'{method}_{steps}_*.json'))
                files=[p for p in files if '_contended_' not in p.name]
                data=[json.loads(p.read_text()) for p in files]
                assert len(data)==20 and len({x['sample_id'] for x in data})==20
                assert all(x['uncontended'] and x['finite'] and x['nfe']==(steps if method=='FM4PDE' else 2*steps-1) for x in data)
                seconds=np.array([x['seconds'] for x in data])
                rows.append(dict(pde='nsnonbounded',method=method,steps=steps,n=20,
                    mean_seconds=float(seconds.mean()),sd_seconds=float(seconds.std(ddof=1)),
                    min_seconds=float(seconds.min()),max_seconds=float(seconds.max()),gpu_uuid=data[0]['uuid']))
        assert len({x['gpu_uuid'] for x in rows})==1
        save_csv(a.output/'ns_timing_summary.csv',rows);write(a.output/'ns_timing_summary.json',rows)
        return
    rows=[];checks=[]
    for dist in ['id','smooth','rough']:
        for setting in SETTINGS:
            seen=[]
            for path in sorted((a.results/dist/setting).glob('offset*.pt')):
                receipt=json.loads(path.with_suffix('.json').read_text())
                assert digest(path)==receipt['result_sha256']
                d=torch.load(path,map_location='cpu',weights_only=False)
                pred=d['predictions'].numpy().astype('float64');truth=d['truths'].numpy().astype('float64')
                mask=d['masks'].numpy().astype('float64')
                assert np.isfinite(pred).all() and np.isfinite(truth).all()
                errors=np.sqrt(np.square(pred-truth).sum((2,3))/np.square(truth).sum((2,3)))
                counts=mask.sum((2,3))
                obs=np.square((pred-truth)*mask).sum((2,3))/np.maximum(counts,1)
                assert receipt['nfe']==100 and receipt['dist']==dist and receipt['setting']==setting
                for j,i in enumerate(receipt['ids']):
                    reported=receipt['rows'][j]
                    assert np.allclose(errors[j],[reported['error_a'],reported['error_u']],rtol=1e-7,atol=1e-9)
                    assert np.allclose(obs[j],[reported['obs_a'],reported['obs_u']],rtol=2e-5,atol=1e-10)
                    assert np.isfinite(reported['pde_mse'])
                    rows.append(dict(dist=dist,setting=setting,offset=i,error_a=errors[j,0],error_u=errors[j,1],
                        obs_a=obs[j,0],obs_u=obs[j,1],pde_mse=reported['pde_mse'],
                        observed_a=int(counts[j,0]),observed_u=int(counts[j,1]),nfe=receipt['nfe'],
                        tf32=receipt['tf32'],gpu=receipt['gpu'],
                        runtime_batch_size=receipt['batch_size'],worker=receipt['worker'],result=str(path)))
                    seen.append(i)
            assert sorted(seen)==EVAL,(dist,setting,len(seen))
            checks.append(dict(dist=dist,setting=setting,n=len(seen),exact_ids=True))
    summaries=[]
    for dist in ['id','smooth','rough']:
        for setting in SETTINGS:
            cell=[r for r in rows if r['dist']==dist and r['setting']==setting]
            row=dict(dist=dist,setting=setting,n=len(cell))
            for metric in ['error_a','error_u','obs_a','obs_u','pde_mse']:
                values=np.array([r[metric] for r in cell])
                row[metric+'_mean']=float(values.mean());row[metric+'_sd']=float(values.std(ddof=1))
            summaries.append(row)
    save_csv(a.output/'ns_main_per_sample.csv',rows);save_csv(a.output/'ns_main_summary.csv',summaries)
    write(a.output/'ns_main_summary.json',summaries)
    write(a.output/'ns_main_validation.json',dict(status='pass',examples=15000,cells=checks,
        reductions='NumPy float64 independent relative L2 and masked observation MSE; sample SD ddof=1',
        all_predictions_finite=True,all_nfe_100=True,all_result_hashes_match=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['main','timing'])
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

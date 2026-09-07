"""Audit saved NS loss-exchange predictions and compute physical diagnostics.

Partial exports are explicitly marked and cannot certify the formal study.
No new sampling, result selection, or modification of source receipts occurs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv,gzip,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_ns_loss_study import sha,write,TASKS,VARIANTS
from ns_loss_exchange import fm_pde_loss,diffusion_pde_loss
from spectral_diagnostics import spectral_record,self_check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--baselines',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import torch
    from sampling.config import AblationConfig
    torch.set_num_threads(2)
    source=json.loads((args.inputs/'source.json').read_text());data=np.load(args.inputs/'fields_masks.npz')
    assert source['joint_shared_mask'] and sha(args.inputs/'fields_masks.npz')==source['fields_sha256']
    protocol=json.loads((args.results/'protocol.json').read_text());ph=sha(args.results/'protocol.json')
    assert protocol['source_sha256']==sha(args.inputs/'source.json')
    for name,digest in protocol['code_sha256'].items():assert sha(ROOT/name)==digest,name
    ids=source['evaluation_ids'];seeds=source['inference_seeds'];assert len(ids)==32 and seeds==[0,1,2]
    expected={(t,m,n,e,i,s) for t in TASKS for m,n,e in VARIANTS for i in ids for s in seeds}
    assert len(expected)==1728
    statuses=Counter();seen=set();metrics=[];spectra=[];traces=[];hashes={};examples={}
    params={k:torch.tensor([v],dtype=torch.float64) for k,v in source['pde_params'].items()}
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    def ground_truth(i):return [torch.from_numpy(data[f'{f}_{i}']) for f in ['a','u']]
    def masks(task,i):
        ma=torch.from_numpy(data[f'mask_a_{i}']);mu=torch.from_numpy(data[f'mask_u_{i}'])
        if task=='both':mu=ma.clone()
        if task=='forward':mu=torch.zeros_like(mu)
        if task=='inverse':ma=torch.zeros_like(ma)
        return [ma,mu]
    for path in sorted((args.results/'results').rglob('*.json')):
        r=json.loads(path.read_text());key=tuple(r[k] for k in ['task','method','steps','exchange','sample_id','seed'])
        assert key in expected and key not in seen and r['protocol_sha256']==ph,key
        seen.add(key);statuses[r['status']]+=1
        t,m,n,e,i,s=key
        if r['status']=='unstable':
            assert r.get('error');continue
        tensor=path.with_suffix('.pt');assert sha(tensor)==r['prediction_sha256'];hashes[str(tensor)]=r['prediction_sha256']
        d=torch.load(tensor,map_location='cpu',weights_only=False)
        assert all(d['receipt'][k]==v for k,v in r.items() if k!='prediction_sha256')
        assert r['nfe']==(n if m=='FM4PDE' else 2*n-1)
        dtype=torch.float32 if m=='FM4PDE' else torch.float64
        for f,pred,truth,mask,gt,mm in zip(['a','u'],d['prediction'],d['truth'],d['masks'],ground_truth(i),masks(t,i)):
            assert pred.dtype==truth.dtype==dtype and tuple(pred.shape)==(1,1,128,128)
            assert torch.equal(truth,gt.to(dtype)) and torch.equal(mask,mm.to(dtype))
            assert int(mask.sum()) in [0,500]
        finite=all(bool(torch.isfinite(x).all()) for x in d['prediction'])
        assert finite==(r['status']=='complete')
        if not finite:continue
        a,u=[x.double() for x in d['prediction']]
        cfg=AblationConfig(**source['fm_configs'][t])
        with torch.no_grad():
            fm=float(fm_pde_loss(a,u,cfg,params));dm=float(diffusion_pde_loss(a,u))
        assert np.isfinite([fm,dm]).all() and min(fm,dm)>=0
        row={k:r[k] for k in ['task','method','steps','exchange','sample_id','seed','seconds','nfe']}
        row.update(fm_residual_mse_f64=fm,diffusion_spatial_loss_f64=dm)
        for j,(f,pred,truth,mask) in enumerate(zip(['a','u'],[a,u],d['truth'],d['masks'])):
            gt=truth.double();rel=float((pred-gt).norm()/gt.norm())
            assert np.isclose(rel,r['relative_l2'][j],rtol=1e-6,atol=2e-8)
            row[f'rel_l2_{f}']=rel;row[f'obs_rel_l2_{f}']=float(((pred-gt)*mask).norm()/(gt*mask).norm()) if bool(mask.any()) else None
            spec=spectral_record(pred[0,0].numpy(),gt[0,0].numpy(),'periodic_fft')
            assert np.isclose(spec['rel_l2'],rel,rtol=2e-12)
            spectra.append(dict(**{k:row[k] for k in ['task','method','steps','exchange','sample_id','seed']},field=f,**spec))
            if i==ids[0] and s==0:
                tag=f'{t}_{m}_{n}_{int(e)}_{f}'
                examples[tag]=pred[0,0].numpy();examples[t+'_truth_'+f]=gt[0,0].numpy();examples[t+'_mask_'+f]=mask[0,0].numpy()
        metrics.append(row)
        assert len(d['trace'])==n and [x['step'] for x in d['trace']]==list(range(n))
        c=source['fm_configs'][t] if m=='FM4PDE' else source['diffusion_configs'][f'{t}_{n}']['config']['generate']
        za,zu=float(c['zeta_obs_a']),float(c['zeta_obs_u'])
        if t=='forward':zu=0.
        if t=='inverse':za=0.
        for point in d['trace']:
            obs=za*point['obs_a_gradient_norm']+zu*point['obs_u_gradient_norm']
            pde=point['zeta_pde']*point['pde_gradient_norm']
            traces.append(dict(**{k:row[k] for k in ['task','method','steps','exchange','sample_id','seed']},**point,
                weighted_pde_norm=pde,sum_weighted_observation_norms=obs,component_norm_ratio=pde/obs if obs>0 else None))
    baseline_protocol=json.loads((args.baselines/'protocol.json').read_text());bh=sha(args.baselines/'protocol.json')
    assert baseline_protocol['source_sha256']==sha(args.inputs/'source.json') and baseline_protocol['tf32'] is False
    assert json.loads((args.baselines/'complete.json').read_text())==dict(status='complete',calls=288,protocol_sha256=bh)
    baseline_rows=[]
    for t in TASKS:
        for m in ['recfno','senseiver','voronoicnn']:
            folder=args.baselines/t/m;pilot=json.loads((folder/'pilot.json').read_text());assert pilot['status']=='pass' and pilot['protocol_sha256']==bh
            assert all(c['repeat']==c['hidden']==0 and c['sensitivity']>0 for c in pilot['checks'])
            assert json.loads((folder/'complete.json').read_text())['calls']==32
            for i in ids:
                path=folder/f'sample{i}.pt';r=json.loads(path.with_suffix('.json').read_text());assert r['protocol_sha256']==bh and sha(path)==r['sha256']
                hashes[str(path)]=r['sha256'];d=torch.load(path,map_location='cpu',weights_only=False)
                assert torch.isfinite(d['prediction']).all() and d['prediction'].dtype==torch.float32
                gt=ground_truth(i);mm=masks(t,i)
                fields=['u'] if t=='forward' else ['a'] if t=='inverse' else ['a','u']
                expected_truth=gt[1] if t=='forward' else gt[0] if t=='inverse' else torch.cat(gt,1)
                obs=gt[0]*mm[0] if t=='forward' else gt[1]*mm[1] if t=='inverse' else torch.cat([x*y for x,y in zip(gt,mm)],1)
                mask=mm[0] if t=='forward' else mm[1] if t=='inverse' else torch.cat(mm,1)
                assert torch.equal(d['truth'],expected_truth.float()) and torch.equal(d['observations'],obs.float()) and torch.equal(d['mask'],mask.float())
                for j,f in enumerate(fields):
                    pred=d['prediction'][0,j].double().numpy();truth=d['truth'][0,j].double().numpy()
                    spec=spectral_record(pred,truth,'periodic_fft');assert np.isclose(spec['rel_l2'],r['relative_l2'][j],rtol=2e-12)
                    meta=dict(task=t,method={'recfno':'RecFNO','senseiver':'Senseiver','voronoicnn':'VoronoiCNN'}[m],steps=1,exchange=False,sample_id=i,seed=0,field=f)
                    spectra.append(dict(**meta,**spec));baseline_rows.append(dict(**meta,rel_l2=spec['rel_l2']))
                    if i==ids[0]:examples[f'{t}_{meta["method"]}_1_0_{f}']=pred
    complete=seen==expected
    if args.require_complete:
        assert complete, f'Only {len(seen)}/1728 calls have completed'
        for shard in [0,1]:
            rr=json.loads((args.results/f'complete_{shard}.json').read_text());assert rr==dict(status='complete',protocol_sha256=ph,jobs=864)
    for name,rows in [('ns_loss_per_run.csv',metrics),('ns_baseline_per_field.csv',baseline_rows)]:
        with (args.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name,rows in [('ns_frequency_records.json.gz',spectra),('ns_guidance_traces.json.gz',traces)]:
        with gzip.open(args.output/name,'wt') as f:json.dump(rows,f,default=convert,allow_nan=False)
    np.savez_compressed(args.output/'ns_example_fields.npz',**examples)
    write(args.output/'ns_audit_manifest.json',dict(status='complete' if complete else 'partial',calls_verified=len(seen),expected_calls=1728,
        outcome_counts=dict(statuses),finite_predictions=len(metrics),baseline_predictions_verified=288,baseline_field_metrics=len(baseline_rows),
        protocol_sha256=ph,baseline_protocol_sha256=bh,source_hashes=hashes,checks=self_check(),script_sha256=sha(Path(__file__)),
        scope='Prediction hashes, native arithmetic, source truths/masks, NFE, independent physical errors, both PDE scalars in float64, and exact Fourier Parseval decomposition. Partial results cannot enter final loss-exchange comparisons.',
        trace_ratio='Norm of weighted PDE component divided by sum of norms of weighted observation components; not the norm ratio of summed vectors and not a direct update-size ratio.',
        outputs={f.name:sha(f) for f in args.output.iterdir() if f.name!='ns_audit_manifest.json' and f.is_file()}))
    print(json.dumps(dict(status='complete' if complete else 'partial',calls=len(seen),outcomes=dict(statuses),baselines=288)))


if __name__=='__main__':main()

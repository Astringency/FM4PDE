"""Offline truth-only audit of all temporal residuals; never sampling input.

Use real ID/Smooth/Rough data and the normal data/parameter loaders. Report
each component's MSE, its sum, and concatenated RMS separately. Sparse
near-endpoint checks use the production observation loader and retain its
metadata. Also freeze small true-data extracts for local inspection.
"""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,sys,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sampling.config import load_config
from sampling.data import load_ground_truth,attach_near_endpoint_observations
from sampling.losses import _pde_params_with_residual_options
from sampling.pde_residuals import compute_pde_residual

PDES=['heat','wave','advection_diffusion','reaction_diffusion','shallow_water','nsnonbounded','burger']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    return v

def main(args):
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    rows=[];details=[];errors=[]
    for pde in args.pdes:
        for dist in args.distributions:
            try:
                cfg=load_config(ROOT/f'configs/main/both/{pde}.yaml',overrides=dict(
                    device='cpu',dtype='float64',test_type=dist,batch_size=args.samples,offset=0,
                    residual_mode='full_trajectory_fd',allow_synthetic_data=False))
                cfg.data_path=cfg.data_paths[dist]
                gt=load_ground_truth(cfg);assert not gt.metadata['synthetic']
                params=_pde_params_with_residual_options(gt.pde_params,cfg)
                cache=args.output/f'{pde}_{dist}_truth.pt'
                torch.save(dict(coef=gt.coef,sol=gt.sol,params=params,metadata=gt.metadata),cache)
                modes=['full_trajectory_fd'] if pde=='burger' else ['full_trajectory_fd','endpoint_secant','hermite_bridge','near_endpoint_full','near_endpoint_sparse']
                for mode in modes:
                    pp=copy.deepcopy(params)
                    if mode.startswith('near_endpoint'):
                        nearcfg=copy.deepcopy(cfg);nearcfg.residual_mode='near_endpoint_temporal'
                        a=torch.ones_like(gt.coef);u=torch.ones_like(gt.sol)
                        if mode.endswith('sparse'):
                            a.zero_();u.zero_()
                            gen=torch.Generator().manual_seed(20260910)
                            for sample in range(args.samples):
                                for m in [a,u]:
                                    ix=torch.randperm(m.shape[-1]*m.shape[-2],generator=gen)[:500]
                                    m[sample].reshape(m.shape[1],-1)[:,ix]=1
                        near_gt=attach_near_endpoint_observations(nearcfg,copy.deepcopy(gt),SimpleNamespace(coef=a,sol=u))
                        pp=_pde_params_with_residual_options(near_gt.pde_params,nearcfg)
                    actual_mode='near_endpoint_temporal' if mode.startswith('near_endpoint') else mode
                    out=compute_pde_residual(pde,gt.coef,gt.sol,pde_params=pp,residual_mode=actual_mode)
                    parts=out.components or {'interior':out.residual}
                    for i in range(args.samples):
                        values={k:(float(v[i].square().mean()) if v is not None and v[i].numel() else 0.) for k,v in parts.items()}
                        total=values.get('interior',0.)+cfg.bc_weight*values.get('boundary',0.)+cfg.endpoint_bc_weight*values.get('endpoint',0.)
                        row=dict(pde=pde,distribution=dist,sample=i,mode=mode,
                            interior_mse=values.get('interior',0.),boundary_mse=values.get('boundary',0.),
                            endpoint_mse=values.get('endpoint',0.),component_mse_sum=total,
                            concatenated_rms=float(out.residual[i].square().mean().sqrt()))
                        assert np.isfinite(list(row.values())[4:]).all()
                        rows.append(row)
                    details.append(dict(pde=pde,distribution=dist,mode=mode,source=cfg.data_path,
                        truth_sha256=sha(cache),data_metadata=gt.metadata,residual_metadata=out.metadata))
                    print(pde,dist,mode,'mean MSE',np.mean([r['component_mse_sum'] for r in rows[-args.samples:]]),flush=True)
            except Exception as exc:
                errors.append(dict(pde=pde,distribution=dist,error=repr(exc),traceback=traceback.format_exc()))
                print('FAILED',pde,dist,repr(exc),flush=True)
    if rows:
        with (args.output/'per_sample.csv').open('w') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'audit.json').write_text(json.dumps(serial(dict(
        status='incomplete' if errors else 'complete',samples=args.samples,rows=len(rows),
        pdes=args.pdes,distributions=args.distributions,precision='float64 loaded from stored fields',
        script_sha256=sha(__file__),residual_sha256=sha(ROOT/'sampling/pde_residuals.py'),
        python=sys.version,torch=torch.__version__,details=details,errors=errors)),indent=2)+'\n')
    if errors:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pdes',nargs='+',default=PDES)
    p.add_argument('--distributions',nargs='+',default=['id','smooth','rough'])
    p.add_argument('--samples',type=int,default=3)
    main(p.parse_args())

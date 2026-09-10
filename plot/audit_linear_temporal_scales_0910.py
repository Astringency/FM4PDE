"""Separate temporal approximation, spatial discretization, and data precision.

Exact Fourier evolution of true initial fields under (a) the generator's
spectral operator and (b) the residual's centered-difference operator.
No learned model is used, and no results are intended for the manuscript.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sampling.pde_residuals import compute_pde_residual,_rhs_time_dependent
from audit_temporal_truth_0910 import sha,serial

def main(args):
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    rows=[];checks=[]
    for pde in ['heat','advection_diffusion','wave']:
        for dist in ['id','smooth','rough']:
            path=args.truths/f'{pde}_{dist}_truth.pt'
            if not path.exists():continue
            cache=torch.load(path,map_location='cpu',weights_only=False)
            q0=cache['coef'].double();real_qT=cache['sol'].double()
            b,c,h,w=q0.shape;assert h==w
            pp={k:v for k,v in cache['params'].items() if k not in
                ['trajectory','trajectory_time_values','trajectory_is_observed_ground_truth']}
            def param(key,default=None):
                v=pp.get(key,default);assert v is not None,(pde,key)
                return torch.as_tensor(v,dtype=q0.dtype).reshape(-1,1,1,1)
            k=2*torch.pi*torch.fft.fftfreq(h,d=1/h,dtype=q0.dtype)
            kx=k.reshape(1,1,h,1);ky=k.reshape(1,1,1,w)
            spectral_lap=-(kx*kx+ky*ky)
            discrete_lap=-4*h*h*(torch.sin(kx/(2*h))**2+torch.sin(ky/(2*h))**2)
            for operator in ['spectral','centered_difference']:
                lap=spectral_lap if operator=='spectral' else discrete_lap
                if pde=='heat': symbol=param('alpha')*lap
                elif pde=='advection_diffusion':
                    dx=kx if operator=='spectral' else h*torch.sin(kx/h)
                    dy=ky if operator=='spectral' else h*torch.sin(ky/h)
                    symbol=-1j*(param('b_x')*dx+param('b_y')*dy)+param('kappa')*lap
                else:freq=param('c',1.)*torch.sqrt(-lap)
                qhat=torch.fft.fft2(q0)
                def evolve(t):
                    if pde!='wave':return torch.fft.ifft2(qhat*torch.exp(symbol*t)).real
                    uh,vh=qhat[:,:1],qhat[:,1:]
                    cos=torch.cos(freq*t);sin=torch.sin(freq*t)
                    safe=torch.where(freq>0,freq,torch.ones_like(freq))
                    sin_over=torch.where(freq>0,sin/safe,torch.full_like(freq,t))
                    return torch.fft.ifft2(torch.cat([uh*cos+vh*sin_over,-uh*freq*sin+vh*cos],dim=1)).real
                def exact_rhs(q):
                    if pde!='wave':return torch.fft.ifft2(torch.fft.fft2(q)*symbol).real
                    return torch.cat([q[:,1:],torch.fft.ifft2(torch.fft.fft2(q[:,:1])*param('c',1.)**2*lap).real],dim=1)
                T=float(param('T',1.)[0]);assert torch.all(param('T',1.)==T)
                delta=exact_rhs(q0)-_rhs_time_dependent(pde,q0,pp)
                check=dict(pde=pde,distribution=dist,operator=operator,source_sha256=sha(path),
                    rhs_discrepancy_rms=float(delta.square().mean().sqrt()),
                    endpoint_reconstruction_relative_error=float((evolve(T)-real_qT).norm()/real_qT.norm()))
                if operator=='centered_difference':assert check['rhs_discrepancy_rms']<1e-8,check
                checks.append(check)
                for horizon in [1.,.1,.01,.001,.0001]:
                    params=dict(pp,T=horizon)
                    qT=evolve(horizon)
                    dt=horizon/10
                    params['near_endpoint_temporal']=dict(q_dt=evolve(dt),q_T_minus_dt=evolve(horizon-dt),
                        dt=dt,mask_0=torch.ones_like(q0),mask_T=torch.ones_like(qT))
                    for mode in ['endpoint_secant','hermite_bridge','near_endpoint_temporal']:
                        out=compute_pde_residual(pde,q0,qT,pde_params=params,residual_mode=mode)
                        vals={k:float(v.square().mean()) if v is not None else 0. for k,v in out.components.items()}
                        rows.append(dict(pde=pde,distribution=dist,operator=operator,horizon=horizon,
                            mode=mode,interior_rms=np.sqrt(vals['interior']),
                            endpoint_rms=np.sqrt(vals.get('endpoint',0.))))
    with (args.output/'time_refinement.csv').open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'operator_checks.json').write_text(json.dumps(serial(dict(
        script_sha256=sha(__file__),residual_sha256=sha(ROOT/'sampling/pde_residuals.py'),
        checks=checks,rows=len(rows))),indent=2)+'\n')
    for c in checks:
        if c['distribution']=='id':print(c['pde'],c['operator'],'RHS gap',c['rhs_discrepancy_rms'],'data rel gap',c['endpoint_reconstruction_relative_error'])
    print('Wrote',len(rows),'analytic time-refinement conditions.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--truths',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

"""Independent nonlinear-operator and true-trajectory diagnostics, outside paper."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sampling.pde_residuals import _rhs_time_dependent,_time_scale_field,compute_pde_residual
from audit_temporal_truth_0910 import sha,serial

def main(args):
 torch.set_num_threads(2);results=[]
 for dist in ['id','smooth','rough']:
  d=torch.load(args.truths/f'shallow_water_{dist}_truth.pt',weights_only=False);a,u,p=d['coef'],d['sol'],d['params'];T=_time_scale_field(p,a,1.)[0]
  f0=_rhs_time_dependent('shallow_water',a,p);fT=_rhs_time_dependent('shallow_water',u,p)
  item=dict(pde='shallow_water',distribution=dist,truth_min_depth=float(p['trajectory'][:,:,0].min()),bridge=[])
  for s in [.25,.5,.75]:
   q=(2*s**3-3*s*s+1)*a+(s**3-2*s*s+s)*T*f0+(-2*s**3+3*s*s)*u+(s**3-s*s)*T*fT
   item['bridge'].append(dict(s=s,min_depth=float(q[:,0].min()),nonpositive_fraction=float((q[:,0]<=0).double().mean())))
  results.append(item)
  d=torch.load(args.truths/f'nsnonbounded_{dist}_truth.pt',weights_only=False);q=d['coef'];p=d['params'];n=q.shape[-1]
  k=2*torch.pi*torch.fft.fftfreq(n,d=1/n,dtype=q.dtype);kx=k[None,None,:,None];ky=k[None,None,None,:];lap=kx*kx+ky*ky;lap[...,0,0]=1
  wh=torch.fft.fft2(q,norm='forward');psi=wh/lap
  U=torch.fft.ifft2(1j*ky*psi,norm='forward');V=torch.fft.ifft2(-1j*kx*psi,norm='forward');wx=torch.fft.ifft2(1j*kx*wh,norm='forward');wy=torch.fft.ifft2(1j*ky*wh,norm='forward')
  mask=(kx.abs()<=2*torch.pi*(2/3)*(n//2))&(ky.abs()<=2*torch.pi*(2/3)*(n//2))
  x=torch.arange(n,dtype=q.dtype)/n;xy=x[:,None]+x[None,:]
  f=.1*(torch.sin(2*torch.pi*xy)+torch.cos(2*torch.pi*xy));nu=p['nu'].reshape(-1,1,1,1)
  rhs=torch.fft.ifft2((-(torch.fft.fft2(U*wx+V*wy,norm='forward'))+torch.fft.fft2(f,norm='forward'))*mask-nu*lap*wh,norm='forward').real
  actual=_rhs_time_dependent('nsnonbounded',q,p);diff=rhs-actual
  results.append(dict(pde='nsnonbounded',distribution=dist,rhs_rms=float(rhs.square().mean().sqrt()),generator_complex_vs_real_spectral_rhs_rms=float(diff.square().mean().sqrt()),difference_max=float(diff.abs().max())))
  d=torch.load(args.truths/f'reaction_diffusion_{dist}_truth.pt',weights_only=False);q=d['coef'];p=d['params'];b,c,h,w=q.shape
  # Independent ghost-cell finite-volume stencil matches the sparse Neumann matrix.
  arr=q.numpy();pad=np.pad(arr,((0,0),(0,0),(1,1),(1,1)),mode='edge')
  dx=2/w;dy=2/h;lap=(pad[:,:,1:-1,2:]-2*arr+pad[:,:,1:-1,:-2])/dx**2+(pad[:,:,2:,1:-1]-2*arr+pad[:,:,:-2,1:-1])/dy**2
  u,v=arr[:,0:1],arr[:,1:2]
  par=lambda k:p[k].numpy().reshape(-1,1,1,1)
  independent=np.concatenate([par('D_u')*lap[:,0:1]+u-u**3-par('k')-v,par('D_v')*lap[:,1:2]+u-v],axis=1)
  gap=torch.as_tensor(independent)-_rhs_time_dependent('reaction_diffusion',q,p)
  results.append(dict(pde='reaction_diffusion',distribution=dist,rhs_discrepancy_rms=float(gap.square().mean().sqrt()),rhs_rms=float(torch.as_tensor(independent).square().mean().sqrt())))
  for pde in ['reaction_diffusion','nsnonbounded','shallow_water']:
   d=torch.load(args.truths/f'{pde}_{dist}_truth.pt',weights_only=False);params=d['params'];trajectory=params['trajectory']
   times=np.asarray(params['trajectory_time_values'],dtype=float);short=[]
   for end in [1,2,5,10]:
    pp={k:v for k,v in params.items() if k not in ['trajectory','trajectory_time_values']};pp['T']=float(times[end]-times[0])
    for mode in ['endpoint_secant','hermite_bridge']:
     out=compute_pde_residual(pde,trajectory[:,0],trajectory[:,end],pde_params=pp,residual_mode=mode)
     short.append(dict(horizon=pp['T'],mode=mode,component_mse_sum=sum(float(v.square().mean()) for v in out.components.values() if v is not None)))
   results.append(dict(pde=pde,distribution=dist,stored_subintervals=short))
 args.output.mkdir(parents=True,exist_ok=True)
 (args.output/'operator_checks.json').write_text(json.dumps(serial(dict(script_sha256=sha(__file__),checks=results)),indent=2)+'\n')
 for r in results:print(r)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--truths',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

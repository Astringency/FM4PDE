"""Enrich low DCT proposals using the full learned-generator Gauss--Newton action.

The extra vectors include correlations with all remaining latent coordinates.
Their construction uses observations only through the fixed quadratic energy,
and never uses reconstruction errors or formal-cohort data.
"""
import argparse,time,json,subprocess
from pathlib import Path
import torch
from experiments.build_fm_tilt_geometry import setup
from experiments.fm_tilt_geometry import full_trajectory_vjp,fit_reference
from scripts.train.resume_study import file_sha,write


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['source','pilot-inputs','output','geometry']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-extra',type=int,default=64)
    p.add_argument('--epsilon',type=float,default=.02)
    args=p.parse_args()
    assert args.output.is_absolute() and '/outputs/' in str(args.output)
    out=args.output/'geometry_enriched';out.mkdir(exist_ok=True)
    assert not (out/'complete.json').exists()
    a,ip,_=setup(args.source,args.pilot_inputs,1103,out)
    old=torch.load(args.geometry,map_location='cpu',weights_only=False)
    basis=old['basis'].double();j=old['jacobian'].double().flatten(1)
    values,vectors=torch.linalg.eigh(old['hessian'])
    selected=torch.where(values>1.)[0][-args.max_extra:]
    projected_j=vectors[:,selected].T@j
    center=old['center'].cuda()
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    endpoint,_=a.target(center.repeat(4,1,1,1))
    assert torch.cuda.max_memory_allocated()<40*2**30
    with torch.enable_grad():
        zero=torch.zeros_like(endpoint).double().requires_grad_(True)
        gzero,=torch.autograd.grad(a.potential(zero).sum(),zero)
    actions=[]
    for begin in range(0,len(selected),4):
        direction=projected_j[begin:begin+4].reshape(-1,2,128,128).cuda()
        size=len(direction)
        if size<4:
            direction=torch.cat([direction,torch.zeros_like(direction[:1]).repeat(4-size,1,1,1)])
        with torch.enable_grad():
            direction.requires_grad_(True)
            gd,=torch.autograd.grad(a.potential(direction).sum(),direction)
        seed=(gd-gzero).to(endpoint)
        action=full_trajectory_vjp(a,seed)[:size].flatten(1).double().cpu()
        actions.append(action)
        progress=dict(stage='full_Gauss_Newton_actions',vectors=min(begin+4,len(selected)),
            total=len(selected),seconds=time.monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated())
        assert progress['peak_bytes']<40*2**30
        write(out/'progress.json',progress);print('ACTION',progress,flush=True)
    action=torch.cat(actions)
    complement=action-(action@basis.T)@basis
    complement-= (complement@basis.T)@basis
    _,singular,vt=torch.linalg.svd(complement,full_matrices=False)
    keep=singular>singular.max()*1e-6
    extra=vt[keep]
    new_basis=torch.cat([basis,extra])
    orthogonality=(new_basis@new_basis.T-torch.eye(len(new_basis))).abs().max()
    assert orthogonality<2e-6,float(orthogonality)
    # Save the completed curvature action before the new finite-difference pass.
    torch.save(dict(actions=action,singular_values=singular,basis=new_basis),out/'subspace.pt')
    new_j=[]
    for begin in range(0,len(extra),2):
        directions=extra[begin:begin+2].to(center).reshape(-1,2,128,128)
        size=len(directions)
        if size<2:
            directions=torch.cat([directions,torch.zeros_like(directions[:1])])
        with torch.no_grad():
            r,_=a.target(torch.cat([center+args.epsilon*directions,center-args.epsilon*directions]))
        derivative=(r[:2].double()-r[2:].double())/(2*args.epsilon)
        new_j.append(derivative[:size].cpu())
        if (begin+2)%16==0 or begin+2>=len(extra):
            progress=dict(stage='enriched_jacobian',vectors=min(begin+2,len(extra)),
                total=len(extra),seconds=time.monotonic()-start)
            write(out/'progress.json',progress);print('JACOBIAN',progress,flush=True)
    jacobian=torch.cat([old['jacobian'],torch.cat(new_j)])
    ref,report=fit_reference(new_basis,jacobian,center,old['endpoint'].cuda(),a)
    protocol=dict(old['protocol'],parent_geometry_sha256=file_sha(args.geometry),
        rank=len(new_basis),original_rank=len(basis),extra_rank=len(extra),
        subspace='Original low DCT modes plus orthogonal full-generator Gauss-Newton directions',
        max_extra=args.max_extra,epsilon=args.epsilon,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    torch.save(dict(basis=new_basis.float(),jacobian=jacobian,center=old['center'],endpoint=old['endpoint'],
        reference_root=report.pop('root'),hessian=report.pop('hessian'),protocol=protocol),out/'geometry.pt')
    report.update(seconds=time.monotonic()-start,geometry_sha256=file_sha(out/'geometry.pt'),
        rank=len(new_basis),extra_rank=len(extra),orthogonality_error=float(orthogonality),
        complement_singular_values=singular.tolist(),peak_bytes=torch.cuda.max_memory_allocated())
    write(out/'protocol.json',protocol);write(out/'complete.json',report)
    print('DONE',dict(seconds=report['seconds'],rank=report['rank'],peak_bytes=report['peak_bytes']),flush=True)


if __name__=='__main__':main()

"""Physics-preserving, joint input/solution distribution shift (1000 tests by default).

Run from the repository root with ``python -m data.DataGen.generate_joint_ood``.
ID calibration uses cases 100:1100, disjoint from the comparison's first 100.
No trained model or prediction is read by this generator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[2]
from data.DataGen.static_solvers import solve_static

PDES = ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger')
PROFILE = 'joint_ood'
VERSION = 1


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def git_commit(root):
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def id_name(pde):
    folder='burgers' if pde=='burger' else pde
    shape='128-128-10' if pde=='nsnonbounded' else '128-128'
    return f'{folder}/{pde}_test_10000-{shape}_id.mat'


def read_fields(pde,path,start=0,count=1000):
    """Read existing physical generator formats without model/baseline dependencies."""
    stop=start+count
    if pde in ('poisson','helmholtz','burger'):
        keys={'poisson':('f_data','phi_data'),'helmholtz':('f_data','psi_data'),
              'burger':('input','output')}[pde]
        data=loadmat(path,variable_names=list(keys))
        u=np.asarray(data[keys[1]][start:stop],dtype=np.float32)
        a=u[:,0] if pde=='burger' else np.asarray(data[keys[0]][start:stop],dtype=np.float32)
    else:
        with h5py.File(path,'r') as f:
            if pde=='darcy':
                def read(ds):
                    if ds.shape[0]==ds.shape[1]:return np.moveaxis(ds[...,start:stop],-1,0)
                    return ds[start:stop]
                a,u=read(f['thresh_a_data']),read(f['thresh_p_data'])
            else:a,u=f['w0'][start:stop],f['w'][start:stop,...,-1]
        a=np.asarray(a,dtype=np.float32);u=np.asarray(u,dtype=np.float32)
    assert len(a)==len(u)==count and np.isfinite(a).all() and np.isfinite(u).all()
    return a,u


def output_name(pde, count=1000):
    folder = 'burgers' if pde == 'burger' else pde
    shape = '128-128-10' if pde == 'nsnonbounded' else '128-128'
    return f'{folder}/{pde}_test_{count}-{shape}_{PROFILE}.mat'


def features(v):
    v = np.asarray(v, dtype=np.float64)
    axes = tuple(range(1, v.ndim))
    mean = v.mean(axis=axes)
    sd = v.std(axis=axes)
    grad = sum(np.mean(np.diff(v, axis=ax)**2, axis=axes) for ax in axes)
    return dict(mean=mean, rms=np.sqrt(np.mean(v*v, axis=axes)), sd=sd,
                relative_gradient=np.sqrt(grad)/np.maximum(sd, 1e-30))


def field_features(pde, a, u):
    out = {'a': features(a), 'u': features(u)}
    if pde == 'burger':
        out['u_terminal'] = features(u[:, -1])
    return out


def stats(x):
    return dict(mean=float(np.mean(x)), min=float(np.min(x)), max=float(np.max(x)),
                q005=float(np.quantile(x, .005)), q995=float(np.quantile(x, .995)))


def calibrate(args):
    import torch
    torch.set_num_threads(2)
    for pde in (PDES if args.pde == 'all' else (args.pde,)):
        path = Path(args.output_root)/'calibration'/f'{pde}.json'
        if path.exists():
            print('CALIBRATION EXISTS', path, flush=True)
            continue
        src = Path(args.data_root)/id_name(pde)
        a,u=read_fields(pde,src,start=100,count=1000)
        ff = field_features(pde, a, u)
        write_json(path, dict(pde=pde, source=str(src), source_sha256=sha256(src),
            indices=list(range(100,1100)), code_commit=git_commit(ROOT),
            stats={f:{k:stats(x) for k,x in fs.items()} for f,fs in ff.items()}))
        print('CALIBRATED', pde, flush=True)


def static_sample(pde, rng, calibration):
    n = 128
    if pde in ('poisson', 'helmholtz'):
        x = np.linspace(0,1,n)
        u = np.zeros((n,n))
        for _ in range(8):
            m, k = rng.integers(4,10,2)
            u += rng.normal()*np.outer(np.sin(m*np.pi*x), np.sin(k*np.pi*x))
        u[[0,-1],:] = 0; u[:,[0,-1]] = 0
        target = calibration['stats']['u']['rms']['mean']*rng.uniform(.8,1.2)
        u *= target/np.sqrt(np.mean(u*u))
        a = np.zeros_like(u)
        a[1:-1,1:-1] = (u[:-2,1:-1]+u[2:,1:-1]+u[1:-1,:-2]+u[1:-1,2:]-4*u[1:-1,1:-1])*(n-1)**2
        if pde == 'helmholtz':
            a += u  # k=1, exactly the existing generator's sign convention.
        replay, meta = solve_static(pde, a)
        meta['relative_replay_error'] = float(np.linalg.norm(replay-u)/np.linalg.norm(u))
        assert meta['relative_replay_error'] < 1e-9
        return a.astype('float32'), u.astype('float32'), meta
    x = (np.arange(n)+.5)/n
    xx, yy = np.meshgrid(x,x,indexing='ij')
    theta = rng.uniform(0,2*np.pi)
    cx, cy = rng.uniform(.38,.62,2)
    rx = (xx-cx)*np.cos(theta)+(yy-cy)*np.sin(theta)
    ry = -(xx-cx)*np.sin(theta)+(yy-cy)*np.cos(theta)
    # Thin, connected, closed barriers embedded in a high-permeability phase.
    radial = np.sqrt((rx/rng.uniform(.22,.31))**2+(ry/rng.uniform(.14,.22))**2)
    low = np.abs(radial-1) < rng.uniform(.045,.10)
    low |= ((np.abs(ry) < rng.uniform(.008,.016)) & (np.abs(rx)<.25))
    low &= (xx>.06)&(xx<.94)&(yy>.06)&(yy<.94)
    a = np.where(low,4.,12.)
    u, meta = solve_static('darcy',a)
    assert meta['minimum_edge_coefficient'] > 0
    return a.astype('float32'),u.astype('float32'),meta


def periodic_initial(pde, rng, calibration):
    x = np.arange(128)/128
    if pde == 'burger':
        v = sum(rng.normal()*np.cos(2*np.pi*k*x+rng.uniform(0,2*np.pi)) for k in range(2,6))
        v *= rng.uniform(.18,.28)/np.sqrt(np.mean(v*v))
        return (v+rng.choice([-1,1])*rng.uniform(.35,.55)).astype('float32')
    xx, yy = np.meshgrid(x,x,indexing='ij')
    v = np.zeros((128,128))
    for _ in range(12):
        kx, ky = rng.integers(-5,6,2)
        while not 3 <= np.hypot(kx,ky) <= 5:
            kx, ky = rng.integers(-5,6,2)
        v += rng.normal()*np.cos(2*np.pi*(kx*xx+ky*yy)+rng.uniform(0,2*np.pi))
    v -= v.mean()
    # Resolved intermediate modes survive viscous evolution at the original T=1.
    target = calibration['stats']['a']['rms']['q995']*rng.uniform(2.5,3.5)
    v *= target/np.sqrt(np.mean(v*v))
    return v.astype('float32')


def burger_batch(initial, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    source, dest = directory/'initial.mat', directory/'trajectory.mat'
    savemat(source,dict(initial=initial))
    def quote(p): return str(p).replace("'", "''")
    expr = (f"addpath('{quote(ROOT/'data/DataGen/static')}');"
            f"generate_burgers_from_initial('{quote(source)}','{quote(dest)}','{quote(args.chebfun_root)}');")
    subprocess.run([args.matlab,'-singleCompThread','-batch',expr], check=True)
    v = loadmat(dest)['trajectory'].astype('float32')
    assert np.isfinite(v).all() and np.array_equal(v[:,0], initial)
    # Periodic unforced Burgers conserves its spatial mean.
    assert np.max(np.abs(v.mean(-1)-initial.mean(-1)[:,None])) < 2e-5
    return v


def ns_batch(initial, args):
    import torch
    from data.DataGen.time_dependent.no_bound_ns.ns_2d import navier_stokes_2d
    w0 = torch.as_tensor(initial,device=args.device)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    grid = torch.arange(128,device=args.device)/128
    x,y = torch.meshgrid(grid,grid,indexing='ij')
    forcing = .1*(torch.sin(2*np.pi*(x+y))+torch.cos(2*np.pi*(x+y)))
    with torch.no_grad():
        vals = navier_stokes_2d(w0,forcing,.001,1.,1e-4,10)
    names = ('vx0','vy0','w','vx','vy','t')
    out = {k:v.detach().cpu().numpy().astype('float32') for k,v in zip(names,vals)}
    assert all(np.isfinite(v).all() for v in out.values())
    if args.device.startswith('cuda'):
        print('NS PEAK ALLOCATED',torch.cuda.max_memory_allocated(args.device),'bytes',flush=True)
    return out


def diagnose(pde, a, u, calibration):
    ff = field_features(pde,a,u)
    # Predeclared primary features: shape for the elliptic forcing problems,
    # phase fraction / solution amplitude for Darcy, amplitude for NS, and
    # conserved signed mean for Burgers. No model scores enter this decision.
    primary = {'poisson':('relative_gradient','relative_gradient'),
        'helmholtz':('relative_gradient','relative_gradient'),
        'darcy':('mean','rms'),'nsnonbounded':('rms','rms'),'burger':('mean','mean')}[pde]
    results = {}
    for field, fs in ff.items():
        key = primary[0 if field=='a' else 1]
        ref = calibration['stats'][field][key]
        outside = (fs[key] < ref['q005']) | (fs[key] > ref['q995'])
        results[field] = dict(primary_feature=key, outside_id_99_percent_fraction=float(outside.mean()),
            stats={k:stats(v) for k,v in fs.items()},
            mean_ratio_to_id={k:float(np.mean(v)/calibration['stats'][field][k]['mean'])
                              for k,v in fs.items() if abs(calibration['stats'][field][k]['mean'])>1e-12})
    return dict(fields=results, passed=all(v['outside_id_99_percent_fraction']>=.95 for v in results.values()),
        definition='At least 95% of new cases lie outside the ID calibration central 99% interval in each predeclared primary feature. Empirical shift, not a claim of disjoint mathematical support.')


def generate(args):
    import torch
    torch.set_num_threads(2)
    calibration_path = Path(args.output_root)/'calibration'/f'{args.pde}.json'
    calibration = json.loads(calibration_path.read_text())
    pde, count = args.pde,args.count
    assert count > 0 and args.batch_size > 0
    path = Path(args.data_root)/output_name(pde,count)
    receipt = path.with_suffix('.json')
    if path.exists():
        if receipt.exists() and json.loads(receipt.read_text())['sha256']==sha256(path):
            print('VERIFIED EXISTING',path,flush=True); return
        raise FileExistsError(f'Refusing to overwrite {path}')
    work = Path(args.output_root)/'generation'/f'{pde}_{count}'
    work.mkdir(parents=True,exist_ok=True)
    config = dict(profile=PROFILE,version=VERSION,pde=pde,count=count,seed=args.seed,
        batch_size=args.batch_size,calibration_sha256=sha256(calibration_path),
        code_commit=git_commit(ROOT),PDE_parameters={'nsnonbounded':dict(nu=.001,T=1.,dt=1e-4,forcing='0.1*(sin(2*pi*(x+y))+cos(2*pi*(x+y)))'),
        'burger':dict(nu=.01,T=1.,dt=1/127), 'darcy':dict(source=1.,phases=[4.,12.],boundary='zero Dirichlet'),
        'poisson':dict(operator='laplacian',boundary='zero Dirichlet'),
        'helmholtz':dict(operator='laplacian + k^2',k=1.,boundary='zero Dirichlet')}[pde])
    frozen = work/'config.json'
    if frozen.exists():
        previous=json.loads(frozen.read_text())
        assert {k:v for k,v in previous.items() if k!='code_commit'} == {k:v for k,v in config.items() if k!='code_commit'}
    else: write_json(frozen,config)
    started=time.time()
    for lo in range(0,count,args.batch_size):
        hi=min(lo+args.batch_size,count); dest=work/f'batch_{lo:04d}_{hi:04d}.npz'
        if dest.exists(): continue
        initials=[]; solutions=[]; meta=[]
        for i in range(lo,hi):
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,PDES.index(pde),i]))
            if pde in ('poisson','helmholtz','darcy'):
                a,u,m=static_sample(pde,rng,calibration);initials.append(a);solutions.append(u);meta.append(m)
            else: initials.append(periodic_initial(pde,rng,calibration))
        a=np.stack(initials)
        if pde=='nsnonbounded':
            extra=ns_batch(a,args);u=extra['w'][...,-1]
        elif pde=='burger':
            u=burger_batch(a,work/f'matlab_{lo:04d}_{hi:04d}',args);extra={}
        else: u=np.stack(solutions);extra={}
        assert np.isfinite(a).all() and np.isfinite(u).all()
        with dest.with_suffix('.tmp').open('wb') as f:np.savez(f,a=a,u=u,**extra)
        dest.with_suffix('.tmp').replace(dest)
        write_json(dest.with_suffix('.json'),dict(start=lo,stop=hi,sha256=sha256(dest),solver_checks=meta))
        print(f'GENERATED {pde} {hi}/{count} elapsed={time.time()-started:.1f}s',flush=True)
    batches=sorted(work.glob('batch_*.npz'))
    a=np.concatenate([np.load(p)['a'] for p in batches]);u=np.concatenate([np.load(p)['u'] for p in batches])
    assert len(a)==count
    diagnostic=diagnose(pde,a,u,calibration)
    write_json(work/'shift.json',diagnostic)
    if not diagnostic['passed']:
        raise ValueError(f'Joint OOD verification failed; inspect {work}/shift.json before evaluating any model')
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    seeds=np.arange(count,dtype=np.int64)
    if pde in ('poisson','helmholtz','burger'):
        names={'poisson':('f_data','phi_data'),'helmholtz':('f_data','psi_data'),'burger':('input','output')}[pde]
        savemat(tmp,{names[0]:a,names[1]:u,'sample_index':seeds,'generation_seed':args.seed},appendmat=False)
    else:
        with h5py.File(tmp,'w') as f:
            f.attrs.update(dataset_type=PROFILE,pde_name=pde,n_samples=count,generation_seed=args.seed)
            if pde=='darcy':
                # Preserve the existing MATLAB H,W,N layout for both loaders.
                f.create_dataset('thresh_a_data',data=np.moveaxis(a,0,-1));f.create_dataset('thresh_p_data',data=np.moveaxis(u,0,-1))
            else:
                f.create_dataset('w0',data=a)
                for key in ('w','vx','vy','vx0','vy0'):
                    shape=(count,)+np.load(batches[0])[key].shape[1:]
                    ds=f.create_dataset(key,shape=shape,dtype='float32')
                    for batch in batches:
                        b=np.load(batch);lo=int(batch.stem.split('_')[1]);ds[lo:lo+len(b['a'])]=b[key]
                f.create_dataset('t',data=np.load(batches[0])['t'])
                f.attrs.update(T=1.,dt=1e-4,viscosity=.001)
            f.create_dataset('sample_index',data=seeds)
    tmp.replace(path)
    write_json(receipt,dict(source=str(path),sha256=sha256(path),count=count,config=config,
        calibration=str(calibration_path),diagnostic=diagnostic,seconds=time.time()-started))
    print('COMPLETE',path,json.dumps(diagnostic),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['calibrate','generate'])
    p.add_argument('--pde',choices=[*PDES,'all'],required=True)
    p.add_argument('--data-root',default='/large_storage/zhangxf/PDEdata')
    p.add_argument('--output-root',required=True)
    p.add_argument('--baseline-root',help=argparse.SUPPRESS)  # Accepted for earlier launcher compatibility.
    p.add_argument('--count',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=60000000)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--matlab',default='/usr/local/bin/matlab')
    p.add_argument('--chebfun-root',default='/research_data/users/zhangxifeng/C04Matlab/package/chebfun')
    args=p.parse_args()
    if args.mode=='generate' and args.pde=='all':p.error('Generate each PDE in a separate process.')
    (calibrate if args.mode=='calibrate' else generate)(args)

if __name__=='__main__':main()

"""Reference Navier–Stokes and Burgers re-solving helpers."""
from pathlib import Path
import json
import numpy as np
import scipy.io
import torch
import subprocess
from experiments.paper.run import ROOT, digest as sha256, write_json

def ns_solve(initial,device):
    from data.DataGen.time_dependent.no_bound_ns.ns_2d import navier_stokes_2d
    grid=torch.linspace(0,1,129,device=device)[:-1]; xx,yy=torch.meshgrid(grid,grid,indexing='ij')
    forcing=.1*(torch.sin(2*torch.pi*(xx+yy))+torch.cos(2*torch.pi*(xx+yy)))
    results=[]
    for lo in range(0,len(initial),16):
        _,_,trajectory,_,_,_=navier_stokes_2d(initial[lo:lo+16].float().to(device),forcing,.001,1.,.0001,1)
        assert torch.isfinite(trajectory).all()
        results.append(trajectory[...,-1].cpu().double())
    return torch.cat(results)

def solve(initial,folder,args):
    folder.mkdir(parents=True,exist_ok=True)
    inp=folder/'initial.mat';out=folder/'solved.mat';receipt=folder/'solve.json'
    if receipt.exists():
        r=json.loads(receipt.read_text());assert sha256(out)==r['output_sha256']
        assert np.array_equal(scipy.io.loadmat(inp)['initial'],initial)
    else:
        scipy.io.savemat(inp,dict(initial=initial))
        quote=lambda s:"'"+str(s).replace("'","''")+"'"
        expression=f"addpath({quote(Path(__file__).parent)}); addpath({quote(ROOT/'data/DataGen/static')}); burger_solve({quote(inp)},{quote(out)},{quote(args.chebfun)});"
        with (folder/'matlab.log').open('w') as log:
            subprocess.run([args.matlab,'-singleCompThread','-batch',expression],stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(receipt,dict(input_sha256=sha256(inp),output_sha256=sha256(out),
            generator_sha256=sha256(ROOT/'data/DataGen/static/burgers1.m'),
            wrapper_sha256=sha256(Path(__file__).with_name('burger_solve.m')),
            parameters=dict(viscosity=.01,spatial_points=128,time_points=128,T=1,dt=1/127)))
    return torch.from_numpy(scipy.io.loadmat(out)['trajectory'])[:,None]

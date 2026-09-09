"""Separate development-only fixed-noise cross-device numerical check."""
from pathlib import Path
import argparse
import dataclasses
import json
import os
import sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'plot')]
import numpy as np
import torch
import sampling.runner as r
import sampling.sampler_wrappers as sw
from sampling.model_io import load_fm4pde_checkpoint_bundle
from run_ns_main_revision_0909 import configuration,observations,infer,write,digest,DEV


def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    protocol=json.loads((a.inputs/'protocol.json').read_text())
    cfg=configuration(protocol,'sparse_joint','smooth',a.inputs/'weights.pth')
    data=np.load(a.inputs/'smooth.npz');fields=torch.from_numpy(data['fields'][list(data['ids']).index(DEV[0]):list(data['ids']).index(DEV[0])+1])
    gt,masks=observations(fields,[DEV[0]],cfg,protocol['pde_params'])
    noise=torch.load(a.noise,weights_only=False,map_location='cpu')
    assert torch.equal(masks.coef.cpu(),noise['mask_a']) and torch.equal(masks.sol.cpu(),noise['mask_u'])
    # Diagnose the native RNG independently of model arithmetic.
    r._set_seed(0)
    native=[]
    for _ in range(3):native.append(torch.randn(1000,2,128,128,device='cuda:0')[0].cpu())
    counter=[0]
    def draw(*args,**kwargs):
        value=noise['noise'][counter[0]].to('cuda:0');counter[0]+=1;return value
    r._sample_initial_noise=draw;sw._stochastic_bridge_noise_like=draw
    bundle=load_fm4pde_checkpoint_bundle(str(a.inputs/'weights.pth'),'nsnonbounded','cuda:0',model_profile='auto')
    prediction,receipt=infer(cfg,bundle,gt,masks,[0],fused=False)
    assert counter[0]==101
    receipt.update(torch=torch.__version__,gpu=torch.cuda.get_device_name(),noise_sha256=digest(a.noise),masks_identical=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(prediction=prediction,native_noise=torch.stack(native),receipt=receipt),a.output)
    write(a.output.with_suffix('.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--noise',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

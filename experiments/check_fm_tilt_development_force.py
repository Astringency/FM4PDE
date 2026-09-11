"""Audit approximate proposal gradients at saved real-FM development states."""
from __future__ import annotations
import argparse,json,hashlib,io
from pathlib import Path
import torch
from experiments.run_fm_tilt import repeated_case
from experiments.fm_tilt_adapter import FMTiltTarget
from sampling.config import AblationConfig
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.masks import make_pair_masks
from scripts.train.resume_study import write


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot-inputs',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    ip=json.loads((args.output/'inputs/protocol.json').read_text())
    state_path=args.output/'pilot_trajectory20_scaled/ids_1103_1103/state.pt'
    encoded=state_path.read_bytes();state=torch.load(io.BytesIO(encoded),map_location='cpu',weights_only=False)
    cache=torch.load(args.pilot_inputs/'truths.pt',map_location='cpu',weights_only=False)
    one=cache[1103];mask=make_pair_masks(one.coef.shape,one.sol.shape,500,'random',True,0)
    cases={1103:dict(truth=one.pair,params=one.pde_params,masks=dict(coef=mask.coef,sol=mask.sol))}
    gt,masks=repeated_case(ip,cases,[1103],4,'cuda:0')
    cfg=AblationConfig(**dict(ip['config'],checkpoint_path=ip['checkpoint_path'],device='cuda:0',batch_size=4))
    bundle=load_fm4pde_checkpoint_bundle(ip['checkpoint_path'],ip['pde'],'cuda:0',model_profile=ip['config']['model_profile'])
    a=FMTiltTarget(bundle,cfg,gt,masks,force_steps=100,force_kind='trajectory_adjoint')
    z=state['z'].to('cuda:0');r,e=a.target(z)
    exact=a.force(z).double().flatten(1)
    a.force_steps=20;approx=1.235*a.force(z).double().flatten(1)
    out=args.output/'development_force_check';out.mkdir(exist_ok=True)
    torch.save(dict(z=z.cpu(),sample_id=1103,iteration=state['iteration']),out/'input.pt')
    report=dict(iteration=state['iteration'],source_state_sha256=hashlib.sha256(encoded).hexdigest(),
        scope='Four finite-chain development states; not a stationary posterior claim.',
        energy=e.tolist(),cosine=((exact*approx).sum(1)/exact.norm(dim=1)/approx.norm(dim=1)).tolist(),
        relative_error=((approx-exact).norm(dim=1)/exact.norm(dim=1)).tolist(),
        exact_norm=exact.norm(dim=1).tolist(),approximate_norm=approx.norm(dim=1).tolist())
    write(out/'complete.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()

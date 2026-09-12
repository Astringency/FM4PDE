"""Audit proposal-force fidelity after whitening; full target remains fixed."""
import argparse, time
from pathlib import Path
import torch
from experiments.build_fm_tilt_geometry import setup
from experiments.fm_tilt_geometry import GaussianReference
from scripts.train.resume_study import file_sha, write


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['source','pilot-inputs','output','run']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    out=args.output/'force_audit';out.mkdir(exist_ok=True)
    assert not (out/'complete.json').exists()
    a,ip,_=setup(args.source,args.pilot_inputs,1103,out)
    state_path=args.run/'state.pt'
    # Snapshot bytes first, so the identity cannot race with a worker save.
    snapshot=out/'input_state.pt'
    snapshot.write_bytes(state_path.read_bytes())
    state=torch.load(snapshot,map_location='cpu',weights_only=False)
    spec=torch.load(args.run/'reference.pt',map_location='cpu',weights_only=False)
    ref=GaussianReference(*[spec[k].cuda() for k in ['basis','root','mean']])
    w=state['w'].cuda();z=ref.decode(w)
    start=time.monotonic();r,e=a.target(z)
    original,whitened={},{}
    for count,scale in [(20,1.235),(100,1.)]:
        a.force_steps=count
        g=scale*a.force(z)
        original[count]=g
        whitened[count]=ref.residual_force(w,z,g)
    result=dict(iteration=state['iteration'],input_state_sha256=file_sha(snapshot),energy=e.tolist(),
        seconds=time.monotonic()-start,checkpoint_sha256=ip['checkpoint_sha256'])
    for name,forces in [('original',original),('whitened',whitened)]:
        exact=forces[100].double().flatten(1);approx=forces[20].double().flatten(1)
        result[name]=dict(exact_norm=exact.norm(dim=1).tolist(),approximate_norm=approx.norm(dim=1).tolist(),
            error_norm=(approx-exact).norm(dim=1).tolist(),
            relative_error=((approx-exact).norm(dim=1)/exact.norm(dim=1)).tolist(),
            cosine=((approx*exact).sum(1)/approx.norm(dim=1)/exact.norm(dim=1)).tolist())
    write(out/'complete.json',result)
    print(result,flush=True)


if __name__=='__main__':main()

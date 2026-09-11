"""Measure proposal-force fidelity, numerical precision and real batch costs."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from experiments.run_fm_tilt import repeated_case
from experiments.fm_tilt_adapter import FMTiltTarget
from sampling.config import AblationConfig
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.masks import make_pair_masks
from scripts.train.resume_study import file_sha,write


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot-inputs',type=Path,required=True)
    p.add_argument('--adjoint-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    ip=json.loads((args.output/'inputs/protocol.json').read_text())
    assert file_sha(ip['checkpoint_path'])==ip['checkpoint_sha256']
    cache=torch.load(args.pilot_inputs/'truths.pt',map_location='cpu',weights_only=False)
    one=cache[1102]
    mask=make_pair_masks(one.coef.shape,one.sol.shape,500,'random',True,0)
    cases={1102:dict(truth=one.pair,params=one.pde_params,masks=dict(coef=mask.coef,sol=mask.sol))}
    bundle=load_fm4pde_checkpoint_bundle(ip['checkpoint_path'],ip['pde'],'cuda:0',model_profile=ip['config']['model_profile'])
    out=args.output/('benchmark_adjoint' if args.adjoint_only else 'benchmark');out.mkdir(exist_ok=True)
    rows=[]
    for count in ([4] if args.adjoint_only else [4,16,32]):
        gt,masks=repeated_case(ip,cases,[1102]*(count//4),4,'cuda:0')
        conf=dict(ip['config'],checkpoint_path=ip['checkpoint_path'],device='cuda:0',batch_size=count,output_dir=str(out))
        cfg=AblationConfig(**conf);cfg.validate()
        a=FMTiltTarget(bundle,cfg,gt,masks,force_steps=20)
        if args.adjoint_only:a.force_kind='trajectory_adjoint'
        rng=torch.Generator(device='cuda:0').manual_seed(9812)
        z=torch.randn(gt.pair.shape,device='cuda:0',generator=rng)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        tic=time.monotonic();ref,energy=a.target(z);torch.cuda.synchronize();elapsed=time.monotonic()-tic
        record=dict(chains=count,target_fp32_seconds=elapsed,energy=energy.tolist())
        forces={}
        for steps in ([4,20,100] if count==4 else [20]):
            a.force_steps=steps
            tic=time.monotonic();forces[steps]=a.force(z);torch.cuda.synchronize()
            record[f'force{steps}_seconds']=time.monotonic()-tic
        if count==4:
            exact=forces[100].double().flatten(1)
            record['force_comparison']={}
            for steps,g in forces.items():
                g=g.double().flatten(1)
                record['force_comparison'][steps]=dict(
                    cosine=((g*exact).sum(1)/g.norm(dim=1)/exact.norm(dim=1)).tolist(),
                    relative_error=((g-exact).norm(dim=1)/exact.norm(dim=1)).tolist())
            again,_=a.target(z)
            record['fp32_generator_bitwise_deterministic']=bool(torch.equal(again,ref))
        record['fp32_peak_bytes']=torch.cuda.max_memory_allocated()
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.cuda.reset_peak_memory_stats()
        # Warm algorithms before timing TensorFloat-32.
        other,eother=a.target(z)
        tic=time.monotonic();other,eother=a.target(z);torch.cuda.synchronize()
        record['target_tf32_seconds']=time.monotonic()-tic
        record['tf32_endpoint_relative_difference']=((other-ref).double().flatten(1).norm(dim=1)/ref.double().flatten(1).norm(dim=1)).tolist()
        record['tf32_energy_difference']=(eother-energy).tolist()
        record['tf32_peak_bytes']=torch.cuda.max_memory_allocated()
        rows.append(record);write(out/'progress.json',dict(rows=rows))
        print('BENCH',json.dumps(record),flush=True)
        assert record['fp32_peak_bytes']<40*2**30
        del a,gt,masks,z,forces,ref,other
        torch.cuda.empty_cache()
    write(out/'complete.json',dict(rows=rows,checkpoint_sha256=ip['checkpoint_sha256'],
        scope='Development input 1102 only; no test errors inspected; TF32 benchmark does not change the production target.'))


if __name__=='__main__':main()

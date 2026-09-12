"""Convergence development for the full learned FM terminal target.

This runner only accepts development/validation IDs outside the frozen 1000.
The proposal is fixed before retained sampling. No reconstruction metric is
used to choose a sampler, a draw, a stopping point, or a convergence threshold.
"""
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np
import torch
from experiments.build_fm_tilt_geometry import setup
from experiments.fm_tilt_geometry import fit_reference
from experiments.fm_tilt_mh import transition, adapt_beta, chain_diagnostics
from scripts.train.resume_study import file_sha, write


def monitor(adapter, endpoint, energy, latent):
    from scipy.fft import dctn
    base = adapter.monitor(endpoint, energy)
    spec = dctn(endpoint.detach().double().cpu().numpy(), axes=(-2,-1), norm='ortho')
    extra = []
    for c in range(2):
        for y, x in [(4,4), (8,0), (0,8), (16,0), (0,16), (8,8)]:
            extra.append(spec[:,c,y,x])
    extra.extend([latent.double().square().flatten(1).sum(1).cpu().numpy(),
        endpoint[:,0].double().square().flatten(1).sum(1).cpu().numpy(),
        endpoint[:,1].double().square().flatten(1).sum(1).cpu().numpy()])
    return np.concatenate([base, np.stack(extra, axis=1)], axis=1)


def diagnostics(trace):
    from experiments.strict_chain_diagnostics import diagnose
    return diagnose(trace)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pilot-inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--id', type=int, default=1103)
    p.add_argument('--force', choices=['none','trajectory'], default='trajectory')
    p.add_argument('--force-steps', type=int, default=20)
    p.add_argument('--force-scale', type=float, default=1.235)
    p.add_argument('--beta', type=float, default=.15)
    p.add_argument('--warmup', type=int, default=256)
    p.add_argument('--keep', type=int, default=512)
    p.add_argument('--seed', type=int, default=290912)
    p.add_argument('--extend-from', type=Path)
    args = p.parse_args()
    assert args.output.is_absolute() and '/outputs/' in str(args.output)
    assert 0<args.beta<1 and args.warmup>=0 and args.keep>=8
    out = args.output / args.name
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'complete.json').exists()
    adapter, ip, cases = setup(args.source, args.pilot_inputs, args.id, out)
    adapter.force_steps = args.force_steps
    geom = torch.load(args.geometry, map_location='cpu', weights_only=False)
    assert geom['protocol']['checkpoint_sha256'] == ip['checkpoint_sha256']
    # Refit only the observation-dependent Gaussian proposal mean/curvature.
    ref, fit = fit_reference(geom['basis'], geom['jacobian'], geom['center'],
        geom['endpoint'].cuda(), adapter)
    fit.pop('root'); fit.pop('hessian')
    spec = dict(input_id=args.id, chains=4, warmup=args.warmup, keep=args.keep,
        force=args.force, force_steps=args.force_steps, force_scale=args.force_scale,
        seed=args.seed, beta=args.beta, initial_overdispersion=dict(fitted_subspace=2.,complement=1.),
        geometry_sha256=file_sha(args.geometry), checkpoint_sha256=ip['checkpoint_sha256'],
        native_config=ip['config'], inputs_protocol_sha256=file_sha(args.source/'inputs/protocol.json'),
        target='exp(-||z||^2/2 - original_L_phys(G_100(z); observed_y)); all 32768 latent coordinates',
        proposal='Fixed Gaussian reference, whitened pCN/pCNL with exact full-target MH correction',
        limitations='Finite-chain convergence diagnostics; direct terminal tilt, not a stepwise exact-velocity ablation',
        convergence_gate=dict(rank_folded_split_rhat=1.01, bulk_ess=100, tail_ess=100,
            additional='Independent validation inputs and a doubled retained budget before expanding to 1000'),
        selection_rule='Tune proposal using convergence and ESS/second only; no reconstruction-error selection',
        extend_from_sha256=file_sha(args.extend_from) if args.extend_from else None,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()) == spec
    else:
        write(out/'protocol.json', spec)
    write(out/'reference_fit.json', fit)
    torch.save(dict(basis=ref.basis.cpu(), root=ref.root.cpu(), mean=ref.mean.cpu()), out/'reference.pt')

    def target(white):
        z = ref.decode(white)
        endpoint, energy = adapter.target(z)
        return endpoint, ref.residual_energy(white, z, energy)

    def force(white):
        if args.force == 'none':
            return torch.zeros_like(white)
        z = ref.decode(white)
        gradient = args.force_scale * adapter.force(z)
        return ref.residual_force(white, z, gradient)

    rng = torch.Generator(device='cuda:0').manual_seed(args.seed+args.id*1009)
    w = torch.randn((4,2,128,128), generator=rng, device='cuda:0')
    w += ((w.flatten(1) @ ref.basis.T) @ ref.basis).reshape_as(w)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    r, value = target(w)
    gradient = force(w)
    peak = torch.cuda.max_memory_allocated()
    assert peak < 40*2**30
    write(out/'probe.json',dict(seconds=time.monotonic()-start,peak_bytes=peak,
        energy=adapter.potential(r).tolist(),residual_energy=value.tolist(),
        residual_gradient_norm=gradient.flatten(1).norm(dim=1).tolist()))
    print('PROBE', json.loads((out/'probe.json').read_text()), flush=True)
    beta = torch.full((4,),args.beta,device='cuda:0')
    trace, accepted, mean = [], torch.zeros(4,device='cuda:0'), torch.zeros_like(r,dtype=torch.float64)
    elapsed, iteration = 0., 0
    state = out/'state.pt'
    if args.extend_from and not state.exists():
        assert args.warmup == 0, 'An extension freezes the calibrated proposal'
        previous = json.loads((args.extend_from.parent/'protocol.json').read_text())
        for key in ['input_id','force','force_steps','force_scale','geometry_sha256','checkpoint_sha256']:
            assert spec[key] == previous[key], key
        s = torch.load(args.extend_from,map_location='cpu',weights_only=False)
        assert s['protocol_sha256'] == file_sha(args.extend_from.parent/'protocol.json')
        w,r,value,gradient,beta=[s[k].cuda() for k in ['w','r','value','gradient','beta']]
        rng.set_state(s['rng_state'])
    if state.exists():
        s = torch.load(state,map_location='cpu',weights_only=False)
        assert s['protocol_sha256'] == file_sha(out/'protocol.json')
        w,r,value,gradient,beta,accepted,mean=[s[k].cuda() for k in ['w','r','value','gradient','beta','accepted','mean']]
        trace,elapsed,iteration=s['trace'],s['seconds'],s['iteration']
        rng.set_state(s['rng_state'])
    start=time.monotonic()
    for it in range(iteration,args.warmup+args.keep):
        w,r,value,gradient,accept,prob=transition(w,r,value,gradient,beta,target=target,force=force,rng=rng)
        if it<args.warmup:
            beta=adapt_beta(beta,prob,it,target_accept=.3 if args.force=='none' else .574)
        else:
            trace.append(monitor(adapter,r,adapter.potential(r),ref.decode(w)))
            accepted+=accept;mean+=r.double()
        if (it+1)%16==0 or it+1==args.warmup+args.keep:
            seconds=elapsed+time.monotonic()-start
            diag=diagnostics(trace) if len(trace)>=8 else {}
            progress=dict(iteration=it+1,total_iterations=args.warmup+args.keep,retained=len(trace),seconds=seconds,
                beta=beta.tolist(),energy=adapter.potential(r).tolist(),
                acceptance=(accepted/max(1,len(trace))).tolist(),diagnostics=diag,
                latest_acceptance_probability=prob.tolist(),
                peak_bytes=torch.cuda.max_memory_allocated())
            tmp=state.with_suffix('.writing')
            torch.save(dict(w=w.cpu(),r=r.cpu(),value=value.cpu(),gradient=gradient.cpu(),beta=beta.cpu(),
                accepted=accepted.cpu(),mean=mean.cpu(),trace=trace,iteration=it+1,seconds=seconds,
                rng_state=rng.get_state(),protocol_sha256=file_sha(out/'protocol.json')),tmp)
            tmp.replace(state);write(out/'progress.json',progress)
            brief={k:v for k,v in progress.items() if k!='diagnostics'}
            brief.update({k:diag[k] for k in ['max_rhat','min_bulk_ess','min_tail_ess'] if k in diag})
            print('STEP',brief,flush=True)
    prediction=adapter.normalizer.inverse_transform(r).cpu()
    posterior_mean=adapter.normalizer.inverse_transform((mean/args.keep).mean(0,keepdim=True)).cpu()
    truth=cases[args.id]['truth']
    torch.save(dict(all_last_draws=prediction,primary_draw=prediction[:1],finite_chain_mean=posterior_mean,
        truth=truth,trace=np.asarray(trace),latent=ref.decode(w).cpu()),out/'result.pt')
    errors={name:(torch.linalg.vector_norm((pred-truth).double().flatten(2),dim=2)/
        torch.linalg.vector_norm(truth.double().flatten(2),dim=2)).tolist()
        for name,pred in [('all_last_draws',prediction),('finite_chain_mean',posterior_mean)]}
    write(out/'complete.json',dict(**progress,errors=errors,result_sha256=file_sha(out/'result.pt'),
        independent_validation_passed=False,formal_1000_completed=0))


if __name__=='__main__':
    main()

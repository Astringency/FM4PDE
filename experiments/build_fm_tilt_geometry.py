"""Calibrate a fixed low-rank Gaussian proposal with the actual 100-step G."""
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
import torch
from experiments.fm_tilt_geometry import cosine_basis, fit_reference
from experiments.fm_tilt_adapter import FMTiltTarget
from experiments.run_fm_tilt import repeated_case
from sampling.config import AblationConfig
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.masks import make_pair_masks
from scripts.train.resume_study import file_sha, write


def setup(source, pilot_inputs, sid, output, count=4):
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    ip = json.loads((source / 'inputs/protocol.json').read_text())
    assert sid not in ip['sample_ids']
    assert file_sha(ip['checkpoint_path']) == ip['checkpoint_sha256']
    pp = json.loads((pilot_inputs / 'protocol.json').read_text())
    assert file_sha(pilot_inputs / 'truths.pt') == pp['truth_sha256']
    one = torch.load(pilot_inputs / 'truths.pt', map_location='cpu', weights_only=False)[sid]
    assert not one.metadata['synthetic']
    masks = make_pair_masks(one.coef.shape, one.sol.shape, ip['config']['num_obs'],
        ip['config']['sensor_mode'], ip['config']['shared_mask'], ip['config']['mask_seed'])
    cases = {sid: dict(truth=one.pair, params=one.pde_params,
        masks=dict(coef=masks.coef, sol=masks.sol))}
    gt, masks = repeated_case(ip, cases, [sid], count, 'cuda:0')
    cfg = AblationConfig(**dict(ip['config'], checkpoint_path=ip['checkpoint_path'], device='cuda:0',
        batch_size=count, output_dir=str(output), save_plots=False, save_intermediate=False))
    cfg.validate()
    bundle = load_fm4pde_checkpoint_bundle(ip['checkpoint_path'], ip['pde'], 'cuda:0', model_profile=cfg.model_profile)
    adapter = FMTiltTarget(bundle, cfg, gt, masks, force_kind='trajectory_adjoint', force_steps=20)
    return adapter, ip, cases


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pilot-inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--center-state', type=Path, required=True)
    p.add_argument('--id', type=int, default=1103)
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--epsilon', type=float, default=.02)
    args = p.parse_args()
    assert args.output.is_absolute() and '/outputs/' in str(args.output)
    out = args.output / 'geometry'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'complete.json').exists()
    adapter, ip, cases = setup(args.source, args.pilot_inputs, args.id, out)
    basis = cosine_basis(width=args.width, device='cuda:0')
    old = torch.load(args.center_state, map_location='cpu', weights_only=False)
    low = old['z'].mean(0, keepdim=True).to(basis).flatten(1) @ basis.T
    rng = torch.Generator(device='cuda:0').manual_seed(936701)
    noise = torch.randn((1, 2, 128, 128), generator=rng, device='cuda:0')
    center = noise + ((low - noise.flatten(1) @ basis.T) @ basis).reshape_as(noise)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.no_grad():
        endpoints, values = adapter.target(center.repeat(4, 1, 1, 1))
    peak = torch.cuda.max_memory_allocated()
    assert peak < 40 * 2**30
    protocol = dict(input_id=args.id, checkpoint_sha256=ip['checkpoint_sha256'],
        inputs_protocol_sha256=file_sha(args.source / 'inputs/protocol.json'),
        center_state_sha256=file_sha(args.center_state), width=args.width, epsilon=args.epsilon,
        rank=len(basis), full_latent_dimension=32768, subspace='Orthonormal low DCT modes in both latent channels',
        role='Proposal only: no mode truncation of the target, no changes to G, PDE energy, masks or weights',
        generator_steps=100, target_precision='FP32 G, TF32 off, FP64 energy',
        preflight_peak_bytes=peak, preflight_seconds=time.monotonic()-start,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
    write(out / 'protocol.json', protocol)
    print('PREFLIGHT', protocol, flush=True)
    jacobian = []
    for k in range(0, len(basis), 2):
        directions = basis[k:k+2].reshape(-1, 2, 128, 128)
        z = torch.cat([center + args.epsilon * directions, center - args.epsilon * directions])
        with torch.no_grad():
            r, _ = adapter.target(z)
        j = (r[:2].double() - r[2:].double()) / (2 * args.epsilon)
        jacobian.append(j.cpu())
        if (k+2) % 16 == 0:
            progress = dict(columns=k+2, total=len(basis), seconds=time.monotonic()-start,
                peak_bytes=torch.cuda.max_memory_allocated())
            write(out / 'progress.json', progress); print('JACOBIAN', progress, flush=True)
    j = torch.cat(jacobian)
    reference, report = fit_reference(basis, j, center, endpoints[:1], adapter)
    # FD consistency at twice the perturbation scale is a proposal QA check.
    consistency = []
    for k in [0, 8, 64, 72]:
        directions = basis[[k, k+1]].reshape(-1, 2, 128, 128)
        with torch.no_grad():
            r, _ = adapter.target(torch.cat([center + 2*args.epsilon*directions, center - 2*args.epsilon*directions]))
        check = ((r[:2].double()-r[2:].double())/(4*args.epsilon)).cpu()
        consistency.extend(((check-j[k:k+2]).flatten(1).norm(dim=1)/j[k:k+2].flatten(1).norm(dim=1)).tolist())
    torch.save(dict(basis=basis.cpu(), jacobian=j, center=center.cpu(), endpoint=endpoints[:1].cpu(),
        reference_root=report.pop('root'), hessian=report.pop('hessian'),
        protocol=protocol), out / 'geometry.pt')
    report.update(finite_difference_relative_changes=consistency, seconds=time.monotonic()-start,
        geometry_sha256=file_sha(out / 'geometry.pt'), peak_bytes=torch.cuda.max_memory_allocated())
    write(out / 'complete.json', report)
    print('DONE', report, flush=True)


if __name__ == '__main__':
    main()

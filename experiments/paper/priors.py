"""The eleven appendix draws: 100 unguided Euler steps and reference re-solving."""
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.paper.run import ROOT, digest, write_json


def run(spec, args):
    from sampling.config import load_config, parse_cli_overrides
    pdes = [p for p in spec['pdes'] if not args.pdes or p in args.pdes]
    pdes = [p for p in pdes if not args.job_ids or p in args.job_ids]
    pdes = pdes[:args.limit][args.shard_index::args.num_shards]
    if (not pdes or (args.methods and 'FM4PDE' not in args.methods)
            or (args.steps and spec['steps'] not in args.steps)
            or (args.phases and 'deterministic' not in args.phases)):
        raise ValueError('No appendix prior draws match the selection')
    if args.tasks or args.test_types:
        raise ValueError('Unconditional draws use --pdes; observation tasks and test distributions do not apply')
    extra = parse_cli_overrides(args.override)
    allowed = {'checkpoint_path', 'model_profile', 'model_gradient_checkpointing'}
    if set(extra) - allowed:
        raise ValueError(f'Appendix draws fix their seeds and 100-step algorithm; unsupported overrides: {sorted(set(extra)-allowed)}')
    configs = []
    for pde in pdes:
        cfg = load_config(ROOT/f'configs/main/both/{pde}.yaml', dict(
            extra, task='unconditional', guidance_components='noguide', sampler_phase='deterministic',
            time_grid='uniform', num_steps=spec['steps'], batch_size=1, device=args.device,
            num_obs=0, zeta_obs_a=0., zeta_obs_u=0., zeta_pde=0.,
            sample_seed=spec.get('seeds', {}).get(pde, spec['seed'])))
        configs.append(cfg)
    if args.plan_only:
        print(json.dumps(dict(paragraph=spec['paragraph'], draws=[c.asdict() for c in configs],
                              physical_parameters={p: spec['physical_parameters'].get(p, {}) for p in pdes}), indent=2))
        return

    import torch
    import numpy as np
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from sampling.runner import _scalar_conditioning_for_sampling
    from sampling.sampler_wrappers import _call_velocity_model
    from sampling.state import standardized_to_physical_state
    from experiments.paper.prior_solvers import solve
    torch.set_num_threads(args.threads)
    sources = {str(p.relative_to(ROOT)): digest(p)
               for directory in ['sampling', 'models', 'data', 'experiments/paper']
               for p in (ROOT/directory).rglob('*.py')}
    records = []
    for cfg in configs:
        pde = cfg.pde
        folder = args.output/pde
        folder.mkdir(parents=True, exist_ok=True)
        parameters = spec['physical_parameters'].get(pde, {})
        identity = dict(config=cfg.asdict(), physical_parameters=parameters,
                        checkpoint_sha256=digest(cfg.checkpoint_path), sources=sources)
        receipt = folder/'receipt.json'
        if args.resume and receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['identity'] != identity or digest(folder/'fields.npz') != saved['fields_sha256']:
                raise ValueError(f'{folder}: changed settings, source or fields; choose a new output directory')
            records.append(saved)
            continue
        net, normalizer, payload = load_fm4pde_checkpoint_bundle(
            cfg.checkpoint_path, pde, args.device, wrap=True, model_profile=cfg.model_profile)
        params = {k: torch.as_tensor(v, device=args.device, dtype=torch.float32)
                  for k, v in parameters.items() if isinstance(v, list)}
        truth = SimpleNamespace(pde_params=params, pair=torch.empty(0, device=args.device))
        extra, conditioning = _scalar_conditioning_for_sampling(
            checkpoint_payload=payload, gt=truth, config=cfg, device=args.device)
        torch.manual_seed(cfg.sample_seed)
        x = torch.randn(1, cfg.img_channels, cfg.img_resolution, cfg.img_resolution, device=args.device)
        import hashlib
        noise_hash = hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
        with torch.no_grad():
            for k in range(cfg.num_steps):
                t = torch.tensor(k/cfg.num_steps, device=x.device, dtype=x.dtype)
                x = x + (1/cfg.num_steps)*_call_velocity_model(net, x, t, extra)
            fields = standardized_to_physical_state(x, pde, normalizer=normalizer)
        a = fields.coef[0].cpu().double().numpy()
        u = fields.sol[0].cpu().double().numpy()
        if not np.isfinite(a).all() or not np.isfinite(u).all():
            raise FloatingPointError(f'{pde}: nonfinite unconditional draw')
        resolved, solver_details = solve(pde, a, u, parameters, folder, args)
        if resolved.shape != u.shape or not np.isfinite(resolved).all():
            raise FloatingPointError(f'{pde}: invalid reference solution')
        errors = [float(np.linalg.norm(v-w)/np.linalg.norm(w)) for v, w in zip(u, resolved)]
        if not np.isfinite(errors).all():
            raise FloatingPointError(f'{pde}: undefined solver-relative error')
        np.savez_compressed(folder/'fields.npz', a=a, u=u, resolved=resolved)
        saved = dict(pde=pde, identity=identity, initial_noise_sha256=noise_hash,
                     scalar_conditioning=conditioning, component_relative_l2=errors,
                     solver_details=solver_details, fields_sha256=digest(folder/'fields.npz'),
                     torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(x.device) if x.is_cuda else None)
        write_json(receipt, saved)
        records.append(saved)
        write_json(args.output/'priors.json', records)
        print(pde, 'solver-relative errors:', errors, flush=True)
        del net, payload, normalizer, fields, x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output/'priors.json', records)
    write_json(args.output/'completion.json', dict(expected=len(configs), completed=len(records), failed=0))

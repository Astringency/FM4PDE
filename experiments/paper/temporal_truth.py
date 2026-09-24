"""Compare all three temporal residuals on the same 32 real endpoint pairs."""
from experiments.paper.run import ROOT, write_json


def evaluate(spec, args):
    import torch
    from sampling.config import load_config
    from sampling.data import load_ground_truth, attach_near_endpoint_observations
    from sampling.masks import PairMasks
    from sampling.losses import compute_guidance_losses
    from sampling.state import SplitState
    rows = []
    seen = set()
    selected = [j for j in spec['jobs'] if not args.pdes or j['pde'] in args.pdes]
    for job in selected[:args.limit]:
        c = job['overrides']
        for offset in range(c['offset'], c['offset']+c['batch_size']):
            key = (job['pde'], offset)
            if key in seen: continue
            seen.add(key)
            cfg = load_config(ROOT/f"configs/main/both/{job['pde']}.yaml",
                              dict(device=args.device, batch_size=1, offset=offset,
                                   residual_mode='near_endpoint_temporal'))
            truth = load_ground_truth(cfg)
            masks = PairMasks(torch.ones_like(truth.coef), torch.ones_like(truth.sol), {})
            truth = attach_near_endpoint_observations(cfg, truth, masks)
            for mode in ['endpoint_secant','hermite_bridge','near_endpoint_temporal']:
                cfg.residual_mode = mode
                with torch.no_grad():
                    loss = compute_guidance_losses(SplitState(truth.coef, truth.sol), truth, masks, cfg)
                if loss.pde_residual_status == 'error':
                    raise RuntimeError(f'{key}: temporal residual failed')
                rows.append(dict(pde=key[0], offset=offset, residual=mode, loss=float(loss.L_pde)))
    write_json(args.output/'true_temporal_losses.json', rows)

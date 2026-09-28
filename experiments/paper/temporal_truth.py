"""Compare all three temporal residuals on the same 32 real endpoint pairs."""
from experiments.paper.run import ROOT, write_json, selected_jobs, digest


def evaluate(spec, args):
    import torch
    from sampling.config import load_config
    from sampling.data import load_ground_truth, attach_near_endpoint_observations
    from sampling.masks import PairMasks
    from sampling.losses import compute_guidance_losses
    from sampling.state import SplitState
    rows = []
    seen = set()
    inputs = {}
    for job in selected_jobs(spec, args):
        c = job['overrides']
        for offset in range(c['offset'], c['offset']+c['batch_size']):
            key = (job['pde'], offset)
            if key in seen: continue
            seen.add(key)
            cfg = load_config(ROOT/f"configs/main/both/{job['pde']}.yaml",
                              dict(device=args.device, batch_size=1, offset=offset,
                                   residual_mode='near_endpoint_temporal'))
            truth = load_ground_truth(cfg)
            if job['pde'] not in inputs:
                inputs[job['pde']] = dict(
                    data_path=cfg.data_path, data_sha256=digest(cfg.data_path),
                    config=cfg.asdict())
            masks = PairMasks(torch.ones_like(truth.coef), torch.ones_like(truth.sol), {})
            truth = attach_near_endpoint_observations(cfg, truth, masks)
            for mode in ['endpoint_secant','hermite_bridge','near_endpoint_temporal']:
                cfg.residual_mode = mode
                with torch.no_grad():
                    loss = compute_guidance_losses(SplitState(truth.coef, truth.sol), truth, masks, cfg)
                if loss.pde_residual_status == 'error':
                    raise RuntimeError(f'{key}: temporal residual failed')
                rows.append(dict(pde=key[0], offset=offset, residual=mode, loss=float(loss.L_pde),
                                 secant_rhs_evaluation='endpoint_average' if mode=='endpoint_secant' else None))
    write_json(args.output/'true_temporal_losses.json', rows)
    write_json(args.output/'true_temporal_receipt.json', dict(
        secant_formula='(u-a)/T - (G_h(a)+G_h(u))/2',
        residual_source_sha256=digest(ROOT/'sampling/pde_residuals.py'),
        driver_sha256=digest(__file__),
        results_sha256=digest(args.output/'true_temporal_losses.json'),
        inputs=inputs, sample_offsets={pde: sorted(i for p, i in seen if p == pde)
                                      for pde in inputs},
        pairs=len(seen), evaluations=len(rows)))

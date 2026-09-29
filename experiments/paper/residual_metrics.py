"""Physical consistency metrics using the current shared PDE residual."""
from sampling.pde_residuals import compute_pde_residual


def full_pde_loss(data):
    """Evaluate the manuscript loss with the saved boundary and residual settings.

    `residual_mse` below is an older interior diagnostic and is not this loss.
    This function also works on archived predictions without resampling.
    """
    import torch
    from types import SimpleNamespace
    from sampling.losses import compute_guidance_losses
    from sampling.masks import PairMasks
    from sampling.metrics import pde_loss_per_sample
    from sampling.state import SplitState

    cfg = SimpleNamespace(**data['config'])
    a, u = data['coef_final'].double(), data['sol_final'].double()
    params = dict(data.get('pde_params', {}))
    if 'near_endpoint_temporal' in params:
        near = dict(params['near_endpoint_temporal'])
        # Saved artifacts retain only sparse measurements, with explicit names.
        for key in ('q_dt', 'q_T_minus_dt'):
            if key not in near and key + '_obs' in near:
                near[key] = near[key + '_obs']
        params['near_endpoint_temporal'] = near
    truth = SimpleNamespace(coef=data['coef_ground_truth'].double(),
                            sol=data['sol_ground_truth'].double(), pde_params=params)
    masks = PairMasks(torch.zeros_like(a), torch.zeros_like(u), {})
    with torch.no_grad():
        losses = compute_guidance_losses(SplitState(a, u), truth, masks, cfg)
    return pde_loss_per_sample(losses)


def evaluate_fields(pde, a, u, true_a, true_u, parameters=None):
    """Keep the published spatial region; replace only the residual definition."""
    import torch
    params=dict(parameters or {})
    params['enforce_boundary_conditions']=False
    if pde=='nsnonbounded':
        params.setdefault('T',1.0)
        params.setdefault('nu',0.001)
    mode='endpoint_secant' if pde=='nsnonbounded' else 'auto'
    with torch.no_grad():
        pred=compute_pde_residual(pde,a,u,pde_params=params,residual_mode=mode)
        truth=compute_pde_residual(pde,true_a,true_u,pde_params=params,residual_mode=mode)
    if pde=='nsnonbounded' and pred.metadata.get('secant_rhs_evaluation')!='endpoint_average':
        raise ValueError('Physical consistency requires endpoint-averaged secant dynamics')
    region=lambda x: x if pde=='nsnonbounded' else x[...,1:-1,1:-1]
    rp,rg=region(pred.residual).flatten(1),region(truth.residual).flatten(1)
    if not torch.isfinite(rp).all() or not torch.isfinite(rg).all():
        raise ValueError('Non-finite PDE residual')
    return dict(residual_mse=rp.square().mean(1).tolist(),
                truth_residual_mse=rg.square().mean(1).tolist(),
                residual_difference_mse=(rp-rg).square().mean(1).tolist(),
                residual_mode=mode,
                secant_rhs_evaluation=pred.metadata.get('secant_rhs_evaluation'),
                residual_region='full' if pde=='nsnonbounded' else 'interior')

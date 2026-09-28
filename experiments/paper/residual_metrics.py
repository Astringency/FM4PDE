"""Physical consistency metrics using the current shared PDE residual."""
from sampling.pde_residuals import compute_pde_residual


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

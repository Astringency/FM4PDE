"""Re-solve predicted inputs with the reference data-generation operators."""
from pathlib import Path


def evaluate(path, folder, args):
    import numpy as np
    import torch
    from experiments.paper import consistency_helpers as h
    data = torch.load(path, map_location='cpu', weights_only=False)
    cfg=data['config']; pde=cfg['pde']
    a=data['coef_final'].double(); u=data['sol_final'].double()
    ta=data['coef_ground_truth'].double(); tu=data['sol_ground_truth'].double()
    if pde in {'poisson','helmholtz','darcy'}:
        from data.DataGen.static_solvers import solve_static
        solve=lambda fields: torch.from_numpy(np.stack([solve_static(pde, x.numpy())[0] for x in fields[:,0]]))[:,None]
    elif pde=='nsnonbounded':
        solve=lambda fields: h.ns_solve(fields[:,0], args.device)[:,None]
    elif pde=='burger':
        if not args.chebfun:
            raise ValueError('Set CHEBFUN_ROOT for the Burgers reference solver')
        solve=None
    else:
        raise ValueError(pde)
    if solve:
        solved, true_solved=solve(a),solve(ta)
    else:
        solved=h.solve(u[:,0,0].numpy(),Path(folder)/'solver_prediction',args)
        true_solved=h.solve(tu[:,0,0].numpy(),Path(folder)/'solver_truth',args)
    norm=lambda x: x.flatten(1).norm(dim=1)
    relative=norm(u-solved)/norm(solved).clamp_min(1e-12)
    floor=norm(tu-true_solved)/norm(tu).clamp_min(1e-12)
    if not torch.isfinite(relative).all():
        raise ValueError('Reference solver produced non-finite consistency metrics')
    return dict(solver_relative_l2=relative.tolist(), truth_solver_relative_l2=floor.tolist())

"""Isolated NS PDE-loss exchange; recipient solvers/observation losses stay native.

No main sampling module is modified. Diffusion source is extracted with AST;
its float64 state/time grid is retained, unlike the earlier timing adapter.
"""
from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
from pathlib import Path


def diffusion_spatial_residual(a, u):
    import torch
    import torch.nn.functional as F
    dx = u.new_tensor([-1., 0., 1.]).view(1, 1, 1, 3) / 2
    dy = dx.transpose(-1, -2)
    r = F.conv2d(u, dx, padding=(0, 1)) + F.conv2d(u, dy, padding=(1, 0))
    mask = torch.ones_like(r)
    mask[..., (0, -1), :] = 0
    mask[..., :, (0, -1)] = 0
    return r * mask


def diffusion_pde_loss(a, u):
    import torch
    r = diffusion_spatial_residual(a, u)
    return (torch.linalg.vector_norm(r.flatten(1), dim=1) / (u.shape[-2] * u.shape[-1])).mean()


def fm_pde_loss(a, u, config, pde_params):
    """Use the exact main FM loss composer, including its reduction/options."""
    from types import SimpleNamespace
    import torch
    from sampling.losses import compute_guidance_losses
    from sampling.masks import PairMasks
    from sampling.state import SplitState
    masks = PairMasks(torch.zeros_like(a), torch.zeros_like(u), {})
    gt = SimpleNamespace(coef=torch.zeros_like(a), sol=torch.zeros_like(u), pde_params=pde_params)
    return compute_guidance_losses(SplitState(a, u), gt, masks, config).L_pde


@contextmanager
def fm_exchange(enabled, trace):
    """Replace only the PDE scalar consumed by the current FM update path."""
    import sampling.runner as runner
    original = runner.compute_guidance_losses
    gradient_original = runner.compute_guidance_gradient

    def losses(physical, gt, masks, cfg, observations=None):
        out = original(physical, gt, masks, cfg, observations)
        if enabled:
            out.guidance_L_pde = diffusion_pde_loss(physical.coef, physical.sol)
        trace.append(dict(step=len(trace), pde_loss=float(out.guidance_L_pde.detach()),
                          obs_a_loss=float(out.guidance_L_obs_a.detach()),
                          obs_u_loss=float(out.guidance_L_obs_u.detach())))
        return out

    def gradient(losses, target, schedule, config):
        out = gradient_original(losses, target, schedule, config)
        trace[-1].update(pde_gradient_norm=out.grad_norm_pde,
                         obs_a_gradient_norm=out.grad_norm_obs_a,
                         obs_u_gradient_norm=out.grad_norm_obs_u,
                         total_gradient_norm=out.grad_norm_total,
                         zeta_pde=float(schedule.zeta_pde_t.detach()),
                         clip_scale=out.clip_scale)
        return out

    runner.compute_guidance_losses = losses
    runner.compute_guidance_gradient = gradient
    try:
        yield
    finally:
        runner.compute_guidance_losses = original
        runner.compute_guidance_gradient = gradient_original


def _names(node):
    return {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()


def build_native_diffusion(diffusion_root, *, exchange=False, diagnostics=False):
    """Retain the native update loop and replace just L_pde when requested.

    Explicit engineering changes: resident network, caller-supplied sparse
    observations/masks, diagnostic output omitted, detached tensor return.
    Diagnostics=True retains original per-step reference-error arithmetic.
    The full effective Python source is returned for every run's archive.
    """
    path = Path(diffusion_root) / 'scripts/generate_ns_nonbounded.py'
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_ns_nonbounded')
    functions = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n is not original]
    body = original.body
    first = next(i for i, n in enumerate(body) if 'latents' in _names(n))
    loop_idx = next(i for i, n in enumerate(body) if isinstance(n, ast.For))
    final_idx = next(i for i, n in enumerate(body) if 'x_final' in _names(n))
    prefix = []
    for n in copy.deepcopy(body[first:loop_idx]):
        if _names(n) & {'time_start', 'loss'}:
            continue
        if isinstance(n, ast.If) and 'full' in ast.unparse(n.test):
            continue
        prefix.append(n)
    loop = copy.deepcopy(body[loop_idx])
    assert isinstance(loop.iter, ast.Call) and loop.iter.func.attr == 'tqdm'
    loop.iter = loop.iter.args[0]
    if not diagnostics:
        stop = next(i for i, n in enumerate(loop.body) if 'a_eval' in _names(n))
        loop.body = loop.body[:stop]
    guide = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "config['generate']['guide']")
    if exchange:
        idx = next(i for i, n in enumerate(guide.body) if 'L_pde' in _names(n))
        guide.body[idx].value = ast.parse('pde_loss_fn(a_N, u_N)', mode='eval').body
    trace_node = ast.parse("""
trace.append(dict(step=i, pde_loss=float(L_pde.detach()),
                  obs_a_loss=float(L_obs_a.detach()), obs_u_loss=float(L_obs_u.detach()),
                  pde_gradient_norm=float(grad_x_cur_pde.detach().norm()),
                  obs_a_gradient_norm=float(grad_x_cur_obs_a.detach().norm()),
                  obs_u_gradient_norm=float(grad_x_cur_obs_u.detach().norm()),
                  zeta_pde=float(zeta_pde) if i > 0.8 * num_steps else 0.0))
""").body[0]
    guide.body.append(trace_node)
    footer = []
    for n in copy.deepcopy(body[final_idx:]):
        if isinstance(n, ast.If):
            break
        footer.append(n)
    fn = ast.parse('def predict(config, net, a_GT, u_GT, mask_a, mask_u, pde_loss_fn, trace):\n pass').body[0]
    preamble = ast.parse("""
device = config['generate']['device']
obs_size = config['data']['obs_size']
batch_size = 1
seed = config['generate']['seed']
torch.manual_seed(seed)
known_index_a, known_index_u = mask_a, mask_u
loss = {'global_a': [], 'global_u': []}
""").body
    fn.body = preamble + prefix + [loop] + footer + ast.parse('return a_final.detach(), u_final.detach()').body
    module = ast.fix_missing_locations(ast.Module(body=functions + [fn], type_ignores=[]))
    source = ast.unparse(module) + '\n'
    import numpy as np
    import torch
    import torch.nn.functional as F
    scope = {'torch': torch, 'np': np, 'F': F}
    exec(compile(module, str(path) + '[isolated-loss-exchange]', 'exec'), scope)
    return scope['predict'], source

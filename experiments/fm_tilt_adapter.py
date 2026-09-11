"""Actual checkpoint/normalizer/PDE adapter, independent of production sampling."""
from __future__ import annotations
from copy import deepcopy
import torch
from sampling.config import AblationConfig
from sampling.state import standardized_to_physical_state, SplitState
from sampling.masks import PairMasks
from sampling.losses import compute_guidance_losses
from sampling.runner import _scalar_conditioning_for_sampling, _class_conditioning_for_sampling
from scripts.train.main_resume_sampling import ground_truth, slice_params


class FMTiltTarget:
    def __init__(self, bundle, config, gt, masks, *, full_steps=100, force_steps=4, force_kind='coarse_ode'):
        self.net, self.normalizer, self.payload = bundle
        self.cfg = config
        self.gt, self.masks = gt, masks
        self.full_steps, self.force_steps = full_steps, force_steps
        self.force_kind = force_kind
        self.cached_states = None
        assert config.noise_level == 0, "This adapter currently binds noiseless archived observations"
        assert config.obs_guidance_reduction == config.pde_guidance_reduction == 'mse'
        assert full_steps == config.num_steps == 100
        self.net.eval()
        for p in self.net.model.parameters():
            p.requires_grad_(False)
        se, _ = _scalar_conditioning_for_sampling(checkpoint_payload=self.payload,
            gt=gt, config=config, device=str(gt.pair.device))
        ce, _ = _class_conditioning_for_sampling(checkpoint_payload=self.payload,
            pde=config.pde, batch_size=len(gt.pair), device=str(gt.pair.device), cfg_scale=config.cfg_scale)
        self.extras = {**(se or {}), **ce}
        self.singles = []
        for i in range(len(gt.pair)):
            params = slice_params(gt.pde_params, [i], len(gt.pair), gt.pair.device)
            one = ground_truth(config.pde, gt.pair[i:i+1], [i], params, gt.pair.device, source='bound observations')
            mask = PairMasks(masks.coef[i:i+1], masks.sol[i:i+1], masks.metadata)
            self.singles.append((one, mask))

    def generator(self, z, steps):
        # No dropout, reinjection, endpoint shortcut, projection or guidance.
        x = z
        for i in range(steps):
            t = torch.full((len(x),), i / steps, device=x.device, dtype=x.dtype)
            x = x + self.net(x, t, **self.extras) / steps
        return x

    def physical(self, endpoints):
        return standardized_to_physical_state(endpoints, self.cfg.pde, self.cfg.img_channels, self.normalizer)

    def potential(self, endpoints):
        # Sum independently defined per-input energies; never divide by batch size.
        phys = self.physical(endpoints.double())
        assert len(endpoints) == len(self.singles)
        energies = []
        for i, (gt, masks) in enumerate(self.singles):
            losses = compute_guidance_losses(SplitState(phys.coef[i:i+1], phys.sol[i:i+1]), gt, masks, self.cfg)
            energies.append(self.cfg.zeta_obs_a * losses.guidance_L_obs_a
                + self.cfg.zeta_obs_u * losses.guidance_L_obs_u
                + self.cfg.zeta_pde * losses.guidance_L_pde)
        return torch.stack(energies)

    @torch.no_grad()
    def target(self, z):
        if self.force_kind == 'trajectory_adjoint':
            states = [z.detach()]
            for i in range(self.full_steps):
                x = states[-1]
                t = torch.full((len(x),), i/self.full_steps, device=x.device, dtype=x.dtype)
                states.append(x+self.net(x,t,**self.extras)/self.full_steps)
            self.cached_states = states
            endpoints = states[-1]
        else:
            endpoints = self.generator(z, self.full_steps)
        return endpoints, self.potential(endpoints)

    def force(self, z):
        # The short integration is ONLY a deterministic proposal drift.
        # The full 100-step integration is always used in the MH target.
        # A discrete reverse adjoint recomputes one network activation graph at
        # a time. This is the derivative of our Euler map, not an approximate
        # continuous-time adjoint, and makes full-step forces feasible too.
        if getattr(self, 'force_kind', 'coarse_ode') == 'trajectory_adjoint':
            return self.trajectory_force(z)
        states = [z.detach()]
        steps = self.force_steps
        with torch.no_grad():
            for i in range(steps):
                x = states[-1]
                t = torch.full((len(x),), i / steps, device=x.device, dtype=x.dtype)
                states.append(x + self.net(x, t, **self.extras) / steps)
        with torch.enable_grad():
            endpoint = states[-1].detach().requires_grad_(True)
            adjoint, = torch.autograd.grad(self.potential(endpoint).sum(), endpoint)
        for i in reversed(range(steps)):
            with torch.enable_grad():
                x = states[i].detach().requires_grad_(True)
                t = torch.full((len(x),), i / steps, device=x.device, dtype=x.dtype)
                v = self.net(x, t, **self.extras)
                vjp, = torch.autograd.grad(v, x, grad_outputs=adjoint / steps)
            adjoint = (adjoint + vjp).detach()
        return adjoint

    def trajectory_force(self, z):
        """A deterministic approximate VJP along the FULL forward trajectory.

        Early time is resolved finely. A short standalone forward integration
        misses the rapid initial expansion of low-frequency Gaussian modes.
        This approximation affects the proposal only; MH uses full energies.
        """
        import numpy as np
        if self.cached_states is None or not torch.equal(z, self.cached_states[0]):
            self.target(z)
        states=self.cached_states
        if self.force_steps==self.full_steps:
            boundaries=list(range(self.full_steps+1))
        else:
            # Quadratic spacing concentrates VJP evaluations near initial time.
            boundaries=sorted(set([0,self.full_steps]+np.rint(
                self.full_steps*np.linspace(0,1,self.force_steps+1)**2).astype(int).tolist()))
        with torch.enable_grad():
            endpoint=states[-1].detach().requires_grad_(True)
            adjoint,=torch.autograd.grad(self.potential(endpoint).sum(),endpoint)
        for lo,hi in reversed(list(zip(boundaries[:-1],boundaries[1:]))):
            index=(lo+hi-1)//2
            with torch.enable_grad():
                x=states[index].detach().requires_grad_(True)
                t=torch.full((len(x),),index/self.full_steps,device=x.device,dtype=x.dtype)
                v=self.net(x,t,**self.extras)
                vjp,=torch.autograd.grad(v,x,grad_outputs=adjoint*((hi-lo)/self.full_steps))
            adjoint=(adjoint+vjp).detach()
        return adjoint

    def monitor(self, endpoints, energy):
        # DCT low modes for nonperiodic PDEs; signed modes retain alignment.
        from scipy.fft import dctn
        import numpy as np
        x = endpoints.detach().double().cpu().numpy()
        spec = dctn(x, axes=(-2, -1), norm='ortho')
        columns = [energy.detach().cpu().numpy()]
        for channel in range(x.shape[1]):
            for y, k in [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]:
                columns.append(spec[:, channel, y, k])
        return np.stack(columns, axis=1)

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
    def __init__(self, bundle, config, gt, masks, *, full_steps=100, force_steps=4):
        self.net, self.normalizer, self.payload = bundle
        self.cfg = config
        self.gt, self.masks = gt, masks
        self.full_steps, self.force_steps = full_steps, force_steps
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
        endpoints = self.generator(z, self.full_steps)
        return endpoints, self.potential(endpoints)

    def force(self, z):
        # The short integration is ONLY a deterministic proposal drift.
        # The full 100-step integration is always used in the MH target.
        with torch.enable_grad():
            x = z.detach().requires_grad_(True)
            potential = self.potential(self.generator(x, self.force_steps)).sum()
            grad, = torch.autograd.grad(potential, x)
        return grad.detach()

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

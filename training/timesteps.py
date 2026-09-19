"""Explicit timestep distributions with noise at zero and data at one."""
from __future__ import annotations

import math
import torch

TIMESTEP_EPS = 1e-5
TIMESTEP_MODES = ('uniform', 'stratified_uniform', 'logit_normal', 'legacy_skewed')


def sample_timesteps(count, device, mode='uniform', *, generator=None,
                     logit_mean=0.0, logit_std=1.0):
    if count < 1:
        raise ValueError('Timestep sample count must be positive')
    if mode == 'uniform':
        time = torch.rand(count, device=device, generator=generator)
    elif mode == 'stratified_uniform':
        # Permute strata so the assignment to each input has a uniform marginal.
        strata = torch.randperm(count, device=device, generator=generator)
        time = (strata + torch.rand(count, device=device, generator=generator)) / count
    elif mode in ('logit_normal', 'legacy_skewed'):
        if mode == 'legacy_skewed':
            logit_mean, logit_std = 1.2, 1.2
        if not math.isfinite(logit_mean) or not math.isfinite(logit_std) or logit_std <= 0:
            raise ValueError('Logit-normal parameters must be finite, with positive std')
        time = torch.sigmoid(logit_mean + logit_std * torch.randn(count, device=device, generator=generator))
    else:
        raise ValueError(f'Unknown timestep sampling mode: {mode}')
    return time.clamp(TIMESTEP_EPS, 1 - TIMESTEP_EPS)

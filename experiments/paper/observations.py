"""Fixed observation protocols used by the published comparison cohorts."""
from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=8)
def comparison_indices(seed=0, count=100):
    import numpy as np
    rng = np.random.RandomState(seed)
    # Both channels consume a draw, including the unobserved channel.
    return tuple((rng.choice(128 * 128, 500, replace=False),
                  rng.choice(128 * 128, 500, replace=False)) for _ in range(count))


def comparison_masks(truth, indices, task, seed=0):
    import torch
    from sampling.masks import PairMasks
    rows = comparison_indices(seed)
    ma, mu = torch.zeros_like(truth.coef), torch.zeros_like(truth.sol)
    if ma.shape[-2:] != (128, 128) or len(indices) != len(ma):
        raise ValueError('The published comparison uses 128 x 128 fields')
    for row, index in enumerate(indices):
        if task in {'forward', 'both'}:
            ma[row].reshape(-1)[torch.as_tensor(rows[index][0], device=ma.device)] = 1
        if task in {'inverse', 'both'}:
            mu[row].reshape(-1)[torch.as_tensor(rows[index][1], device=mu.device)] = 1
    return PairMasks(ma, mu, dict(sensor_mode='common_ddis_random', mask_seed=seed,
                                sample_indices=list(indices), num_obs=500))


def for_job(job, truth, config):
    protocol = job.get('observation_protocol')
    if protocol is None:
        return None
    if protocol == 'common_comparison':
        return comparison_masks(truth, range(config.offset, config.offset + config.batch_size),
                                config.task, config.mask_seed)
    if protocol in {'baseline_v3', 'physics_smooth'}:
        import torch
        from sampling.masks import PairMasks
        mask = torch.zeros_like(truth.coef)
        for row in range(config.batch_size):
            # The physics baselines identify Smooth files by their original
            # release names, without the later ``_smooth`` filename suffix.
            filename = (f'{config.pde}_test_10000-128-128.mat'
                        if protocol == 'physics_smooth' else Path(config.data_path).name)
            sample_id = f'{filename}:{config.offset + row}'
            payload = f'mask|1|test|{sample_id}|0'.encode()
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**63 - 1)
            indices = torch.randperm(128 * 128, generator=torch.Generator().manual_seed(seed))[:500]
            mask[row].reshape(-1)[indices.to(mask.device)] = 1
        return PairMasks(mask, mask.clone(), dict(sensor_mode='baseline_v3', shared_mask=True,
                         base_seed=1, split='test', num_obs=500))
    raise ValueError(f'Unknown published observation protocol: {protocol}')

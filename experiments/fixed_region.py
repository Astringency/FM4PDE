"""A fixed set of observation locations restricted to the left half-domain."""
from __future__ import annotations

import torch

from sampling.masks import PairMasks


def left_half_pair_masks(coef_shape, sol_shape, num_obs=500, seed=0,
                         device="cpu", dtype=torch.float32):
    """Sample spatial locations once; reuse them across fields/channels/inputs.

    Coordinates are (row, column). The right half has no observations.
    Sharing locations preserves the effective sharing of the old prefix mask.
    A private CPU generator leaves the sampler's global RNG untouched.
    """
    b, ca, h, w = tuple(coef_shape)
    bu, cu, hu, wu = tuple(sol_shape)
    if (b, h, w) != (bu, hu, wu):
        raise ValueError("This paired regional layout requires matching grids/batches")
    width = w // 2
    if not (0 < num_obs <= h * width):
        raise ValueError("Observation count must fit in the left half-domain")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected = torch.randperm(h * width, generator=generator)[:num_obs]
    y, x = selected // width, selected % width
    spatial = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    spatial[0, 0, y.to(device), x.to(device)] = 1
    return PairMasks(
        spatial.repeat(b, ca, 1, 1), spatial.repeat(b, cu, 1, 1),
        dict(sensor_mode="fixed", ablation_layout="left_half_fixed_random_v1",
             region=dict(row_start=0, row_stop=h, column_start=0, column_stop=width),
             num_obs=int(num_obs), shared_mask=True, mask_seed_coef=int(seed),
             mask_seed_sol=int(seed), spatial_locations_per_field=int(num_obs),
             rule="without-replacement fixed-seed draw within left half; same locations for every field, channel, input and sampling step"),
    )

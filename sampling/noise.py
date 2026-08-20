from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ObservationNoise:
    clean: Any
    noisy: Any
    noise: Any
    metadata: dict[str, Any]


def add_observation_noise(
    obs: Any,
    mask: Any,
    noise_level: float,
    relative: bool = True,
    seed: int = 0,
) -> ObservationNoise:
    import torch

    if obs.shape != mask.shape:
        raise ValueError(f"Observation and mask shapes must match, got obs={obs.shape}, mask={mask.shape}")
    clean = obs * mask
    if noise_level <= 0:
        return ObservationNoise(
            clean=clean,
            noisy=clean.clone(),
            noise=torch.zeros_like(clean),
            metadata={"noise_level": noise_level, "noise_seed": seed, "relative": relative},
        )
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    eps = torch.randn(clean.shape, generator=gen, dtype=clean.dtype).to(clean.device)
    if relative:
        batch = int(clean.shape[0])
        clean_flat = clean.reshape(batch, -1)
        mask_flat = mask.reshape(batch, -1)
        counts = mask_flat.sum(dim=1)
        means = (clean_flat * mask_flat).sum(dim=1) / counts.clamp_min(1.0)
        variances = (
            ((clean_flat - means[:, None]) ** 2 * mask_flat).sum(dim=1)
            / counts.clamp_min(1.0)
        )
        scale_values = variances.sqrt().clamp_min(1e-12)
        scale_values = torch.where(counts > 0, scale_values, torch.zeros_like(scale_values))
        scale = scale_values.view(batch, *([1] * (clean.ndim - 1)))
    else:
        scale_values = torch.ones(int(clean.shape[0]), dtype=clean.dtype, device=clean.device)
        scale = scale_values.view(int(clean.shape[0]), *([1] * (clean.ndim - 1)))
    noise = noise_level * scale * eps * mask
    scale_metadata = [float(value) for value in scale_values.detach().cpu().tolist()]
    return ObservationNoise(
        clean=clean,
        noisy=clean + noise,
        noise=noise,
        metadata={
            "noise_level": noise_level,
            "noise_seed": seed,
            "relative": relative,
            "scale": scale_metadata[0] if len(scale_metadata) == 1 else scale_metadata,
            "scale_per_sample": scale_metadata,
            "scale_reduction": "per_sample_observed_std",
        },
    )

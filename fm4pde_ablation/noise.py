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
    observed = clean[mask.bool()]
    if observed.numel() == 0:
        scale = torch.as_tensor(0.0, dtype=clean.dtype, device=clean.device)
    elif relative:
        scale = observed.std(unbiased=False).clamp_min(1e-12)
    else:
        scale = torch.as_tensor(1.0, dtype=clean.dtype, device=clean.device)
    noise = noise_level * scale * eps * mask
    return ObservationNoise(
        clean=clean,
        noisy=clean + noise,
        noise=noise,
        metadata={
            "noise_level": noise_level,
            "noise_seed": seed,
            "relative": relative,
            "scale": float(scale.detach().cpu()) if hasattr(scale, "detach") else float(scale),
        },
    )

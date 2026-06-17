from __future__ import annotations

import numpy as np

try:
    from .common import (
        ChunkResult,
        FuturePDEConfig,
        ensure_finite,
        format_float_range,
        periodic_wavenumbers,
        sample_periodic_grf,
        scalar_field,
    )
except ImportError:  # pragma: no cover
    from common import (
        ChunkResult,
        FuturePDEConfig,
        ensure_finite,
        format_float_range,
        periodic_wavenumbers,
        sample_periodic_grf,
        scalar_field,
    )


B_RANGE = (-1.0, 1.0)
KAPPA_RANGE = (5e-4, 5e-3)


def advection_diffusion_metadata(config: FuturePDEConfig) -> dict[str, object]:
    return {
        "equation": "u_t + b_x u_x + b_y u_y = kappa * Delta u on [0,1]^2",
        "parameter_ranges": {
            "b_x": format_float_range(B_RANGE),
            "b_y": format_float_range(B_RANGE),
            "kappa": format_float_range(KAPPA_RANGE),
        },
        "channel_names_input": ["u0", "b_x", "b_y", "kappa"],
        "channel_names_output": ["uT", "b_x", "b_y", "kappa"],
        "channel_names_data": ["u0", "b_x", "b_y", "kappa", "uT", "b_x", "b_y", "kappa"],
    }


def solve_advection_diffusion_chunk(global_ids: np.ndarray, config: FuturePDEConfig) -> ChunkResult:
    if config.bc != "periodic":
        raise ValueError("advection_diffusion generator currently supports periodic boundary conditions")
    n = len(global_ids)
    s = config.resolution
    times = np.linspace(0.0, config.T, config.n_time, dtype=np.float64)
    kx, ky, ksq = periodic_wavenumbers(s)
    input_data = np.empty((n, 4, s, s), dtype=np.float64)
    output_data = np.empty((n, 4, s, s), dtype=np.float64)
    trajectory = np.empty((n, 1, config.n_time, s, s), dtype=np.float64) if config.save_trajectory else None
    bx_values = np.empty((n,), dtype=np.float64)
    by_values = np.empty((n,), dtype=np.float64)
    kappa_values = np.empty((n,), dtype=np.float64)

    for local_idx, sample_id in enumerate(global_ids):
        rng = np.random.default_rng(config.base_seed_train + int(sample_id))
        bx = rng.uniform(*B_RANGE)
        by = rng.uniform(*B_RANGE)
        kappa = rng.uniform(*KAPPA_RANGE)
        u0 = sample_periodic_grf(rng, s, smoothness=3.2, tau=5.0, scale=1.0)
        coeff = np.fft.fft2(u0)
        phase = bx * kx + by * ky
        frames = [
            np.fft.ifft2(coeff * np.exp(-(kappa * ksq + 1j * phase) * t)).real
            for t in times
        ]
        frames_arr = np.stack(frames, axis=0)
        bx_field = scalar_field(bx, 1, s)[0]
        by_field = scalar_field(by, 1, s)[0]
        kappa_field = scalar_field(kappa, 1, s)[0]
        input_data[local_idx] = np.stack([u0, bx_field, by_field, kappa_field], axis=0)
        output_data[local_idx] = np.stack([frames_arr[-1], bx_field, by_field, kappa_field], axis=0)
        if trajectory is not None:
            trajectory[local_idx, 0] = frames_arr
        bx_values[local_idx] = bx
        by_values[local_idx] = by
        kappa_values[local_idx] = kappa

    data = np.concatenate([input_data, output_data], axis=1)
    ensure_finite(input_data, output_data, data)
    if trajectory is not None:
        ensure_finite(trajectory)
    return ChunkResult(
        input_data=input_data,
        output_data=output_data,
        data=data,
        trajectory=trajectory,
        params={"b_x": bx_values, "b_y": by_values, "kappa": kappa_values},
    )


from __future__ import annotations

import math

import numpy as np

try:
    from .common import (
        ChunkResult,
        FuturePDEConfig,
        ensure_finite,
        finite_difference_periodic_laplacian,
        format_float_range,
        n_internal_steps_for_cfl,
        periodic_wavenumbers,
        sample_periodic_grf,
    )
except ImportError:  # pragma: no cover
    from common import (
        ChunkResult,
        FuturePDEConfig,
        ensure_finite,
        finite_difference_periodic_laplacian,
        format_float_range,
        n_internal_steps_for_cfl,
        periodic_wavenumbers,
        sample_periodic_grf,
    )


C_RANGE = (0.75, 1.25)


def wave_metadata(config: FuturePDEConfig) -> dict[str, object]:
    variable_c = bool(config.extra.get("variable_c", False))
    input_channels = ["u0", "v0", "c"] if variable_c else ["u0", "v0"]
    output_channels = ["uT", "vT", "c"] if variable_c else ["uT", "vT"]
    return {
        "equation": "u_tt = c(x,y)^2 * Delta u on [0,1]^2",
        "parameter_ranges": {"c": format_float_range(C_RANGE) if variable_c else {"fixed": 1.0}},
        "initial_velocity": "random GRF" if config.extra.get("random_v0", False) else "zero",
        "variable_c": variable_c,
        "channel_names_input": input_channels,
        "channel_names_output": output_channels,
        "channel_names_data": input_channels + output_channels,
    }


def solve_wave_chunk(global_ids: np.ndarray, config: FuturePDEConfig) -> ChunkResult:
    if config.bc != "periodic":
        raise ValueError("wave generator currently supports periodic boundary conditions")
    if config.extra.get("variable_c", False):
        return solve_wave_variable_c_chunk(global_ids, config)
    return solve_wave_constant_c_chunk(global_ids, config)


def solve_wave_constant_c_chunk(global_ids: np.ndarray, config: FuturePDEConfig) -> ChunkResult:
    n = len(global_ids)
    s = config.resolution
    times = np.linspace(0.0, config.T, config.n_time, dtype=np.float64)
    kx, _, ksq = periodic_wavenumbers(s)
    kabs = np.sqrt(ksq)
    input_data = np.empty((n, 2, s, s), dtype=np.float64)
    output_data = np.empty((n, 2, s, s), dtype=np.float64)
    trajectory = np.empty((n, 1, config.n_time, s, s), dtype=np.float64) if config.save_trajectory else None
    c_values = np.ones((n,), dtype=np.float64)

    random_v0 = bool(config.extra.get("random_v0", False))
    c = float(config.extra.get("c", 1.0))
    for local_idx, sample_id in enumerate(global_ids):
        rng = np.random.default_rng(config.base_seed_train + int(sample_id))
        u0 = sample_periodic_grf(rng, s, smoothness=5.0, tau=3.0, scale=0.5)
        v0 = sample_periodic_grf(rng, s, smoothness=5.5, tau=3.5, scale=0.1) if random_v0 else np.zeros_like(u0)
        u0_hat = np.fft.fft2(u0)
        v0_hat = np.fft.fft2(v0)
        frames = []
        v_frames = []
        for t in times:
            ck_t = c * kabs * t
            cos_term = np.cos(ck_t)
            sin_over_ck = np.zeros_like(kabs)
            nonzero = kabs > 0.0
            sin_over_ck[nonzero] = np.sin(ck_t[nonzero]) / (c * kabs[nonzero])
            u_hat = u0_hat * cos_term + v0_hat * sin_over_ck
            u_hat[0, 0] = u0_hat[0, 0] + t * v0_hat[0, 0]

            v_hat = -u0_hat * c * kabs * np.sin(ck_t) + v0_hat * cos_term
            v_hat[0, 0] = v0_hat[0, 0]
            frames.append(np.fft.ifft2(u_hat).real)
            v_frames.append(np.fft.ifft2(v_hat).real)
        frames_arr = np.stack(frames, axis=0)
        v_frames_arr = np.stack(v_frames, axis=0)
        input_data[local_idx] = np.stack([u0, v0], axis=0)
        output_data[local_idx] = np.stack([frames_arr[-1], v_frames_arr[-1]], axis=0)
        if trajectory is not None:
            trajectory[local_idx, 0] = frames_arr

    data = np.concatenate([input_data, output_data], axis=1)
    ensure_finite(input_data, output_data, data)
    if trajectory is not None:
        ensure_finite(trajectory)
    return ChunkResult(
        input_data=input_data,
        output_data=output_data,
        data=data,
        trajectory=trajectory,
        params={"c": c_values, "spectral_kx_max": np.full((n,), float(np.max(np.abs(kx))))},
    )


def solve_wave_variable_c_chunk(global_ids: np.ndarray, config: FuturePDEConfig) -> ChunkResult:
    n = len(global_ids)
    s = config.resolution
    times = np.linspace(0.0, config.T, config.n_time, dtype=np.float64)
    input_data = np.empty((n, 3, s, s), dtype=np.float64)
    output_data = np.empty((n, 3, s, s), dtype=np.float64)
    trajectory = np.empty((n, 1, config.n_time, s, s), dtype=np.float64) if config.save_trajectory else None
    c_mean = np.empty((n,), dtype=np.float64)
    c_max = np.empty((n,), dtype=np.float64)
    dt_used = np.empty((n,), dtype=np.float64)

    random_v0 = bool(config.extra.get("random_v0", False))
    for local_idx, sample_id in enumerate(global_ids):
        rng = np.random.default_rng(config.base_seed_train + int(sample_id))
        u = sample_periodic_grf(rng, s, smoothness=5.0, tau=3.0, scale=0.5)
        v = sample_periodic_grf(rng, s, smoothness=5.5, tau=3.5, scale=0.1) if random_v0 else np.zeros_like(u)
        u0 = u.copy()
        v0 = v.copy()
        c_field = np.clip(1.0 + 0.15 * sample_periodic_grf(rng, s, smoothness=4.0, tau=4.0), C_RANGE[0], C_RANGE[1])
        dx = 1.0 / s
        steps_total = n_internal_steps_for_cfl(config.T, s, float(np.max(c_field)), cfl=0.25)
        dt = config.T / steps_total
        record_frames = []
        target_idx = 0
        record_frames.append(u.copy())
        for step in range(1, steps_total + 1):
            lap = finite_difference_periodic_laplacian(u, dx)
            v_half = v + 0.5 * dt * (c_field**2) * lap
            u = u + dt * v_half
            lap_new = finite_difference_periodic_laplacian(u, dx)
            v = v_half + 0.5 * dt * (c_field**2) * lap_new
            t_now = step * dt
            while target_idx + 1 < len(times) and t_now + 0.5 * dt >= times[target_idx + 1]:
                record_frames.append(u.copy())
                target_idx += 1
        while len(record_frames) < len(times):
            record_frames.append(u.copy())
        frames_arr = np.stack(record_frames[: len(times)], axis=0)
        input_data[local_idx] = np.stack([u0, v0, c_field], axis=0)
        output_data[local_idx] = np.stack([frames_arr[-1], v, c_field], axis=0)
        if trajectory is not None:
            trajectory[local_idx, 0] = frames_arr
        c_mean[local_idx] = float(np.mean(c_field))
        c_max[local_idx] = float(np.max(c_field))
        dt_used[local_idx] = dt

    data = np.concatenate([input_data, output_data], axis=1)
    ensure_finite(input_data, output_data, data)
    if trajectory is not None:
        ensure_finite(trajectory)
    cfl = dt_used * c_max * math.sqrt(2.0) * s
    return ChunkResult(
        input_data=input_data,
        output_data=output_data,
        data=data,
        trajectory=trajectory,
        params={"c_mean": c_mean, "c_max": c_max, "dt": dt_used, "cfl": cfl},
    )

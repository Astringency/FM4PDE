from __future__ import annotations

import numpy as np

try:
    from scipy.fft import dctn, idctn
except Exception:  # pragma: no cover - scipy is an existing FM4PDE dependency
    dctn = None
    idctn = None

try:
    from .common import (
        ChunkResult,
        PairH5Config,
        ensure_finite,
        format_float_range,
        neumann_eigenvalues,
        periodic_wavenumbers,
        sample_periodic_grf,
    )
except ImportError:  # pragma: no cover
    from common import (
        ChunkResult,
        PairH5Config,
        ensure_finite,
        format_float_range,
        neumann_eigenvalues,
        periodic_wavenumbers,
        sample_periodic_grf,
    )


ALPHA_RANGE = (5e-4, 5e-3)


def heat_metadata(config: PairH5Config) -> dict[str, object]:
    return {
        "equation": "u_t = alpha * Delta u on [0,1]^2",
        "boundary_condition": config.bc,
        "parameter_ranges": {"alpha": format_float_range(ALPHA_RANGE)},
        "alpha_mode": config.extra.get("alpha_mode", "random"),
        "alpha_random": config.extra.get("alpha_mode", "random") == "random",
        "fixed_alpha": float(config.extra.get("alpha", 1e-3)),
        "hdf5_schema": {
            "input_data": "[N,1,H,W] u0",
            "output_data": "[N,1,H,W] uT",
            "alpha": "[N] only when alpha_mode=random",
            "fixed_alpha": "root attr only when alpha_mode=fixed",
        },
        "loader_materialized_schema_random_alpha": ["u0", "alpha", "uT", "alpha"],
        "loader_materialized_schema_fixed_alpha": ["u0", "uT"],
        "channel_names_input": ["u0"],
        "channel_names_output": ["uT"],
        "channel_names_model_random_alpha": ["u0", "alpha", "uT", "alpha"],
        "channel_names_model_fixed_alpha": ["u0", "uT"],
    }


def solve_heat_chunk(global_ids: np.ndarray, config: PairH5Config) -> ChunkResult:
    n = len(global_ids)
    s = config.resolution
    times = np.linspace(0.0, config.T, config.n_time, dtype=np.float64)
    input_data = np.empty((n, 1, s, s), dtype=np.float64)
    output_data = np.empty((n, 1, s, s), dtype=np.float64)
    trajectory = np.empty((n, 1, config.n_time, s, s), dtype=np.float64) if config.save_trajectory else None
    alpha_mode = config.extra.get("alpha_mode", "random")
    if alpha_mode not in {"random", "fixed"}:
        raise ValueError(f"Unsupported alpha_mode={alpha_mode!r}")
    fixed_alpha = float(config.extra.get("alpha", 1e-3))
    alpha = np.empty((n,), dtype=np.float64)

    if config.bc == "periodic":
        _, _, ksq = periodic_wavenumbers(s)
    elif config.bc == "neumann":
        if dctn is None or idctn is None:
            raise RuntimeError("scipy.fft.dctn/idctn are required for --bc neumann")
        ksq = neumann_eigenvalues(s)
    else:
        raise ValueError(f"Unsupported heat boundary condition: {config.bc}")

    for local_idx, sample_id in enumerate(global_ids):
        rng = np.random.default_rng(config.base_seed_train + int(sample_id))
        alpha_i = rng.uniform(*ALPHA_RANGE) if alpha_mode == "random" else fixed_alpha
        u0 = sample_periodic_grf(rng, s, smoothness=3.2, tau=5.0, scale=1.0)
        alpha[local_idx] = alpha_i
        input_data[local_idx, 0] = u0
        if config.bc == "periodic":
            coeff = np.fft.fft2(u0)
            frames = [np.fft.ifft2(coeff * np.exp(-alpha_i * ksq * t)).real for t in times]
        else:
            coeff = dctn(u0, type=2, norm="ortho")
            frames = [idctn(coeff * np.exp(-alpha_i * ksq * t), type=2, norm="ortho") for t in times]
        frames_arr = np.stack(frames, axis=0)
        output_data[local_idx, 0] = frames_arr[-1]
        if trajectory is not None:
            trajectory[local_idx, 0] = frames_arr

    data = None
    if config.materialize_constant_fields:
        if alpha_mode == "random":
            alpha_field = alpha[:, None, None, None] * np.ones((n, 1, s, s), dtype=np.float64)
            data = np.concatenate([input_data, alpha_field, output_data, alpha_field], axis=1)
        else:
            data = np.concatenate([input_data, output_data], axis=1)
    ensure_finite(input_data, output_data)
    if data is not None:
        ensure_finite(data)
    if trajectory is not None:
        ensure_finite(trajectory)
    return ChunkResult(
        input_data=input_data,
        output_data=output_data,
        data=data,
        trajectory=trajectory,
        params={"alpha": alpha} if alpha_mode == "random" else {},
    )

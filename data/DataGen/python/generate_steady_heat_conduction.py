from __future__ import annotations

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
except Exception:  # pragma: no cover - scipy is an existing FM4PDE dependency
    coo_matrix = None
    spsolve = None

try:
    from .common import ChunkResult, PairH5Config, ensure_finite, format_float_range
except ImportError:  # pragma: no cover
    from common import ChunkResult, PairH5Config, ensure_finite, format_float_range


UD_RANGE = (288.0, 308.0)
SOURCE_COUNT_RANGE = (1, 4)
SOURCE_AMPLITUDE_RANGE = (10.0, 60.0)
SOURCE_SIGMA_RANGE = (0.04, 0.12)
LAMBDA_MIN = 0.1


def steady_heat_conduction_metadata(config: PairH5Config) -> dict[str, object]:
    return {
        "equation": "-div(lambda(u) grad u) = f, lambda(u)=1+0.05*(u-298)",
        "boundary_condition": "bottom Dirichlet u=u_D; top/left/right zero Neumann",
        "parameter_ranges": {
            "u_D": format_float_range(UD_RANGE),
            "n_sources": {"min": SOURCE_COUNT_RANGE[0], "max": SOURCE_COUNT_RANGE[1]},
            "source_amplitude": format_float_range(SOURCE_AMPLITUDE_RANGE),
            "source_sigma": format_float_range(SOURCE_SIGMA_RANGE),
        },
        "recfno_style": True,
        "time_dependent": False,
        "lambda_min_clamp": LAMBDA_MIN,
        "hdf5_schema": {
            "input_data": "[N,1,H,W] heat source f",
            "output_data": "[N,1,H,W] temperature u",
            "u_D": "[N]",
            "picard_iters": "[N]",
            "converged": "[N]",
            "residual_norm": "[N]",
        },
        "channel_names_input": ["f"],
        "channel_names_output": ["u"],
        "channel_names_model": ["f", "u_D", "u", "u_D"],
    }


def solve_steady_heat_conduction_chunk(global_ids: np.ndarray, config: PairH5Config) -> ChunkResult:
    if coo_matrix is None or spsolve is None:
        raise RuntimeError("scipy.sparse is required for steady_heat_conduction")
    n = len(global_ids)
    s = config.resolution
    input_data = np.empty((n, 1, s, s), dtype=np.float64)
    output_data = np.empty((n, 1, s, s), dtype=np.float64)
    u_d_values = np.empty((n,), dtype=np.float64)
    residual_norm = np.empty((n,), dtype=np.float64)
    picard_iters = np.empty((n,), dtype=np.int32)
    converged = np.empty((n,), dtype=np.int8)
    max_sources = SOURCE_COUNT_RANGE[1]
    n_sources_arr = np.empty((n,), dtype=np.int32)
    source_x = np.zeros((n, max_sources), dtype=np.float64)
    source_y = np.zeros((n, max_sources), dtype=np.float64)
    source_amp = np.zeros((n, max_sources), dtype=np.float64)
    source_sigma = np.zeros((n, max_sources), dtype=np.float64)

    for local_idx, sample_id in enumerate(global_ids):
        rng = np.random.default_rng(config.base_seed_train + int(sample_id))
        f, params = sample_heat_sources(rng, s)
        u_d = rng.uniform(*UD_RANGE)
        result = solve_nonlinear_heat(f, u_d, config)
        input_data[local_idx, 0] = f
        output_data[local_idx, 0] = result["u"]
        u_d_values[local_idx] = u_d
        residual_norm[local_idx] = result["residual_norm"]
        picard_iters[local_idx] = result["picard_iters"]
        converged[local_idx] = 1 if result["converged"] else 0
        count = int(params["n_sources"])
        n_sources_arr[local_idx] = count
        source_x[local_idx, :count] = params["x"]
        source_y[local_idx, :count] = params["y"]
        source_amp[local_idx, :count] = params["amp"]
        source_sigma[local_idx, :count] = params["sigma"]

    data = None
    if config.materialize_constant_fields:
        u_d_field = u_d_values[:, None, None, None] * np.ones((n, 1, s, s), dtype=np.float64)
        data = np.concatenate([input_data, u_d_field, output_data, u_d_field], axis=1)
    ensure_finite(input_data, output_data, residual_norm)
    if data is not None:
        ensure_finite(data)
    return ChunkResult(
        input_data=input_data,
        output_data=output_data,
        data=data,
        trajectory=None,
        params={
            "u_D": u_d_values,
            "residual_norm": residual_norm,
            "picard_iters": picard_iters,
            "converged": converged,
            "n_sources": n_sources_arr,
            "source_x": source_x,
            "source_y": source_y,
            "source_amp": source_amp,
            "source_sigma": source_sigma,
        },
    )


def sample_heat_sources(rng: np.random.Generator, resolution: int) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    x = np.linspace(0.0, 1.0, resolution)
    y = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    n_sources = int(rng.integers(SOURCE_COUNT_RANGE[0], SOURCE_COUNT_RANGE[1] + 1))
    xs = rng.uniform(0.15, 0.85, size=n_sources)
    ys = rng.uniform(0.15, 0.90, size=n_sources)
    amps = rng.uniform(*SOURCE_AMPLITUDE_RANGE, size=n_sources)
    sigmas = rng.uniform(*SOURCE_SIGMA_RANGE, size=n_sources)
    f = np.zeros((resolution, resolution), dtype=np.float64)
    for x0, y0, amp, sigma in zip(xs, ys, amps, sigmas):
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        f += amp * np.exp(-r2 / (2.0 * sigma**2))
    return f, {"n_sources": n_sources, "x": xs, "y": ys, "amp": amps, "sigma": sigmas}


def solve_nonlinear_heat(f: np.ndarray, u_d: float, config: PairH5Config) -> dict[str, object]:
    s = f.shape[0]
    max_iter = int(config.extra.get("picard_max_iter", 30))
    tol = float(config.extra.get("picard_tol", 1e-5))
    u = np.full_like(f, u_d, dtype=np.float64)
    converged = False
    for it in range(1, max_iter + 1):
        conductivity = conductivity_lambda(u)
        matrix, rhs = build_linear_system(conductivity, f, u_d)
        u_new = np.asarray(spsolve(matrix, rhs), dtype=np.float64).reshape(s, s)
        delta = np.linalg.norm(u_new - u) / (np.linalg.norm(u) + 1e-12)
        u = u_new
        if delta < tol:
            converged = True
            break
    u[0, :] = u_d
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    u[-1, :] = u[-2, :]
    conductivity = conductivity_lambda(u)
    residual = nonlinear_residual(u, conductivity, f)
    return {
        "u": u,
        "residual_norm": float(np.linalg.norm(residual) / np.sqrt(max(residual.size, 1))),
        "picard_iters": it,
        "converged": converged,
    }


def conductivity_lambda(u: np.ndarray) -> np.ndarray:
    return np.maximum(1.0 + 0.05 * (u - 298.0), LAMBDA_MIN)


def build_linear_system(conductivity: np.ndarray, f: np.ndarray, u_d: float) -> tuple[object, np.ndarray]:
    s = conductivity.shape[0]
    dx = 1.0 / (s - 1)
    inv_dx2 = 1.0 / (dx * dx)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(s * s, dtype=np.float64)

    def idx(i: int, j: int) -> int:
        return i * s + j

    for i in range(s):
        for j in range(s):
            row = idx(i, j)
            if i == 0:
                rows.append(row)
                cols.append(row)
                vals.append(1.0)
                rhs[row] = u_d
                continue
            if j == 0:
                rows.extend([row, row])
                cols.extend([idx(i, 0), idx(i, 1)])
                vals.extend([1.0, -1.0])
                continue
            if j == s - 1:
                rows.extend([row, row])
                cols.extend([idx(i, s - 1), idx(i, s - 2)])
                vals.extend([1.0, -1.0])
                continue
            if i == s - 1:
                rows.extend([row, row])
                cols.extend([idx(s - 1, j), idx(s - 2, j)])
                vals.extend([1.0, -1.0])
                continue

            center = 0.0
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                face = 0.5 * (conductivity[i, j] + conductivity[ni, nj]) * inv_dx2
                center += face
                rows.append(row)
                cols.append(idx(ni, nj))
                vals.append(-face)
            rows.append(row)
            cols.append(row)
            vals.append(center)
            rhs[row] = f[i, j]

    return coo_matrix((vals, (rows, cols)), shape=(s * s, s * s)).tocsr(), rhs


def nonlinear_residual(u: np.ndarray, conductivity: np.ndarray, f: np.ndarray) -> np.ndarray:
    s = u.shape[0]
    dx = 1.0 / (s - 1)
    inv_dx2 = 1.0 / (dx * dx)
    residual = np.zeros((max(s - 2, 0), max(s - 2, 0)), dtype=np.float64)
    for i in range(1, s - 1):
        for j in range(1, s - 1):
            accum = 0.0
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                face = 0.5 * (conductivity[i, j] + conductivity[ni, nj]) * inv_dx2
                accum += face * (u[i, j] - u[ni, nj])
            residual[i - 1, j - 1] = accum - f[i, j]
    return residual


import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
scipy_interpolate = pytest.importorskip("scipy.interpolate")

from sampling.pde_residuals import compute_pde_residual


def _cubic_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return scipy_interpolate.CubicSpline(source, np.eye(source.size), axis=0)(target)


def test_darcy_residual_reconstructs_the_generator_nodal_system():
    n = 12
    cell = (np.arange(n) + 0.5) / n
    nodal = np.linspace(0.0, 1.0, n)
    cell_to_nodal = _cubic_matrix(cell, nodal)
    nodal_to_cell = _cubic_matrix(nodal, cell)

    rng = np.random.default_rng(7)
    coeff_cell = np.where(rng.standard_normal((n, n)) >= 0.0, 12.0, 4.0)
    coeff_nodal = cell_to_nodal @ coeff_cell @ cell_to_nodal.T
    h = 1.0 / (n - 1)
    interior_size = (n - 2) ** 2
    matrix = np.zeros((interior_size, interior_size), dtype=np.float64)
    rhs = np.ones(interior_size, dtype=np.float64)

    def index(i: int, j: int) -> int:
        return (i - 1) * (n - 2) + (j - 1)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            row = index(i, j)
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                face = 0.5 * (coeff_nodal[i, j] + coeff_nodal[ni, nj]) / h**2
                matrix[row, row] += face
                if 0 < ni < n - 1 and 0 < nj < n - 1:
                    matrix[row, index(ni, nj)] -= face
    solution_nodal = np.zeros((n, n), dtype=np.float64)
    solution_nodal[1:-1, 1:-1] = np.linalg.solve(matrix, rhs).reshape(n - 2, n - 2)
    solution_cell = nodal_to_cell @ solution_nodal @ nodal_to_cell.T

    coeff = torch.tensor(coeff_cell, dtype=torch.float64).reshape(1, 1, n, n)
    solution = torch.tensor(solution_cell, dtype=torch.float64).reshape(1, 1, n, n)
    output = compute_pde_residual("darcy", coeff, solution)
    assert output.metadata["discretization"] == "matlab_spline_nodal_conservative"
    assert output.components["interior"].square().mean().sqrt().item() < 1e-8
    assert output.components["boundary"].square().mean().sqrt().item() < 1e-8


def test_helmholtz_residual_matches_the_actual_kronecker_generator_rows():
    n = 10
    k = 2
    h = 1.0 / (n - 1)
    one_d = np.diag(np.full(n, -2.0))
    one_d += np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    one_d /= h**2
    one_d[0] = 0.0
    one_d[0, 0] = 1.0
    one_d[-1] = 0.0
    one_d[-1, -1] = 1.0
    rng = np.random.default_rng(3)
    source = rng.standard_normal((n, n))
    rhs = source.copy()
    rhs[[0, -1], :] = 0.0
    rhs[:, [0, -1]] = 0.0
    matrix = np.kron(np.eye(n), one_d) + np.kron(one_d, np.eye(n)) + k**2 * np.eye(n * n)
    solution = np.linalg.solve(matrix, rhs.reshape(-1, order="F")).reshape(n, n, order="F")
    output = compute_pde_residual(
        "helmholtz",
        torch.tensor(source, dtype=torch.float64).reshape(1, 1, n, n),
        torch.tensor(solution, dtype=torch.float64).reshape(1, 1, n, n),
        k=k,
    )
    assert output.metadata["discretization"] == "matlab_kronecker_generator"
    assert output.components["boundary"] is None
    assert output.components["interior"].square().mean().sqrt().item() < 1e-8


def test_burgers_uses_128_saved_frames_with_dt_one_over_127():
    n = 128
    times = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    trajectory = times.reshape(1, 1, n, 1).expand(1, 1, n, n).clone()
    output = compute_pde_residual("burger", trajectory, trajectory, pde_params={"nu": 0.0})
    interior = output.components["interior"]
    assert interior.shape == (1, 1, 126, 128)
    assert torch.allclose(interior, torch.ones_like(interior), atol=1e-12, rtol=1e-12)
    assert output.metadata["dt"] == pytest.approx(1.0 / 127.0)
    assert output.metadata["dx"] == pytest.approx(1.0 / 128.0)


def test_wave_full_trajectory_accepts_generator_displacement_only_storage():
    n = 16
    time_points = 9
    dt = 0.02
    grid = torch.arange(n, dtype=torch.float64) / n
    xx, _ = torch.meshgrid(grid, grid, indexing="ij")
    spatial = torch.sin(2.0 * math.pi * xx)
    theta = math.acos(1.0 - 0.5 * dt**2 * (2.0 * math.pi) ** 2)
    trajectory = torch.stack([math.cos(i * theta) * spatial for i in range(time_points)], dim=0)
    trajectory = trajectory.reshape(1, time_points, 1, n, n)
    q0 = torch.zeros(1, 2, n, n, dtype=torch.float64)
    qT = torch.zeros_like(q0)
    q0[:, :1] = trajectory[:, 0]
    qT[:, :1] = trajectory[:, -1]
    output = compute_pde_residual(
        "wave",
        q0,
        qT,
        pde_params={"trajectory": trajectory, "trajectory_dt": dt, "c": 1.0},
        residual_mode="full_trajectory_fd",
    )
    assert output.metadata["wave_state_form"] == "displacement_only_second_order"
    assert output.components["interior"].square().mean().sqrt().item() < 1e-8


def test_ns_full_trajectory_does_not_treat_internal_solver_dt_as_snapshot_dt():
    n = 8
    trajectory = torch.zeros(1, 11, 1, n, n, dtype=torch.float64)
    q0 = trajectory[:, 0]
    qT = trajectory[:, -1]
    output = compute_pde_residual(
        "nsnonbounded",
        q0,
        qT,
        pde_params={"trajectory": trajectory, "dt": 1e-4, "T": 1.0, "forcing": 0.0},
        residual_mode="full_trajectory_fd",
    )
    assert output.metadata["time_delta"]["source"] == "T/(n_time-1)"
    assert output.metadata["time_delta"]["values"] == pytest.approx(0.1)


def test_advection_diffusion_x_is_the_first_stored_spatial_axis():
    n = 32
    grid = torch.arange(n, dtype=torch.float64) / n
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    field = torch.sin(2.0 * math.pi * xx) + 0.3 * torch.cos(4.0 * math.pi * yy)
    q = field.reshape(1, 1, n, n)
    output = compute_pde_residual(
        "advection_diffusion",
        q,
        q,
        pde_params={"b_x": 1.0, "b_y": 0.0, "kappa": 0.0, "T": 1.0},
        residual_mode="endpoint_secant",
    )
    expected = (2.0 * math.pi * torch.cos(2.0 * math.pi * xx)).reshape(1, 1, n, n)
    assert torch.allclose(output.components["interior"], expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "pde,params,missing",
    [
        ("heat", {"T": 1.0}, "alpha"),
        ("advection_diffusion", {"b_x": 0.0, "b_y": 0.0, "T": 1.0}, "kappa"),
    ],
)
def test_random_sample_parameters_must_not_silently_default(pde, params, missing):
    q = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match=missing):
        compute_pde_residual(pde, q, q, pde_params=params, residual_mode="endpoint_secant")


def test_steady_heat_random_boundary_temperature_must_not_silently_default():
    q = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="u_D"):
        compute_pde_residual("steady_heat_conduction", q, q, pde_params={})

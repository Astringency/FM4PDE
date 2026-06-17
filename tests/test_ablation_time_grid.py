import pytest

torch = pytest.importorskip("torch")

from fm4pde_ablation.time_grid import affine_coefficients, make_time_grid, scheduler_coefficients


def test_time_grid_has_num_steps_plus_one():
    grid = make_time_grid("uniform", 5)
    assert len(grid) == 6
    assert grid[0].item() == 0
    assert grid[-1].item() == 1


def test_scheduler_no_tuple_for_vp():
    coeffs = scheduler_coefficients(torch.tensor(0.5), scheduler="VP")
    assert hasattr(coeffs.alpha_t, "shape")
    affine = affine_coefficients(coeffs)
    assert torch.isfinite(affine.b_t)

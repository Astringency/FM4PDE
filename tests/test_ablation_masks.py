import pytest

torch = pytest.importorskip("torch")

from sampling.masks import make_mask, make_pair_masks, residual_region_mask


def test_random_mask_reproducible():
    a = make_mask((1, 1, 8, 8), 5, "random", seed=7)
    b = make_mask((1, 1, 8, 8), 5, "random", seed=7)
    assert torch.equal(a, b)
    assert int(a.sum()) == 5


def test_random_mask_is_shared_across_batch():
    mask = make_mask((3, 1, 8, 8), 5, "random", seed=7)
    assert torch.equal(mask[0], mask[1])
    assert torch.equal(mask[1], mask[2])
    assert int(mask.sum()) == 15


def test_per_sample_random_mask_is_independent_across_batch():
    mask = make_mask((3, 1, 8, 8), 5, "per_sample_random", seed=7)
    assert not torch.equal(mask[0], mask[1])
    assert not torch.equal(mask[1], mask[2])
    assert int(mask.sum()) == 15


def test_pair_shared_mask_channel_repeat():
    masks = make_pair_masks((1, 1, 8, 8), (1, 2, 8, 8), 4, "grid", True, 0)
    assert masks.coef.shape == (1, 1, 8, 8)
    assert masks.sol.shape == (1, 2, 8, 8)
    assert torch.equal(masks.sol[:, 0:1], masks.coef)


def test_sensor_column_budget_counts_complete_columns():
    mask = make_mask(
        (2, 1, 6, 9),
        999,
        "sensor_column",
        seed=3,
        num_sensor_columns=4,
    )
    assert int(mask[0].sum()) == 4 * 6
    assert torch.equal(mask[0], mask[1])
    with pytest.raises(ValueError, match="exceeds spatial width"):
        make_mask((1, 1, 6, 3), 0, "sensor_column", 0, num_sensor_columns=4)


def test_residual_observation_regions_are_task_aware():
    coef = torch.zeros(1, 1, 4, 4)
    sol = torch.zeros_like(coef)
    coef[..., 0, 0] = 1
    sol[..., 3, 3] = 1
    forward = residual_region_mask("active_obs_union", coef, sol, (1, 1, 4, 4), task="forward")
    inverse = residual_region_mask("active_obs_union", coef, sol, (1, 1, 4, 4), task="inverse")
    both = residual_region_mask("active_obs_union", coef, sol, (1, 1, 4, 4), task="both")
    assert torch.equal(forward, coef)
    assert torch.equal(inverse, sol)
    assert torch.equal(both, torch.maximum(coef, sol))
    with pytest.raises(ValueError, match="not active"):
        residual_region_mask("coef_obs", coef, sol, (1, 1, 4, 4), task="inverse")

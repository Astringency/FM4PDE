import pytest

torch = pytest.importorskip("torch")

from fm4pde_ablation.masks import make_mask, make_pair_masks


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

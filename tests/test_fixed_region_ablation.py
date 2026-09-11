import pytest
import torch

from experiments.fixed_region import left_half_pair_masks
from sampling.masks import make_mask


def test_half_domain_count_shared_channels_and_rng_independence():
    torch.manual_seed(73)
    state = torch.random.get_rng_state().clone()
    masks = left_half_pair_masks((2, 1, 128, 128), (2, 3, 128, 128), 500, 0)
    assert torch.equal(state, torch.random.get_rng_state())
    for mask in (masks.coef, masks.sol):
        assert torch.equal(mask.sum((-1, -2)), torch.full(mask.shape[:2], 500.0))
        assert not mask[..., 64:].any()
        assert torch.equal(mask, masks.coef[:, :1].expand_as(mask))
    assert (masks.coef[0, 0].sum(1) > 0).sum() > 100
    assert torch.equal(masks.coef, left_half_pair_masks((2, 1, 128, 128), (2, 3, 128, 128), 500, 0).coef)
    assert not torch.equal(masks.coef, left_half_pair_masks((2, 1, 128, 128), (2, 3, 128, 128), 500, 1).coef)


def test_production_fixed_keeps_original_contract():
    legacy = make_mask((1, 1, 128, 128), 500, "fixed", 0).flatten()
    assert legacy[:500].all() and not legacy[500:].any()


def test_oversubscribed_region_fails_instead_of_silently_changing_budget():
    with pytest.raises(ValueError, match="fit"):
        left_half_pair_masks((1, 1, 8, 8), (1, 1, 8, 8), 33)

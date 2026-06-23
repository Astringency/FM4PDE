from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.losses import _masked_mse, _pde_loss, compute_guidance_losses
from sampling.masks import PairMasks
from sampling.state import SplitState


@dataclass
class GT:
    coef: object
    sol: object


def test_obs_only_losses():
    cfg = AblationConfig(task="both", guidance_components="obs_only", loss_type="mse")
    pred = SplitState(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4) * 2)
    gt = GT(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    masks = PairMasks(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), {})
    out = compute_guidance_losses(pred, gt, masks, cfg)
    assert out.L_obs_a.item() == pytest.approx(1.0)
    assert out.L_obs_u.item() == pytest.approx(4.0)
    assert out.L_pde.item() == pytest.approx(0.0)
    assert out.metadata["loss_reduction"]["obs_a"] == "masked_mse_over_observed_entries"
    assert out.metadata["obs_counts"]["coef"] == pytest.approx(16.0)


def test_masked_mse_uses_observed_denominator():
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    assert _masked_mse(pred, target, mask).item() == pytest.approx(2.0)


def test_masked_mse_broadcasts_single_channel_mask():
    pred = torch.ones(1, 2, 2, 2)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    assert mask.sum().item() == 2.0
    assert _masked_mse(pred, target, mask).item() == pytest.approx(1.0)


def test_masked_mse_zero_mask_is_finite_zero():
    pred = torch.ones(1, 1, 2, 2)
    target = torch.zeros_like(pred)
    mask = torch.zeros_like(pred)

    out = _masked_mse(pred, target, mask)
    assert torch.isfinite(out)
    assert out.item() == pytest.approx(0.0)


def test_pde_loss_uses_mse_not_legacy_l2_norm():
    residual = torch.tensor([1.0, 2.0, 3.0])

    assert _pde_loss(residual, "mse").item() == pytest.approx(14.0 / 3.0)

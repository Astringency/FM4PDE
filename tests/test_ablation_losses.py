from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.losses import compute_guidance_losses
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

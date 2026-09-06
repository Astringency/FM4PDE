from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.losses import _componentwise_pde_rms_loss, _mean_per_sample_rms, compute_guidance_losses
from sampling.masks import PairMasks
from sampling.state import SplitState


def test_rms_uses_each_samples_active_count_and_broadcasts_mask():
    residual = torch.tensor([[[[3., 4., 99.]]], [[[12., 99., 99.]]]]).expand(-1, 2, -1, -1)
    mask = torch.tensor([[[[1., 1., 0.]]], [[[1., 0., 0.]]]])
    assert _mean_per_sample_rms(residual, mask).item() == pytest.approx((5 / 2**0.5 + 12) / 2)


@pytest.mark.parametrize("empty_mask", [False, True])
def test_rms_zero_loss_has_finite_zero_gradient(empty_mask):
    residual = torch.ones(2, 1, 3, 3) if empty_mask else torch.zeros(2, 1, 3, 3)
    residual.requires_grad_()
    mask = torch.zeros_like(residual) if empty_mask else torch.ones_like(residual)
    loss = _mean_per_sample_rms(residual, mask)
    loss.backward()
    assert loss.item() == 0
    assert torch.isfinite(residual.grad).all()
    assert torch.count_nonzero(residual.grad) == 0


def test_rms_loss_scales_linearly_but_gradient_does_not_grow_with_residual():
    residual = torch.tensor([[[3., 4.]]], requires_grad=True)
    large = (residual.detach() * 1000).requires_grad_()
    loss = _mean_per_sample_rms(residual)
    large_loss = _mean_per_sample_rms(large)
    loss.backward()
    large_loss.backward()
    assert large_loss.item() == pytest.approx(1000 * loss.item())
    assert torch.allclose(large.grad, residual.grad)


def test_rms_preserves_separate_component_weights_and_empty_components():
    interior = torch.tensor([[[3., 4.]]])
    total, parts = _componentwise_pde_rms_loss(
        {"interior": interior, "boundary": torch.tensor([[[4.]]]), "endpoint": torch.empty(1, 0)},
        bc_weight=0.5, endpoint_weight=2, fallback_field=interior,
    )
    assert total.item() == pytest.approx(5 / 2**0.5 + 2)
    assert parts["boundary"] == 4
    assert parts["endpoint"] == 0
    assert parts["total"] == pytest.approx(total.item())


def test_rms_guidance_preserves_evaluation_losses_and_observation_guidance():
    cfg = AblationConfig(pde="poisson", guidance_components="obs_pde", boundary_condition_mode="none")
    coef = torch.full((2, 1, 8, 8), 3.)
    sol = torch.zeros_like(coef)
    pred = SplitState(coef, sol)
    gt = SplitState(torch.zeros_like(coef), torch.zeros_like(sol))
    masks = PairMasks(torch.ones_like(coef), torch.ones_like(sol), {})
    mse = compute_guidance_losses(pred, gt, masks, cfg)
    rms = compute_guidance_losses(pred, gt, masks, replace(cfg, pde_guidance_reduction="rms"))
    assert rms.L_pde.item() == mse.L_pde.item()
    assert torch.equal(rms.pde_residual, mse.pde_residual)
    assert rms.guidance_L_pde.item() == pytest.approx(mse.L_pde.item()**0.5)
    assert rms.guidance_L_obs_a.item() == mse.guidance_L_obs_a.item()
    assert rms.guidance_L_obs_u.item() == mse.guidance_L_obs_u.item()
    assert rms.metadata["pde"]["guidance_component_losses"]["total"] == pytest.approx(rms.guidance_L_pde.item())
    assert rms.metadata["guidance_loss_reduction"]["pde"] == "componentwise_mean_per_sample_rms_sum"


def test_rms_config_has_distinct_run_name():
    cfg = AblationConfig(pde_guidance_reduction="rms")
    cfg.validate()
    assert "_pdered-rms" in cfg.resolved_ablation_name()
    assert "_pdered-" not in AblationConfig().resolved_ablation_name()
    with pytest.raises(ValueError, match="requires guidance_operator"):
        AblationConfig(pde_guidance_reduction="rms", guidance_operator="legacy").validate()

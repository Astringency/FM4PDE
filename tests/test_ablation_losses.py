from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.losses import _componentwise_pde_mse_loss, _masked_mse, compute_guidance_losses
from sampling.masks import PairMasks
from sampling.state import SplitState


@dataclass
class GT:
    coef: object
    sol: object


def test_obs_only_losses():
    cfg = AblationConfig(task="both", guidance_components="obs_only")
    pred = SplitState(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4) * 2)
    gt = GT(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    masks = PairMasks(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), {})
    out = compute_guidance_losses(pred, gt, masks, cfg)
    assert out.L_obs_a.item() == pytest.approx(1.0)
    assert out.L_obs_u.item() == pytest.approx(4.0)
    assert out.L_pde.item() > 0.0
    assert out.guidance_L_pde.item() == pytest.approx(0.0)
    assert out.metadata["loss_reduction"]["obs_a"] == "masked_mse_over_observed_entries"
    assert out.metadata["loss_reduction"]["pde"] == "componentwise_mse_sum"
    assert out.metadata["obs_counts"]["coef"] == pytest.approx(16.0)


def test_eval_pde_loss_is_independent_of_guidance_flags_and_zeta():
    coef = torch.ones(1, 1, 5, 5)
    sol = torch.zeros_like(coef)
    gt = GT(torch.zeros_like(coef), torch.zeros_like(sol))
    gt.pde_params = {}
    masks = PairMasks(torch.ones_like(coef), torch.ones_like(sol), {})
    obs_cfg = AblationConfig(
        pde="poisson",
        task="both",
        guidance_components="obs_only",
        zeta_pde=0.0,
        boundary_condition_mode="none",
    )
    pde_cfg = AblationConfig(
        pde="poisson",
        task="both",
        guidance_components="pde_only",
        zeta_pde=7.0,
        boundary_condition_mode="none",
    )
    obs_out = compute_guidance_losses(SplitState(coef, sol), gt, masks, obs_cfg)
    pde_out = compute_guidance_losses(SplitState(coef, sol), gt, masks, pde_cfg)
    assert torch.allclose(obs_out.L_pde, pde_out.L_pde)
    assert obs_out.guidance_L_pde.item() == pytest.approx(0.0)
    assert torch.allclose(pde_out.guidance_L_pde, pde_out.L_pde)


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


def test_pde_componentwise_mse_sums_components_with_separate_denominators():
    interior = torch.tensor([1.0, 2.0, 3.0])
    boundary = torch.tensor([4.0])
    initial = None
    endpoint = torch.tensor([2.0, 2.0])

    total, parts = _componentwise_pde_mse_loss(
        {
            "interior": interior,
            "boundary": boundary,
            "initial": initial,
            "endpoint": endpoint,
        },
        bc_weight=0.5,
        ic_weight=1.0,
        endpoint_weight=2.0,
        fallback_field=interior,
    )

    expected = (1.0 + 4.0 + 9.0) / 3.0 + 0.5 * 16.0 + 2.0 * 4.0
    assert total.item() == pytest.approx(expected)
    assert parts["interior"] == pytest.approx(14.0 / 3.0)
    assert parts["boundary"] == pytest.approx(16.0)
    assert parts["endpoint"] == pytest.approx(4.0)


def test_pde_interior_mask_is_normalized_over_active_entries_per_sample():
    interior = torch.ones(2, 1, 4, 4)
    mask = torch.zeros_like(interior)
    mask[0, :, 0, 0] = 1.0
    mask[1, :, :2, :2] = 1.0

    total, parts = _componentwise_pde_mse_loss(
        {
            "interior": interior * mask,
            "interior_mask": mask,
            "boundary": None,
            "initial": None,
            "endpoint": None,
        },
        bc_weight=1.0,
        ic_weight=1.0,
        endpoint_weight=1.0,
        fallback_field=interior,
    )

    assert total.item() == pytest.approx(1.0)
    assert parts["interior"] == pytest.approx(1.0)


def test_pde_guidance_residual_error_fails_closed():
    cfg = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="pde_only",
        residual_mode="endpoint_secant",
        boundary_condition_mode="periodic",
        initial_condition_mode="none",
    )
    q = torch.zeros(1, 1, 8, 8)
    gt = GT(q, q)
    gt.pde_params = {}
    masks = PairMasks(torch.zeros_like(q), torch.zeros_like(q), {})

    with pytest.raises(RuntimeError, match="PDE residual computation failed for generated"):
        compute_guidance_losses(SplitState(q, q), gt, masks, cfg)


def test_pde_evaluation_error_fails_even_when_pde_guidance_is_disabled():
    cfg = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="obs_only",
        residual_mode="endpoint_secant",
        boundary_condition_mode="periodic",
        initial_condition_mode="none",
    )
    q = torch.zeros(1, 1, 8, 8)
    gt = GT(q, q)
    gt.pde_params = {}
    masks = PairMasks(torch.ones_like(q), torch.ones_like(q), {})

    with pytest.raises(RuntimeError, match="PDE residual computation failed for generated"):
        compute_guidance_losses(SplitState(q, q), gt, masks, cfg)


def test_observed_full_trajectory_is_rejected_for_endpoint_models():
    cfg = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="obs_only",
        residual_mode="full_trajectory_fd",
    )
    q0 = torch.zeros(1, 1, 8, 8)
    qT = torch.full_like(q0, 123.0)
    gt = GT(q0, qT)
    gt.pde_params = {
        "trajectory": torch.zeros(1, 3, 1, 8, 8),
        "trajectory_is_observed_ground_truth": True,
        "alpha": torch.tensor([1e-3]),
    }
    masks = PairMasks(torch.ones_like(q0), torch.ones_like(qT), {})

    with pytest.raises(ValueError, match="Only Burgers"):
        compute_guidance_losses(SplitState(q0, qT), gt, masks, cfg)


def test_nsnonbounded_pde_only_guidance_loss_is_nonzero_from_default_forcing():
    cfg = AblationConfig(
        pde="nsnonbounded",
        task="both",
        guidance_components="pde_only",
        residual_mode="endpoint_secant",
        boundary_condition_mode="periodic",
        initial_condition_mode="none",
    )
    w0 = torch.zeros(1, 1, 8, 8)
    wT = torch.zeros_like(w0)
    gt = GT(w0, wT)
    gt.pde_params = {"T": torch.tensor([1.0]), "nu": torch.tensor([1e-3])}
    masks = PairMasks(torch.zeros_like(w0), torch.zeros_like(wT), {})

    out = compute_guidance_losses(SplitState(w0, wT), gt, masks, cfg)

    assert out.pde_residual_status == "approximate"
    assert out.L_pde.item() > 0.0
    assert out.metadata["pde"]["loss_reduction"] == "componentwise_mse_sum"
    assert out.metadata["pde"]["rhs_equation"] == "2D vorticity Navier-Stokes"

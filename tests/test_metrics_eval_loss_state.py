from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sampling.losses import GuidanceLossOutput
from sampling.masks import PairMasks
from sampling.metrics import (
    final_metrics,
    obs_relative_l2,
    pde_residual_norm,
    pde_residual_norm_per_sample,
    per_sample_metrics,
    relative_l2,
    relative_l2_per_sample,
    step_metrics,
)
from sampling.state import SplitState


def _losses(l_pde, residual_value):
    zero = torch.tensor(0.0)
    residual = torch.full((1, 1, 2, 2), float(residual_value))
    return GuidanceLossOutput(
        L_obs_a=zero + 0.1,
        L_obs_u=zero + 0.2,
        L_pde=zero + float(l_pde),
        obs_a_residual=residual,
        obs_u_residual=residual,
        pde_residual=residual,
        clean_L_obs_a=zero + 0.3,
        clean_L_obs_u=zero + 0.4,
        pde_residual_status="approximate",
        metadata={"pde": {"equation": "toy"}},
    )


def test_final_metrics_use_post_update_eval_losses():
    step_output = SimpleNamespace(
        t=torch.tensor(0.0),
        t_next=torch.tensor(1.0),
        phase="deterministic",
        loss_state="endpoint",
    )
    guidance_losses = _losses(10.0, 5.0)
    eval_losses = _losses(1.0, 2.0)
    phys_state = SplitState(coef=torch.ones(1, 1, 2, 2) * 2.0, sol=torch.ones(1, 1, 2, 2) * 3.0)
    ground_truth = SimpleNamespace(coef=torch.ones(1, 1, 2, 2), sol=torch.ones(1, 1, 2, 2))
    masks = PairMasks(coef=torch.ones(1, 1, 2, 2), sol=torch.ones(1, 1, 2, 2), metadata={})

    row = step_metrics(
        0,
        step_output,
        guidance_losses,
        eval_losses,
        None,
        phys_state,
        ground_truth,
        masks,
        wall_time=0.01,
    )
    final = final_metrics([row])

    assert final["L_pde"] == pytest.approx(1.0)
    assert final["eval_L_pde"] == pytest.approx(1.0)
    assert final["guidance_L_pde"] == pytest.approx(10.0)
    assert final["guidance_L_pde"] != final["eval_L_pde"]
    assert final["pde_residual_norm"] == pytest.approx(final["eval_pde_residual_norm"])
    assert final["guidance_pde_residual_norm"] != final["eval_pde_residual_norm"]
    assert final["rel_l2_a"] == pytest.approx(1.0)
    assert final["rel_l2_u"] == pytest.approx(2.0)


def test_relative_errors_are_computed_per_sample_then_averaged():
    target = torch.ones(2, 1, 2, 2)
    pred = torch.stack([target[0] * 2.0, target[1] * 3.0])
    mask = torch.ones_like(target)

    assert relative_l2_per_sample(pred, target) == pytest.approx([1.0, 2.0])
    assert relative_l2(pred, target) == pytest.approx(1.5)
    assert obs_relative_l2(pred, target, mask) == pytest.approx(1.5)


def test_final_step_metrics_retain_per_sample_relative_errors():
    step_output = SimpleNamespace(
        t=torch.tensor(0.0),
        t_next=torch.tensor(1.0),
        phase="deterministic",
        loss_state="endpoint",
    )
    losses = _losses(1.0, 2.0)
    target = torch.ones(2, 1, 2, 2)
    state = SplitState(
        coef=torch.stack([target[0] * 2.0, target[1] * 3.0]),
        sol=torch.stack([target[0] * 4.0, target[1] * 5.0]),
    )
    ground_truth = SimpleNamespace(coef=target, sol=target)
    masks = PairMasks(coef=torch.ones_like(target), sol=torch.ones_like(target), metadata={})

    row = step_metrics(
        0,
        step_output,
        losses,
        losses,
        None,
        state,
        ground_truth,
        masks,
        wall_time=0.01,
    )
    final = final_metrics([row])

    assert final["rel_l2_a_per_sample"] == pytest.approx([1.0, 2.0])
    assert final["rel_l2_a"] == pytest.approx(1.5)
    assert final["relative_l2_reduction"] == "mean_of_per_sample_relative_l2"


def test_pde_residual_norm_is_mean_of_per_sample_rms():
    residual = torch.stack([torch.ones(1, 2, 2), torch.ones(1, 2, 2) * 3.0])

    assert pde_residual_norm_per_sample(residual) == pytest.approx([1.0, 3.0])
    assert pde_residual_norm(residual) == pytest.approx(2.0)


def test_missing_pde_residual_is_nan_not_a_fake_perfect_zero():
    assert pde_residual_norm(None) != pde_residual_norm(None)


def test_per_sample_artifact_has_one_core_row_per_batch_item():
    losses = _losses(1.0, 2.0)
    target = torch.ones(2, 1, 2, 2)
    state = SplitState(coef=target * torch.tensor([2.0, 3.0]).view(2, 1, 1, 1), sol=target)
    ground_truth = SimpleNamespace(
        coef=target,
        sol=target,
        metadata={"sample_ids": ["sample-a", "sample-b"]},
    )
    masks = PairMasks(torch.ones_like(target), torch.ones_like(target), {})
    rows = per_sample_metrics(state, ground_truth, masks, losses)
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "sample-a"
    assert rows[1]["rel_l2_a"] == pytest.approx(2.0)
    assert set(rows[0]) == {
        "sample_index",
        "sample_id",
        "rel_l2_a",
        "rel_l2_u",
        "obs_rel_l2_a",
        "obs_rel_l2_u",
        "pde_residual_norm",
    }

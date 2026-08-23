from dataclasses import dataclass, field

import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.guidance import compute_guidance_gradient, make_zeta_schedule
from sampling.losses import GROUND_TRUTH_FIELD_PDE_PARAM_KEYS, compute_guidance_losses
from sampling.masks import PairMasks
from sampling.pde_residuals import compute_pde_residual
from sampling.state import SplitState


@dataclass
class GT:
    coef: object
    sol: object
    pde_params: dict = field(default_factory=dict)


def _poisson_config(task: str, guidance_components: str = "obs_pde") -> AblationConfig:
    return AblationConfig(
        pde="poisson",
        task=task,
        guidance_components=guidance_components,
        boundary_condition_mode="none",
    )


@pytest.mark.parametrize("task", ["forward", "inverse", "both"])
def test_pde_loss_is_invariant_to_ground_truth_fields_for_every_task(task):
    values = torch.linspace(-1.0, 1.0, 36).reshape(1, 1, 6, 6)
    prediction = SplitState(coef=0.2 + values, sol=values.square() + 0.3 * values)
    masks = PairMasks(torch.ones_like(values), torch.ones_like(values), {})
    first_gt = GT(torch.zeros_like(values), torch.ones_like(values))
    second_gt = GT(torch.full_like(values, 100.0), torch.full_like(values, -250.0))

    first = compute_guidance_losses(prediction, first_gt, masks, _poisson_config(task))
    second = compute_guidance_losses(prediction, second_gt, masks, _poisson_config(task))

    assert torch.allclose(first.pde_residual, second.pde_residual)
    assert torch.allclose(first.L_pde, second.L_pde)
    assert first.metadata["pde"]["field_input_sources"] == {
        "coef": "model_output",
        "sol": "model_output",
    }
    assert first.metadata["pde"]["uses_ground_truth_fields"] is False


@pytest.mark.parametrize("task", ["forward", "inverse", "both"])
def test_both_predicted_fields_enter_pde_residual_for_every_task(task):
    base = torch.zeros(1, 1, 6, 6)
    changed_solution = base.clone()
    changed_solution[..., 2:4, 2:4] = 1.0
    gt = GT(torch.full_like(base, 31.0), torch.full_like(base, -17.0))
    masks = PairMasks(torch.ones_like(base), torch.ones_like(base), {})
    config = _poisson_config(task, guidance_components="pde_only")

    baseline = compute_guidance_losses(SplitState(base, base), gt, masks, config)
    changed_coef = compute_guidance_losses(SplitState(torch.ones_like(base), base), gt, masks, config)
    changed_sol = compute_guidance_losses(SplitState(base, changed_solution), gt, masks, config)

    assert not torch.allclose(baseline.pde_residual, changed_coef.pde_residual)
    assert not torch.allclose(baseline.pde_residual, changed_sol.pde_residual)


def test_ground_truth_derived_pde_params_are_removed_from_sampling_loss():
    prediction = torch.linspace(-0.5, 0.5, 36).reshape(1, 1, 6, 6)
    hidden = torch.full_like(prediction, 999.0)
    params = {
        "observed_initial": hidden,
        "true_initial": hidden,
        "initial_mask": torch.ones_like(hidden),
        "allow_true_initial_condition": True,
        "initial_condition_source": "ground_truth",
        "trajectory": hidden.unsqueeze(1),
        "full_trajectory": hidden.unsqueeze(1),
        "trajectory_is_observed_ground_truth": True,
        "near_endpoint_temporal": {"q_dt": hidden, "q_T_minus_dt": hidden},
    }
    gt = GT(hidden, -hidden, params)
    masks = PairMasks(torch.zeros_like(hidden), torch.zeros_like(hidden), {})

    output = compute_guidance_losses(
        SplitState(prediction, prediction.square()),
        gt,
        masks,
        _poisson_config("both", guidance_components="pde_only"),
    )

    assert set(output.metadata["excluded_ground_truth_field_params"]) == set(params)
    assert set(output.metadata["pde"]["excluded_ground_truth_field_params"]) == set(params)
    assert not (set(output.metadata["pde_params_used"]) & GROUND_TRUTH_FIELD_PDE_PARAM_KEYS)


def test_near_endpoint_exception_exposes_only_sparse_observed_values():
    q0 = torch.zeros(1, 1, 6, 6, requires_grad=True)
    qT = torch.ones(1, 1, 6, 6, requires_grad=True)
    mask_0 = torch.zeros_like(q0)
    mask_T = torch.zeros_like(qT)
    mask_0[..., 2, 3] = 1.0
    mask_T[..., 4, 1] = 1.0
    q_dt = torch.full_like(q0, 1000.0, requires_grad=True)
    q_T_minus_dt = torch.full_like(qT, -1000.0, requires_grad=True)
    with torch.no_grad():
        q_dt[..., 2, 3] = 0.2
        q_T_minus_dt[..., 4, 1] = 0.8

    config = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="pde_only",
        residual_mode="near_endpoint_temporal",
        boundary_condition_mode="periodic",
    )
    masks = PairMasks(torch.zeros_like(q0), torch.zeros_like(qT), {})

    def losses_for(near_q_dt, near_q_tm):
        gt = GT(
            torch.full_like(q0, 77.0),
            torch.full_like(qT, -88.0),
            {
                "alpha": torch.tensor([1e-3]),
                "near_endpoint_temporal": {
                    "q_dt": near_q_dt,
                    "q_T_minus_dt": near_q_tm,
                    "dt": torch.tensor([0.1]),
                    "mask_0": mask_0,
                    "mask_T": mask_T,
                },
            },
        )
        return compute_guidance_losses(SplitState(q0, qT), gt, masks, config)

    baseline = losses_for(q_dt, q_T_minus_dt)
    changed_unobserved_q_dt = torch.full_like(q_dt, -5e5)
    changed_unobserved_q_tm = torch.full_like(q_T_minus_dt, 9e5)
    changed_unobserved_q_dt[..., 2, 3] = 0.2
    changed_unobserved_q_tm[..., 4, 1] = 0.8
    changed_unobserved = losses_for(changed_unobserved_q_dt, changed_unobserved_q_tm)

    assert torch.allclose(baseline.pde_residual, changed_unobserved.pde_residual)
    assert torch.allclose(baseline.L_pde, changed_unobserved.L_pde)
    assert baseline.metadata["pde"]["uses_ground_truth_fields"] is True
    assert baseline.metadata["pde"]["uses_ground_truth_endpoint_fields"] is False
    assert baseline.metadata["pde"]["ground_truth_field_exception"] == (
        "near_endpoint_temporal_sparse_observations"
    )
    assert baseline.metadata["pde"]["auxiliary_field_input_sources"] == {
        "q_dt": "sparse_ground_truth_observations",
        "q_T_minus_dt": "sparse_ground_truth_observations",
    }

    changed_observed_q_dt = changed_unobserved_q_dt.clone()
    changed_observed_q_dt[..., 2, 3] = 0.4
    changed_observed = losses_for(changed_observed_q_dt, changed_unobserved_q_tm)
    assert not torch.allclose(baseline.pde_residual, changed_observed.pde_residual)

    baseline.L_pde.backward()
    assert q0.grad is not None
    assert qT.grad is not None
    assert q_dt.grad is None
    assert q_T_minus_dt.grad is None


def test_near_endpoint_exception_rejects_empty_sparse_masks():
    q0 = torch.zeros(1, 1, 5, 5)
    qT = torch.ones_like(q0)
    zero_mask = torch.zeros_like(q0)
    config = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="pde_only",
        residual_mode="near_endpoint_temporal",
        boundary_condition_mode="periodic",
    )
    gt = GT(
        q0,
        qT,
        {
            "alpha": 1e-3,
            "near_endpoint_temporal": {
                "q_dt": q0,
                "q_T_minus_dt": qT,
                "dt": torch.tensor([0.1]),
                "mask_0": zero_mask,
                "mask_T": zero_mask,
            },
        },
    )
    masks = PairMasks(zero_mask, zero_mask, {})

    with pytest.raises(ValueError, match="at least one point"):
        compute_guidance_losses(SplitState(q0, qT), gt, masks, config)


def _burgers_trajectory(batch: int = 1, time_points: int = 7, space_points: int = 8):
    time = torch.linspace(0.0, 1.0, time_points).view(1, 1, time_points, 1)
    space = torch.arange(space_points, dtype=torch.float32).view(1, 1, 1, space_points) / space_points
    field = (1.0 + time + 0.2 * time.square()) * torch.sin(2.0 * torch.pi * space)
    return field.repeat(batch, 1, 1, 1)


def test_burgers_sampling_residual_uses_predicted_full_time_field_only():
    prediction = _burgers_trajectory(batch=2)
    zeros = torch.zeros_like(prediction)
    hidden = torch.full_like(prediction, 1234.0)
    pde_params = {
        "nu": torch.tensor([0.1, 0.2]),
        "T": torch.tensor([1.0, 2.0]),
        "domain_length": torch.tensor([1.0, 3.0]),
    }
    config = AblationConfig(
        pde="burger",
        task="both",
        guidance_components="pde_only",
        residual_mode="full_trajectory_fd",
        boundary_condition_mode="periodic",
    )
    masks = PairMasks(torch.zeros_like(prediction), torch.zeros_like(prediction), {})

    first = compute_guidance_losses(
        SplitState(prediction, prediction), GT(zeros, zeros, pde_params), masks, config
    )
    second = compute_guidance_losses(
        SplitState(prediction, prediction), GT(hidden, -hidden, pde_params), masks, config
    )

    assert torch.allclose(first.pde_residual, second.pde_residual)
    assert torch.allclose(first.L_pde, second.L_pde)
    assert first.metadata["pde"]["field_input_source"] == "model_output_full_time_space"
    assert first.metadata["pde"]["uses_generated_trajectory"] is True
    assert first.metadata["pde"]["uses_ground_truth_fields"] is False
    assert tuple(first.metadata["pde"]["nu"]) == pytest.approx((0.1, 0.2))

    changed = prediction.clone()
    changed[:, :, 3] += 0.5
    changed_output = compute_guidance_losses(
        SplitState(changed, changed), GT(hidden, hidden, pde_params), masks, config
    )
    assert not torch.allclose(first.pde_residual, changed_output.pde_residual)

    with pytest.raises(ValueError, match="aliases"):
        compute_pde_residual("burger", prediction, prediction + 1.0, pde_params=pde_params)


def test_burgers_batch_uses_each_samples_own_physical_parameters():
    target = _burgers_trajectory()
    other = 3.0 * _burgers_trajectory()
    target_params = {"nu": torch.tensor([0.2]), "T": torch.tensor([2.0]), "domain_length": torch.tensor([3.0])}
    alone = compute_pde_residual("burger", target, target, pde_params=target_params)

    batched = torch.cat([other, target], dim=0)
    batch_params = {
        "nu": torch.tensor([5.0, 0.2]),
        "T": torch.tensor([0.5, 2.0]),
        "domain_length": torch.tensor([0.25, 3.0]),
    }
    together = compute_pde_residual("burger", batched, batched, pde_params=batch_params)

    assert torch.allclose(
        alone.components["interior"][0],
        together.components["interior"][1],
        atol=1e-6,
        rtol=1e-6,
    )


def test_pde_guidance_gradient_is_independent_of_other_batch_samples():
    config = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="pde_only",
        residual_mode="endpoint_secant",
        boundary_condition_mode="periodic",
        clip_mode="none",
    )
    schedule = make_zeta_schedule(config, torch.tensor(0.2), torch.tensor(0.4), torch.tensor(1.0))
    sample = torch.linspace(-1.0, 1.0, 72).reshape(1, 2, 6, 6)
    other = 20.0 * sample

    def gradient_for(states, alpha, total_time):
        pair = torch.cat(states, dim=0).requires_grad_(True)
        split = SplitState(pair[:, 0:1], pair[:, 1:2])
        zero = torch.zeros_like(split.coef)
        gt = GT(zero, zero, {"alpha": torch.tensor(alpha), "T": torch.tensor(total_time)})
        masks = PairMasks(torch.zeros_like(zero), torch.zeros_like(zero), {})
        losses = compute_guidance_losses(split, gt, masks, config)
        return compute_guidance_gradient(losses, pair, schedule, config).grad_total.detach()

    alone = gradient_for([sample], [0.01], [1.5])[0]
    together = gradient_for([other, sample], [3.0, 0.01], [0.2, 1.5])[1]

    assert torch.allclose(alone, together, atol=1e-5, rtol=1e-5)

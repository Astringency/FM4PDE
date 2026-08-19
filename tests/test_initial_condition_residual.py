from dataclasses import dataclass, field

import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.losses import compute_guidance_losses
from sampling.masks import PairMasks
from sampling.pde_residuals import compute_pde_residual
from sampling.state import SplitState


@dataclass
class GT:
    coef: object
    sol: object
    pde_params: dict = field(default_factory=dict)


def test_hermite_initial_residual_uses_q0_not_qT():
    q0 = torch.zeros(1, 1, 4, 4)
    qT = torch.ones_like(q0)
    mask = torch.ones_like(q0)
    out = compute_pde_residual(
        "heat",
        q0,
        qT,
        pde_params={"alpha": 1e-3, "observed_initial": q0, "initial_mask": mask, "boundary_condition_mode": "periodic"},
        residual_mode="hermite_bridge",
    )
    assert torch.allclose(out.components["initial"], torch.zeros_like(q0))
    out_bad = compute_pde_residual(
        "heat",
        q0,
        qT,
        pde_params={"alpha": 1e-3, "observed_initial": qT, "initial_mask": mask, "boundary_condition_mode": "periodic"},
        residual_mode="hermite_bridge",
    )
    assert out_bad.components["initial"].abs().sum() > 0


def test_near_endpoint_initial_residual_uses_q0_not_qT():
    q0 = torch.zeros(1, 1, 4, 4)
    qT = torch.ones_like(q0)
    mask = torch.ones_like(q0)
    near = {
        "q_dt": q0,
        "q_T_minus_dt": qT,
        "dt": torch.tensor([0.1]),
        "mask_0": mask,
        "mask_T": mask,
    }
    out = compute_pde_residual(
        "heat",
        q0,
        qT,
        pde_params={"alpha": 1e-3, "near_endpoint_temporal": near, "observed_initial": q0, "initial_mask": mask, "boundary_condition_mode": "periodic"},
        residual_mode="near_endpoint_temporal",
    )
    assert torch.allclose(out.components["initial"], torch.zeros_like(q0))
    out_bad = compute_pde_residual(
        "heat",
        q0,
        qT,
        pde_params={"alpha": 1e-3, "near_endpoint_temporal": near, "observed_initial": qT, "initial_mask": mask, "boundary_condition_mode": "periodic"},
        residual_mode="near_endpoint_temporal",
    )
    assert out_bad.components["initial"].abs().sum() > 0


def test_inverse_does_not_inject_initial_observation():
    coef = torch.zeros(1, 1, 4, 4)
    sol = torch.ones_like(coef)
    cfg = AblationConfig(
        pde="heat",
        task="inverse",
        guidance_components="obs_pde",
        residual_mode="endpoint_secant",
        initial_condition_mode="auto",
        boundary_condition_mode="periodic",
    )
    masks = PairMasks(torch.ones_like(coef), torch.ones_like(sol), {})
    out = compute_guidance_losses(SplitState(coef, sol), GT(coef, sol, {"alpha": 1e-3}), masks, cfg)
    assert out.metadata["pde"]["ic_residual_enabled"] is False
    assert out.metadata["pde"]["initial_condition_source"] == "not_available"


def test_pde_only_does_not_inject_initial_observation():
    coef = torch.zeros(1, 1, 4, 4)
    sol = torch.ones_like(coef)
    cfg = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="pde_only",
        residual_mode="endpoint_secant",
        initial_condition_mode="auto",
        boundary_condition_mode="periodic",
    )
    masks = PairMasks(torch.ones_like(coef), torch.ones_like(sol), {})
    out = compute_guidance_losses(SplitState(coef, sol), GT(coef, sol, {"alpha": 1e-3}), masks, cfg)
    assert out.metadata["pde"]["ic_residual_enabled"] is False
    assert out.metadata["pde"]["initial_condition_source"] == "not_available"


def test_full_trajectory_ground_truth_cannot_be_used_as_endpoint_guidance():
    coef = torch.zeros(1, 1, 4, 4)
    sol = torch.ones_like(coef)
    cfg = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="pde_only",
        residual_mode="full_trajectory_fd",
        boundary_condition_mode="periodic",
    )
    gt = GT(
        coef,
        sol,
        {
            "alpha": 1e-3,
            "trajectory": torch.zeros(1, 3, 1, 4, 4),
            "trajectory_is_observed_ground_truth": True,
        },
    )
    masks = PairMasks(torch.zeros_like(coef), torch.zeros_like(sol), {})
    with pytest.raises(ValueError, match="evaluation-only"):
        compute_guidance_losses(SplitState(coef, sol), gt, masks, cfg)


def test_observed_initial_mode_requires_obs_a():
    coef = torch.zeros(1, 1, 4, 4)
    sol = torch.ones_like(coef)
    cfg = AblationConfig(
        pde="heat",
        task="inverse",
        guidance_components="obs_pde",
        residual_mode="endpoint_secant",
        initial_condition_mode="observed_initial",
        boundary_condition_mode="periodic",
    )
    masks = PairMasks(torch.ones_like(coef), torch.ones_like(sol), {})
    with pytest.raises(ValueError, match="requires coefficient/initial observations"):
        compute_guidance_losses(SplitState(coef, sol), GT(coef, sol), masks, cfg)

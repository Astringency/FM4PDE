from dataclasses import dataclass, field

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
    pde_params: dict = field(default_factory=dict)


def test_inverse_does_not_inject_initial_observation():
    coef = torch.zeros(1, 1, 4, 4)
    sol = torch.ones_like(coef)
    cfg = AblationConfig(
        pde="heat",
        task="inverse",
        guidance_components="obs_pde",
        residual_mode="endpoint_secant",
        boundary_condition_mode="periodic",
    )
    masks = PairMasks(torch.ones_like(coef), torch.ones_like(sol), {})
    out = compute_guidance_losses(SplitState(coef, sol), GT(coef, sol, {"alpha": 1e-3}), masks, cfg)
    assert "initial" not in out.metadata["pde"]["component_norms"]


def test_pde_only_does_not_inject_initial_observation():
    coef = torch.zeros(1, 1, 4, 4)
    sol = torch.ones_like(coef)
    cfg = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="pde_only",
        residual_mode="endpoint_secant",
        boundary_condition_mode="periodic",
    )
    masks = PairMasks(torch.ones_like(coef), torch.ones_like(sol), {})
    out = compute_guidance_losses(SplitState(coef, sol), GT(coef, sol, {"alpha": 1e-3}), masks, cfg)
    assert "initial" not in out.metadata["pde"]["component_norms"]


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
    with pytest.raises(ValueError, match="must not be mixed into PDE loss"):
        compute_guidance_losses(SplitState(coef, sol), gt, masks, cfg)

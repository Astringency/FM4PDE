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


def test_boundary_excluded_masks_only_interior_not_boundary():
    coef = torch.zeros(1, 1, 6, 6)
    sol = torch.zeros_like(coef)
    sol[..., 0, :] = 5.0
    cfg = AblationConfig(
        pde="poisson",
        task="both",
        guidance_components="pde_only",
        pde_residual_region="boundary_excluded",
        boundary_condition_mode="dirichlet_zero",
    )
    masks = PairMasks(torch.zeros_like(coef), torch.zeros_like(sol), {})
    out = compute_guidance_losses(SplitState(coef, sol), GT(coef, sol), masks, cfg)
    assert out.metadata["pde"]["component_norms"]["boundary"] > 0
    assert out.metadata["pde"]["pde_residual_region_applied_to"] == "interior_only"
    assert out.metadata["pde"]["boundary_region_masked"] is False
    assert out.L_pde.item() > 0


def test_near_endpoint_temporal_uses_sparse_observation_exception_and_skips_second_region_mask():
    q0 = torch.zeros(1, 1, 5, 5)
    qT = torch.ones_like(q0)
    mask_0 = torch.zeros_like(q0)
    mask_T = torch.zeros_like(q0)
    mask_0[..., 2, 2] = 1.0
    mask_T[..., 1, 3] = 1.0
    cfg = AblationConfig(
        pde="heat",
        task="both",
        guidance_components="pde_only",
        residual_mode="near_endpoint_temporal",
        pde_residual_region="active_obs_union",
        boundary_condition_mode="periodic",
    )
    gt = GT(
        q0,
        qT,
        {
            "alpha": 1e-3,
            "near_endpoint_temporal": {
                "q_dt": q0 + 0.1,
                "q_T_minus_dt": qT - 0.1,
                "dt": torch.tensor([0.1]),
                "mask_0": mask_0,
                "mask_T": mask_T,
            }
        },
    )
    masks = PairMasks(torch.zeros_like(q0), torch.zeros_like(qT), {})
    out = compute_guidance_losses(SplitState(q0, qT), gt, masks, cfg)
    assert out.metadata["pde"]["pde_residual_region_skipped"] is True
    assert out.metadata["pde"]["reason"] == "near_endpoint_temporal interior is already sparse-temporal masked"
    assert out.metadata["pde"]["uses_ground_truth_fields"] is True
    assert out.metadata["pde"]["uses_ground_truth_endpoint_fields"] is False
    assert out.metadata["pde"]["ground_truth_field_exception"] == (
        "near_endpoint_temporal_sparse_observations"
    )
    assert out.metadata["pde"]["field_input_sources"] == {
        "coef": "model_output",
        "sol": "model_output",
    }
    assert out.pde_residual.abs().sum() > 0

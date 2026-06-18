from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sampling.losses import GuidanceLossOutput
from sampling.masks import PairMasks
from sampling.metrics import final_metrics, step_metrics
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

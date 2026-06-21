import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.runner import _sanitize_pde_params_for_artifact


def test_sanitize_pde_params_recursive_cpu():
    cfg = AblationConfig(residual_mode="endpoint_secant")
    params = {
        "alpha": torch.tensor([1.0]),
        "nested": {"items": [torch.ones(1, 1), (torch.zeros(1),)]},
    }
    out = _sanitize_pde_params_for_artifact(params, cfg)
    assert out["alpha"].device.type == "cpu"
    assert out["nested"]["items"][0].device.type == "cpu"
    assert out["nested"]["items"][1][0].device.type == "cpu"


def test_near_endpoint_temporal_sanitize_omits_hidden_frames():
    cfg = AblationConfig(residual_mode="near_endpoint_temporal", num_near_endpoint_obs=1)
    q_dt = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    q_tm = q_dt + 100
    mask_0 = torch.zeros_like(q_dt)
    mask_t = torch.zeros_like(q_dt)
    mask_0[..., 1, 1] = 1.0
    mask_t[..., 2, 2] = 1.0
    out = _sanitize_pde_params_for_artifact(
        {
            "near_endpoint_temporal": {
                "q_dt": q_dt,
                "q_T_minus_dt": q_tm,
                "mask_0": mask_0,
                "mask_T": mask_t,
                "dt": torch.tensor([0.1]),
                "metadata": {"source": "test"},
            }
        },
        cfg,
    )
    near = out["near_endpoint_temporal"]
    assert "q_dt" not in near
    assert "q_T_minus_dt" not in near
    assert near["q_dt_obs"].device.type == "cpu"
    assert near["q_T_minus_dt_obs"].device.type == "cpu"
    assert near["q_dt_obs"].sum().item() == q_dt[..., 1, 1].item()
    assert near["q_T_minus_dt_obs"].sum().item() == q_tm[..., 2, 2].item()
    assert near["mask_0"].device.type == "cpu"
    assert near["mask_T"].device.type == "cpu"
    assert near["metadata"]["full_near_endpoint_frames_saved"] is False
    assert near["metadata"]["sparse_temporal_observations_saved"] is True

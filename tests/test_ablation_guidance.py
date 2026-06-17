import pytest

torch = pytest.importorskip("torch")

from fm4pde_ablation.config import AblationConfig
from fm4pde_ablation.guidance import compute_guidance_gradient, make_zeta_schedule
from fm4pde_ablation.losses import GuidanceLossOutput, guidance_component_flags
from fm4pde_ablation.runner import _disable_unreliable_pde_guidance


def test_guidance_global_clip_records_scale():
    cfg = AblationConfig(task="both", guidance_components="obs_only", clip_mode="global_norm", clip_threshold=1.0)
    x = torch.ones(1, 2, 4, 4, requires_grad=True)
    loss = (x**2).mean()
    zero = x.sum() * 0.0
    losses = GuidanceLossOutput(loss, zero, zero, x, x, None, loss, zero, "disabled", {})
    schedule = make_zeta_schedule(cfg, torch.tensor(0.2), torch.tensor(0.4), torch.tensor(1.0))
    grad = compute_guidance_gradient(losses, x, schedule, cfg)
    assert grad.grad_norm_total <= 1.0 + 1e-6
    assert grad.clip_scale <= 1.0


def test_obs_only_is_task_gated_and_both_obs_is_explicit_both_task():
    assert guidance_component_flags("obs_only", "forward") == {"obs_a": True, "obs_u": False, "pde": False}
    assert guidance_component_flags("obs_only", "inverse") == {"obs_a": False, "obs_u": True, "pde": False}
    assert guidance_component_flags("obs_only", "both") == {"obs_a": True, "obs_u": True, "pde": False}
    assert guidance_component_flags("both_obs", "both") == {"obs_a": True, "obs_u": True, "pde": False}
    with pytest.raises(ValueError):
        guidance_component_flags("both_obs", "forward")


def test_ns_pde_guidance_is_mapped_to_non_pde_guidance():
    cfg = AblationConfig(pde="nsnonbounded", task="both", guidance_components="pde_only")
    with pytest.warns(RuntimeWarning):
        _disable_unreliable_pde_guidance(cfg)
    assert cfg.guidance_components == "noguide"
    assert cfg.zeta_pde == 0.0
    assert cfg.extra["guidance_components_requested"] == "pde_only"
    assert cfg.extra["guidance_components_effective"] == "noguide"

    cfg = AblationConfig(pde="nsnonbounded", task="both", guidance_components="obs_pde")
    with pytest.warns(RuntimeWarning):
        _disable_unreliable_pde_guidance(cfg)
    assert cfg.guidance_components == "obs_only"
    assert cfg.extra["guidance_components_requested"] == "obs_pde"
    assert cfg.extra["guidance_components_effective"] == "obs_only"

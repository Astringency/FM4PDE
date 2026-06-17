import pytest

torch = pytest.importorskip("torch")

from fm4pde_ablation.config import AblationConfig
from fm4pde_ablation.guidance import compute_guidance_gradient, make_zeta_schedule
from fm4pde_ablation.losses import GuidanceLossOutput


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

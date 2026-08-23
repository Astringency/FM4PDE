import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.guidance import apply_guidance_update, compute_guidance_gradient, make_zeta_schedule
from sampling.losses import GuidanceLossOutput, compute_guidance_losses, guidance_component_flags
from sampling.masks import PairMasks
from sampling.runner import _disable_unreliable_pde_guidance
from sampling.state import SplitState


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


def test_ns_pde_guidance_is_not_mapped_to_non_pde_guidance():
    cfg = AblationConfig(pde="nsnonbounded", task="both", guidance_components="pde_only")
    _disable_unreliable_pde_guidance(cfg)
    assert cfg.guidance_components == "pde_only"
    assert cfg.zeta_pde != 0.0
    assert "guidance_components_requested" not in cfg.runtime_metadata

    cfg = AblationConfig(pde="nsnonbounded", task="both", guidance_components="obs_pde")
    _disable_unreliable_pde_guidance(cfg)
    assert cfg.guidance_components == "obs_pde"
    assert cfg.zeta_pde != 0.0
    assert "guidance_components_requested" not in cfg.runtime_metadata


def test_disabled_pde_guidance_is_still_mapped_to_non_pde_guidance():
    cfg = AblationConfig(pde="unsupported_disabled_pde", task="both", guidance_components="obs_pde")
    with pytest.warns(RuntimeWarning):
        _disable_unreliable_pde_guidance(cfg)
    assert cfg.guidance_components == "obs_only"
    assert cfg.runtime_metadata["guidance_components_requested"] == "obs_pde"
    assert cfg.runtime_metadata["guidance_components_effective"] == "obs_only"


def test_stochastic_guidance_scale_defaults_to_current_time():
    cfg = AblationConfig(
        task="both",
        guidance_components="obs_only",
        stochastic_guidance_coeff=0.5,
        stochastic_guidance_time="t",
    )
    x = torch.ones(1, 1, 2, 2, requires_grad=True)
    loss = (x**2).sum()
    zero = x.sum() * 0.0
    losses = GuidanceLossOutput(loss, zero, zero, x, x, None, loss, zero, "disabled", {})
    schedule = make_zeta_schedule(cfg, torch.tensor(0.2), torch.tensor(0.8), torch.tensor(1.0))
    grad = compute_guidance_gradient(losses, x, schedule, cfg)
    step_output = type(
        "Step",
        (),
        {
            "phase": "stochastic",
            "t": torch.tensor(0.2),
            "t_next": torch.tensor(0.8),
            "step_size": torch.tensor(0.6),
        },
    )()

    updated = apply_guidance_update(torch.zeros_like(x), grad, step_output, schedule, cfg)

    assert grad.metadata["stochastic_guidance_time"] == "t"
    assert grad.metadata["guidance_update_scale"] == pytest.approx(0.4)
    assert torch.allclose(updated, -0.4 * grad.grad_total)


@pytest.mark.parametrize("clip_mode", ["none", "global_norm", "per_component_norm"])
def test_guidance_gradient_is_independent_of_other_batch_samples(clip_mode):
    cfg = AblationConfig(
        pde="poisson",
        task="forward",
        guidance_components="obs_only",
        clip_mode=clip_mode,
        clip_threshold=0.2,
        boundary_condition_mode="dirichlet_zero",
    )
    schedule = make_zeta_schedule(cfg, torch.tensor(0.2), torch.tensor(0.4), torch.tensor(1.0))

    def gradient_for(values):
        x = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1, 2, 2).requires_grad_(True)
        zero = torch.zeros_like(x)
        gt = type("GT", (), {"coef": zero, "sol": zero, "pde_params": {}})()
        masks = PairMasks(torch.ones_like(x), torch.zeros_like(x), {})
        losses = compute_guidance_losses(SplitState(x, zero), gt, masks, cfg)
        return compute_guidance_gradient(losses, x, schedule, cfg).grad_total.detach()

    sample = [1.0, 2.0, 3.0, 4.0]
    alone = gradient_for([sample])[0]
    batched = gradient_for([sample, [100.0, 200.0, 300.0, 400.0]])[0]

    assert torch.allclose(alone, batched, atol=1e-7, rtol=1e-6)

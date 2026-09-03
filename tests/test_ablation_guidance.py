from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.guidance import _update_scale, apply_guidance_update, compute_guidance_gradient, make_zeta_schedule
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


@pytest.mark.parametrize(
    "mode,coeff,max_scale,expected",
    [
        ("legacy", 1.0, 0.1, 9999.99),
        ("zero_at_t0", 1.0, 0.1, 0.0),
        ("t_next", 1.0, 0.1, 0.99),
        ("clipped", 1.0, 0.1, 0.1),
        ("clipped", 1.0, 0.03, 0.03),
        ("clipped_zero_at_t0", 1.0, 0.03, 0.0),
        ("stochastic_like", 0.1, 0.1, 0.1),
        ("capped_stochastic_like", 0.1, 0.0075, 0.0075),
    ],
)
def test_deterministic_bt_modes_regularize_the_t0_update(mode, coeff, max_scale, expected):
    cfg = AblationConfig(
        deterministic_bt_mode=mode,
        deterministic_guidance_coeff=coeff,
        deterministic_bt_max_scale=max_scale,
    )
    scale = _update_scale(
        "deterministic",
        torch.tensor(0.0),
        torch.tensor(0.01),
        torch.tensor(0.01),
        torch.tensor(999999.0),
        cfg,
    )
    assert float(scale) == pytest.approx(expected, rel=1e-5)


def test_zero_at_t0_only_disables_the_singular_first_update():
    cfg = AblationConfig(deterministic_bt_mode="zero_at_t0", deterministic_guidance_coeff=0.1)
    scale = _update_scale(
        "deterministic",
        torch.tensor(0.01),
        torch.tensor(0.02),
        torch.tensor(0.01),
        torch.tensor(99.0),
        cfg,
    )
    assert float(scale) == pytest.approx(0.099)


def test_capped_stochastic_like_recovers_the_stochastic_tail():
    cfg = AblationConfig(
        deterministic_bt_mode="capped_stochastic_like",
        deterministic_guidance_coeff=0.1,
        deterministic_bt_max_scale=0.0075,
    )
    scale = _update_scale(
        "deterministic",
        torch.tensor(0.99),
        torch.tensor(1.0),
        torch.tensor(0.01),
        torch.tensor(0.010101),
        cfg,
    )
    assert float(scale) == pytest.approx(0.001)


def test_deterministic_guidance_gate_skips_and_smoothly_ramps_all_components():
    cfg = AblationConfig(
        deterministic_bt_mode="clipped",
        deterministic_bt_max_scale=0.1,
        deterministic_guidance_start_ratio=0.2,
        deterministic_guidance_ramp_ratio=0.2,
    )
    gradient = SimpleNamespace(grad_total=torch.ones(1, 1, 2, 2), metadata={})
    schedule = SimpleNamespace(bt=torch.tensor(1.0))

    before = SimpleNamespace(
        phase="deterministic",
        t=torch.tensor(0.1),
        t_next=torch.tensor(0.2),
        step_size=torch.tensor(0.1),
    )
    unchanged = apply_guidance_update(
        torch.zeros_like(gradient.grad_total), gradient, before, schedule, cfg
    )
    assert torch.count_nonzero(unchanged).item() == 0
    assert gradient.metadata["deterministic_guidance_factor"] == pytest.approx(0.0)

    halfway = SimpleNamespace(
        phase="deterministic",
        t=torch.tensor(0.3),
        t_next=torch.tensor(0.4),
        step_size=torch.tensor(0.1),
    )
    updated = apply_guidance_update(
        torch.zeros_like(gradient.grad_total), gradient, halfway, schedule, cfg
    )
    assert torch.allclose(updated, torch.full_like(updated, -0.05))
    assert gradient.metadata["deterministic_guidance_factor"] == pytest.approx(0.5)


def test_deterministic_correction_is_clipped_by_per_sample_rms():
    cfg = AblationConfig(
        deterministic_bt_mode="clipped",
        deterministic_bt_max_scale=0.1,
        deterministic_correction_max_rms=0.02,
    )
    gradient = SimpleNamespace(grad_total=torch.ones(2, 1, 2, 2), metadata={})
    schedule = SimpleNamespace(bt=torch.tensor(1.0))
    step = SimpleNamespace(
        phase="deterministic",
        t=torch.tensor(0.5),
        t_next=torch.tensor(0.6),
        step_size=torch.tensor(0.1),
    )

    updated = apply_guidance_update(
        torch.zeros_like(gradient.grad_total), gradient, step, schedule, cfg
    )

    assert torch.allclose(updated, torch.full_like(updated, -0.02))
    assert gradient.metadata["guidance_correction_rms"] == pytest.approx(0.02)
    assert gradient.metadata["correction_clip_scale"] == pytest.approx(0.2)


def test_deterministic_numerical_guard_rejects_nonfinite_sample_update():
    cfg = AblationConfig(deterministic_bt_mode="clipped", deterministic_bt_max_scale=0.1)
    grad_total = torch.ones(2, 1, 2, 2)
    grad_total[1, 0, 0, 0] = torch.inf
    gradient = SimpleNamespace(grad_total=grad_total, metadata={})
    schedule = SimpleNamespace(bt=torch.tensor(1.0))
    step = SimpleNamespace(
        phase="deterministic",
        t=torch.tensor(0.5),
        t_next=torch.tensor(0.6),
        step_size=torch.tensor(0.1),
    )

    updated = apply_guidance_update(torch.zeros_like(grad_total), gradient, step, schedule, cfg)

    assert torch.allclose(updated[0], torch.full_like(updated[0], -0.1))
    assert torch.count_nonzero(updated[1]).item() == 0
    assert gradient.metadata["nonfinite_correction_samples"] == 1


def test_pde_guidance_start_and_ramp_do_not_change_observation_weights():
    cfg = AblationConfig(
        task="both",
        guidance_components="obs_pde",
        zeta_obs_a=2.0,
        zeta_obs_u=3.0,
        zeta_pde=10.0,
        pde_guidance_start_ratio=0.5,
        pde_guidance_ramp_ratio=0.25,
    )

    before = make_zeta_schedule(cfg, torch.tensor(0.25), torch.tensor(0.26), torch.tensor(1.0))
    halfway = make_zeta_schedule(cfg, torch.tensor(0.625), torch.tensor(0.635), torch.tensor(1.0))
    after = make_zeta_schedule(cfg, torch.tensor(0.8), torch.tensor(0.81), torch.tensor(1.0))

    assert before.zeta_obs_a_t.item() == pytest.approx(2.0)
    assert before.zeta_obs_u_t.item() == pytest.approx(3.0)
    assert before.zeta_pde_t.item() == pytest.approx(0.0)
    assert before.metadata["pde_guidance_factor"] == pytest.approx(0.0)
    assert halfway.zeta_pde_t.item() == pytest.approx(5.0)
    assert halfway.metadata["pde_guidance_factor"] == pytest.approx(0.5)
    assert after.zeta_pde_t.item() == pytest.approx(10.0)
    assert after.metadata["pde_guidance_factor"] == pytest.approx(1.0)


def test_zero_pde_ramp_is_a_hard_switch_and_default_starts_at_eighty_percent():
    default_cfg = AblationConfig(zeta_pde=4.0)
    default_before = make_zeta_schedule(
        default_cfg, torch.tensor(0.799), torch.tensor(0.8), torch.tensor(1.0)
    )
    default_at_start = make_zeta_schedule(
        default_cfg, torch.tensor(0.8), torch.tensor(0.81), torch.tensor(1.0)
    )
    assert default_before.zeta_pde_t.item() == pytest.approx(0.0)
    assert default_at_start.zeta_pde_t.item() == pytest.approx(4.0)

    legacy_cfg = AblationConfig(
        zeta_pde=4.0,
        pde_guidance_start_ratio=0.0,
        pde_guidance_ramp_ratio=0.0,
    )
    legacy_schedule = make_zeta_schedule(
        legacy_cfg, torch.tensor(0.0), torch.tensor(0.01), torch.tensor(1.0)
    )
    assert legacy_schedule.zeta_pde_t.item() == pytest.approx(4.0)

    gated_cfg = AblationConfig(
        zeta_pde=4.0,
        pde_guidance_start_ratio=0.5,
        pde_guidance_ramp_ratio=0.0,
    )
    before = make_zeta_schedule(gated_cfg, torch.tensor(0.499), torch.tensor(0.5), torch.tensor(1.0))
    at_start = make_zeta_schedule(gated_cfg, torch.tensor(0.5), torch.tensor(0.51), torch.tensor(1.0))
    assert before.zeta_pde_t.item() == pytest.approx(0.0)
    assert at_start.zeta_pde_t.item() == pytest.approx(4.0)


def test_pde_gradient_is_inactive_before_start_while_observation_gradient_remains_active():
    cfg = AblationConfig(
        task="both",
        guidance_components="obs_pde",
        zeta_obs_a=2.0,
        zeta_pde=10.0,
        pde_guidance_start_ratio=0.5,
        clip_mode="none",
    )
    x = torch.ones(1, 2, 2, 2, requires_grad=True)
    obs_loss = x[:, :1].square().mean()
    pde_loss = x[:, 1:].square().mean()
    zero = x.sum() * 0.0
    losses = GuidanceLossOutput(
        L_obs_a=obs_loss,
        L_obs_u=zero,
        L_pde=pde_loss,
        obs_a_residual=x[:, :1],
        obs_u_residual=x[:, 1:],
        pde_residual=x[:, 1:],
        clean_L_obs_a=obs_loss,
        clean_L_obs_u=zero,
        pde_residual_status="measured",
        metadata={"enabled": {"obs_a": True, "obs_u": False, "pde": True}},
    )
    schedule = make_zeta_schedule(cfg, torch.tensor(0.25), torch.tensor(0.26), torch.tensor(1.0))

    grad = compute_guidance_gradient(losses, x, schedule, cfg)

    assert grad.metadata["scheduled_components_active"] == {
        "obs_a": True,
        "obs_u": False,
        "pde": False,
    }
    assert torch.count_nonzero(grad.grad_pde).item() == 0
    assert torch.count_nonzero(grad.grad_total[:, :1]).item() > 0
    assert torch.count_nonzero(grad.grad_total[:, 1:]).item() == 0


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

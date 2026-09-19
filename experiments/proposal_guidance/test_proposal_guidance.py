"""Analytic checks for where and with respect to what endpoint guidance is taken."""
import pytest
import torch

from sampling.config import AblationConfig
from sampling.sampler_wrappers import sampler_step
from sampling.runner import _gradient_target_tensor


class LinearVelocity:
    def __init__(self):
        self.calls = []

    def __call__(self, x, t):
        self.calls.append((float(t), torch.is_grad_enabled()))
        return (1 + t) * x


@pytest.mark.parametrize("phase", ["deterministic", "stochastic"])
def test_post_proposal_endpoint_and_jacobian(phase):
    x = torch.arange(1., 5.).reshape(1, 1, 2, 2).requires_grad_(True)
    t, tn = torch.tensor(.25), torch.tensor(.5)
    net = LinearVelocity()
    torch.manual_seed(123)
    before = torch.get_rng_state()
    out = sampler_step(net, x, t, tn, phase, "euler", "endpoint",
                       gradient_target="proposal_state_chain_rule")
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    old = sampler_step(LinearVelocity(), x, t, tn, phase, "euler", "endpoint")
    assert torch.equal(out.x_raw_next, old.x_raw_next)
    assert torch.equal(after, torch.get_rng_state())
    assert out.x_raw_next.is_leaf and out.x_raw_next.requires_grad
    assert net.calls == [(.25, False), (.5, True)]
    scale = 1 + (1 - tn) * (1 + tn)
    assert torch.allclose(out.x_endpoint, scale * out.x_raw_next)
    cfg = AblationConfig(gradient_target="proposal_state_chain_rule")
    cfg.validate()
    target = _gradient_target_tensor(cfg, x, out)
    grad, old_grad = torch.autograd.grad(out.x_loss_state.square().sum(), (target, x), allow_unused=True)
    assert old_grad is None
    assert torch.allclose(grad, 2 * scale**2 * out.x_raw_next)
    assert out.endpoint_model_evaluations == 2


def test_terminal_endpoint_is_identity():
    net = LinearVelocity()
    out = sampler_step(net, torch.ones(1, 1, 2, 2), torch.tensor(.9), torch.tensor(1.),
                       "stochastic", "euler", "endpoint", gradient_target="proposal_state_chain_rule")
    assert len(net.calls) == 1
    assert out.x_endpoint is out.x_raw_next
    grad, = torch.autograd.grad(out.x_endpoint.square().sum(), out.x_raw_next)
    assert torch.equal(grad, 2 * out.x_raw_next)


@pytest.mark.parametrize("override", [{"loss_state": "xt"}, {"step_method": "midpoint"},
                                     {"deterministic_endpoint_mode": "rollout"}])
def test_unsupported_endpoint_definitions_rejected(override):
    with pytest.raises(ValueError, match="proposal_state_chain_rule requires"):
        AblationConfig(gradient_target="proposal_state_chain_rule", **override).validate()


@pytest.mark.parametrize("phase", ["stochastic", "deterministic"])
def test_runner_proposal_mode_writes_real_metrics(tmp_path, phase):
    from pathlib import Path
    from sampling.runner import run_single_ablation
    cfg = AblationConfig(
        gradient_target="proposal_state_chain_rule", sampler_phase=phase,
        output_dir=str(tmp_path), dry_run=True, allow_synthetic_data=True,
        num_steps=3, img_resolution=8, batch_size=1, num_obs=8,
        save_plots=False, zeta_obs_a=1., zeta_obs_u=1., zeta_pde=0.,
    )
    result = run_single_ablation(cfg)
    assert result["status"] == "ok"
    assert result["gradient_target"] == "proposal_state_chain_rule"
    assert (Path(result["run_dir"]) / "metrics_per_sample.csv").exists()


def test_new_mode_has_distinct_run_name():
    assert AblationConfig().resolved_ablation_name() != AblationConfig(
        gradient_target="proposal_state_chain_rule").resolved_ablation_name()

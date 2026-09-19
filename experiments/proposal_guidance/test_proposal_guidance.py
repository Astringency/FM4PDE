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

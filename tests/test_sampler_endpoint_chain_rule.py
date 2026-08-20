import pytest

torch = pytest.importorskip("torch")

from sampling.config import AblationConfig
from sampling.guidance import compute_guidance_gradient, make_zeta_schedule
from sampling.losses import GuidanceLossOutput
from sampling.sampler_wrappers import sampler_step


class EndpointDoubler:
    def __call__(self, x, t):
        return x / (1.0 - t)


def _loss_output(loss, reference):
    zero = reference.sum() * 0.0
    return GuidanceLossOutput(zero, zero, loss, reference, reference, reference, zero, zero, "approximate", {})


def _schedule(cfg):
    return make_zeta_schedule(cfg, torch.tensor(0.25), torch.tensor(0.5), torch.tensor(1.0))


def test_endpoint_guidance_current_state_uses_chain_rule():
    cfg = AblationConfig(guidance_components="pde_only", gradient_target="current_state_chain_rule", clip_mode="none")
    x_cur = torch.arange(1.0, 5.0, dtype=torch.float32).view(1, 1, 2, 2).requires_grad_(True)
    out = sampler_step(
        net=EndpointDoubler(),
        x_cur=x_cur,
        t=torch.tensor(0.25),
        t_next=torch.tensor(0.5),
        phase="deterministic",
        step_method="euler",
        loss_state="endpoint",
    )
    loss = (out.x_loss_state**2).sum()

    grad = compute_guidance_gradient(_loss_output(loss, out.x_loss_state), x_cur, _schedule(cfg), cfg)

    assert torch.allclose(out.x_loss_state, 2.0 * x_cur)
    assert torch.allclose(grad.grad_pde, 8.0 * x_cur)
    assert grad.metadata["gradient_target"] == "current_state_chain_rule"


def test_endpoint_guidance_loss_state_direct_matches_endpoint_gradient():
    cfg = AblationConfig(guidance_components="pde_only", gradient_target="loss_state_direct", clip_mode="none")
    x_cur = torch.arange(1.0, 5.0, dtype=torch.float32).view(1, 1, 2, 2).requires_grad_(True)
    out = sampler_step(
        net=EndpointDoubler(),
        x_cur=x_cur,
        t=torch.tensor(0.25),
        t_next=torch.tensor(0.5),
        phase="deterministic",
        step_method="euler",
        loss_state="endpoint",
    )
    loss = (out.x_loss_state**2).sum()

    grad = compute_guidance_gradient(_loss_output(loss, out.x_loss_state), out.x_loss_state, _schedule(cfg), cfg)

    assert torch.allclose(grad.grad_pde, 2.0 * out.x_loss_state)
    assert torch.allclose(grad.grad_pde, 4.0 * x_cur)
    assert grad.metadata["gradient_target"] == "loss_state_direct"


def test_xt_current_state_and_direct_gradients_match():
    x_cur = torch.randn(1, 1, 3, 3, requires_grad=True)
    cfg_current = AblationConfig(guidance_components="pde_only", loss_state="xt", gradient_target="current_state_chain_rule", clip_mode="none")
    out = sampler_step(
        net=EndpointDoubler(),
        x_cur=x_cur,
        t=torch.tensor(0.25),
        t_next=torch.tensor(0.5),
        phase="deterministic",
        step_method="euler",
        loss_state="xt",
    )
    loss = (out.x_loss_state**2).sum()
    grad_current = compute_guidance_gradient(_loss_output(loss, out.x_loss_state), x_cur, _schedule(cfg_current), cfg_current)

    x_cur_direct = x_cur.detach().clone().requires_grad_(True)
    cfg_direct = AblationConfig(guidance_components="pde_only", loss_state="xt", gradient_target="loss_state_direct", clip_mode="none")
    out_direct = sampler_step(
        net=EndpointDoubler(),
        x_cur=x_cur_direct,
        t=torch.tensor(0.25),
        t_next=torch.tensor(0.5),
        phase="deterministic",
        step_method="euler",
        loss_state="xt",
    )
    loss_direct = (out_direct.x_loss_state**2).sum()
    grad_direct = compute_guidance_gradient(
        _loss_output(loss_direct, out_direct.x_loss_state),
        out_direct.x_loss_state,
        _schedule(cfg_direct),
        cfg_direct,
    )

    assert torch.allclose(grad_current.grad_pde, grad_direct.grad_pde)
    assert torch.allclose(grad_current.grad_pde, 2.0 * x_cur)


def test_enabled_guidance_with_disconnected_target_raises():
    cfg = AblationConfig(
        guidance_components="pde_only",
        loss_state="x_next",
        gradient_target="next_state_direct",
        clip_mode="none",
    )
    connected = torch.ones(1, 1, 2, 2, requires_grad=True)
    disconnected = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss = connected.square().sum()
    with pytest.raises(RuntimeError, match="not connected"):
        compute_guidance_gradient(
            _loss_output(loss, connected),
            disconnected,
            _schedule(cfg),
            cfg,
        )

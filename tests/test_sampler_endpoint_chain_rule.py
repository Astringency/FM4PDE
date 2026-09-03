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
    return GuidanceLossOutput(
        zero,
        zero,
        loss,
        reference,
        reference,
        reference,
        zero,
        zero,
        "approximate",
        {},
    )


def _config(gradient_target, *, loss_state="endpoint"):
    return AblationConfig(
        guidance_components="pde_only",
        gradient_target=gradient_target,
        loss_state=loss_state,
        pde_guidance_start_ratio=0.0,
        clip_mode="none",
    )


def _schedule(cfg):
    return make_zeta_schedule(
        cfg,
        torch.tensor(0.25),
        torch.tensor(0.5),
        torch.tensor(1.0),
    )


def _step(x_cur, loss_state):
    return sampler_step(
        net=EndpointDoubler(),
        x_cur=x_cur,
        t=torch.tensor(0.25),
        t_next=torch.tensor(0.5),
        phase="deterministic",
        step_method="euler",
        loss_state=loss_state,
    )


@pytest.mark.parametrize(
    ("gradient_target", "expected_factor"),
    (("current_state_chain_rule", 8.0), ("loss_state_direct", 4.0)),
)
def test_endpoint_guidance_gradient_target(gradient_target, expected_factor):
    cfg = _config(gradient_target)
    x_cur = torch.arange(1.0, 5.0, dtype=torch.float32).view(1, 1, 2, 2).requires_grad_(True)
    out = _step(x_cur, "endpoint")
    loss = (out.x_loss_state**2).sum()
    grad_target = x_cur if gradient_target == "current_state_chain_rule" else out.x_loss_state

    grad = compute_guidance_gradient(
        _loss_output(loss, out.x_loss_state), grad_target, _schedule(cfg), cfg
    )

    assert torch.allclose(out.x_loss_state, 2.0 * x_cur)
    assert torch.allclose(grad.grad_pde, expected_factor * x_cur)
    assert grad.metadata["gradient_target"] == gradient_target


def test_xt_current_state_and_direct_gradients_match():
    x_cur = torch.randn(1, 1, 3, 3, requires_grad=True)
    cfg_current = _config("current_state_chain_rule", loss_state="xt")
    out = _step(x_cur, "xt")
    loss = (out.x_loss_state**2).sum()
    grad_current = compute_guidance_gradient(
        _loss_output(loss, out.x_loss_state),
        x_cur,
        _schedule(cfg_current),
        cfg_current,
    )

    x_cur_direct = x_cur.detach().clone().requires_grad_(True)
    cfg_direct = _config("loss_state_direct", loss_state="xt")
    out_direct = _step(x_cur_direct, "xt")
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
    cfg = _config("next_state_direct", loss_state="x_next")
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

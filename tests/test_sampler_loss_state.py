import pytest

torch = pytest.importorskip("torch")

from fm4pde_ablation.sampler_wrappers import choose_loss_state, sampler_step


class AffineVelocity:
    def __call__(self, x, t):
        return x + t


def test_choose_loss_state_selects_expected_tensor():
    x_cur = torch.ones(1, 1, 2, 2)
    x_next = x_cur + 1
    x_endpoint = x_cur + 2
    assert choose_loss_state("xt", x_cur, x_next, x_endpoint) is x_cur
    assert choose_loss_state("x_next", x_cur, x_next, x_endpoint) is x_next
    assert choose_loss_state("endpoint", x_cur, x_next, x_endpoint) is x_endpoint


def test_midpoint_endpoint_uses_midpoint_state_and_time():
    x_cur = torch.ones(1, 1, 2, 2)
    t = torch.tensor(0.2)
    t_next = torch.tensor(0.4)
    out = sampler_step(
        net=AffineVelocity(),
        x_cur=x_cur,
        t=t,
        t_next=t_next,
        phase="deterministic",
        step_method="midpoint",
        loss_state="endpoint",
    )
    v = x_cur + t
    t_mid = t + 0.5 * (t_next - t)
    x_mid = x_cur + 0.5 * (t_next - t) * v
    expected_endpoint = x_mid + (1.0 - t_mid) * (x_mid + t_mid)
    assert torch.allclose(out.x_endpoint, expected_endpoint)
    assert torch.allclose(out.x_loss_state, expected_endpoint)
    assert out.loss_state == "endpoint"
    assert out.wall_time >= 0.0

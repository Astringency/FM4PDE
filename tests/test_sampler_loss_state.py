import pytest

torch = pytest.importorskip("torch")

from sampling.sampler_wrappers import _call_velocity_model, choose_loss_state, sampler_step


class AffineVelocity:
    def __call__(self, x, t):
        return x + t


class ExtraCapturingVelocity:
    def __init__(self):
        self.extras = []

    def __call__(self, x, t, extra=None):
        self.extras.append(extra)
        return torch.ones_like(x)


class TimeCapturingVelocity:
    def __init__(self):
        self.times = []

    def __call__(self, x, t):
        self.times.append(float(t))
        return torch.ones_like(x)


class LabelScalarVelocity:
    def __init__(self):
        self.extras = []

    def __call__(self, x, t, extra=None):
        self.extras.append(extra)
        label = extra["label"].to(x).view(-1, 1, 1, 1)
        scalar = extra["scalar_conditioning"].to(x).view(-1, 1, 1, 1)
        return torch.ones_like(x) * (10.0 * label + scalar)


class StateVelocity:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, t):
        self.calls += 1
        return x


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


def test_stochastic_midpoint_uses_current_step_midpoint():
    net = TimeCapturingVelocity()
    sampler_step(
        net=net,
        x_cur=torch.zeros(1, 1, 2, 2),
        t=torch.tensor(0.2),
        t_next=torch.tensor(0.4),
        phase="stochastic",
        step_method="midpoint",
        loss_state="endpoint",
    )
    assert net.times == pytest.approx([0.2, 0.3])


@pytest.mark.parametrize("scale,expected", [(0.0, 23.0), (1.0, 13.0), (2.0, 3.0)])
def test_cfg_uses_standard_formula_and_retains_scalar_conditioning(scale, expected):
    net = LabelScalarVelocity()
    x = torch.zeros(1, 1, 2, 2)
    result = _call_velocity_model(
        net,
        x,
        torch.tensor(0.2),
        {
            "label": torch.tensor([1]),
            "scalar_conditioning": torch.tensor([[3.0]]),
            "_cfg_scale": scale,
            "_cfg_null_label": 2,
        },
    )
    assert torch.allclose(result, torch.full_like(x, expected))
    assert all("scalar_conditioning" in extra for extra in net.extras)


@pytest.mark.parametrize("phase", ["deterministic", "stochastic"])
@pytest.mark.parametrize("step_method, expected_calls", [("euler", 1), ("midpoint", 2)])
def test_sampler_step_passes_model_extra_to_velocity_calls(phase, step_method, expected_calls):
    net = ExtraCapturingVelocity()
    x_cur = torch.zeros(1, 1, 2, 2)
    model_extra = {"scalar_conditioning": torch.tensor([[1.5]])}

    sampler_step(
        net=net,
        x_cur=x_cur,
        t=torch.tensor(0.2),
        t_next=torch.tensor(0.4),
        phase=phase,
        step_method=step_method,
        loss_state="endpoint",
        model_extra=model_extra,
    )

    assert len(net.extras) == expected_calls
    assert all(extra is model_extra for extra in net.extras)


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_deterministic_rollout_uses_original_grid_and_retains_endpoint_gradient(use_checkpoint):
    net = StateVelocity()
    x_cur = torch.ones(1, 1, 2, 2, requires_grad=True)
    grid = torch.tensor([0.0, 0.5, 1.0])
    out = sampler_step(
        net=net,
        x_cur=x_cur,
        t=grid[0],
        t_next=grid[1],
        phase="deterministic",
        step_method="euler",
        loss_state="endpoint",
        deterministic_endpoint_mode="rollout",
        deterministic_endpoint_time_grid=grid,
        deterministic_rollout_checkpoint=use_checkpoint,
    )

    assert torch.allclose(out.x_raw_next, 1.5 * x_cur)
    assert torch.allclose(out.x_endpoint, 2.25 * x_cur)
    assert out.x_loss_state is out.x_endpoint
    assert out.endpoint_prediction_mode == "rollout"
    assert out.endpoint_model_evaluations == 2
    gradient = torch.autograd.grad(out.x_endpoint.square().sum(), x_cur)[0]
    assert torch.allclose(gradient, torch.full_like(x_cur, 10.125))
    # The forward endpoint uses exactly two evaluations. Depending on whether
    # autograd saved enough intermediates, non-reentrant checkpoint may or may
    # not need additional recomputation during this deliberately simple test.
    assert net.calls >= 2


def test_deterministic_rollout_requires_remaining_time_grid():
    with pytest.raises(ValueError, match="remaining sampling time grid"):
        sampler_step(
            net=StateVelocity(),
            x_cur=torch.ones(1, 1, 2, 2),
            t=torch.tensor(0.0),
            t_next=torch.tensor(0.5),
            phase="deterministic",
            step_method="euler",
            loss_state="endpoint",
            deterministic_endpoint_mode="rollout",
        )

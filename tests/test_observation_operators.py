import types

import pytest
import torch

from sampling.observation_operators import BoxAverageObservation, PointObservation
from sampling.losses import ObservationTargets, compute_guidance_losses
from sampling.config import AblationConfig
from sampling.masks import PairMasks
from sampling.state import SplitState
from experiments.observation_stress.measurements import build_measurements


@pytest.mark.parametrize("k", [1, 4, 8, 16])
def test_window_means_and_adjoint(k):
    x = torch.arange(32*32, dtype=torch.float64).reshape(1, 1, 32, 32).requires_grad_()
    operator = BoxAverageObservation(k)
    y = operator(x)
    lo = 16-k//2
    expected = x[..., lo:lo+k, lo:lo+k].mean()
    torch.testing.assert_close(y[0, 0, 16, 16], expected)
    gradient, = torch.autograd.grad(y[0, 0, 16, 16], x)
    wanted = torch.zeros_like(x)
    wanted[..., lo:lo+k, lo:lo+k] = 1/(k*k)
    torch.testing.assert_close(gradient, wanted)
    mask = torch.zeros_like(x)
    mask[..., 16, 16] = 1
    operator.validate_mask(mask)
    if k > 1:
        mask[..., 0, 0] = 1
        with pytest.raises(ValueError, match="inside"):
            operator.validate_mask(mask)


def test_point_operator_preserves_losses_and_average_changes_only_observation_terms():
    torch.manual_seed(7)
    f = torch.randn(2, 1, 16, 16, requires_grad=True)
    u = torch.randn_like(f, requires_grad=True)
    mask = torch.zeros_like(f)
    mask[..., 8, 8] = 1
    zero = torch.zeros_like(f)
    gt = types.SimpleNamespace(coef=zero, sol=zero, pde_params={})
    cfg = AblationConfig(pde="poisson", task="both", guidance_components="obs_pde")
    masks = PairMasks(mask, mask.clone(), {})
    state = SplitState(f, u)
    original = compute_guidance_losses(state, gt, masks, cfg)
    point = compute_guidance_losses(state, gt, masks, cfg,
        ObservationTargets(zero, zero, zero, zero, PointObservation()))
    assert torch.equal(point.L_obs_a, original.L_obs_a)
    assert torch.equal(point.L_obs_u, original.L_obs_u)
    average = compute_guidance_losses(state, gt, masks, cfg,
        ObservationTargets(zero, zero, zero, zero, BoxAverageObservation(4)))
    expected = f[..., 6:10, 6:10].mean((-2, -1)).square().mean()
    torch.testing.assert_close(average.L_obs_a, expected)
    assert torch.equal(original.L_pde, average.L_pde)
    average.L_obs_a.backward()
    assert f.grad[..., 6:10, 6:10].count_nonzero() == 32
    assert f.grad.count_nonzero() == 32


def test_frozen_layouts_noise_and_windows_share_measurements():
    torch.manual_seed(13)
    full = torch.randn(2, 2, 128, 128)
    ids = ["file:0", "file:1"]
    reference = torch.zeros(2, 1, 128, 128)
    reference.flatten(2)[..., :500] = 1
    for case in ["random", "grid", "columns", "hole10", "hole25", "strip25"]:
        v = build_measurements(full, ids, reference, case)
        assert (v["mask"].flatten(1).sum(1) == (640 if case == "columns" else 500)).all()
        assert not (v["mask"].bool() & v["excluded"]).any()
        torch.testing.assert_close(v["clean"], full*v["mask"])
    low = build_measurements(full, ids, reference, "noise01")
    high = build_measurements(full, ids, reference, "noise10")
    torch.testing.assert_close((low["noisy"]-low["clean"])*10,
                               high["noisy"]-high["clean"], atol=3e-6, rtol=1e-4)
    windows = [build_measurements(full, ids, reference, f"average{k}") for k in [1, 4, 8, 16]]
    assert all(torch.equal(windows[0]["mask"], v["mask"]) for v in windows)
    for k, v in zip([1, 4, 8, 16], windows):
        yy, xx = v["mask"][0, 0].nonzero()[0]
        yy, xx = int(yy), int(xx)
        manual = full[0, :, yy-k//2:yy-k//2+k, xx-k//2:xx-k//2+k].mean((-1, -2))
        torch.testing.assert_close(v["clean"][0, :, yy, xx], manual)

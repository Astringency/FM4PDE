from types import SimpleNamespace

import pytest

from sampling.config import AblationConfig
from sampling.runner import _calibrate_l2_observation_zeta, _sample_initial_noise
from sampling.sampler_wrappers import _stochastic_bridge_noise_like


torch = pytest.importorskip("torch")


def test_initial_noise_can_replay_one_row_from_a_larger_batch():
    ground_truth = SimpleNamespace(pair=torch.zeros(1, 2, 4, 4))

    full_cfg = AblationConfig(batch_size=4)
    torch.manual_seed(123)
    full = _sample_initial_noise(full_cfg, ground_truth, torch.device("cpu"))

    replay_cfg = AblationConfig(
        batch_size=1,
        extra={
            "initial_noise_source_batch_size": 4,
            "initial_noise_source_indices": [2],
        },
    )
    torch.manual_seed(123)
    replay = _sample_initial_noise(replay_cfg, ground_truth, torch.device("cpu"))

    assert torch.equal(replay, full[2:3])


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"initial_noise_source_batch_size": 0}, "at least batch_size"),
        (
            {"initial_noise_source_batch_size": 4, "initial_noise_source_indices": [0, 1]},
            "exactly batch_size",
        ),
        (
            {"initial_noise_source_batch_size": 4, "initial_noise_source_indices": [4]},
            "inside the source batch",
        ),
    ],
)
def test_initial_noise_replay_validates_source_selection(extra, match):
    ground_truth = SimpleNamespace(pair=torch.zeros(1, 2, 4, 4))
    cfg = AblationConfig(batch_size=1, extra=extra)
    with pytest.raises(ValueError, match=match):
        _sample_initial_noise(cfg, ground_truth, torch.device("cpu"))


def test_batch1_l2_zeta_calibration_matches_reference_mse_gradient_scale():
    cfg = AblationConfig(
        task="inverse",
        batch_size=1,
        obs_guidance_reduction="l2_norm",
        zeta_obs_u=123.0,
        extra={"obs_l2_reference_mse_zeta_u": 3200.0},
    )
    losses = SimpleNamespace(
        L_obs_a=torch.tensor(0.25),
        L_obs_u=torch.tensor(0.04),
        metadata={"obs_counts": {"coef": 500.0, "sol": 500.0}},
    )

    calibration = _calibrate_l2_observation_zeta(cfg, losses, step=0)

    expected_factor = 500.0**0.5 / (2.0 * 0.04**0.5)
    assert cfg.zeta_obs_u == pytest.approx(3200.0 / expected_factor)
    assert calibration["u"]["weighted_gradient_ratio_l2_over_mse"] == pytest.approx(1.0)


def test_stochastic_bridge_noise_can_replay_one_row_from_every_larger_batch_draw():
    full_state = torch.zeros(4, 2, 4, 4)
    torch.manual_seed(456)
    full = _stochastic_bridge_noise_like(full_state, device=torch.device("cpu"))

    replay_state = torch.zeros(1, 2, 4, 4)
    torch.manual_seed(456)
    replay = _stochastic_bridge_noise_like(
        replay_state,
        device=torch.device("cpu"),
        source_batch_size=4,
        source_indices=[2],
    )

    assert torch.equal(replay, full[2:3])

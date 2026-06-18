import pytest

torch = pytest.importorskip("torch")

from sampling.noise import add_observation_noise


def test_noise_only_on_mask():
    obs = torch.ones(1, 1, 4, 4)
    mask = torch.zeros_like(obs)
    mask[..., 0, 0] = 1
    out = add_observation_noise(obs, mask, 0.1, seed=1)
    assert torch.all(out.noisy[mask == 0] == 0)
    assert out.clean.sum() == 1


def test_zero_noise_is_clean():
    obs = torch.randn(1, 1, 4, 4)
    mask = torch.ones_like(obs)
    out = add_observation_noise(obs, mask, 0.0)
    assert torch.equal(out.clean, out.noisy)

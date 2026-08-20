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


def test_relative_noise_scale_is_independent_per_batch_sample():
    base = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    other = base * 1000.0
    mask = torch.ones_like(base)

    alone = add_observation_noise(base, mask, 0.1, seed=7)
    batched = add_observation_noise(
        torch.cat([base, other]),
        torch.cat([mask, mask]),
        0.1,
        seed=7,
    )

    assert batched.metadata["scale_per_sample"][0] == pytest.approx(alone.metadata["scale"])
    assert torch.allclose(batched.noise[0], alone.noise[0])
    assert batched.metadata["scale_per_sample"][1] == pytest.approx(alone.metadata["scale"] * 1000.0)

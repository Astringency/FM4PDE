import pytest

torch = pytest.importorskip("torch")

from data.transform import PDEStandardizer


def test_standardizer_roundtrip_and_stats(tmp_path):
    data = torch.randn(8, 4, 16, 16) * torch.tensor([1.0, 2.0, 0.5, 4.0]).view(1, 4, 1, 1)
    data = data + torch.tensor([0.5, -2.0, 3.0, 10.0]).view(1, 4, 1, 1)

    normalizer = PDEStandardizer.fit(data)
    z = normalizer.transform(data)
    restored = normalizer.inverse_transform(z)

    assert (restored - data).abs().max().item() < 1e-5
    assert torch.allclose(z.mean(dim=(0, 2, 3)), torch.zeros(4), atol=1e-6)
    assert torch.allclose(z.std(dim=(0, 2, 3), unbiased=False), torch.ones(4), atol=1e-6)

    path = normalizer.save(tmp_path / "normalizer.pt")
    loaded = PDEStandardizer.load(path)
    assert torch.allclose(loaded.transform(data), z)
    assert (tmp_path / "normalization.json").exists()

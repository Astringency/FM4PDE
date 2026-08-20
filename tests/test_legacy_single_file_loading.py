import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from data.load import PDEloader


def test_explicit_legacy_file_is_loaded_once_even_when_size_requests_multiple_shards(tmp_path):
    path = tmp_path / "poisson.mat"
    f_data = np.stack([np.full((4, 4), 1.0), np.full((4, 4), 2.0)]).astype(np.float32)
    phi_data = f_data * 10.0
    scipy_io.savemat(path, {"f_data": f_data, "phi_data": phi_data})

    data, labels = PDEloader("poisson").load_data(path, size=5)

    assert tuple(data.shape) == (2, 2, 4, 4)
    assert tuple(labels.shape) == (2,)
    assert data[:, 0, 0, 0].tolist() == pytest.approx([1.0, 2.0])

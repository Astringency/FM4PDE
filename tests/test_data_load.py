import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from data.load import PDEloader


def _write_pair_h5(path, n_samples=2, input_channels=1, output_channels=1):
    with h5py.File(path, "w") as file:
        file.create_dataset("input_data", data=np.zeros((n_samples, input_channels, 4, 4), dtype=np.float32))
        file.create_dataset("output_data", data=np.ones((n_samples, output_channels, 4, 4), dtype=np.float32))
        file.create_dataset(
            "materialized_data",
            data=np.full((n_samples, input_channels + output_channels + 2, 4, 4), 9.0, dtype=np.float32),
        )


def test_pair_h5_sample_scalars_are_cached_not_materialized(tmp_path):
    path = tmp_path / "heat_1-4-4_1.h5"
    _write_pair_h5(path, n_samples=3)
    with h5py.File(path, "a") as file:
        file.create_dataset("alpha", data=np.array([0.1, 0.2, 0.3], dtype=np.float32))
        file.attrs["T"] = 2.0

    loader = PDEloader("heat")
    data, labels = loader.load_data(str(path))

    assert tuple(data.shape) == (3, 2, 4, 4)
    assert torch.equal(data[:, 0], torch.zeros(3, 4, 4))
    assert torch.equal(data[:, 1], torch.ones(3, 4, 4))
    assert torch.equal(labels, torch.full((3,), 7, dtype=torch.long))
    assert set(loader.pde_params) == {"alpha", "T"}
    assert torch.allclose(loader.pde_params["alpha"], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(loader.pde_params["T"], torch.tensor([2.0, 2.0, 2.0]))


def test_pair_h5_fixed_attr_scalars_are_broadcast_and_cached(tmp_path):
    path = tmp_path / "wave_1-4-4_1.h5"
    _write_pair_h5(path, n_samples=2, input_channels=2, output_channels=2)
    with h5py.File(path, "a") as file:
        file.attrs["c"] = 2.5

    loader = PDEloader("wave")
    data, _ = loader.load_data(str(path))

    assert tuple(data.shape) == (2, 4, 4, 4)
    assert torch.allclose(loader.pde_params["c"], torch.tensor([2.5, 2.5]))


def test_pair_h5_required_scalars_accept_dataset_or_attr(tmp_path):
    path = tmp_path / "advection_diffusion_1-4-4_1.h5"
    _write_pair_h5(path, n_samples=2)
    with h5py.File(path, "a") as file:
        file.attrs["b_x"] = 1.0
        file.create_dataset("b_y", data=np.float32(2.0))
        file.create_dataset("kappa", data=np.array([0.01, 0.02], dtype=np.float32))

    loader = PDEloader("advection_diffusion")
    data, labels = loader.load_data(str(path))

    assert tuple(data.shape) == (2, 2, 4, 4)
    assert torch.equal(labels, torch.full((2,), 9, dtype=torch.long))
    assert torch.allclose(loader.pde_params["b_x"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(loader.pde_params["b_y"], torch.tensor([2.0, 2.0]))
    assert torch.allclose(loader.pde_params["kappa"], torch.tensor([0.01, 0.02]))


def test_pair_h5_reads_optional_time_scale_dataset(tmp_path):
    path = tmp_path / "wave_1-4-4_1.h5"
    _write_pair_h5(path, n_samples=2, input_channels=2, output_channels=2)
    with h5py.File(path, "a") as file:
        file.attrs["c"] = 1.5
        file.create_dataset("dt", data=np.array([0.25, 0.5], dtype=np.float32))

    loader = PDEloader("wave")
    data, _ = loader.load_data(str(path))

    assert tuple(data.shape) == (2, 4, 4, 4)
    assert torch.allclose(loader.pde_params["dt"], torch.tensor([0.25, 0.5]))
    assert loader.pde_param_sources["dt"] == "dataset"

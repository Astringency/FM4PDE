import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from data.load import PDEloader


def _write_pair_h5(path, n_samples=3, input_channels=1, output_channels=1, attrs=None, datasets=None):
    with h5py.File(path, "w") as file:
        file.create_dataset("input_data", data=np.zeros((n_samples, input_channels, 6, 6), dtype=np.float32))
        file.create_dataset("output_data", data=np.ones((n_samples, output_channels, 6, 6), dtype=np.float32))
        for key, value in (attrs or {}).items():
            file.attrs[key] = value
        for key, value in (datasets or {}).items():
            file.create_dataset(key, data=value)


@pytest.mark.parametrize(
    ("pde", "input_channels", "output_channels", "attrs", "datasets", "expected_channels", "param_names"),
    [
        ("heat", 1, 1, {}, {"alpha": np.array([0.1, 0.2, 0.3], dtype=np.float32)}, 2, {"alpha"}),
        ("wave", 2, 2, {"c": 2.5}, {}, 4, {"c"}),
        (
            "advection_diffusion",
            1,
            1,
            {"b_x": 1.0, "b_y": 2.0},
            {"kappa": np.array([0.01, 0.02, 0.03], dtype=np.float32)},
            2,
            {"b_x", "b_y", "kappa"},
        ),
        ("steady_heat_conduction", 1, 1, {"u_D": 5.0}, {}, 2, {"u_D"}),
    ],
)
def test_pair_h5_loader_shapes_and_scalar_metadata(
    tmp_path,
    pde,
    input_channels,
    output_channels,
    attrs,
    datasets,
    expected_channels,
    param_names,
):
    path = tmp_path / f"{pde}_1-6-6_1.h5"
    _write_pair_h5(path, input_channels=input_channels, output_channels=output_channels, attrs=attrs, datasets=datasets)

    loader = PDEloader(pde)
    data, labels = loader.load_data(str(path))

    assert tuple(data.shape) == (3, expected_channels, 6, 6)
    assert data.dtype == torch.float32
    assert labels.dtype == torch.long
    assert set(loader.pde_params) == param_names
    for value in loader.pde_params.values():
        assert tuple(value.shape) == (3,)

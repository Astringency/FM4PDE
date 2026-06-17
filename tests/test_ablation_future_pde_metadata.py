import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data.load import PDEloader
from fm4pde_ablation.config import AblationConfig
from fm4pde_ablation.data import load_ground_truth


def _write_heat_h5(path):
    with h5py.File(path, "w") as file:
        file.create_dataset("input_data", data=np.zeros((2, 1, 5, 5), dtype=np.float32))
        file.create_dataset("output_data", data=np.ones((2, 1, 5, 5), dtype=np.float32))
        file.create_dataset("alpha", data=np.array([0.2, 0.4], dtype=np.float32))


def test_future_loader_returns_metadata_without_scalar_fields(tmp_path):
    path = tmp_path / "heat_2-5-5_1.h5"
    _write_heat_h5(path)

    loader = PDEloader("heat")
    data, labels, metadata = loader.load_data(str(path), return_metadata=True)

    assert tuple(data.shape) == (2, 2, 5, 5)
    assert labels.tolist() == [7, 7]
    assert set(metadata["pde_params"]) == {"alpha"}
    assert torch.allclose(metadata["pde_params"]["alpha"], torch.tensor([0.2, 0.4]))
    assert metadata["channel_names"] == ["u0", "uT"]


def test_ablation_ground_truth_reads_sample_level_pde_params(tmp_path):
    path = tmp_path / "heat_2-5-5_1.h5"
    _write_heat_h5(path)
    cfg = AblationConfig(
        pde="heat",
        task="both",
        data_path=str(path),
        data_config_path="",
        checkpoint_path="",
        loadby="future_h5",
        coef_name="input_data",
        solution_name="output_data",
        img_channels=2,
        img_resolution=5,
        batch_size=2,
        offset=0,
        device="cpu",
        allow_synthetic_data=False,
    )

    gt = load_ground_truth(cfg)

    assert tuple(gt.coef.shape) == (2, 1, 5, 5)
    assert tuple(gt.sol.shape) == (2, 1, 5, 5)
    assert tuple(gt.pair.shape) == (2, 2, 5, 5)
    assert set(gt.pde_params) == {"alpha"}
    assert gt.metadata["pde_params_keys"] == ["alpha"]
    assert gt.metadata["pde_params_sources"] == {"alpha": "dataset"}

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from data.load import PDEloader
from sampling.config import AblationConfig
from sampling.data import load_ground_truth


def _write_heat_h5(path):
    with h5py.File(path, "w") as file:
        file.create_dataset("input_data", data=np.zeros((2, 1, 5, 5), dtype=np.float32))
        file.create_dataset("output_data", data=np.ones((2, 1, 5, 5), dtype=np.float32))
        file.create_dataset("alpha", data=np.array([0.2, 0.4], dtype=np.float32))


def test_pair_h5_loader_returns_metadata_without_scalar_fields(tmp_path):
    path = tmp_path / "heat_2-5-5_1.h5"
    _write_heat_h5(path)

    loader = PDEloader("heat")
    data, labels, metadata = loader.load_data(str(path), return_metadata=True)

    assert tuple(data.shape) == (2, 2, 5, 5)
    assert labels.tolist() == [7, 7]
    assert set(metadata["pde_params"]) == {"alpha"}
    assert torch.allclose(metadata["pde_params"]["alpha"], torch.tensor([0.2, 0.4]))
    assert metadata["channel_names"] == ["u0", "uT"]


def test_sampling_ground_truth_reads_pair_h5_params(tmp_path):
    path = tmp_path / "heat_2-5-5_1.h5"
    _write_heat_h5(path)
    cfg = AblationConfig(
        pde="heat",
        task="both",
        data_path=str(path),
        data_config_path="",
        checkpoint_path="",
        loadby="pair_h5",
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


def test_sampling_ground_truth_reads_pair_h5_time_scale_params(tmp_path):
    path = tmp_path / "heat_2-5-5_1.h5"
    _write_heat_h5(path)
    with h5py.File(path, "a") as file:
        file.attrs["total_time"] = 2.5
    cfg = AblationConfig(
        pde="heat",
        task="both",
        data_path=str(path),
        data_config_path="",
        checkpoint_path="",
        loadby="pair_h5",
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

    assert set(gt.pde_params) == {"alpha", "total_time"}
    assert torch.allclose(gt.pde_params["total_time"], torch.tensor([2.5, 2.5]))
    assert gt.metadata["pde_params_sources"]["total_time"] == "attrs:total_time"


def test_near_endpoint_temporal_loader_requires_trajectory(tmp_path):
    path = tmp_path / "heat_2-5-5_1.h5"
    _write_heat_h5(path)
    cfg = AblationConfig(
        pde="heat",
        task="both",
        data_path=str(path),
        data_config_path="",
        checkpoint_path="",
        loadby="pair_h5",
        coef_name="input_data",
        solution_name="output_data",
        img_channels=2,
        img_resolution=5,
        batch_size=1,
        offset=0,
        device="cpu",
        allow_synthetic_data=False,
        residual_mode="near_endpoint_temporal",
        num_near_endpoint_obs=3,
    )

    with pytest.raises(ValueError, match="pair_h5 input/output endpoint data does not contain"):
        load_ground_truth(cfg)


def test_near_endpoint_temporal_loader_from_pair_h5_trajectory(tmp_path):
    path = tmp_path / "heat_2-5-5_1.h5"
    trajectory = np.stack(
        [
            np.zeros((1, 5, 5), dtype=np.float32),
            np.ones((1, 5, 5), dtype=np.float32),
            np.ones((1, 5, 5), dtype=np.float32) * 2,
            np.ones((1, 5, 5), dtype=np.float32) * 3,
        ],
        axis=0,
    )
    with h5py.File(path, "w") as file:
        file.create_dataset("input_data", data=trajectory[None, 0])
        file.create_dataset("output_data", data=trajectory[None, -1])
        file.create_dataset("full_trajectory", data=trajectory[None])
        file.create_dataset("alpha", data=np.array([0.2], dtype=np.float32))
        file.attrs["T"] = 3.0
    cfg = AblationConfig(
        pde="heat",
        task="both",
        data_path=str(path),
        data_config_path="",
        checkpoint_path="",
        loadby="pair_h5",
        coef_name="input_data",
        solution_name="output_data",
        img_channels=2,
        img_resolution=5,
        batch_size=1,
        offset=0,
        device="cpu",
        allow_synthetic_data=False,
        residual_mode="near_endpoint_temporal",
        num_near_endpoint_obs=4,
        near_endpoint_mask_seed=123,
    )

    gt = load_ground_truth(cfg)

    near = gt.pde_params["near_endpoint_temporal"]
    assert torch.allclose(near["q_dt"], torch.ones(1, 1, 5, 5))
    assert torch.allclose(near["q_T_minus_dt"], torch.ones(1, 1, 5, 5) * 2)
    assert torch.allclose(near["dt"], torch.tensor([1.0]))
    assert near["mask_0"].sum().item() == pytest.approx(4.0)
    assert near["mask_T"].sum().item() == pytest.approx(4.0)
    assert gt.metadata["near_endpoint_temporal"]["extra_observation_budget"] is True
    assert gt.metadata["near_endpoint_temporal"]["frame_metadata"][0]["trajectory_dataset"] == "full_trajectory"

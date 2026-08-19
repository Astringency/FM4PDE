import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from data.load import PDEloader
from data.specs import get_pde_spec
from sampling.config import AblationConfig
from sampling.data import _attach_boundary_metadata_params, load_ground_truth
from sampling.data import finalize_ground_truth_config


def test_boundary_metadata_detects_mixed_before_neumann():
    class File:
        attrs = {"boundary_condition": "bottom Dirichlet; top/left/right zero Neumann"}

    params = {}
    sources = {}
    _attach_boundary_metadata_params(AblationConfig(pde="steady_heat_conduction"), File(), params, sources)
    assert params["boundary_condition_kind"] == "mixed"


def _write_heat_h5(path):
    with h5py.File(path, "w") as file:
        file.create_dataset("input_data", data=np.zeros((2, 1, 5, 5), dtype=np.float32))
        file.create_dataset("output_data", data=np.ones((2, 1, 5, 5), dtype=np.float32))
        file.create_dataset("alpha", data=np.array([0.2, 0.4], dtype=np.float32))


def _write_ns_h5(path):
    w = np.zeros((2, 6, 6, 4), dtype=np.float32)
    for sample_idx in range(2):
        for time_idx in range(4):
            w[sample_idx, :, :, time_idx] = sample_idx + time_idx + 1
    with h5py.File(path, "w") as file:
        file.create_dataset("w0", data=np.stack([np.full((6, 6), i, dtype=np.float32) for i in range(2)]))
        file.create_dataset("w", data=w)
        file.create_dataset("t", data=np.linspace(0.5, 2.0, 4, dtype=np.float32))
        file.attrs["nu"] = 0.002
        file.create_dataset("viscosity", data=np.array([0.003, 0.004], dtype=np.float32))
        file.attrs["T"] = 2.0
        file.create_dataset("total_time", data=np.array([2.5, 3.5], dtype=np.float32))
        file.create_dataset("dt", data=np.array([0.25, 0.5], dtype=np.float32))


def test_pair_h5_loader_returns_metadata_without_scalar_fields(tmp_path):
    path = tmp_path / "heat_2-5-5_1.h5"
    _write_heat_h5(path)

    loader = PDEloader("heat")
    data, labels, metadata = loader.load_data(str(path), return_metadata=True)

    assert tuple(data.shape) == (2, 2, 5, 5)
    assert labels.tolist() == [get_pde_spec("heat").label_id, get_pde_spec("heat").label_id]
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
    assert set(gt.pde_params) == {"alpha", "boundary_condition_kind"}
    assert gt.pde_params["boundary_condition_kind"] == "periodic"
    assert gt.metadata["pde_params_keys"] == ["alpha", "boundary_condition_kind"]
    assert gt.metadata["pde_params_sources"] == {
        "alpha": "dataset",
        "boundary_condition_kind": "loader_confirmed_default",
    }
    assert gt.metadata["boundary_condition"] == "periodic"
    assert gt.metadata["boundary_condition_source"] == "loader_confirmed_default"


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

    assert set(gt.pde_params) == {"alpha", "total_time", "boundary_condition_kind"}
    assert gt.metadata["pde_params_keys"] == ["alpha", "boundary_condition_kind", "total_time"]
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


def test_nsnonbounded_h5py_reads_scalar_params(tmp_path):
    path = tmp_path / "ns.h5"
    _write_ns_h5(path)
    cfg = AblationConfig(
        pde="nsnonbounded",
        task="both",
        data_path=str(path),
        data_config_path="",
        checkpoint_path="",
        loadby="h5py",
        coef_name="w0",
        solution_name="w",
        img_channels=2,
        img_resolution=6,
        batch_size=2,
        offset=0,
        device="cpu",
        allow_synthetic_data=False,
    )

    gt = load_ground_truth(cfg)

    assert tuple(gt.coef.shape) == (2, 1, 6, 6)
    assert tuple(gt.sol.shape) == (2, 1, 6, 6)
    assert set(gt.pde_params) == {"nu", "T", "solver_dt"}
    assert torch.allclose(gt.pde_params["nu"], torch.tensor([0.002, 0.002]))
    assert torch.allclose(gt.pde_params["T"], torch.tensor([2.0, 2.0]))
    assert torch.allclose(gt.pde_params["solver_dt"], torch.tensor([0.25, 0.5]))
    assert gt.metadata["pde_params_sources"]["nu"] == "attrs:nu"
    assert gt.metadata["pde_params_sources"]["T"] == "attrs:T"
    assert gt.metadata["pde_params_sources"]["solver_dt"] == "dataset:dt"


def test_nsnonbounded_h5py_full_trajectory_fd_reads_w(tmp_path):
    path = tmp_path / "ns.h5"
    _write_ns_h5(path)
    cfg = AblationConfig(
        pde="nsnonbounded",
        task="both",
        data_path=str(path),
        data_config_path="",
        checkpoint_path="",
        loadby="h5py",
        coef_name="w0",
        solution_name="w",
        img_channels=2,
        img_resolution=6,
        batch_size=1,
        offset=1,
        device="cpu",
        allow_synthetic_data=False,
        residual_mode="full_trajectory_fd",
    )

    gt = load_ground_truth(cfg)

    trajectory = gt.pde_params["trajectory"]
    assert tuple(trajectory.shape) == (1, 5, 1, 6, 6)
    assert torch.allclose(trajectory[:, 0], gt.coef)
    assert torch.allclose(trajectory[:, -1], gt.sol)
    assert np.allclose(gt.pde_params["trajectory_time_values"], np.linspace(0.0, 2.0, 5))
    assert gt.metadata["full_trajectory_fd"]["source"] == "h5py"
    assert gt.metadata["full_trajectory_fd"]["frame_metadata"][0]["trajectory_dataset"] == "w"


def test_helmholtz_k_is_inferred_from_generator_filename():
    cfg = AblationConfig(pde="helmholtz", data_path="/data/helmholtz_test_1000-128-128-k10.mat", k=1)
    assert finalize_ground_truth_config(cfg).k == 10

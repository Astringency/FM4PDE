import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from data.load import PDEloader
from data.specs import get_pde_spec
from sampling.config import AblationConfig
from sampling.data import _attach_boundary_metadata_params, attach_near_endpoint_observations, load_ground_truth
from sampling.data import finalize_ground_truth_config
from sampling.masks import make_pair_masks
from sampling.pde_residuals import compute_pde_residual


def _load_and_attach(cfg):
    gt = load_ground_truth(cfg)
    masks = make_pair_masks(
        gt.coef.shape,
        gt.sol.shape,
        cfg.num_obs,
        cfg.sensor_mode,
        cfg.shared_mask,
        cfg.mask_seed,
        device=gt.coef.device,
        dtype=gt.coef.dtype,
        num_sensor_columns=cfg.num_sensor_columns,
    )
    return attach_near_endpoint_observations(cfg, gt, masks)


def test_boundary_metadata_detects_mixed_before_neumann():
    class File:
        attrs = {"boundary_condition": "bottom Dirichlet; top/left/right zero Neumann"}

    params = {}
    sources = {}
    _attach_boundary_metadata_params(AblationConfig(pde="steady_heat_conduction"), File(), params, sources)
    assert params["boundary_condition_kind"] == "mixed"


def _write_heat_h5(path):
    with h5py.File(path, "w") as file:
        file.attrs["split"] = "train"
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


def _write_rd_h5(path):
    trajectory = np.zeros((5, 4, 4, 2), dtype=np.float32)
    for time_idx in range(5):
        trajectory[time_idx, :, :, 0] = time_idx + 1
        trajectory[time_idx, :, :, 1] = 10 * (time_idx + 1)
    with h5py.File(path, "w") as file:
        file.attrs["T"] = 1.0
        group = file.create_group("000000")
        group.create_dataset("data", data=trajectory)


def _write_swe_h5(path):
    with h5py.File(path, "w") as file:
        file.attrs["T"] = 1.0
        file.attrs["g"] = 1.0
        group = file.create_group("000000")
        data = group.create_group("data")
        for channel_idx, name in enumerate(("h", "hu", "hv"), start=1):
            trajectory = np.zeros((5, 4, 4, 1), dtype=np.float32)
            for time_idx in range(5):
                trajectory[time_idx, :, :, 0] = channel_idx * (time_idx + 1)
            data.create_dataset(name, data=trajectory)


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

    gt = _load_and_attach(cfg)

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

    gt = _load_and_attach(cfg)

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
        num_obs=3,
    )

    with pytest.raises(ValueError, match="pair_h5 input/output endpoint data does not contain"):
        _load_and_attach(cfg)


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
        file.attrs["split"] = "train"
    cfg = AblationConfig(
        pde="heat",
        task="both",
        data_path=str(path),
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
        num_obs=4,
        mask_seed=123,
    )

    gt = _load_and_attach(cfg)

    near = gt.pde_params["near_endpoint_temporal"]
    assert torch.allclose(near["q_dt"], near["mask_0"])
    assert torch.allclose(near["q_T_minus_dt"], near["mask_T"] * 2)
    assert torch.count_nonzero(near["q_dt"]).item() == 4
    assert torch.count_nonzero(near["q_T_minus_dt"]).item() == 4
    assert torch.allclose(near["dt"], torch.tensor([1.0]))
    assert near["mask_0"].sum().item() == pytest.approx(4.0)
    assert near["mask_T"].sum().item() == pytest.approx(4.0)
    assert gt.metadata["near_endpoint_temporal"]["extra_observation_budget"] is False
    assert gt.metadata["near_endpoint_temporal"]["mask_alignment"] == {
        "q_dt": "coef/q0",
        "q_T_minus_dt": "sol/qT",
    }
    assert gt.metadata["near_endpoint_temporal"]["observed_values_only"] is True
    assert gt.metadata["near_endpoint_temporal"]["unobserved_values_zeroed"] is True
    assert gt.metadata["near_endpoint_temporal"]["full_near_endpoint_frames_retained"] is False
    assert gt.metadata["near_endpoint_temporal"]["frame_metadata"][0]["trajectory_dataset"] == "full_trajectory"


def test_wave_near_endpoint_reconstructs_velocity_from_legacy_displacement_trajectory(tmp_path):
    path = tmp_path / "wave_1-4-4_1.h5"
    times = np.array([0.0, 0.1, 0.4, 0.9, 1.0], dtype=np.float32)
    displacement = np.stack(
        [np.full((4, 4), time**2, dtype=np.float32) for time in times],
        axis=0,
    )
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "input_data",
            data=np.stack([displacement[0], np.zeros((4, 4), dtype=np.float32)], axis=0)[None],
        )
        file.create_dataset(
            "output_data",
            data=np.stack([displacement[-1], np.full((4, 4), 2.0, dtype=np.float32)], axis=0)[None],
        )
        # Legacy formal Wave files use [N,1,T,H,W], while endpoints are [u,v].
        file.create_dataset("full_trajectory", data=displacement[None, None])
        file.create_dataset("t", data=times)
        file.attrs["T"] = 1.0
        file.attrs["fixed_c"] = 1.0

    cfg = AblationConfig(
        pde="wave",
        task="both",
        data_path=str(path),
        checkpoint_path="",
        loadby="pair_h5",
        coef_name="input_data",
        solution_name="output_data",
        img_channels=4,
        img_resolution=4,
        batch_size=1,
        offset=0,
        device="cpu",
        allow_synthetic_data=False,
        residual_mode="near_endpoint_temporal",
        num_obs=4,
        mask_seed=123,
    )

    gt = _load_and_attach(cfg)

    near = gt.pde_params["near_endpoint_temporal"]
    assert tuple(near["q_dt"].shape) == (1, 2, 4, 4)
    assert tuple(near["q_T_minus_dt"].shape) == (1, 2, 4, 4)
    assert torch.allclose(near["q_dt"][:, 0:1], near["mask_0"] * 0.01, atol=1e-6)
    assert torch.allclose(near["q_dt"][:, 1:2], near["mask_0"] * 0.2, atol=1e-6)
    assert torch.allclose(near["q_T_minus_dt"][:, 0:1], near["mask_T"] * 0.81, atol=1e-6)
    assert torch.allclose(near["q_T_minus_dt"][:, 1:2], near["mask_T"] * 1.8, atol=1e-6)
    assert torch.allclose(near["dt"], torch.tensor([0.1]), atol=1e-7)
    frame_meta = gt.metadata["near_endpoint_temporal"]["frame_metadata"][0]
    assert frame_meta["near_endpoint_state_source"] == (
        "saved_displacement_trajectory_with_reconstructed_velocity"
    )
    assert frame_meta["wave_velocity_reconstructed"] is True
    assert frame_meta["wave_velocity_reconstruction"] == "three_point_lagrange_derivative"
    assert frame_meta["stored_trajectory_channels"] == 1
    assert frame_meta["expected_state_channels"] == 2
    assert frame_meta["dt_source"] == "dataset:t_endpoint_intervals"
    residual = compute_pde_residual(
        "wave",
        gt.coef,
        gt.sol,
        pde_params=gt.pde_params,
        residual_mode="near_endpoint_temporal",
    )
    assert residual.metadata["resolved_residual_mode"] == "near_endpoint_temporal"
    assert residual.status == "approximate"


def test_nsnonbounded_h5py_reads_scalar_params(tmp_path):
    path = tmp_path / "ns.h5"
    _write_ns_h5(path)
    cfg = AblationConfig(
        pde="nsnonbounded",
        task="both",
        data_path=str(path),
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

    gt = _load_and_attach(cfg)

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

    gt = _load_and_attach(cfg)

    trajectory = gt.pde_params["trajectory"]
    assert tuple(trajectory.shape) == (1, 5, 1, 6, 6)
    assert torch.allclose(trajectory[:, 0], gt.coef)
    assert torch.allclose(trajectory[:, -1], gt.sol)
    assert np.allclose(gt.pde_params["trajectory_time_values"], np.linspace(0.0, 2.0, 5))
    assert gt.metadata["full_trajectory_fd"]["source"] == "h5py"
    assert gt.metadata["full_trajectory_fd"]["frame_metadata"][0]["trajectory_dataset"] == "w"


@pytest.mark.parametrize("stored_channels", [1, 2])
def test_wave_truth_trajectory_loader_accepts_legacy_and_full_state(tmp_path, stored_channels):
    """Both real archive schemas must reach the offline residual with all channels."""
    path = tmp_path / "wave.h5"
    times = np.linspace(0., 1., 5, dtype=np.float32)
    displacement = np.broadcast_to(times[:, None, None], (5, 8, 8)).copy()
    state = np.stack([displacement, np.ones_like(displacement)], axis=0)
    with h5py.File(path, "w") as file:
        file.create_dataset("input_data", data=state[:, 0][None])
        file.create_dataset("output_data", data=state[:, -1][None])
        file.create_dataset("full_trajectory", data=state[None, :stored_channels])
        file.create_dataset("t", data=times)
        file.attrs["T"] = 1.
        file.attrs["fixed_c"] = 1.
    cfg = AblationConfig(pde="wave", task="both", data_path=str(path),
        checkpoint_path="", loadby="pair_h5", coef_name="input_data",
        solution_name="output_data", img_channels=4, img_resolution=8,
        batch_size=1, device="cpu", dtype="float64", allow_synthetic_data=False,
        residual_mode="full_trajectory_fd")
    gt = load_ground_truth(cfg)
    trajectory = gt.pde_params["trajectory"]
    assert trajectory.shape == (1, 5, stored_channels, 8, 8)
    assert torch.equal(trajectory[0], torch.as_tensor(state[:stored_channels].transpose(1, 0, 2, 3)).double())
    assert gt.metadata["full_trajectory_fd"]["uses_generated_trajectory"] is False
    assert gt.metadata["full_trajectory_fd"]["guidance_compatible"] is False
    output = compute_pde_residual("wave", gt.coef, gt.sol,
        pde_params=gt.pde_params, residual_mode="full_trajectory_fd")
    assert output.components["interior"].abs().max() < 1e-12
    assert output.metadata["guidance_compatible"] is False
    expected = "first_order_state" if stored_channels == 2 else "displacement_only_second_order"
    assert output.metadata["wave_state_form"] == expected


def test_nsnonbounded_near_endpoint_uses_first_saved_frame_and_correct_dt(tmp_path):
    path = tmp_path / "ns.h5"
    _write_ns_h5(path)
    cfg = AblationConfig(
        pde="nsnonbounded",
        task="both",
        data_path=str(path),
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
        residual_mode="near_endpoint_temporal",
        num_obs=3,
        shared_mask=False,
        mask_seed=7,
    )

    gt = _load_and_attach(cfg)

    near = gt.pde_params["near_endpoint_temporal"]
    expected_q_dt = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1) * near["mask_0"]
    expected_q_tm = torch.tensor([3.0, 4.0]).reshape(2, 1, 1, 1) * near["mask_T"]
    assert torch.allclose(near["q_dt"], expected_q_dt)
    assert torch.allclose(near["q_T_minus_dt"], expected_q_tm)
    assert torch.allclose(near["dt"], torch.tensor([0.5, 0.5]))
    frame_meta = gt.metadata["near_endpoint_temporal"]["frame_metadata"]
    assert [item["q_dt_frame"] for item in frame_meta] == [0, 0]
    assert all(item["initial_frame_stored_separately"] for item in frame_meta)


@pytest.mark.parametrize(
    ("pde", "loadby", "writer", "channels", "first_values", "near_end_values"),
    [
        ("reaction_diffusion", "rd", _write_rd_h5, 2, (2.0, 20.0), (4.0, 40.0)),
        ("shallow_water", "swe", _write_swe_h5, 3, (2.0, 4.0, 6.0), (4.0, 8.0, 12.0)),
    ],
)
def test_grouped_temporal_pdes_retain_only_sparse_near_endpoint_observations(
    tmp_path,
    pde,
    loadby,
    writer,
    channels,
    first_values,
    near_end_values,
):
    path = tmp_path / f"{pde}.h5"
    writer(path)
    cfg = AblationConfig(
        pde=pde,
        task="both",
        data_path=str(path),
        checkpoint_path="",
        loadby=loadby,
        img_channels=2 * channels,
        img_resolution=4,
        batch_size=1,
        offset=0,
        device="cpu",
        allow_synthetic_data=False,
        residual_mode="near_endpoint_temporal",
        num_obs=3,
        mask_seed=17,
    )

    gt = _load_and_attach(cfg)

    near = gt.pde_params["near_endpoint_temporal"]
    assert tuple(near["q_dt"].shape) == (1, channels, 4, 4)
    assert tuple(near["q_T_minus_dt"].shape) == (1, channels, 4, 4)
    for channel, value in enumerate(first_values):
        assert torch.allclose(near["q_dt"][:, channel : channel + 1], near["mask_0"] * value)
    for channel, value in enumerate(near_end_values):
        assert torch.allclose(
            near["q_T_minus_dt"][:, channel : channel + 1], near["mask_T"] * value
        )
    assert torch.allclose(near["dt"], torch.tensor([0.25]))


def test_burgers_full_trajectory_mode_does_not_load_ground_truth_as_pde_param(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    path = tmp_path / "burger.mat"
    trajectories = np.arange(2 * 7 * 8, dtype=np.float32).reshape(2, 7, 8)
    scipy_io.savemat(path, {"output": trajectories})
    cfg = AblationConfig(
        pde="burger",
        task="both",
        data_path=str(path),
        checkpoint_path="",
        loadby="scipy",
        coef_name="output",
        solution_name="output",
        img_channels=1,
        img_resolution=8,
        batch_size=2,
        offset=0,
        device="cpu",
        allow_synthetic_data=False,
        residual_mode="full_trajectory_fd",
    )

    gt = load_ground_truth(cfg)

    assert tuple(gt.pair.shape) == (2, 1, 7, 8)
    assert "trajectory" not in gt.pde_params
    assert "full_trajectory" not in gt.pde_params
    assert gt.metadata["full_trajectory_fd"]["ground_truth_auxiliary_loaded"] is False
    assert gt.metadata["full_trajectory_fd"]["source"] == "model_output_at_guidance_and_evaluation_time"


def test_helmholtz_k_is_inferred_from_generator_filename():
    cfg = AblationConfig(pde="helmholtz", data_path="/data/helmholtz_test_1000-128-128-k10.mat", k=1)
    assert finalize_ground_truth_config(cfg).k == 10

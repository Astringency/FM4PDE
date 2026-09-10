from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from data.specs import get_pde_spec
from sampling.config import AblationConfig, normalize_residual_mode


@dataclass
class PDEGroundTruth:
    pde: str
    coef: Any
    sol: Any
    pair: Any
    pde_params: dict[str, Any]
    channel_names_coef: list[str]
    channel_names_sol: list[str]
    metadata: dict[str, Any]


def finalize_ground_truth_config(config: AblationConfig) -> AblationConfig:
    spec = get_pde_spec(config.pde)
    if not config.loadby:
        config.loadby = spec.default_loadby
    if not config.coef_name:
        config.coef_name = "input_data" if config.loadby == "pair_h5" else config.coef_name
    if not config.solution_name:
        config.solution_name = "output_data" if config.loadby == "pair_h5" else config.solution_name
    if config.pde == "burger":
        config.img_channels = 1
    else:
        config.img_channels = spec.coef_channels + spec.sol_channels
    if config.pde == "helmholtz" and config.data_path:
        match = re.search(r"(?:_|-)k(\d+)(?:\.[^.]+)?$", Path(config.data_path).name)
        if match:
            file_k = int(match.group(1))
            if config.k not in {1, file_k}:
                raise ValueError(f"Helmholtz filename encodes k={file_k}, but config sets k={config.k}")
            config.k = file_k
    return config


def load_ground_truth(config: AblationConfig) -> PDEGroundTruth:
    """Load one or more test samples using the configured FM4PDE data format."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("load_ground_truth requires torch") from exc

    config = finalize_ground_truth_config(config)

    if not config.data_path or not Path(config.data_path).exists():
        if normalize_residual_mode(config.residual_mode) == "near_endpoint_temporal":
            raise FileNotFoundError(
                "near_endpoint_temporal mode requires extra near-endpoint sparse temporal observations and masks; "
                f"data path does not exist: {config.data_path}"
            )
        if not config.allow_synthetic_data:
            raise FileNotFoundError(f"Data path does not exist: {config.data_path}")
        return make_synthetic_ground_truth(config)

    raw = _load_raw_data(config)
    coef_list = []
    sol_list = []
    offsets = []
    for batch_idx in range(config.batch_size):
        offset = config.offset + batch_idx
        offsets.append(offset)
        coef, sol = _extract_single_sample(config, raw, offset)
        expected = _channel_counts(config.pde, config.img_channels)
        coef_list.append(_ensure_bchw(coef, config.pde, "coef", config.device, config.dtype, expected[0]))
        sol_list.append(_ensure_bchw(sol, config.pde, "sol", config.device, config.dtype, expected[1]))

    coef_t = torch.cat(coef_list, dim=0)
    sol_t = torch.cat(sol_list, dim=0)
    _assert_spatial_compatible(coef_t, sol_t, config.pde)
    pair = coef_t if config.pde == "burger" else torch.cat([coef_t, sol_t], dim=1)
    pde_params: dict[str, Any] = {}
    pde_param_sources: dict[str, str] = {}
    if config.loadby == "pair_h5":
        pde_params, pde_param_sources = _pair_h5_params_for_offsets(raw["__h5__"], config.pde, offsets, pair.device)
    elif config.loadby == "h5py":
        pde_params, pde_param_sources = _h5py_params_for_offsets(raw["__h5__"], config.pde, offsets, pair.device)
    elif config.loadby == "rd":
        pde_params, pde_param_sources = _rd_params_for_offsets(raw["__h5__"], offsets, pair.device)
    elif config.loadby == "swe":
        pde_params, pde_param_sources = _swe_params_for_offsets(raw["__h5__"], offsets, pair.device)
    _attach_boundary_metadata_params(config, raw.get("__h5__") if isinstance(raw, dict) else None, pde_params, pde_param_sources)
    trajectory_metadata: dict[str, Any] | None = None
    if normalize_residual_mode(config.residual_mode) == "full_trajectory_fd" and config.pde != "burger":
        trajectory, trajectory_metadata = _full_trajectory_for_offsets(config, raw, offsets, coef_t, pde_params)
        pde_params["trajectory"] = trajectory
        pde_param_sources["trajectory"] = "full_trajectory_observations"
        pde_params["trajectory_is_observed_ground_truth"] = True
        pde_param_sources["trajectory_is_observed_ground_truth"] = "loader_semantics"
        time_values = _full_trajectory_time_values(config, raw, offsets)
        if time_values is not None:
            pde_params["trajectory_time_values"] = time_values
            pde_param_sources["trajectory_time_values"] = "saved_snapshot_times"
    elif normalize_residual_mode(config.residual_mode) == "full_trajectory_fd" and config.pde == "burger":
        trajectory_metadata = {
            "source": "model_output_at_guidance_and_evaluation_time",
            "shape": list(coef_t.shape),
            "ground_truth_auxiliary_loaded": False,
            "uses_generated_trajectory": True,
            "endpoint_only": False,
        }
    spec = get_pde_spec(config.pde)
    channel_names_coef = _channel_names(spec.coef_channel_names, int(coef_t.shape[1]), "coef")
    channel_names_sol = _channel_names(spec.sol_channel_names, int(sol_t.shape[1]), "sol")
    metadata = {
        "data_path": config.data_path,
        "offset": config.offset,
        "loadby": config.loadby,
        "synthetic": False,
        "batch_size": config.batch_size,
        "sample_offsets": offsets,
        "sample_ids": [str(offset) for offset in offsets],
        "channel_names": list(spec.channel_names),
        "channel_names_coef": channel_names_coef,
        "channel_names_sol": channel_names_sol,
        "pde_data_spec": spec.to_metadata(),
        "scalar_param_names": list(spec.scalar_param_names),
        "residual_family": spec.residual_family,
        "endpoint_pair": config.pde != "burger",
        "pde_params_keys": sorted(pde_params),
        "pde_params_sources": pde_param_sources,
        "scalar_params_loaded": bool(pde_params),
        "boundary_condition": pde_params.get("boundary_condition_kind", None),
        "boundary_condition_source": pde_param_sources.get("boundary_condition_kind", None),
    }
    if trajectory_metadata is not None:
        metadata["full_trajectory_fd"] = trajectory_metadata
        metadata["pde_params_keys"] = sorted(pde_params)
        metadata["pde_params_sources"] = pde_param_sources
    ground_truth = PDEGroundTruth(
        pde=config.pde,
        coef=coef_t,
        sol=sol_t,
        pair=pair,
        pde_params=pde_params,
        channel_names_coef=channel_names_coef,
        channel_names_sol=channel_names_sol,
        metadata=metadata,
    )
    handle = raw.get("__h5__") if isinstance(raw, dict) else None
    if handle is not None:
        handle.close()
    return ground_truth


def attach_near_endpoint_observations(
    config: AblationConfig,
    ground_truth: PDEGroundTruth,
    masks: Any,
) -> PDEGroundTruth:
    """Attach the sole allowed true-field PDE auxiliary using endpoint-aligned masks."""
    if normalize_residual_mode(config.residual_mode) != "near_endpoint_temporal":
        return ground_truth
    if not config.data_path or not Path(config.data_path).exists():
        raise FileNotFoundError(
            "near_endpoint_temporal requires real temporal data containing q(dt) and q(T-dt)"
        )
    offsets = [int(value) for value in ground_truth.metadata.get("sample_offsets", [])]
    if len(offsets) != int(ground_truth.coef.shape[0]):
        offsets = [int(config.offset) + idx for idx in range(int(ground_truth.coef.shape[0]))]
    raw = _load_raw_data(config)
    try:
        near_params, near_metadata = _near_endpoint_temporal_for_offsets(
            config,
            raw,
            offsets,
            ground_truth.coef,
            ground_truth.sol,
            ground_truth.pde_params,
            masks,
        )
    finally:
        handle = raw.get("__h5__") if isinstance(raw, dict) else None
        if handle is not None:
            handle.close()
    ground_truth.pde_params["near_endpoint_temporal"] = near_params
    sources = ground_truth.metadata.setdefault("pde_params_sources", {})
    sources["near_endpoint_temporal"] = "endpoint_aligned_sparse_temporal_observations"
    ground_truth.metadata["near_endpoint_temporal"] = near_metadata
    ground_truth.metadata["pde_params_keys"] = sorted(ground_truth.pde_params)
    return ground_truth


def make_synthetic_ground_truth(config: AblationConfig) -> PDEGroundTruth:
    """Create deterministic synthetic tensors for dry-run tests when PDE data is unavailable."""
    import torch

    if normalize_residual_mode(config.residual_mode) == "near_endpoint_temporal":
        raise ValueError(
            "near_endpoint_temporal mode requires extra near-endpoint sparse temporal observations and masks; "
            "synthetic endpoint-only dry-run data cannot provide them"
        )

    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    channels = _channel_counts(config.pde, config.img_channels)
    h = w = int(config.img_resolution)
    gen = torch.Generator(device="cpu").manual_seed(int(config.sample_seed))
    total_channels = 1 if config.pde == "burger" else sum(channels)
    base = torch.randn(
        config.batch_size,
        total_channels,
        h,
        w,
        generator=gen,
        dtype=dtype,
    ).to(device)
    # Smooth-ish fields are enough to exercise transforms, masks, losses and residuals.
    yy = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w)
    smooth = torch.sin(2 * torch.pi * xx) * torch.cos(2 * torch.pi * yy)
    base = 0.1 * base + smooth
    if config.pde == "burger":
        coef = base
        sol = base
        pair = base
    else:
        coef = base[:, : channels[0]]
        sol = base[:, channels[0] :]
        pair = torch.cat([coef, sol], dim=1)
    pde_params = _synthetic_pde_params(config, device, dtype)
    spec = get_pde_spec(config.pde)
    channel_names_coef = _channel_names(spec.coef_channel_names, channels[0], "coef")
    channel_names_sol = _channel_names(spec.sol_channel_names, channels[1], "sol")
    return PDEGroundTruth(
        pde=config.pde,
        coef=coef,
        sol=sol,
        pair=pair,
        pde_params=pde_params,
        channel_names_coef=channel_names_coef,
        channel_names_sol=channel_names_sol,
        metadata={
            "data_path": config.data_path,
            "offset": config.offset,
            "loadby": config.loadby,
            "synthetic": True,
            "reason": "configured data path was missing",
            "batch_size": config.batch_size,
            "channel_names": list(spec.channel_names),
            "channel_names_coef": channel_names_coef,
            "channel_names_sol": channel_names_sol,
            "pde_data_spec": spec.to_metadata(),
            "scalar_param_names": list(spec.scalar_param_names),
            "residual_family": spec.residual_family,
            "endpoint_pair": config.pde != "burger",
            "pde_params_keys": sorted(pde_params),
            "pde_params_sources": {key: "synthetic_default" for key in pde_params},
            "scalar_params_loaded": False,
        },
    )


def _load_raw_data(config: AblationConfig) -> dict[str, Any]:
    if config.loadby == "scipy":
        import scipy.io

        return scipy.io.loadmat(config.data_path)
    if config.loadby in {"h5py", "swe", "rd", "pair_h5"}:
        import h5py

        return {"__h5__": h5py.File(config.data_path, "r")}
    if config.loadby == "numpy":
        import numpy as np

        loaded = np.load(config.data_path)
        return dict(loaded.items()) if hasattr(loaded, "items") else {"array": loaded}
    raise ValueError(f"Unsupported loadby={config.loadby!r} for {config.pde}")


def _extract_single_sample(config: AblationConfig, raw: dict[str, Any], offset: int) -> tuple[Any, Any]:
    pde = config.pde
    if config.loadby == "swe":
        file = raw["__h5__"]
        key = _sample_group_key(file, offset)
        group = file[key]["data"]
        import numpy as np

        coef = np.stack(
            [
                group["h"][0, :, :, 0],
                group["hu"][0, :, :, 0],
                group["hv"][0, :, :, 0],
            ],
            axis=0,
        )
        sol = np.stack(
            [
                group["h"][-1, :, :, 0],
                group["hu"][-1, :, :, 0],
                group["hv"][-1, :, :, 0],
            ],
            axis=0,
        )
        return coef, sol

    if config.loadby == "rd":
        file = raw["__h5__"]
        key = _sample_group_key(file, offset)
        arr = file[key]["data"][:]
        return arr[0, :, :, :], arr[-1, :, :, :]

    if config.loadby == "pair_h5":
        file = raw["__h5__"]
        import numpy as np

        input_data = np.asarray(file["input_data"][offset], dtype=np.float32)
        output_data = np.asarray(file["output_data"][offset], dtype=np.float32)
        return input_data, output_data

    if config.loadby == "h5py":
        file = raw["__h5__"]
        if pde == "darcy":
            return file[config.coef_name][:, :, offset], file[config.solution_name][:, :, offset]
        if pde == "nsnonbounded":
            return file[config.coef_name][offset, :, :], file[config.solution_name][offset, :, :, -1]
        coef_raw = file[config.coef_name]
        sol_raw = file[config.solution_name]
    else:
        coef_raw = raw[config.coef_name]
        sol_raw = raw[config.solution_name]

    if pde == "burger":
        field = coef_raw[offset, :, :]
        return field, field
    return coef_raw[offset, :, :], sol_raw[offset, :, :]


def _near_endpoint_temporal_for_offsets(
    config: AblationConfig,
    raw: dict[str, Any],
    offsets: list[int],
    coef_t: Any,
    sol_t: Any,
    pde_params: dict[str, Any],
    masks: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    q_dt_list = []
    q_T_minus_dt_list = []
    dt_values = []
    frame_metadata = []
    for batch_idx, offset in enumerate(offsets):
        q_dt_np, q_T_minus_dt_np, dt_value, sample_meta = _extract_near_endpoint_single(
            config, raw, offset, batch_idx, pde_params
        )
        expected = int(coef_t.shape[1])
        q_dt = _ensure_bchw(q_dt_np, config.pde, "near_q_dt", config.device, config.dtype, expected)
        q_T_minus_dt = _ensure_bchw(
            q_T_minus_dt_np,
            config.pde,
            "near_q_T_minus_dt",
            config.device,
            config.dtype,
            expected,
        )
        q_dt_list.append(q_dt)
        q_T_minus_dt_list.append(q_T_minus_dt)
        dt_values.append(float(dt_value))
        frame_metadata.append(sample_meta)

    q_dt_t = torch.cat(q_dt_list, dim=0).to(coef_t.device, dtype=coef_t.dtype)
    q_T_minus_dt_t = torch.cat(q_T_minus_dt_list, dim=0).to(coef_t.device, dtype=coef_t.dtype)
    if q_dt_t.shape != coef_t.shape or q_T_minus_dt_t.shape != sol_t.shape:
        raise ValueError(
            "near_endpoint_temporal observations must match endpoint shapes; "
            f"got q_dt={tuple(q_dt_t.shape)}, q_T_minus_dt={tuple(q_T_minus_dt_t.shape)}, "
            f"coef={tuple(coef_t.shape)}, sol={tuple(sol_t.shape)}"
        )
    mask_0 = torch.as_tensor(masks.coef, dtype=coef_t.dtype, device=coef_t.device).detach()
    mask_T = torch.as_tensor(masks.sol, dtype=sol_t.dtype, device=sol_t.device).detach()
    if mask_0.shape != q_dt_t.shape or mask_T.shape != q_T_minus_dt_t.shape:
        raise ValueError(
            "near_endpoint_temporal reuses endpoint masks, which must match the temporal state shapes; "
            f"mask_0={tuple(mask_0.shape)}, q_dt={tuple(q_dt_t.shape)}, "
            f"mask_T={tuple(mask_T.shape)}, q_T_minus_dt={tuple(q_T_minus_dt_t.shape)}"
        )
    # Retain only the explicitly observed values. The full true near-endpoint
    # frames are temporary loader inputs and must not enter guidance artifacts
    # or the PDE residual outside the selected sparse sensor locations.
    q_dt_obs = q_dt_t * mask_0
    q_T_minus_dt_obs = q_T_minus_dt_t * mask_T
    dt_t = torch.as_tensor(dt_values, dtype=coef_t.dtype, device=coef_t.device)
    metadata = {
        "source": config.loadby,
        "frame_metadata": frame_metadata,
        "dt": dt_values,
        "sensor_mode": config.sensor_mode,
        "num_obs": int(config.num_obs),
        "num_sensor_columns": config.num_sensor_columns,
        "mask_alignment": {"q_dt": "coef/q0", "q_T_minus_dt": "sol/qT"},
        "uses_endpoint_sensor_budget": True,
        "extra_observation_budget": False,
        "ground_truth_field_exception": "near_endpoint_temporal_sparse_observations",
        "observed_values_only": True,
        "unobserved_values_zeroed": True,
        "full_near_endpoint_frames_retained": False,
        "observed_count_q_dt_per_sample": mask_0.reshape(mask_0.shape[0], -1).sum(dim=1).detach().cpu().tolist(),
        "observed_count_q_T_minus_dt_per_sample": mask_T.reshape(mask_T.shape[0], -1).sum(dim=1).detach().cpu().tolist(),
    }
    return (
        {
            "q_dt": q_dt_obs,
            "q_T_minus_dt": q_T_minus_dt_obs,
            "dt": dt_t,
            "mask_0": mask_0,
            "mask_T": mask_T,
            "metadata": metadata,
        },
        metadata,
    )


def _extract_near_endpoint_single(
    config: AblationConfig,
    raw: dict[str, Any],
    offset: int,
    batch_idx: int,
    pde_params: dict[str, Any],
) -> tuple[Any, Any, float, dict[str, Any]]:
    if config.loadby == "swe":
        file = raw["__h5__"]
        key = _sample_group_key(file, offset)
        group = file[key]["data"]
        n_time = int(group["h"].shape[0])
        if n_time < 3:
            raise ValueError("near_endpoint_temporal mode requires at least three shallow-water time frames")
        start_idx, near_start_idx, near_end_idx, final_idx = 0, 1, n_time - 2, n_time - 1
        dt_value, dt_source = _near_endpoint_dt(config.pde, pde_params, batch_idx, final_idx - start_idx)
        return (
            _swe_frame(group, near_start_idx),
            _swe_frame(group, near_end_idx),
            dt_value,
            {
                "sample_offset": int(offset),
                "dataset_key": str(key),
                "start_frame": start_idx,
                "q_dt_frame": near_start_idx,
                "q_T_minus_dt_frame": near_end_idx,
                "final_frame": final_idx,
                "dt_source": dt_source,
            },
        )
    if config.loadby == "rd":
        file = raw["__h5__"]
        key = _sample_group_key(file, offset)
        arr = file[key]["data"][:]
        n_time = int(arr.shape[0])
        start_idx, near_start_idx, near_end_idx, final_idx = 0, 1, n_time - 2, n_time - 1
        if n_time < 3:
            raise ValueError("near_endpoint_temporal mode requires at least three reaction-diffusion time frames")
        dt_value, dt_source = _near_endpoint_dt(config.pde, pde_params, batch_idx, final_idx - start_idx)
        return (
            arr[near_start_idx, :, :, :],
            arr[near_end_idx, :, :, :],
            dt_value,
            {
                "sample_offset": int(offset),
                "dataset_key": str(key),
                "start_frame": start_idx,
                "q_dt_frame": near_start_idx,
                "q_T_minus_dt_frame": near_end_idx,
                "final_frame": final_idx,
                "dt_source": dt_source,
            },
        )
    if config.loadby == "pair_h5":
        file = raw["__h5__"]
        explicit = _pair_h5_explicit_near_endpoint(file, config, offset)
        if explicit is not None:
            q_dt, q_T_minus_dt, dt_value, source = explicit
            return (
                q_dt,
                q_T_minus_dt,
                dt_value,
                {
                    "sample_offset": int(offset),
                    "source": source,
                    "dt_source": "near_dt_dataset_or_attr",
                },
            )
        dataset_name = _pair_h5_trajectory_dataset_name(file)
        if dataset_name is None:
            raise ValueError(
                "near_endpoint_temporal mode requires extra near-endpoint sparse temporal observations and masks; "
                "pair_h5 input/output endpoint data does not contain a full_trajectory, trajectory, states, "
                "or solution_trajectory dataset"
            )
        expected_channels = _channel_counts(config.pde, config.img_channels)[0]
        trajectory = _pair_h5_trajectory_sample(file[dataset_name], offset)
        stored_channels = expected_channels
        try:
            n_time = _trajectory_time_length(trajectory, stored_channels)
        except ValueError:
            # Legacy Wave datasets stored only displacement u(t) in the saved
            # trajectory even though endpoint states use the first-order
            # representation [u, v].  Preserve support for the current full
            # two-channel schema and reconstruct v only for that legacy case.
            if config.pde != "wave":
                raise
            stored_channels = 1
            n_time = _trajectory_time_length(trajectory, stored_channels)
        if n_time < 3:
            raise ValueError(f"{dataset_name} must contain at least three time frames for near_endpoint_temporal")
        start_idx, near_start_idx, near_end_idx, final_idx = 0, 1, n_time - 2, n_time - 1
        time_values, dt_value, dt_source = _pair_h5_near_endpoint_times(
            file,
            config.pde,
            pde_params,
            batch_idx,
            offset,
            n_time,
        )
        if config.pde == "wave" and stored_channels == 1:
            q_dt, q_T_minus_dt = _wave_near_endpoint_states_from_displacement(
                trajectory,
                time_values,
            )
            state_source = "saved_displacement_trajectory_with_reconstructed_velocity"
            velocity_metadata = {
                "wave_velocity_reconstructed": True,
                "wave_velocity_reconstruction": "three_point_lagrange_derivative",
                "stored_trajectory_channels": 1,
                "expected_state_channels": expected_channels,
            }
        else:
            q_dt = _trajectory_frame_to_chw(trajectory, near_start_idx, expected_channels)
            q_T_minus_dt = _trajectory_frame_to_chw(trajectory, near_end_idx, expected_channels)
            state_source = "saved_full_state_trajectory"
            velocity_metadata = {
                "wave_velocity_reconstructed": False,
                "stored_trajectory_channels": stored_channels,
                "expected_state_channels": expected_channels,
            }
        return (
            q_dt,
            q_T_minus_dt,
            dt_value,
            {
                "sample_offset": int(offset),
                "trajectory_dataset": dataset_name,
                "near_endpoint_state_source": state_source,
                "start_frame": start_idx,
                "q_dt_frame": near_start_idx,
                "q_T_minus_dt_frame": near_end_idx,
                "final_frame": final_idx,
                "dt_source": dt_source,
                **velocity_metadata,
            },
        )
    if config.loadby == "h5py":
        file = raw["__h5__"]
        if config.pde != "nsnonbounded":
            raise ValueError(
                "near_endpoint_temporal loadby='h5py' is currently defined only for nsnonbounded"
            )
        # Formal NS files store w at t=dt,...,T and store w0 separately.
        # Read one sample only: loading the whole HDF5 trajectory here can use
        # hundreds of MB and also obscures the fact that w does not include t=0.
        trajectory = _ns_h5py_trajectory_sample(
            file[config.solution_name], offset, config.solution_name
        )
        n_time = int(trajectory.shape[0])
        if n_time < 3:
            raise ValueError("near_endpoint_temporal mode requires at least three time frames in h5py trajectory")
        # w[0] is q(dt), w[-2] is q(T-dt), and there are n_time
        # intervals from the separately stored q0 to w[-1]=q(T).
        q_dt_np = trajectory[0]
        q_T_minus_dt_np = trajectory[-2]
        dt_value, dt_source = _near_endpoint_dt(config.pde, pde_params, batch_idx, n_time)
        return (
            q_dt_np,
            q_T_minus_dt_np,
            dt_value,
            {
                "sample_offset": int(offset),
                "q_dt_frame": 0,
                "q_T_minus_dt_frame": n_time - 2,
                "final_frame": n_time - 1,
                "dt_source": dt_source,
                "initial_frame_stored_separately": True,
                "auto_constructed": True,
            },
        )
    raise ValueError(
        "near_endpoint_temporal mode requires extra near-endpoint sparse temporal observations and masks; "
        f"loadby={config.loadby!r} is not supported"
    )


def _sample_group_key(file: Any, offset: int) -> str:
    for candidate in (f"{offset:06d}", f"{offset:05d}", f"{offset:04d}", str(offset)):
        if candidate in file and hasattr(file[candidate], "keys") and "data" in file[candidate]:
            return candidate
    keys = sorted(
        key
        for key in file.keys()
        if hasattr(file[key], "keys") and "data" in file[key]
    )
    if offset < 0 or offset >= len(keys):
        raise IndexError(f"Sample offset {offset} is out of range for {len(keys)} HDF5 sample groups")
    return keys[offset]


def _attach_boundary_metadata_params(config: AblationConfig, file: Any, params: dict[str, Any], sources: dict[str, str]) -> None:
    if "boundary_condition_kind" in params:
        return
    raw = None
    source = None
    if file is not None:
        for name in ("boundary_condition_kind", "boundary_condition", "bc"):
            if name in file.attrs:
                raw = file.attrs[name]
                source = f"root_attr:{name}"
                break
    if raw is None:
        defaults = {
            "heat": "periodic" if getattr(config, "boundary_condition_mode", "auto") == "auto" else None,
            "wave": "periodic",
            "advection_diffusion": "periodic",
            "reaction_diffusion": "neumann",
            "shallow_water": "open",
            "steady_heat_conduction": "mixed",
        }
        raw = defaults.get(config.pde)
        source = "loader_confirmed_default" if raw is not None else None
    if raw is None:
        return
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    lowered = text.lower()
    if "periodic" in lowered:
        kind = "periodic"
    elif "dirichlet" in lowered and "neumann" in lowered:
        kind = "mixed"
    elif "neumann" in lowered or "extrap" in lowered or "open" in lowered:
        kind = "open" if "extrap" in lowered or "open" in lowered else "neumann"
    elif "dirichlet" in lowered:
        kind = "dirichlet"
    elif lowered in {"periodic", "neumann", "dirichlet", "mixed", "open", "wall"}:
        kind = lowered
    else:
        kind = text
    params["boundary_condition_kind"] = kind
    sources["boundary_condition_kind"] = source or "metadata"


def _rd_params_for_offsets(file: Any, offsets: list[int], device: Any) -> tuple[dict[str, Any], dict[str, str]]:
    import torch

    keys = [_sample_group_key(file, offset) for offset in offsets]
    specs = {
        "T": ("T",),
        "D_u": ("D_u", "Du"),
        "D_v": ("D_v", "Dv"),
        "k": ("k",),
        "n_save_steps": ("n_save_steps",),
        "tdim": ("tdim",),
        "x_left": ("x_left",),
        "x_right": ("x_right",),
        "y_bottom": ("y_bottom",),
        "y_top": ("y_top",),
        "dx": ("dx",),
        "dy": ("dy",),
    }
    params = {}
    sources = {}
    for canonical, aliases in specs.items():
        values = []
        source = None
        for key in keys:
            group = file[key]
            value = None
            for alias in aliases:
                if alias in group.attrs:
                    value = group.attrs[alias]
                    source = f"group_attr:{alias}"
                    break
            if value is None:
                for alias in aliases:
                    if alias in file.attrs:
                        value = file.attrs[alias]
                        source = f"root_attr:{alias}"
                        break
            if value is None:
                values = []
                break
            values.append(float(value))
        if values:
            params[canonical] = torch.as_tensor(values, dtype=torch.float32, device=device)
            sources[canonical] = source or "attr"
    for range_name, left_name, right_name in (
        ("x_range", "x_left", "x_right"),
        ("y_range", "y_bottom", "y_top"),
    ):
        if left_name in params and right_name in params:
            continue
        left_values = []
        right_values = []
        source = None
        for key in keys:
            group = file[key]
            value = None
            if range_name in group.attrs:
                value = group.attrs[range_name]
                source = f"group_attr:{range_name}"
            elif range_name in file.attrs:
                value = file.attrs[range_name]
                source = f"root_attr:{range_name}"
            if value is None:
                left_values = []
                right_values = []
                break
            arr = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
            if arr.numel() < 2:
                left_values = []
                right_values = []
                break
            left_values.append(arr[0])
            right_values.append(arr[1])
        if left_values and left_name not in params:
            params[left_name] = torch.stack(left_values)
            sources[left_name] = source or "attr"
        if right_values and right_name not in params:
            params[right_name] = torch.stack(right_values)
            sources[right_name] = source or "attr"
    return params, sources


def _swe_params_for_offsets(file: Any, offsets: list[int], device: Any) -> tuple[dict[str, Any], dict[str, str]]:
    import torch

    keys = [_sample_group_key(file, offset) for offset in offsets]
    params: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for canonical, aliases in {"g": ("grav", "g"), "T": ("T", "total_time")}.items():
        values = []
        source = None
        for key in keys:
            group = file[key]
            value = None
            for alias in aliases:
                if alias in group.attrs:
                    value = group.attrs[alias]
                    source = f"group_attr:{alias}"
                    break
                if alias in file.attrs:
                    value = file.attrs[alias]
                    source = f"root_attr:{alias}"
                    break
            if value is None:
                values = []
                break
            values.append(float(value))
        if values:
            params[canonical] = torch.as_tensor(values, dtype=torch.float32, device=device)
            sources[canonical] = source or "attr"
    for range_name, left_name, right_name in (
        ("x_range", "x_left", "x_right"),
        ("y_range", "y_bottom", "y_top"),
    ):
        ranges = []
        source = None
        for key in keys:
            group = file[key]
            value = group.attrs.get(range_name, file.attrs.get(range_name))
            if value is None:
                ranges = []
                break
            ranges.append(torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1))
            source = f"group_attr:{range_name}" if range_name in group.attrs else f"root_attr:{range_name}"
        if ranges and all(value.numel() >= 2 for value in ranges):
            params[left_name] = torch.stack([value[0] for value in ranges])
            params[right_name] = torch.stack([value[1] for value in ranges])
            sources[left_name] = source or "attr"
            sources[right_name] = source or "attr"
    if "x_left" in params and "x_right" in params:
        params["dx"] = (params["x_right"] - params["x_left"]) / float(file[keys[0]]["data"]["h"].shape[1])
        sources["dx"] = "x_range/num_cells"
    if "y_bottom" in params and "y_top" in params:
        params["dy"] = (params["y_top"] - params["y_bottom"]) / float(file[keys[0]]["data"]["h"].shape[2])
        sources["dy"] = "y_range/num_cells"
    return params, sources


def _h5py_params_for_offsets(file: Any, pde: str, offsets: list[int], device: Any) -> tuple[dict[str, Any], dict[str, str]]:
    import torch

    spec = get_pde_spec(pde)
    params: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name in spec.scalar_param_names:
        storage_name = _h5py_param_storage_name(file, spec, name)
        if storage_name is None:
            if name in spec.optional_scalar_param_names:
                continue
            raise KeyError(f"Missing sample-level scalar PDE parameter {name!r} for loadby='h5py'")
        values = [_read_h5py_scalar_value(file, storage_name, offset) for offset in offsets]
        params[name] = torch.as_tensor(values, dtype=torch.float32, device=device)
        sources[name] = f"dataset:{storage_name}" if storage_name in file else f"attrs:{storage_name}"
    return params, sources


def _h5py_param_storage_name(file: Any, spec: Any, name: str) -> str | None:
    for candidate in spec.param_aliases.get(name, (name,)):
        if candidate in file or candidate in file.attrs:
            return candidate
    return None


def _read_h5py_scalar_value(file: Any, name: str, offset: int) -> float:
    import numpy as np

    if name in file:
        dataset = file[name]
        if dataset.shape == ():
            value = dataset[()]
        elif dataset.shape[0] == 1:
            value = dataset[0]
        elif offset < dataset.shape[0]:
            value = dataset[offset]
        else:
            raise IndexError(
                f"Sample offset {offset} is out of range for scalar PDE parameter dataset {name!r} "
                f"with shape {tuple(dataset.shape)}"
            )
    elif name in file.attrs:
        values = np.asarray(file.attrs[name])
        if values.ndim == 0:
            value = values
        elif values.shape[0] == 1:
            value = values.reshape(-1)[0]
        elif offset < values.shape[0]:
            value = values[offset]
        else:
            raise IndexError(
                f"Sample offset {offset} is out of range for scalar PDE parameter attr {name!r} "
                f"with shape {tuple(values.shape)}"
            )
    else:
        raise KeyError(f"Missing sample-level scalar PDE parameter {name!r}")
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Sample-level scalar PDE parameter {name!r} must be scalar per sample, got shape {tuple(arr.shape)}")
    return float(arr[0])


def _swe_frame(group: Any, frame_idx: int) -> Any:
    import numpy as np

    return np.stack(
        [
            group["h"][frame_idx, :, :, 0],
            group["hu"][frame_idx, :, :, 0],
            group["hv"][frame_idx, :, :, 0],
        ],
        axis=0,
    )


def _pair_h5_explicit_near_endpoint(file: Any, config: AblationConfig, offset: int) -> tuple[Any, Any, float, str] | None:
    q_dt_name = _first_existing_dataset(file, ("near_q_dt", "q_dt", "near_endpoint_q_dt"))
    q_T_minus_dt_name = _first_existing_dataset(
        file,
        ("near_q_T_minus_dt", "q_T_minus_dt", "near_endpoint_q_T_minus_dt", "q_t_minus_dt"),
    )
    if q_dt_name is None and q_T_minus_dt_name is None:
        return None
    if q_dt_name is None or q_T_minus_dt_name is None:
        raise ValueError("pair_h5 near endpoint data must include both q_dt and q_T_minus_dt datasets")
    dt_name = _first_existing_dataset(file, ("near_dt", "dt"))
    if dt_name is not None:
        dt_value = _read_pair_h5_value(file, dt_name, offset)
    elif "near_dt" in file.attrs:
        dt_value = _read_pair_h5_value(file, "near_dt", offset)
    elif "dt" in file.attrs:
        dt_value = _read_pair_h5_value(file, "dt", offset)
    else:
        raise ValueError("pair_h5 explicit near endpoint data requires near_dt or dt")
    expected_channels = _channel_counts(config.pde, config.img_channels)[0]
    q_dt = _ensure_chw_array(_pair_h5_endpoint_sample(file[q_dt_name], offset), expected_channels)
    q_T_minus_dt = _ensure_chw_array(_pair_h5_endpoint_sample(file[q_T_minus_dt_name], offset), expected_channels)
    return q_dt, q_T_minus_dt, float(dt_value), f"{q_dt_name},{q_T_minus_dt_name}"


def _pair_h5_trajectory_dataset_name(file: Any) -> str | None:
    return _first_existing_dataset(
        file,
        (
            "full_trajectory",
            "trajectory",
            "states",
            "solution_trajectory",
            "state_trajectory",
            "u_trajectory",
            "w_trajectory",
        ),
    )


def _full_trajectory_for_offsets(
    config: AblationConfig,
    raw: dict[str, Any],
    offsets: list[int],
    coef_t: Any,
    pde_params: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch

    trajectories = []
    frame_metadata = []
    expected_channels = int(coef_t.shape[1])
    for batch_idx, offset in enumerate(offsets):
        trajectory_np, sample_meta = _extract_full_trajectory_single(config, raw, offset, batch_idx, pde_params)
        try:
            trajectory = _trajectory_to_btchw(trajectory_np, expected_channels, config.device, config.dtype)
        except ValueError:
            if config.pde != "wave":
                raise
            # Legacy Wave files store displacement alone, while current files
            # store both displacement and velocity. Preserve both schemas.
            trajectory = _trajectory_to_btchw(trajectory_np, 1, config.device, config.dtype)
        trajectories.append(trajectory)
        frame_metadata.append(sample_meta)
    trajectory_t = torch.cat(trajectories, dim=0).to(coef_t.device, dtype=coef_t.dtype)
    metadata = {
        "source": config.loadby,
        "frame_metadata": frame_metadata,
        "shape": list(trajectory_t.shape),
        "uses_generated_trajectory": False,
        "uses_observed_ground_truth_trajectory": True,
        "guidance_compatible": False,
        "endpoint_only": False,
    }
    return trajectory_t, metadata


def _full_trajectory_time_values(config: AblationConfig, raw: dict[str, Any], offsets: list[int]) -> Any | None:
    import numpy as np

    file = raw.get("__h5__")
    if file is None:
        return None
    if config.loadby in {"pair_h5", "h5py"} and "t" in file:
        values = np.asarray(file["t"][:], dtype=np.float32)
        if config.pde == "nsnonbounded" and (values.size == 0 or abs(float(values[0])) > 1e-7):
            values = np.concatenate([np.zeros(1, dtype=np.float32), values])
        return values
    if config.loadby in {"rd", "swe"}:
        key = _sample_group_key(file, offsets[0])
        group = file[key]
        if "grid" in group and "t" in group["grid"]:
            return np.asarray(group["grid"]["t"][:], dtype=np.float32)
    return None


def _extract_full_trajectory_single(
    config: AblationConfig,
    raw: dict[str, Any],
    offset: int,
    batch_idx: int,
    pde_params: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if config.loadby == "pair_h5":
        file = raw["__h5__"]
        dataset_name = _pair_h5_trajectory_dataset_name(file)
        if dataset_name is None:
            raise ValueError(
                "full_trajectory_fd mode requires explicit full trajectory observations; "
                "pair_h5 input/output endpoint data does not contain a full_trajectory, trajectory, states, "
                "solution_trajectory, state_trajectory, u_trajectory, or w_trajectory dataset"
            )
        trajectory = _pair_h5_trajectory_sample(file[dataset_name], offset)
        return trajectory, {"sample_offset": int(offset), "trajectory_dataset": dataset_name}
    if config.loadby == "rd":
        file = raw["__h5__"]
        key = _sample_group_key(file, offset)
        return file[key]["data"][:], {"sample_offset": int(offset), "dataset_key": str(key)}
    if config.loadby == "swe":
        file = raw["__h5__"]
        key = _sample_group_key(file, offset)
        group = file[key]["data"]
        import numpy as np

        trajectory = np.stack(
            [
                group["h"][:, :, :, 0],
                group["hu"][:, :, :, 0],
                group["hv"][:, :, :, 0],
            ],
            axis=1,
        )
        return trajectory, {"sample_offset": int(offset), "dataset_key": str(key)}
    if config.loadby == "h5py" and config.pde == "nsnonbounded":
        file = raw["__h5__"]
        if config.solution_name not in file:
            raise ValueError(
                "full_trajectory_fd mode requires explicit full trajectory observations; "
                f"nsnonbounded h5py data is missing solution dataset {config.solution_name!r}"
            )
        trajectory = _ns_h5py_trajectory_sample(file[config.solution_name], offset, config.solution_name)
        initial = file[config.coef_name][offset]
        import numpy as np

        initial = np.asarray(initial, dtype=np.float32)[None, None, :, :]
        trajectory = np.concatenate([initial, trajectory], axis=0)
        return trajectory, {"sample_offset": int(offset), "trajectory_dataset": config.solution_name}
    raise ValueError(
        "full_trajectory_fd mode requires explicit full trajectory observations; "
        f"loadby={config.loadby!r} is not supported"
    )


def _ns_h5py_trajectory_sample(dataset: Any, offset: int, dataset_name: str) -> Any:
    import numpy as np

    if dataset.ndim < 4:
        raise ValueError(
            "full_trajectory_fd mode requires nsnonbounded h5py solution data with a sample and time axis; "
            f"{dataset_name!r} has shape {tuple(dataset.shape)}"
        )
    if offset < 0 or offset >= dataset.shape[0]:
        raise IndexError(f"Sample offset {offset} is out of range for {dataset_name!r} with shape {tuple(dataset.shape)}")
    sample = np.asarray(dataset[offset], dtype=np.float32)
    if sample.ndim == 3:
        # Formal NS files usually store w as [N,H,W,T], so a single sample is HWT.
        if sample.shape[0] == sample.shape[1] and sample.shape[-1] != sample.shape[-2]:
            return np.moveaxis(sample, -1, 0)[:, None, :, :]
        if sample.shape[-2] == sample.shape[-1]:
            return sample[:, None, :, :]
        raise ValueError(
            "Cannot infer nsnonbounded h5py trajectory layout; expected sample shape [H,W,T] or [T,H,W], "
            f"got {tuple(sample.shape)}"
        )
    if sample.ndim == 4:
        return sample
    raise ValueError(
        "Cannot infer nsnonbounded h5py trajectory layout; expected sample shape [H,W,T], [T,H,W], "
        f"[T,C,H,W], [C,T,H,W], or [T,H,W,C], got {tuple(sample.shape)}"
    )


def _trajectory_to_btchw(value: Any, expected_channels: int, device: str, dtype_name: str) -> Any:
    import torch

    dtype = _torch_dtype(dtype_name)
    tensor = torch.as_tensor(value, dtype=dtype)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"trajectory must be [B,T,C,H,W] or [B,C,T,H,W], got shape {tuple(tensor.shape)}")
    if tensor.shape[2] == expected_channels:
        pass
    elif tensor.shape[1] == expected_channels:
        tensor = tensor.permute(0, 2, 1, 3, 4)
    elif tensor.shape[-1] == expected_channels:
        tensor = tensor.permute(0, 1, 4, 2, 3)
    else:
        raise ValueError(
            "Cannot infer trajectory channel axis; expected [B,T,C,H,W], [B,C,T,H,W], or [B,T,H,W,C] "
            f"with {expected_channels} channels, got {tuple(tensor.shape)}"
        )
    target = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    return tensor.to(target)


def _first_existing_dataset(file: Any, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in file:
            return name
    return None


def _pair_h5_endpoint_sample(dataset: Any, offset: int) -> Any:
    import numpy as np

    shape = tuple(dataset.shape) if hasattr(dataset, "shape") else np.asarray(dataset).shape
    if len(shape) == 4 and shape[0] > offset:
        return np.asarray(dataset[offset], dtype=np.float32)
    return np.asarray(dataset, dtype=np.float32)


def _pair_h5_trajectory_sample(dataset: Any, offset: int) -> Any:
    import numpy as np

    shape = tuple(dataset.shape) if hasattr(dataset, "shape") else np.asarray(dataset).shape
    if len(shape) == 5 and shape[0] > offset:
        return np.asarray(dataset[offset], dtype=np.float32)
    return np.asarray(dataset, dtype=np.float32)


def _trajectory_time_length(trajectory: Any, expected_channels: int) -> int:
    if trajectory.ndim != 4:
        raise ValueError(f"trajectory dataset sample must be 4D, got shape {tuple(trajectory.shape)}")
    if trajectory.shape[1] == expected_channels:
        return int(trajectory.shape[0])
    if trajectory.shape[-1] == expected_channels:
        return int(trajectory.shape[0])
    if trajectory.shape[0] == expected_channels:
        return int(trajectory.shape[1])
    raise ValueError(
        "Cannot infer trajectory layout; expected one of TCHW, THWC, or CTHW with "
        f"{expected_channels} channels, got {tuple(trajectory.shape)}"
    )


def _trajectory_frame_to_chw(trajectory: Any, frame_idx: int, expected_channels: int) -> Any:
    if trajectory.shape[1] == expected_channels:
        return trajectory[frame_idx, :, :, :]
    if trajectory.shape[-1] == expected_channels:
        return _ensure_chw_array(trajectory[frame_idx, :, :, :], expected_channels)
    if trajectory.shape[0] == expected_channels:
        return trajectory[:, frame_idx, :, :]
    raise ValueError(f"Cannot infer trajectory layout for shape {tuple(trajectory.shape)}")


def _pair_h5_near_endpoint_times(
    file: Any,
    pde: str,
    pde_params: dict[str, Any],
    batch_idx: int,
    offset: int,
    n_time: int,
) -> tuple[Any, float, str]:
    import numpy as np

    if "t" in file:
        raw_values = np.asarray(file["t"][:], dtype=np.float64)
        if raw_values.ndim == 2:
            if raw_values.shape[0] == 1:
                raw_values = raw_values[0]
            elif offset < raw_values.shape[0]:
                raw_values = raw_values[offset]
        values = np.asarray(raw_values).squeeze()
        if values.ndim != 1 or values.size != n_time:
            raise ValueError(
                "pair_h5 trajectory time dataset must be one-dimensional and match the trajectory; "
                f"t={tuple(values.shape)}, trajectory_time_points={n_time}"
            )
        if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
            raise ValueError("pair_h5 trajectory time values must be finite and strictly increasing")
        start_dt = float(values[1] - values[0])
        end_dt = float(values[-1] - values[-2])
        if not np.isclose(start_dt, end_dt, rtol=1e-5, atol=1e-8):
            raise ValueError(
                "near_endpoint_temporal requires equal first and last saved time intervals because its "
                f"schema carries one dt; got start_dt={start_dt:g}, end_dt={end_dt:g}"
            )
        return values, 0.5 * (start_dt + end_dt), "dataset:t_endpoint_intervals"

    dt_value, dt_source = _near_endpoint_dt(pde, pde_params, batch_idx, n_time - 1)
    values = np.arange(n_time, dtype=np.float64) * float(dt_value)
    return values, dt_value, dt_source


def _wave_near_endpoint_states_from_displacement(
    trajectory: Any,
    time_values: Any,
) -> tuple[Any, Any]:
    """Build [u, v] near-endpoint states from a legacy u-only Wave trajectory."""
    import numpy as np

    n_time = _trajectory_time_length(trajectory, 1)
    if n_time < 3:
        raise ValueError("Wave velocity reconstruction requires at least three trajectory time points")
    times = np.asarray(time_values, dtype=np.float64)
    if times.ndim != 1 or times.size != n_time:
        raise ValueError(
            f"Wave trajectory time values must contain {n_time} entries, got shape {tuple(times.shape)}"
        )

    def state_at(frame_idx: int) -> Any:
        indices = (frame_idx - 1, frame_idx, frame_idx + 1)
        selected_times = times[list(indices)]
        frames = [
            _trajectory_frame_to_chw(trajectory, index, 1).astype(np.float64, copy=False)
            for index in indices
        ]
        velocity = sum(
            _lagrange_derivative_weight(selected_times, local_idx, selected_times[1]) * frame
            for local_idx, frame in enumerate(frames)
        )
        displacement = frames[1]
        return np.concatenate([displacement, velocity], axis=0).astype(np.float32, copy=False)

    return state_at(1), state_at(n_time - 2)


def _lagrange_derivative_weight(nodes: Any, node_idx: int, x: float) -> float:
    """Derivative at x of one quadratic Lagrange basis polynomial."""
    weight = 0.0
    for differentiated_idx in range(3):
        if differentiated_idx == node_idx:
            continue
        term = 1.0
        for other_idx in range(3):
            if other_idx in {node_idx, differentiated_idx}:
                continue
            denominator = float(nodes[node_idx] - nodes[other_idx])
            if denominator == 0.0:
                raise ValueError("Wave trajectory time values must be distinct")
            term *= float(x - nodes[other_idx]) / denominator
        denominator = float(nodes[node_idx] - nodes[differentiated_idx])
        if denominator == 0.0:
            raise ValueError("Wave trajectory time values must be distinct")
        weight += term / denominator
    return weight


def _ensure_chw_array(value: Any, expected_channels: int) -> Any:
    import numpy as np

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2 and expected_channels == 1:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected near endpoint frame to be CHW or HWC, got shape {tuple(arr.shape)}")
    if arr.shape[0] == expected_channels:
        return arr
    if arr.shape[-1] == expected_channels:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"Cannot infer channel axis for near endpoint frame shape {tuple(arr.shape)}")


def _near_endpoint_dt(
    pde: str,
    pde_params: dict[str, Any],
    batch_idx: int,
    interval_count: int,
) -> tuple[float, str]:
    if interval_count <= 0:
        raise ValueError("near_endpoint temporal interval count must be positive")
    for name in ("T", "total_time"):
        if name in pde_params:
            return _batch_scalar(pde_params[name], batch_idx) / float(interval_count), f"{name}/frame_interval_count"
    if "dt" in pde_params:
        return _batch_scalar(pde_params["dt"], batch_idx), "dt"
    default_total = 1.0
    return default_total / float(interval_count), "pde_default_total_time/frame_interval_count"


def _batch_scalar(value: Any, batch_idx: int) -> float:
    import numpy as np

    arr = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot read scalar from empty value")
    if arr.size == 1:
        return float(arr[0])
    if batch_idx >= arr.size:
        raise IndexError(f"Cannot read batch scalar index {batch_idx} from shape {arr.shape}")
    return float(arr[batch_idx])


def _ensure_bchw(value: Any, pde: str, side: str, device: str, dtype_name: str, expected_channels: int | None = None) -> Any:
    import torch

    dtype = _torch_dtype(dtype_name)
    tensor = torch.as_tensor(value, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        expected = expected_channels or _channel_counts(pde, 0)[0 if side == "coef" else 1]
        if tensor.shape[0] == expected:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == expected:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(
                f"Cannot infer channel dimension for {pde} {side} tensor with shape {tuple(tensor.shape)}"
            )
    elif tensor.ndim == 4:
        pass
    else:
        raise ValueError(f"Expected {pde} {side} tensor with 2-4 dims, got {tuple(tensor.shape)}")

    target = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    return tensor.to(target)


def _assert_spatial_compatible(coef: Any, sol: Any, pde: str) -> None:
    if coef.ndim != 4 or sol.ndim != 4:
        raise ValueError(f"{pde} tensors must be BCHW, got coef={coef.shape}, sol={sol.shape}")
    if coef.shape[0] != sol.shape[0]:
        raise ValueError(f"{pde} batch mismatch: coef={coef.shape}, sol={sol.shape}")
    if coef.shape[-2:] != sol.shape[-2:]:
        raise ValueError(f"{pde} spatial mismatch: coef={coef.shape}, sol={sol.shape}")


def _channel_counts(pde: str, img_channels: int) -> tuple[int, int]:
    spec = get_pde_spec(pde)
    return spec.coef_channels, spec.sol_channels


def _pair_h5_params_for_offsets(file: Any, pde: str, offsets: list[int], device: Any) -> tuple[dict[str, Any], dict[str, str]]:
    import torch

    spec = get_pde_spec(pde)
    param_names = spec.scalar_param_names
    optional = spec.optional_scalar_param_names
    params = {}
    sources = {}
    for name in param_names:
        storage_name = _pair_h5_storage_name(file, pde, name)
        if storage_name is None:
            if name in optional:
                continue
            raise KeyError(f"Missing pair_h5 scalar {name!r}")
        values = [_read_pair_h5_value(file, storage_name, offset) for offset in offsets]
        params[name] = torch.as_tensor(values, dtype=torch.float32, device=device)
        sources[name] = "dataset" if storage_name in file else f"attrs:{storage_name}"
    return params, sources


def _pair_h5_storage_name(file: Any, pde: str, name: str) -> str | None:
    spec = get_pde_spec(pde)
    for candidate in spec.param_aliases.get(name, (name,)):
        if candidate in file or candidate in file.attrs:
            return candidate
    return None


def _read_pair_h5_value(file: Any, name: str, offset: int) -> Any:
    import numpy as np

    if name in file:
        dataset = file[name]
        if dataset.shape == ():
            values = np.asarray(dataset[()], dtype=np.float32)
        elif dataset.shape[0] == 1:
            values = np.asarray(dataset[0], dtype=np.float32)
        elif offset < dataset.shape[0]:
            selected = np.asarray(dataset[offset], dtype=np.float32)
            return float(selected) if selected.ndim == 0 else selected
        else:
            raise IndexError(f"pair_h5 dataset {name!r} has shape {tuple(dataset.shape)}, cannot read offset {offset}")
        if values.ndim == 0:
            return float(values)
        if values.shape[0] == 1:
            return values[0]
        return values
    if name in file.attrs:
        values = np.asarray(file.attrs[name], dtype=np.float32)
        flat = values.reshape(-1)
        if flat.size == 1:
            return float(flat[0])
        if offset >= flat.size:
            raise IndexError(f"pair_h5 attr {name!r} has {flat.size} values, cannot read offset {offset}")
        return flat[offset]
    raise KeyError(f"Missing pair_h5 scalar {name!r}")


def _synthetic_pde_params(config: AblationConfig, device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    b = int(config.batch_size)
    ones = lambda value: torch.full((b,), float(value), dtype=dtype, device=device)
    if config.pde == "heat":
        return {"alpha": ones(1e-3), "T": ones(1.0)}
    if config.pde == "wave":
        return {"c": ones(1.0), "T": ones(1.0)}
    if config.pde == "advection_diffusion":
        return {"b_x": ones(0.0), "b_y": ones(0.0), "kappa": ones(1e-3), "T": ones(1.0)}
    if config.pde == "steady_heat_conduction":
        return {"u_D": ones(298.0)}
    return {}


def _torch_dtype(name: str) -> Any:
    import torch

    if name == "float64":
        return torch.float64
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _channel_names(names: tuple[str, ...], n_channels: int, prefix: str) -> list[str]:
    names = list(names)
    if len(names) == n_channels:
        return names
    return [f"{prefix}_{i}" for i in range(n_channels)]

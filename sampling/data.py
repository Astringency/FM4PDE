from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    elif config.loadby == "rd":
        pde_params, pde_param_sources = _rd_params_for_offsets(raw["__h5__"], offsets, pair.device)
    near_metadata: dict[str, Any] | None = None
    if normalize_residual_mode(config.residual_mode) == "near_endpoint_temporal":
        near_params, near_metadata = _near_endpoint_temporal_for_offsets(config, raw, offsets, coef_t, sol_t, pde_params)
        pde_params["near_endpoint_temporal"] = near_params
        pde_param_sources["near_endpoint_temporal"] = "extra_sparse_temporal_observations"
    trajectory_metadata: dict[str, Any] | None = None
    if normalize_residual_mode(config.residual_mode) == "full_trajectory_fd":
        trajectory, trajectory_metadata = _full_trajectory_for_offsets(config, raw, offsets, coef_t, pde_params)
        pde_params["trajectory"] = trajectory
        pde_param_sources["trajectory"] = "full_trajectory_observations"
    spec = get_pde_spec(config.pde)
    channel_names_coef = _channel_names(spec.coef_channel_names, int(coef_t.shape[1]), "coef")
    channel_names_sol = _channel_names(spec.sol_channel_names, int(sol_t.shape[1]), "sol")
    metadata = {
        "data_path": config.data_path,
        "offset": config.offset,
        "loadby": config.loadby,
        "synthetic": False,
        "batch_size": config.batch_size,
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
    }
    if near_metadata is not None:
        metadata["near_endpoint_temporal"] = near_metadata
        metadata["pde_params_keys"] = sorted(pde_params)
        metadata["pde_params_sources"] = pde_param_sources
    if trajectory_metadata is not None:
        metadata["full_trajectory_fd"] = trajectory_metadata
        metadata["pde_params_keys"] = sorted(pde_params)
        metadata["pde_params_sources"] = pde_param_sources
    return PDEGroundTruth(
        pde=config.pde,
        coef=coef_t,
        sol=sol_t,
        pair=pair,
        pde_params=pde_params,
        channel_names_coef=channel_names_coef,
        channel_names_sol=channel_names_sol,
        metadata=metadata,
    )


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
        key = list(file.keys())[offset]
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
        coef_raw = file[config.coef_name][:]
        sol_raw = file[config.solution_name][:]
    else:
        coef_raw = raw[config.coef_name]
        sol_raw = raw[config.solution_name]

    if pde == "darcy":
        return coef_raw[:, :, offset], sol_raw[:, :, offset]
    if pde == "nsnonbounded":
        return coef_raw[offset, :, :], sol_raw[offset, :, :, -1]
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if int(config.num_near_endpoint_obs) <= 0:
        raise ValueError(
            "near_endpoint_temporal mode requires extra near-endpoint sparse temporal observations and masks; "
            "num_near_endpoint_obs must be > 0"
        )
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
    mask_0, mask_T, mask_meta = _make_near_endpoint_masks(
        config,
        batch_size=int(coef_t.shape[0]),
        height=int(coef_t.shape[-2]),
        width=int(coef_t.shape[-1]),
        device=coef_t.device,
        dtype=coef_t.dtype,
    )
    dt_t = torch.as_tensor(dt_values, dtype=coef_t.dtype, device=coef_t.device)
    metadata = {
        "source": config.loadby,
        "frame_metadata": frame_metadata,
        "dt": dt_values,
        "num_near_endpoint_obs": int(config.num_near_endpoint_obs),
        "near_endpoint_sensor_mode": config.near_endpoint_sensor_mode,
        "near_endpoint_mask_seed": int(config.near_endpoint_mask_seed),
        "near_endpoint_shared_mask": bool(config.near_endpoint_shared_mask),
        "extra_observation_budget": True,
        **mask_meta,
    }
    return (
        {
            "q_dt": q_dt_t,
            "q_T_minus_dt": q_T_minus_dt_t,
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
        key = list(file.keys())[offset]
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
        n_time = _trajectory_time_length(trajectory, expected_channels)
        if n_time < 3:
            raise ValueError(f"{dataset_name} must contain at least three time frames for near_endpoint_temporal")
        start_idx, near_start_idx, near_end_idx, final_idx = 0, 1, n_time - 2, n_time - 1
        dt_value, dt_source = _near_endpoint_dt(config.pde, pde_params, batch_idx, final_idx - start_idx)
        return (
            _trajectory_frame_to_chw(trajectory, near_start_idx, expected_channels),
            _trajectory_frame_to_chw(trajectory, near_end_idx, expected_channels),
            dt_value,
            {
                "sample_offset": int(offset),
                "trajectory_dataset": dataset_name,
                "start_frame": start_idx,
                "q_dt_frame": near_start_idx,
                "q_T_minus_dt_frame": near_end_idx,
                "final_frame": final_idx,
                "dt_source": dt_source,
            },
        )
    raise ValueError(
        "near_endpoint_temporal mode requires extra near-endpoint sparse temporal observations and masks; "
        f"loadby={config.loadby!r} is not supported"
    )


def _sample_group_key(file: Any, offset: int) -> str:
    keys = sorted(
        key
        for key in file.keys()
        if hasattr(file[key], "keys") and "data" in file[key]
    )
    if offset < 0 or offset >= len(keys):
        raise IndexError(f"Sample offset {offset} is out of range for {len(keys)} HDF5 sample groups")
    return keys[offset]


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
        trajectory = _trajectory_to_btchw(trajectory_np, expected_channels, config.device, config.dtype)
        trajectories.append(trajectory)
        frame_metadata.append(sample_meta)
    trajectory_t = torch.cat(trajectories, dim=0).to(coef_t.device, dtype=coef_t.dtype)
    metadata = {
        "source": config.loadby,
        "frame_metadata": frame_metadata,
        "shape": list(trajectory_t.shape),
        "uses_generated_trajectory": True,
        "endpoint_only": False,
    }
    return trajectory_t, metadata


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
        key = list(file.keys())[offset]
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
    raise ValueError(
        "full_trajectory_fd mode requires explicit full trajectory observations; "
        f"loadby={config.loadby!r} is not supported"
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

    values = np.asarray(dataset)
    if values.ndim == 4 and values.shape[0] > offset:
        return np.asarray(dataset[offset], dtype=np.float32)
    return np.asarray(dataset, dtype=np.float32)


def _pair_h5_trajectory_sample(dataset: Any, offset: int) -> Any:
    import numpy as np

    values = np.asarray(dataset)
    if values.ndim == 5 and values.shape[0] > offset:
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


def _make_near_endpoint_masks(
    config: AblationConfig,
    *,
    batch_size: int,
    height: int,
    width: int,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(config.near_endpoint_mask_seed))
    masks_0 = []
    masks_T = []
    if bool(config.near_endpoint_shared_mask):
        base = _single_near_endpoint_mask(config.near_endpoint_sensor_mode, int(config.num_near_endpoint_obs), height, width, generator)
        masks_0 = [base.clone() for _ in range(batch_size)]
        masks_T = [base.clone() for _ in range(batch_size)]
    else:
        for _ in range(batch_size):
            masks_0.append(
                _single_near_endpoint_mask(
                    config.near_endpoint_sensor_mode,
                    int(config.num_near_endpoint_obs),
                    height,
                    width,
                    generator,
                )
            )
            masks_T.append(
                _single_near_endpoint_mask(
                    config.near_endpoint_sensor_mode,
                    int(config.num_near_endpoint_obs),
                    height,
                    width,
                    generator,
                )
            )
    mask_0 = torch.stack(masks_0, dim=0).unsqueeze(1).to(device=device, dtype=dtype)
    mask_T = torch.stack(masks_T, dim=0).unsqueeze(1).to(device=device, dtype=dtype)
    counts_0 = mask_0.reshape(batch_size, -1).sum(dim=1).detach().cpu().tolist()
    counts_T = mask_T.reshape(batch_size, -1).sum(dim=1).detach().cpu().tolist()
    return (
        mask_0,
        mask_T,
        {
            "mask_observed_points_0": counts_0,
            "mask_observed_points_T": counts_T,
            "mask_shape": [batch_size, 1, height, width],
        },
    )


def _single_near_endpoint_mask(mode: str, num_obs: int, height: int, width: int, generator: Any) -> Any:
    import math
    import torch

    total = int(height * width)
    n = min(max(int(num_obs), 0), total)
    mask = torch.zeros(total, dtype=torch.float32)
    if n == 0:
        return mask.view(height, width)
    if mode in {"random", "per_sample_random"}:
        indices = torch.randperm(total, generator=generator)[:n]
    elif mode == "fixed":
        indices = torch.arange(n)
    elif mode == "grid":
        side = max(int(math.ceil(math.sqrt(n))), 1)
        ys = torch.linspace(0, height - 1, side).round().long()
        xs = torch.linspace(0, width - 1, side).round().long()
        grid = torch.cartesian_prod(ys, xs)
        indices = torch.unique(grid[:, 0] * width + grid[:, 1])[:n]
        if indices.numel() < n:
            filler = torch.arange(total)
            indices = torch.unique(torch.cat([indices, filler]))[:n]
    elif mode == "sensor_column":
        cols_needed = max(int(math.ceil(n / max(height, 1))), 1)
        cols = torch.linspace(0, width - 1, cols_needed).round().long()
        rows = torch.arange(height)
        grid = torch.cartesian_prod(rows, cols)
        indices = torch.unique(grid[:, 0] * width + grid[:, 1])[:n]
    else:
        raise ValueError(f"Unsupported near_endpoint_sensor_mode={mode!r}")
    mask[indices] = 1.0
    return mask.view(height, width)


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
        values = np.asarray(dataset[()] if dataset.shape == () else dataset[:], dtype=np.float32)
        if values.ndim == 0:
            return float(values)
        if values.shape[0] == 1:
            return values[0]
        return values[offset]
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
        return {"alpha": ones(1.0), "T": ones(1.0)}
    if config.pde == "wave":
        return {"c": ones(1.0), "T": ones(1.0)}
    if config.pde == "advection_diffusion":
        return {"b_x": ones(0.0), "b_y": ones(0.0), "kappa": ones(1.0), "T": ones(1.0)}
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

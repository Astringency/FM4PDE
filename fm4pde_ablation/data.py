from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fm4pde_ablation.config import AblationConfig, load_yaml_file


COEF_CHANNELS = {
    "darcy": ["coef"],
    "poisson": ["source"],
    "helmholtz": ["source"],
    "nsnonbounded": ["w0"],
    "burger": ["u"],
    "reaction_diffusion": ["u0", "v0"],
    "shallow_water": ["h0", "hu0", "hv0"],
    "heat": ["u0"],
    "wave": ["u0", "v0"],
    "advection_diffusion": ["u0"],
    "steady_heat_conduction": ["f"],
}

SOL_CHANNELS = {
    "darcy": ["pressure"],
    "poisson": ["phi"],
    "helmholtz": ["psi"],
    "nsnonbounded": ["wT"],
    "burger": ["u"],
    "reaction_diffusion": ["uT", "vT"],
    "shallow_water": ["hT", "huT", "hvT"],
    "heat": ["uT"],
    "wave": ["uT", "vT"],
    "advection_diffusion": ["uT"],
    "steady_heat_conduction": ["u"],
}

FUTURE_H5_CHANNEL_COUNTS = {
    "heat": (1, 1),
    "wave": (2, 2),
    "advection_diffusion": (1, 1),
    "steady_heat_conduction": (1, 1),
}

FUTURE_H5_PARAM_NAMES = {
    "heat": ("alpha",),
    "wave": ("c",),
    "advection_diffusion": ("b_x", "b_y", "kappa"),
    "steady_heat_conduction": (
        "u_D",
        "residual_norm",
        "picard_iters",
        "converged",
        "n_sources",
        "source_x",
        "source_y",
        "source_amp",
        "source_sigma",
    ),
}

OPTIONAL_FUTURE_H5_PARAMS = {
    "heat": {"alpha"},
    "wave": {"c"},
    "steady_heat_conduction": {
        "residual_norm",
        "picard_iters",
        "converged",
        "n_sources",
        "source_x",
        "source_y",
        "source_amp",
        "source_sigma",
    },
}

FUTURE_H5_PARAM_ALIASES = {
    "alpha": ("alpha", "fixed_alpha"),
    "c": ("c", "fixed_c"),
}


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
    if _needs_legacy_fields(config) and Path(config.data_config_path).exists():
        legacy = load_yaml_file(config.data_config_path)
        config = _merge_legacy_data_fields(config, legacy)
    if config.loadby == "future_h5" and config.pde in FUTURE_H5_CHANNEL_COUNTS:
        counts = FUTURE_H5_CHANNEL_COUNTS[config.pde]
        config.img_channels = sum(counts)
    return config


def load_ground_truth(config: AblationConfig) -> PDEGroundTruth:
    """Load one or more test samples using the legacy FM4PDE data conventions."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("load_ground_truth requires torch") from exc

    config = finalize_ground_truth_config(config)

    if not config.data_path or not Path(config.data_path).exists():
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
    pair = torch.cat([coef_t, sol_t], dim=1)
    pde_params: dict[str, Any] = {}
    pde_param_sources: dict[str, str] = {}
    if config.loadby == "future_h5":
        pde_params, pde_param_sources = _future_h5_params_for_offsets(raw["__h5__"], config.pde, offsets, pair.device)
    channel_names_coef = _channel_names(COEF_CHANNELS, config.pde, int(coef_t.shape[1]), "coef")
    channel_names_sol = _channel_names(SOL_CHANNELS, config.pde, int(sol_t.shape[1]), "sol")
    metadata = {
        "data_path": config.data_path,
        "offset": config.offset,
        "loadby": config.loadby,
        "synthetic": False,
        "batch_size": config.batch_size,
        "channel_names": channel_names_coef + channel_names_sol,
        "channel_names_coef": channel_names_coef,
        "channel_names_sol": channel_names_sol,
        "pde_params_keys": sorted(pde_params),
        "pde_params_sources": pde_param_sources,
        "scalar_params_loaded": bool(pde_params),
    }
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

    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    channels = _channel_counts(config.pde, config.img_channels)
    h = w = int(config.img_resolution)
    gen = torch.Generator(device="cpu").manual_seed(int(config.sample_seed))
    base = torch.randn(
        config.batch_size,
        sum(channels),
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
    coef = base[:, : channels[0]]
    sol = base[:, channels[0] :]
    pde_params = _synthetic_pde_params(config, device, dtype)
    channel_names_coef = _channel_names(COEF_CHANNELS, config.pde, channels[0], "coef")
    channel_names_sol = _channel_names(SOL_CHANNELS, config.pde, channels[1], "sol")
    return PDEGroundTruth(
        pde=config.pde,
        coef=coef,
        sol=sol,
        pair=torch.cat([coef, sol], dim=1),
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
            "channel_names": channel_names_coef + channel_names_sol,
            "channel_names_coef": channel_names_coef,
            "channel_names_sol": channel_names_sol,
            "pde_params_keys": sorted(pde_params),
            "pde_params_sources": {key: "synthetic_default" for key in pde_params},
            "scalar_params_loaded": False,
        },
    )


def _needs_legacy_fields(config: AblationConfig) -> bool:
    return not config.data_path or not config.coef_name or not config.solution_name or not config.loadby


def _merge_legacy_data_fields(config: AblationConfig, legacy: dict[str, Any]) -> AblationConfig:
    data = legacy.get("data", {})
    model = legacy.get("model", {})
    generate = legacy.get("generate", {})
    config.data_path = config.data_path or str(data.get("datapath", "")).strip("'\"")
    config.coef_name = config.coef_name or str(data.get("coef", "")).strip("'\"")
    config.solution_name = config.solution_name or str(data.get("solution", "")).strip("'\"")
    config.loadby = config.loadby or str(data.get("loadby", "")).strip("'\"")
    config.img_resolution = int(config.img_resolution or data.get("img_resolution", 128))
    config.img_channels = int(config.img_channels or data.get("img_channels", 2))
    if not config.checkpoint_path:
        config.checkpoint_path = str(model.get("pre-trained", "")).strip("'\"")
    if config.device == "cpu" and generate.get("device"):
        # Keep CPU as the ablation default unless explicitly overridden.
        pass
    return config


def _load_raw_data(config: AblationConfig) -> dict[str, Any]:
    if config.loadby == "scipy":
        import scipy.io

        return scipy.io.loadmat(config.data_path)
    if config.loadby in {"h5py", "swe", "rd", "future_h5"}:
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
        key = list(file.keys())[offset]
        arr = file[key]["data"][:]
        return arr[50, :, :, :], arr[-1, :, :, :]

    if config.loadby == "future_h5":
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
    if pde == "burger":
        return (1, 1)
    if pde == "reaction_diffusion":
        return (2, 2)
    if pde == "shallow_water":
        return (3, 3)
    if pde in FUTURE_H5_CHANNEL_COUNTS:
        return FUTURE_H5_CHANNEL_COUNTS[pde]
    if pde == "heat":
        return (img_channels // 2, img_channels // 2) if img_channels and img_channels % 2 == 0 else (1, 1)
    if pde == "wave":
        return (img_channels // 2, img_channels // 2) if img_channels and img_channels % 2 == 0 else (2, 2)
    if pde == "advection_diffusion":
        return (img_channels // 2, img_channels // 2) if img_channels and img_channels % 2 == 0 else (1, 1)
    if pde == "steady_heat_conduction":
        return (img_channels // 2, img_channels // 2) if img_channels and img_channels % 2 == 0 else (1, 1)
    if img_channels and img_channels % 2 == 0:
        return (img_channels // 2, img_channels // 2)
    return (1, 1)


def _future_h5_params_for_offsets(file: Any, pde: str, offsets: list[int], device: Any) -> tuple[dict[str, Any], dict[str, str]]:
    import torch

    param_names = FUTURE_H5_PARAM_NAMES.get(pde, ())
    optional = OPTIONAL_FUTURE_H5_PARAMS.get(pde, set())
    params = {}
    sources = {}
    for name in param_names:
        storage_name = _future_h5_storage_name(file, name)
        if storage_name is None:
            if name in optional:
                continue
            raise KeyError(f"Missing future_h5 scalar {name!r}")
        values = [_read_future_h5_value(file, storage_name, offset) for offset in offsets]
        params[name] = torch.as_tensor(values, dtype=torch.float32, device=device)
        sources[name] = "dataset" if storage_name in file else f"attrs:{storage_name}"
    return params, sources


def _future_h5_storage_name(file: Any, name: str) -> str | None:
    for candidate in FUTURE_H5_PARAM_ALIASES.get(name, (name,)):
        if candidate in file or candidate in file.attrs:
            return candidate
    return None


def _read_future_h5_value(file: Any, name: str, offset: int) -> Any:
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
        return float(values.reshape(-1)[0]) if values.ndim == 0 or values.size == 1 else values
    raise KeyError(f"Missing future_h5 scalar {name!r}")


def _synthetic_pde_params(config: AblationConfig, device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    b = int(config.batch_size)
    ones = lambda value: torch.full((b,), float(value), dtype=dtype, device=device)
    if config.pde == "heat":
        return {"alpha": ones(1.0)}
    if config.pde == "wave":
        return {"c": ones(1.0)}
    if config.pde == "advection_diffusion":
        return {"b_x": ones(0.0), "b_y": ones(0.0), "kappa": ones(1.0)}
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


def _channel_names(table: dict[str, list[str]], pde: str, n_channels: int, prefix: str) -> list[str]:
    names = table.get(pde, [])
    if len(names) == n_channels:
        return names
    return [f"{prefix}_{i}" for i in range(n_channels)]

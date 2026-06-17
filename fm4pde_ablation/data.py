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
    "advection_diffusion": ["u0", "b_x", "b_y", "kappa"],
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
    "advection_diffusion": ["uT", "b_x", "b_y", "kappa"],
}


@dataclass
class PDEGroundTruth:
    pde: str
    coef: Any
    sol: Any
    pair: Any
    channel_names_coef: list[str]
    channel_names_sol: list[str]
    metadata: dict[str, Any]


def load_ground_truth(config: AblationConfig) -> PDEGroundTruth:
    """Load one or more test samples using the legacy FM4PDE data conventions."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("load_ground_truth requires torch") from exc

    if _needs_legacy_fields(config) and Path(config.data_config_path).exists():
        legacy = load_yaml_file(config.data_config_path)
        config = _merge_legacy_data_fields(config, legacy)

    if not config.data_path or not Path(config.data_path).exists():
        if not config.allow_synthetic_data:
            raise FileNotFoundError(f"Data path does not exist: {config.data_path}")
        return make_synthetic_ground_truth(config)

    raw = _load_raw_data(config)
    coef_list = []
    sol_list = []
    for batch_idx in range(config.batch_size):
        offset = config.offset + batch_idx
        coef, sol = _extract_single_sample(config, raw, offset)
        coef_list.append(_ensure_bchw(coef, config.pde, "coef", config.device, config.dtype))
        sol_list.append(_ensure_bchw(sol, config.pde, "sol", config.device, config.dtype))

    coef_t = torch.cat(coef_list, dim=0)
    sol_t = torch.cat(sol_list, dim=0)
    _assert_spatial_compatible(coef_t, sol_t, config.pde)
    pair = torch.cat([coef_t, sol_t], dim=1)
    return PDEGroundTruth(
        pde=config.pde,
        coef=coef_t,
        sol=sol_t,
        pair=pair,
        channel_names_coef=COEF_CHANNELS.get(config.pde, [f"coef_{i}" for i in range(coef_t.shape[1])]),
        channel_names_sol=SOL_CHANNELS.get(config.pde, [f"sol_{i}" for i in range(sol_t.shape[1])]),
        metadata={
            "data_path": config.data_path,
            "offset": config.offset,
            "loadby": config.loadby,
            "synthetic": False,
            "batch_size": config.batch_size,
        },
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
    return PDEGroundTruth(
        pde=config.pde,
        coef=coef,
        sol=sol,
        pair=torch.cat([coef, sol], dim=1),
        channel_names_coef=COEF_CHANNELS.get(config.pde, [f"coef_{i}" for i in range(channels[0])]),
        channel_names_sol=SOL_CHANNELS.get(config.pde, [f"sol_{i}" for i in range(channels[1])]),
        metadata={
            "data_path": config.data_path,
            "offset": config.offset,
            "loadby": config.loadby,
            "synthetic": True,
            "reason": "configured data path was missing",
            "batch_size": config.batch_size,
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
        return file[config.coef_name][offset], file[config.solution_name][offset]

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


def _ensure_bchw(value: Any, pde: str, side: str, device: str, dtype_name: str) -> Any:
    import torch

    dtype = _torch_dtype(dtype_name)
    tensor = torch.as_tensor(value, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        expected = _channel_counts(pde, 0)[0 if side == "coef" else 1]
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
    if pde == "heat":
        return (1, 1)
    if pde == "wave":
        return (2, 2)
    if pde == "advection_diffusion":
        return (4, 4)
    if img_channels and img_channels % 2 == 0:
        return (img_channels // 2, img_channels // 2)
    return (1, 1)


def _torch_dtype(name: str) -> Any:
    import torch

    if name == "float64":
        return torch.float64
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32

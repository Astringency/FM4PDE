from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


class WrappedModel:
    def __init__(self, model: Any):
        self.model = model

    def __call__(self, x: Any, t: Any, **extras: Any) -> Any:
        return self.model(x, t, **extras)

    def to(self, device: str | Any) -> "WrappedModel":
        self.model = self.model.to(device)
        return self

    def eval(self) -> "WrappedModel":
        self.model.eval()
        return self


def load_fm4pde_checkpoint(checkpoint_path: str, pde_type: str, device: str | Any, wrap: bool = True) -> Any:
    model, _, _ = load_fm4pde_checkpoint_bundle(checkpoint_path, pde_type, device, wrap=wrap)
    return model


def load_fm4pde_checkpoint_bundle(
    checkpoint_path: str,
    pde_type: str,
    device: str | Any,
    wrap: bool = True,
) -> tuple[Any, Any | None, dict[str, Any]]:
    import torch
    from data.transform import PDEStandardizer
    from models.model_configs import instantiate_model

    path = _resolve_checkpoint_path(checkpoint_path, pde_type)
    payload = torch.load(path, weights_only=False, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    num_channels = _infer_num_channels(payload)
    model = instantiate_model(
        architechture=pde_type,
        use_ema=False,
        in_channels=num_channels,
        out_channels=num_channels,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    normalizer = None
    if isinstance(payload, dict) and payload.get("normalizer") is not None:
        normalizer = PDEStandardizer.from_state_dict(payload["normalizer"])
    else:
        warnings.warn(
            "Checkpoint has no normalizer; sampling will treat model state as physical identity. "
            "Use legacy_minmax=True only for explicit legacy min-max checkpoints.",
            RuntimeWarning,
            stacklevel=2,
        )
    wrapped = WrappedModel(model).to(device) if wrap else model
    return wrapped, normalizer, payload if isinstance(payload, dict) else {"model": payload}


def _infer_num_channels(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("num_channels") is not None:
        return int(payload["num_channels"])
    if payload.get("data_shape") is not None:
        return int(payload["data_shape"][1])
    normalizer = payload.get("normalizer")
    if normalizer is not None and normalizer.get("mean") is not None:
        return int(normalizer["mean"].shape[1])
    state = payload.get("model")
    if isinstance(state, dict):
        for key in ("input_blocks.0.0.weight", "model.input_blocks.0.0.weight"):
            if key in state:
                return int(state[key].shape[1])
    return None


def _resolve_checkpoint_path(checkpoint_path: str, pde_type: str) -> Path:
    path = Path(checkpoint_path)
    if path.exists():
        return path
    fallback_names = {
        "burger": "fm4burgers.pth",
        "darcy": "fm4darcy.pth",
        "poisson": "fm4poisson.pth",
        "helmholtz": "fm4helmholtz.pth",
        "nsnonbounded": "fm4nsnonbounded.pth",
        "reaction_diffusion": "fm4reaction_diffusion.pth",
        "shallow_water": "fm4shallow_water.pth",
        "heat": "fm4heat.pth",
        "wave": "fm4wave.pth",
        "advection_diffusion": "fm4advection_diffusion.pth",
        "steady_heat_conduction": "fm4steady_heat_conduction.pth",
    }
    fallback = Path("output/pretrained") / fallback_names.get(pde_type, f"fm4{pde_type}.pth")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path} (fallback {fallback} also missing)")

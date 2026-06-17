from __future__ import annotations

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
    import torch
    from models.model_configs import instantiate_model

    path = _resolve_checkpoint_path(checkpoint_path, pde_type)
    payload = torch.load(path, weights_only=False, map_location=device)
    model = instantiate_model(architechture=pde_type, use_ema=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return WrappedModel(model).to(device) if wrap else model


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
    }
    fallback = Path("output/pretrained") / fallback_names.get(pde_type, f"fm4{pde_type}.pth")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path} (fallback {fallback} also missing)")

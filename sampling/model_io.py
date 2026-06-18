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


def load_fm4pde_checkpoint(
    checkpoint_path: str,
    pde_type: str,
    device: str | Any,
    wrap: bool = True,
    prefer_ema: bool = True,
    model_profile: str | None = None,
) -> Any:
    model, _, _ = load_fm4pde_checkpoint_bundle(
        checkpoint_path,
        pde_type,
        device,
        wrap=wrap,
        prefer_ema=prefer_ema,
        model_profile=model_profile,
    )
    return model


def load_fm4pde_checkpoint_bundle(
    checkpoint_path: str,
    pde_type: str,
    device: str | Any,
    wrap: bool = True,
    prefer_ema: bool = True,
    model_profile: str | None = None,
) -> tuple[Any, Any | None, dict[str, Any]]:
    import torch
    from data.transform import PDEStandardizer
    from models.model_configs import instantiate_model

    path = _resolve_checkpoint_path(checkpoint_path, pde_type)
    loaded = torch.load(path, weights_only=False, map_location="cpu")
    payload = loaded if isinstance(loaded, dict) else {"model": loaded}
    num_channels = _infer_num_channels(payload)
    model_cfg, selected_profile, selected_metadata = _select_model_config_from_payload(
        payload,
        pde_type=pde_type,
        num_channels=num_channels,
        requested_profile=model_profile,
    )
    model = instantiate_model(architechture=pde_type, use_ema=False, model_config=model_cfg)
    state, selected_inference_weight = _select_inference_state(payload, model, prefer_ema=prefer_ema)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture does not match the current model profile. "
            "Retrain with recommended profile or pass explicit legacy_base profile if this is an old checkpoint. "
            f"checkpoint={path}, pde={pde_type}, selected_profile={selected_profile!r}, "
            f"inference_weight={selected_inference_weight}."
        ) from exc
    model = model.to(device)
    model.eval()
    payload["selected_inference_weight"] = selected_inference_weight
    payload["runtime_requested_model_profile"] = model_profile
    payload["selected_model_profile"] = selected_profile
    payload["selected_model_config_metadata"] = selected_metadata
    payload["selected_architecture_family"] = selected_metadata.get("architecture_family")
    payload["has_ema"] = bool(
        payload.get("has_ema", False)
        or payload.get("model_ema") is not None
        or _has_legacy_ema_shadow_params(payload.get("model"))
    )

    normalizer = None
    if payload.get("normalizer") is not None:
        normalizer = PDEStandardizer.from_state_dict(payload["normalizer"])
    else:
        raise ValueError(
            "Checkpoint has no PDEStandardizer normalizer. "
            "Sampling requires a checkpoint with saved normalization metadata; dry_run is the only path "
            "that creates an explicit identity normalizer."
        )
    wrapped = WrappedModel(model).to(device) if wrap else model
    return wrapped, normalizer, payload


def _select_model_config_from_payload(
    payload: dict[str, Any],
    *,
    pde_type: str,
    num_channels: int | None,
    requested_profile: str | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    from models.model_configs import (
        get_model_config,
        get_model_config_metadata,
        model_config_metadata_from_config,
    )

    checkpoint_cfg = payload.get("model_config")
    if isinstance(checkpoint_cfg, dict) and checkpoint_cfg:
        selected_profile = str(
            payload.get("model_profile")
            or checkpoint_cfg.get("architecture_profile")
            or "checkpoint"
        )
        derived_metadata = model_config_metadata_from_config(checkpoint_cfg)
        metadata = payload.get("model_config_metadata")
        metadata = {
            **derived_metadata,
            **(metadata if isinstance(metadata, dict) else {}),
        }
        metadata.setdefault("architecture_profile", selected_profile)
        return dict(checkpoint_cfg), selected_profile, dict(metadata)

    if requested_profile is None:
        raise ValueError(
            "Checkpoint has no saved model_config. Pass an explicit model_profile, for example "
            "model_profile=legacy_base for old checkpoints, or retrain so checkpoints include model_config."
        )

    model_cfg = get_model_config(
        pde_type,
        profile=requested_profile,
        in_channels=num_channels,
        out_channels=num_channels,
    )
    metadata = get_model_config_metadata(
        pde_type,
        profile=requested_profile,
        in_channels=num_channels,
        out_channels=num_channels,
    )
    return model_cfg, requested_profile, metadata


def _jsonable_model_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, list):
            out[key] = value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _select_inference_state(
    payload: dict[str, Any],
    model: Any,
    prefer_ema: bool = True,
) -> tuple[dict[str, Any], str]:
    if prefer_ema and payload.get("model_ema") is not None:
        state = payload["model_ema"]
        selected = "ema"
    elif "model" in payload:
        state = payload["model"]
        selected = "raw"
    else:
        raise ValueError("Checkpoint payload has neither model_ema nor model state")

    if not isinstance(state, dict):
        raise ValueError("Selected checkpoint state is not a state_dict")

    if _looks_like_legacy_ema_state(state):
        legacy_has_ema = _has_legacy_ema_shadow_params(state)
        selected = "ema" if prefer_ema and legacy_has_ema else "raw"
        state = _convert_legacy_ema_state(state, model, prefer_ema=prefer_ema)
    elif any(str(key).startswith("model.") for key in state):
        state = _strip_model_prefix_state(state)

    return state, selected


def _strip_model_prefix_state(state: dict[str, Any]) -> dict[str, Any]:
    stripped = {
        str(key)[len("model.") :]: _clone_state_value(value)
        for key, value in state.items()
        if str(key).startswith("model.")
    }
    if not stripped:
        raise ValueError("State dict has no model.* keys to strip")
    return stripped


def _convert_legacy_ema_state(
    state: dict[str, Any],
    model: Any,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    raw_state = _strip_model_prefix_state(state)
    if not prefer_ema:
        warnings.warn(
            "Loading raw model.* weights from a legacy EMA wrapper checkpoint.",
            RuntimeWarning,
            stacklevel=2,
        )
        return raw_state

    shadow_params = _legacy_shadow_params(state)
    if not shadow_params:
        warnings.warn(
            "Legacy EMA wrapper checkpoint has no shadow_params; falling back to raw model.* weights.",
            RuntimeWarning,
            stacklevel=2,
        )
        return raw_state

    converted = {key: _clone_state_value(value) for key, value in raw_state.items()}
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if len(shadow_params) != len(trainable_params):
        raise ValueError(
            "Checkpoint is an EMA wrapper state_dict but its shadow_params cannot be "
            "mapped onto the plain UNetModel parameters: "
            f"{len(shadow_params)} shadow params vs {len(trainable_params)} trainable params."
        )
    for (name, _), shadow in zip(trainable_params, shadow_params):
        if name not in converted:
            raise ValueError(
                f"Checkpoint is an EMA wrapper state_dict but parameter {name!r} "
                "is missing from the wrapped model.* state."
            )
        converted[name] = _clone_state_value(shadow)
    warnings.warn(
        "Converted a legacy EMA wrapper checkpoint to plain EMA model weights for sampling.",
        RuntimeWarning,
        stacklevel=2,
    )
    return converted


def _looks_like_legacy_ema_state(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    return (
        "num_updates" in state
        or _has_legacy_ema_shadow_params(state)
        or any(str(key).startswith("model.") for key in state)
    )


def _has_legacy_ema_shadow_params(state: Any) -> bool:
    return isinstance(state, dict) and any(str(key).startswith("shadow_params.") for key in state)


def _legacy_shadow_params(state: dict[str, Any]) -> list[Any]:
    params = []
    for key, value in state.items():
        key_str = str(key)
        if not key_str.startswith("shadow_params."):
            continue
        try:
            index = int(key_str.rsplit(".", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Malformed EMA shadow parameter key: {key_str!r}") from exc
        params.append((index, value))
    return [value for _, value in sorted(params, key=lambda item: item[0])]


def _clone_state_value(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _infer_num_channels(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("num_channels") is not None:
        return int(payload["num_channels"])
    model_config = payload.get("model_config")
    if isinstance(model_config, dict) and model_config.get("in_channels") is not None:
        return int(model_config["in_channels"])
    if payload.get("data_shape") is not None:
        return int(payload["data_shape"][1])
    normalizer = payload.get("normalizer")
    if normalizer is not None and normalizer.get("mean") is not None:
        return int(normalizer["mean"].shape[1])
    for state_key in ("model", "model_ema", "model_for_resume"):
        state = payload.get(state_key)
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
    fallback = Path("outputs/pretrained") / fallback_names.get(pde_type, f"fm4{pde_type}.pth")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path} (fallback {fallback} also missing)")

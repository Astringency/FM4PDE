from __future__ import annotations

from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 3


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
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    from data.transform import PDEStandardizer
    from models.model_configs import instantiate_model, model_config_metadata_from_config

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, weights_only=False, map_location="cpu")
    _validate_checkpoint_payload(payload, path)

    model_config = dict(payload["model_config"])
    selected_profile = str(
        payload.get("model_profile")
        or model_config.get("architecture_profile")
        or "recommended"
    )
    if model_profile is not None and model_profile != selected_profile:
        raise ValueError(
            f"Configured model_profile={model_profile!r} does not match checkpoint "
            f"profile={selected_profile!r}: {path}"
        )

    selected_metadata = model_config_metadata_from_config(model_config)
    saved_metadata = payload.get("model_config_metadata")
    if isinstance(saved_metadata, dict):
        for key in ("joint_pde_names", "pde_label_mapping", "model_arch", "model_profile"):
            if key in saved_metadata:
                selected_metadata[key] = saved_metadata[key]
    selected_metadata["architecture_profile"] = selected_profile

    _validate_joint_checkpoint_conditioning(payload, model_config, selected_metadata, path)
    model = instantiate_model(
        architechture=pde_type,
        use_ema=False,
        model_config=model_config,
    )
    state, selected_inference_weight = _select_inference_state(
        payload,
        prefer_ema=prefer_ema,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint model_config does not match its saved state_dict. "
            f"checkpoint={path}, pde={pde_type}, profile={selected_profile!r}, "
            f"inference_weight={selected_inference_weight}."
        ) from exc

    model = model.to(device)
    model.eval()
    normalizer = PDEStandardizer.from_state_dict(payload["normalizer"])

    payload["selected_inference_weight"] = selected_inference_weight
    payload["runtime_requested_model_profile"] = model_profile
    payload["selected_model_profile"] = selected_profile
    payload["selected_model_config_metadata"] = selected_metadata
    payload["selected_architecture_family"] = selected_metadata.get("architecture_family")
    payload["has_ema"] = isinstance(payload.get("model_ema"), dict)

    wrapped = WrappedModel(model).to(device) if wrap else model
    return wrapped, normalizer, payload


def _validate_checkpoint_payload(payload: Any, path: Path) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    schema = payload.get("checkpoint_schema_version")
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}, got {schema!r}: {path}"
        )
    if not isinstance(payload.get("model_config"), dict) or not payload["model_config"]:
        raise ValueError(f"Checkpoint has no model_config: {path}")
    if not isinstance(payload.get("model_config_metadata"), dict) or not payload["model_config_metadata"]:
        raise ValueError(f"Checkpoint has no model_config_metadata: {path}")
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError(f"Checkpoint has no plain model state_dict: {path}")
    if not isinstance(payload.get("normalizer"), dict):
        raise ValueError(f"Checkpoint has no PDEStandardizer normalizer: {path}")


def _validate_joint_checkpoint_conditioning(
    payload: dict[str, Any],
    model_config: dict[str, Any],
    model_metadata: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    data_metadata = payload.get("data_metadata", {})
    joint_names = model_metadata.get("joint_pde_names")
    if not isinstance(joint_names, list) and isinstance(data_metadata, dict):
        joint_names = data_metadata.get("pde_names")
    if not isinstance(joint_names, list) or len(joint_names) <= 1:
        return
    mapping = model_metadata.get("pde_label_mapping")
    if not isinstance(mapping, dict) and isinstance(data_metadata, dict):
        mapping = data_metadata.get("pde_label_mapping")
    if model_config.get("num_classes") is None or not isinstance(mapping, dict):
        raise ValueError(
            "Joint checkpoint requires a PDE category layer and pde_label_mapping: "
            f"{checkpoint_path}"
        )
    normalized = {str(name): int(index) for name, index in mapping.items()}
    if sorted(normalized.values()) != list(range(int(model_config["num_classes"]))):
        raise ValueError(f"Joint checkpoint has a non-contiguous pde_label_mapping: {normalized}")


def _select_inference_state(
    payload: dict[str, Any],
    *,
    prefer_ema: bool,
) -> tuple[dict[str, Any], str]:
    if prefer_ema and isinstance(payload.get("model_ema"), dict):
        state = payload["model_ema"]
        selected = "ema"
    else:
        state = payload["model"]
        selected = "raw"
    if not state:
        raise ValueError(f"Selected {selected} state_dict is empty")
    invalid = [
        str(key)
        for key in state
        if str(key).startswith(("model.", "shadow_params.")) or key == "num_updates"
    ]
    if invalid:
        raise ValueError(
            f"Selected {selected} state_dict is not a plain model state_dict; "
            f"first invalid key={invalid[0]!r}"
        )
    return state, selected

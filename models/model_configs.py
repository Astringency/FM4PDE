# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Union

from data.specs import PDE_DATA_SPECS, get_pde_spec
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.unet import UNetModel


MODEL_PROFILE_CHOICES = ("recommended", "light", "base", "heavy", "legacy_base")

MODEL_METADATA_KEYS = {
    "architecture_family",
    "architecture_profile",
    "axis_semantics",
    "scalar_conditioning",
    "scalar_conditioning_params",
    "notes",
}

MODEL_PARAMETER_KEYS = {
    "in_channels",
    "model_channels",
    "out_channels",
    "num_res_blocks",
    "attention_resolutions",
    "dropout",
    "channel_mult",
    "conv_resample",
    "dims",
    "num_classes",
    "use_checkpoint",
    "num_heads",
    "num_head_channels",
    "num_heads_upsample",
    "use_scale_shift_norm",
    "resblock_updown",
    "use_new_attention_order",
    "with_fourier_features",
    "fourier_feature_start",
    "fourier_feature_stop",
    "fourier_feature_step",
    "ignore_time",
    "input_projection",
    "image_size",
}


def _base_config(
    in_channels: int,
    model_channels: int = 128,
    out_channels: int | None = None,
    *,
    num_res_blocks: int = 4,
    attention_resolutions: tuple[int, ...] = (16,),
    dropout: float = 0.05,
    channel_mult: tuple[int, ...] = (1, 2, 4),
    use_scale_shift_norm: bool = True,
    use_new_attention_order: bool = True,
    with_fourier_features: bool = False,
    architecture_family: str = "base",
    axis_semantics: str = "spatial_2d",
    architecture_profile: str = "recommended",
    notes: str | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "in_channels": int(in_channels),
        "model_channels": int(model_channels),
        "out_channels": int(out_channels) if out_channels is not None else int(in_channels),
        "num_res_blocks": int(num_res_blocks),
        "attention_resolutions": tuple(int(value) for value in attention_resolutions),
        "dropout": float(dropout),
        "channel_mult": tuple(int(value) for value in channel_mult),
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": bool(use_scale_shift_norm),
        "resblock_updown": False,
        "use_new_attention_order": bool(use_new_attention_order),
        "with_fourier_features": bool(with_fourier_features),
        "fourier_feature_start": 6,
        "fourier_feature_stop": 8,
        "fourier_feature_step": 1,
        "architecture_family": architecture_family,
        "axis_semantics": axis_semantics,
        "architecture_profile": architecture_profile,
        "scalar_conditioning": False,
        "scalar_conditioning_params": (),
    }
    if notes:
        cfg["notes"] = notes
    return cfg


def _light_smooth_config(in_channels: int, *, profile: str = "recommended") -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=96,
        num_res_blocks=3,
        dropout=0.05,
        channel_mult=(1, 2, 4),
        attention_resolutions=(16,),
        with_fourier_features=False,
        architecture_family="light_smooth",
        architecture_profile=profile,
    )


def _elliptic_static_config(in_channels: int, *, profile: str = "recommended") -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=128,
        num_res_blocks=4,
        dropout=0.05,
        channel_mult=(1, 2, 4),
        attention_resolutions=(8, 16),
        with_fourier_features=True,
        use_scale_shift_norm=True,
        architecture_family="elliptic_static",
        architecture_profile=profile,
    )


def _temporal_endpoint_base_config(in_channels: int, *, profile: str = "recommended") -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=128,
        num_res_blocks=4,
        dropout=0.05,
        channel_mult=(1, 2, 4),
        attention_resolutions=(16,),
        with_fourier_features=False,
        architecture_family="temporal_endpoint_base",
        architecture_profile=profile,
    )


def _temporal_endpoint_heavy_config(
    in_channels: int,
    *,
    profile: str = "recommended",
    with_fourier_features: bool,
    notes: str | None = None,
) -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=192,
        num_res_blocks=4,
        dropout=0.05,
        channel_mult=(1, 2, 4, 4),
        attention_resolutions=(8, 16),
        with_fourier_features=with_fourier_features,
        architecture_family="temporal_endpoint_heavy",
        architecture_profile=profile,
        notes=notes,
    )


def _wave_config(in_channels: int = 4, *, profile: str = "recommended") -> dict[str, Any]:
    return _temporal_endpoint_heavy_config(
        in_channels,
        profile=profile,
        with_fourier_features=True,
        notes="wave profile enables Fourier features for oscillatory phase/frequency structure",
    )


def _shallow_water_config(in_channels: int = 6, *, profile: str = "recommended") -> dict[str, Any]:
    return _temporal_endpoint_heavy_config(
        in_channels,
        profile=profile,
        with_fourier_features=False,
        notes=(
            "shallow-water profile keeps Fourier features off initially because conservative "
            "variables can be scale-sensitive"
        ),
    )


def _ns_config(in_channels: int = 2, *, profile: str = "recommended") -> dict[str, Any]:
    return _temporal_endpoint_heavy_config(
        in_channels,
        profile=profile,
        with_fourier_features=True,
        notes="nsnonbounded profile enables Fourier features for periodic vorticity structure",
    )


def _burger_time_space_config(in_channels: int = 1, *, profile: str = "recommended") -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=128,
        num_res_blocks=4,
        dropout=0.05,
        channel_mult=(1, 2, 4),
        attention_resolutions=(16,),
        with_fourier_features=True,
        architecture_family="full_time_space",
        axis_semantics="BCHW_as_time_space_H_time_W_space",
        architecture_profile=profile,
    )


def _profile_config(
    in_channels: int,
    *,
    profile: str,
    architecture_family: str,
    model_channels: int,
    num_res_blocks: int,
    channel_mult: tuple[int, ...],
    attention_resolutions: tuple[int, ...],
    with_fourier_features: bool,
    axis_semantics: str = "spatial_2d",
    notes: str | None = None,
) -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        dropout=0.05,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        with_fourier_features=with_fourier_features,
        architecture_family=architecture_family,
        axis_semantics=axis_semantics,
        architecture_profile=profile,
        notes=notes,
    )


def _light_profile_config(pde: str, in_channels: int) -> dict[str, Any]:
    family = _recommended_family(pde)
    return _profile_config(
        in_channels,
        profile="light",
        architecture_family=family,
        model_channels=96,
        num_res_blocks=2 if pde in {"poisson", "heat"} else 3,
        channel_mult=(1, 2, 4),
        attention_resolutions=(16,),
        with_fourier_features=pde in {"burger"},
        axis_semantics=_axis_semantics(pde),
        notes=_profile_notes(pde, "light"),
    )


def _base_profile_config(pde: str, in_channels: int) -> dict[str, Any]:
    family = _recommended_family(pde)
    return _profile_config(
        in_channels,
        profile="base",
        architecture_family=family,
        model_channels=128,
        num_res_blocks=4,
        channel_mult=(1, 2, 4),
        attention_resolutions=(8, 16) if family in {"elliptic_static", "temporal_endpoint_heavy"} else (16,),
        with_fourier_features=pde in {"darcy", "helmholtz", "steady_heat_conduction", "wave", "nsnonbounded", "burger"},
        axis_semantics=_axis_semantics(pde),
        notes=_profile_notes(pde, "base"),
    )


def _heavy_profile_config(pde: str, in_channels: int) -> dict[str, Any]:
    family = _recommended_family(pde)
    return _profile_config(
        in_channels,
        profile="heavy",
        architecture_family=family,
        model_channels=192,
        num_res_blocks=4,
        channel_mult=(1, 2, 4, 4),
        attention_resolutions=(8, 16),
        with_fourier_features=pde not in {"poisson", "heat", "advection_diffusion", "reaction_diffusion", "shallow_water"},
        axis_semantics=_axis_semantics(pde),
        notes=_profile_notes(pde, "heavy"),
    )


def _legacy_base_config(in_channels: int, model_channels: int = 128, out_channels: int | None = None) -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=4,
        attention_resolutions=(2,),
        dropout=0.1,
        channel_mult=(1, 2, 4),
        use_scale_shift_norm=True,
        use_new_attention_order=True,
        with_fourier_features=False,
        architecture_family="legacy_base",
        architecture_profile="legacy_base",
        notes="explicit legacy reproduction of the old shared _base_config",
    )


def _legacy_elliptic_config(in_channels: int = 2) -> dict[str, Any]:
    cfg = _legacy_base_config(in_channels)
    cfg.update(
        {
            "attention_resolutions": (32,),
            "channel_mult": (1, 2, 2),
            "use_scale_shift_norm": False,
            "use_new_attention_order": False,
            "notes": "explicit legacy reproduction of the old poisson/helmholtz elliptic config",
        }
    )
    return cfg


def _recommended_family(pde: str) -> str:
    if pde in {"poisson", "heat"}:
        return "light_smooth"
    if pde in {"darcy", "helmholtz", "steady_heat_conduction"}:
        return "elliptic_static"
    if pde in {"advection_diffusion", "reaction_diffusion"}:
        return "temporal_endpoint_base"
    if pde in {"wave", "shallow_water", "nsnonbounded"}:
        return "temporal_endpoint_heavy"
    if pde == "burger":
        return "full_time_space"
    raise ValueError(f"Unknown PDE architecture {pde!r}")


def _axis_semantics(pde: str) -> str:
    return "BCHW_as_time_space_H_time_W_space" if pde == "burger" else "spatial_2d"


def _profile_notes(pde: str, profile: str) -> str | None:
    if pde == "burger":
        return f"{profile} profile preserves Burgers BCHW time-space axis semantics"
    if pde == "shallow_water":
        return f"{profile} profile keeps scalar FiLM conditioning disabled; conservative variables stay in native channels"
    return None


def _with_scalar_metadata(pde: str, cfg: dict[str, Any]) -> dict[str, Any]:
    spec = get_pde_spec(pde)
    out = deepcopy(cfg)
    out["scalar_conditioning"] = False
    out["scalar_conditioning_params"] = tuple(spec.scalar_param_names)
    return out


def _build_recommended_configs() -> dict[str, dict[str, Any]]:
    return {
        "poisson": _with_scalar_metadata("poisson", _light_smooth_config(2)),
        "heat": _with_scalar_metadata("heat", _light_smooth_config(2)),
        "darcy": _with_scalar_metadata("darcy", _elliptic_static_config(2)),
        "helmholtz": _with_scalar_metadata("helmholtz", _elliptic_static_config(2)),
        "steady_heat_conduction": _with_scalar_metadata("steady_heat_conduction", _elliptic_static_config(2)),
        "advection_diffusion": _with_scalar_metadata("advection_diffusion", _temporal_endpoint_base_config(2)),
        "reaction_diffusion": _with_scalar_metadata("reaction_diffusion", _temporal_endpoint_base_config(4)),
        "wave": _with_scalar_metadata("wave", _wave_config(4)),
        "shallow_water": _with_scalar_metadata("shallow_water", _shallow_water_config(6)),
        "nsnonbounded": _with_scalar_metadata("nsnonbounded", _ns_config(2)),
        "burger": _with_scalar_metadata("burger", _burger_time_space_config(1)),
    }


def _build_profile_configs(profile: str) -> dict[str, dict[str, Any]]:
    builders = {
        "light": _light_profile_config,
        "base": _base_profile_config,
        "heavy": _heavy_profile_config,
    }
    if profile not in builders:
        raise ValueError(f"Unknown model profile {profile!r}")
    return {
        pde: _with_scalar_metadata(pde, builders[profile](pde, spec.img_channels))
        for pde, spec in PDE_DATA_SPECS.items()
    }


def _build_legacy_configs() -> dict[str, dict[str, Any]]:
    configs = {
        pde: _with_scalar_metadata(pde, _legacy_base_config(spec.img_channels))
        for pde, spec in PDE_DATA_SPECS.items()
    }
    configs["poisson"] = _with_scalar_metadata("poisson", _legacy_elliptic_config(2))
    configs["helmholtz"] = _with_scalar_metadata("helmholtz", _legacy_elliptic_config(2))
    return configs


MODEL_CONFIGS_RECOMMENDED = _build_recommended_configs()
MODEL_CONFIGS_LIGHT = _build_profile_configs("light")
MODEL_CONFIGS_BASE = _build_profile_configs("base")
MODEL_CONFIGS_HEAVY = _build_profile_configs("heavy")
MODEL_CONFIGS_LEGACY_BASE = _build_legacy_configs()

MODEL_CONFIGS_BY_PROFILE: dict[str, dict[str, dict[str, Any]]] = {
    "recommended": MODEL_CONFIGS_RECOMMENDED,
    "light": MODEL_CONFIGS_LIGHT,
    "base": MODEL_CONFIGS_BASE,
    "heavy": MODEL_CONFIGS_HEAVY,
    "legacy_base": MODEL_CONFIGS_LEGACY_BASE,
}

# Backwards-compatible registry name for callers/tests that enumerate PDE names.
MODEL_CONFIGS = MODEL_CONFIGS_RECOMMENDED

ARCHITECTURE_FAMILIES = {
    pde: cfg["architecture_family"] for pde, cfg in MODEL_CONFIGS_RECOMMENDED.items()
}


def get_model_config(
    architecture: str,
    profile: str = "recommended",
    in_channels: int | None = None,
    out_channels: int | None = None,
) -> dict[str, Any]:
    if profile not in MODEL_CONFIGS_BY_PROFILE:
        raise ValueError(f"model_profile={profile!r} is invalid; expected one of {MODEL_PROFILE_CHOICES}")
    profile_configs = MODEL_CONFIGS_BY_PROFILE[profile]
    if architecture not in profile_configs:
        raise ValueError(f"Model architecture {architecture!r} is missing its {profile!r} config.")

    cfg = deepcopy(profile_configs[architecture])
    if in_channels is not None:
        cfg["in_channels"] = int(in_channels)
    if out_channels is not None:
        cfg["out_channels"] = int(out_channels)
    if "out_channels" not in cfg or cfg["out_channels"] is None:
        cfg["out_channels"] = int(cfg["in_channels"])
    cfg["architecture_profile"] = profile
    if architecture in PDE_DATA_SPECS:
        cfg["scalar_conditioning_params"] = tuple(get_pde_spec(architecture).scalar_param_names)
    return cfg


def model_constructor_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only keys accepted by UNetModel."""
    return {key: deepcopy(value) for key, value in config.items() if key in MODEL_PARAMETER_KEYS}


def get_model_config_metadata(
    architecture: str,
    profile: str = "recommended",
    in_channels: int | None = None,
    out_channels: int | None = None,
) -> dict[str, Any]:
    cfg = get_model_config(
        architecture,
        profile=profile,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    return _jsonable_model_config(cfg)


def _jsonable_model_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, list):
            out[key] = [_jsonable_model_value(item) for item in value]
        elif isinstance(value, dict):
            out[key] = {str(child_key): _jsonable_model_value(child) for child_key, child in value.items()}
        else:
            out[key] = _jsonable_model_value(value)
    return out


def _jsonable_model_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_jsonable_model_value(child) for child in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_model_value(child) for key, child in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


MODEL_ARCHITECTURE_METADATA = {
    pde: get_model_config_metadata(pde, profile="recommended")
    for pde in MODEL_CONFIGS_RECOMMENDED
}


def instantiate_model(
    architechture: str,
    use_ema: bool,
    in_channels: int | None = None,
    out_channels: int | None = None,
    num_classes: int | None = None,
    profile: str = "recommended",
    model_config: Mapping[str, Any] | None = None,
) -> Union[UNetModel, DiscreteUNetModel, EMA]:
    if model_config is not None:
        cfg = deepcopy(dict(model_config))
        if in_channels is not None:
            cfg["in_channels"] = int(in_channels)
        if out_channels is not None:
            cfg["out_channels"] = int(out_channels)
    else:
        cfg = get_model_config(
            architechture,
            profile=profile,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    if num_classes is not None:
        cfg["num_classes"] = int(num_classes)

    model = UNetModel(**model_constructor_config(cfg))
    return EMA(model=model) if use_ema else model

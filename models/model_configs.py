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
from models.unet import (
    UNetModel,
    coordinate_fourier_feature_channel_count,
    fourier_feature_channel_count,
)


MODEL_PROFILE_CHOICES = ("recommended", "light", "base", "heavy")

MODEL_METADATA_KEYS = {
    "architecture_family",
    "architecture_profile",
    "axis_semantics",
    "scalar_conditioning",
    "scalar_conditioning_dim",
    "scalar_conditioning_params",
    "fourier_feature_type",
    "fourier_feature_notes",
    "with_value_fourier_features",
    "with_coordinate_fourier_features",
    "value_fourier_feature_channels",
    "coordinate_fourier_feature_channels",
    "fourier_feature_channels",
    "effective_in_channels",
    "coordinate_fourier_coord_range",
    "coordinate_fourier_include_raw_coords",
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
    "use_scale_shift_norm",
    "resblock_updown",
    "with_value_fourier_features",
    "with_coordinate_fourier_features",
    "fourier_feature_start",
    "fourier_feature_stop",
    "fourier_feature_step",
    "coordinate_fourier_start",
    "coordinate_fourier_stop",
    "coordinate_fourier_step",
    "coordinate_fourier_include_raw_coords",
    "coordinate_fourier_coord_range",
    "ignore_time",
    "input_projection",
    "scalar_conditioning",
    "scalar_conditioning_dim",
    "image_size",
}


RECOMMENDED_VALUE_FOURIER_PDES = {"helmholtz", "wave", "nsnonbounded", "burger"}
RECOMMENDED_COORDINATE_FOURIER_PDES = {
    "darcy",
    "helmholtz",
    "steady_heat_conduction",
    "advection_diffusion",
    "wave",
    "burger",
}


def _fourier_feature_type(
    with_value_fourier_features: bool,
    with_coordinate_fourier_features: bool,
) -> str:
    if with_value_fourier_features and with_coordinate_fourier_features:
        return "value_and_coordinate"
    if with_value_fourier_features:
        return "value"
    if with_coordinate_fourier_features:
        return "coordinate"
    return "none"


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
    with_value_fourier_features: bool = False,
    with_coordinate_fourier_features: bool = False,
    coordinate_fourier_start: int = 0,
    coordinate_fourier_stop: int = 4,
    coordinate_fourier_step: int = 1,
    coordinate_fourier_include_raw_coords: bool = True,
    coordinate_fourier_coord_range: str = "unit",
    architecture_family: str = "base",
    axis_semantics: str = "spatial_2d",
    architecture_profile: str = "recommended",
    notes: str | None = None,
) -> dict[str, Any]:
    value_fourier_enabled = bool(with_value_fourier_features)
    coordinate_fourier_enabled = bool(with_coordinate_fourier_features)
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
        "use_scale_shift_norm": bool(use_scale_shift_norm),
        "resblock_updown": False,
        "with_value_fourier_features": value_fourier_enabled,
        "with_coordinate_fourier_features": coordinate_fourier_enabled,
        "fourier_feature_start": 6,
        "fourier_feature_stop": 8,
        "fourier_feature_step": 1,
        "coordinate_fourier_start": int(coordinate_fourier_start),
        "coordinate_fourier_stop": int(coordinate_fourier_stop),
        "coordinate_fourier_step": int(coordinate_fourier_step),
        "coordinate_fourier_include_raw_coords": bool(coordinate_fourier_include_raw_coords),
        "coordinate_fourier_coord_range": str(coordinate_fourier_coord_range),
        "architecture_family": architecture_family,
        "axis_semantics": axis_semantics,
        "architecture_profile": architecture_profile,
        "scalar_conditioning": False,
        "scalar_conditioning_dim": 0,
        "scalar_conditioning_params": (),
        "fourier_feature_type": _fourier_feature_type(
            value_fourier_enabled, coordinate_fourier_enabled
        ),
        "fourier_feature_notes": "Value and coordinate Fourier features are controlled independently.",
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
        with_value_fourier_features=False,
        with_coordinate_fourier_features=False,
        architecture_family="light_smooth",
        architecture_profile=profile,
    )


def _elliptic_static_config(
    in_channels: int,
    *,
    profile: str = "recommended",
    with_value_fourier_features: bool = False,
    with_coordinate_fourier_features: bool = True,
    notes: str | None = None,
) -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=128,
        num_res_blocks=4,
        dropout=0.05,
        channel_mult=(1, 2, 4),
        attention_resolutions=(8, 16),
        with_value_fourier_features=with_value_fourier_features,
        with_coordinate_fourier_features=with_coordinate_fourier_features,
        use_scale_shift_norm=True,
        architecture_family="elliptic_static",
        architecture_profile=profile,
        notes=notes,
    )


def _temporal_endpoint_base_config(
    in_channels: int,
    *,
    profile: str = "recommended",
    with_value_fourier_features: bool = False,
    with_coordinate_fourier_features: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=128,
        num_res_blocks=4,
        dropout=0.05,
        channel_mult=(1, 2, 4),
        attention_resolutions=(16,),
        with_value_fourier_features=with_value_fourier_features,
        with_coordinate_fourier_features=with_coordinate_fourier_features,
        architecture_family="temporal_endpoint_base",
        architecture_profile=profile,
        notes=notes,
    )


def _temporal_endpoint_heavy_config(
    in_channels: int,
    *,
    profile: str = "recommended",
    with_value_fourier_features: bool,
    with_coordinate_fourier_features: bool,
    notes: str | None = None,
) -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=192,
        num_res_blocks=4,
        dropout=0.05,
        channel_mult=(1, 2, 4, 4),
        attention_resolutions=(8, 16),
        with_value_fourier_features=with_value_fourier_features,
        with_coordinate_fourier_features=with_coordinate_fourier_features,
        architecture_family="temporal_endpoint_heavy",
        architecture_profile=profile,
        notes=notes,
    )


def _wave_config(in_channels: int = 4, *, profile: str = "recommended") -> dict[str, Any]:
    return _temporal_endpoint_heavy_config(
        in_channels,
        profile=profile,
        with_value_fourier_features=True,
        with_coordinate_fourier_features=True,
        notes="wave profile enables value and coordinate Fourier features for oscillatory phase/frequency structure",
    )


def _shallow_water_config(in_channels: int = 6, *, profile: str = "recommended") -> dict[str, Any]:
    return _temporal_endpoint_heavy_config(
        in_channels,
        profile=profile,
        with_value_fourier_features=False,
        with_coordinate_fourier_features=False,
        notes=(
            "shallow-water profile keeps Fourier features off initially because conservative "
            "variables can be scale-sensitive"
        ),
    )


def _ns_config(in_channels: int = 2, *, profile: str = "recommended") -> dict[str, Any]:
    return _temporal_endpoint_heavy_config(
        in_channels,
        profile=profile,
        with_value_fourier_features=True,
        with_coordinate_fourier_features=False,
        notes=(
            "nsnonbounded profile keeps coordinate Fourier disabled to preserve periodic "
            "translation-equivariance while retaining value Fourier for vorticity value lift"
        ),
    )


def _burger_time_space_config(in_channels: int = 1, *, profile: str = "recommended") -> dict[str, Any]:
    return _base_config(
        in_channels,
        model_channels=128,
        num_res_blocks=4,
        dropout=0.05,
        channel_mult=(1, 2, 4),
        attention_resolutions=(16,),
        with_value_fourier_features=True,
        with_coordinate_fourier_features=True,
        architecture_family="full_time_space",
        axis_semantics="BCHW_as_time_space_H_time_W_space",
        architecture_profile=profile,
        notes="burger profile uses value Fourier plus coordinate Fourier with H=time and W=space",
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
    with_value_fourier_features: bool,
    with_coordinate_fourier_features: bool,
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
        with_value_fourier_features=with_value_fourier_features,
        with_coordinate_fourier_features=with_coordinate_fourier_features,
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
        with_value_fourier_features=pde in {"burger"},
        with_coordinate_fourier_features=pde in {
            "burger",
            "darcy",
            "helmholtz",
            "steady_heat_conduction",
            "advection_diffusion",
        },
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
        with_value_fourier_features=_recommended_value_fourier(pde),
        with_coordinate_fourier_features=_recommended_coordinate_fourier(pde),
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
        with_value_fourier_features=_recommended_value_fourier(pde),
        with_coordinate_fourier_features=_recommended_coordinate_fourier(pde),
        axis_semantics=_axis_semantics(pde),
        notes=_profile_notes(pde, "heavy"),
    )


def _recommended_family(pde: str) -> str:
    if pde == "poisson":
        return "light_smooth"
    if pde in {"heat", "advection_diffusion", "reaction_diffusion"}:
        return "temporal_endpoint_base"
    if pde in {"darcy", "helmholtz", "steady_heat_conduction"}:
        return "elliptic_static"
    if pde in {"wave", "shallow_water", "nsnonbounded"}:
        return "temporal_endpoint_heavy"
    if pde == "burger":
        return "full_time_space"
    raise ValueError(f"Unknown PDE architecture {pde!r}")


def _axis_semantics(pde: str) -> str:
    return "BCHW_as_time_space_H_time_W_space" if pde == "burger" else "spatial_2d"


def _recommended_value_fourier(pde: str) -> bool:
    return pde in RECOMMENDED_VALUE_FOURIER_PDES


def _recommended_coordinate_fourier(pde: str) -> bool:
    return pde in RECOMMENDED_COORDINATE_FOURIER_PDES


def _profile_notes(pde: str, profile: str) -> str | None:
    if pde == "burger":
        return f"{profile} profile preserves Burgers BCHW time-space axis semantics with coordinate Fourier"
    if pde == "nsnonbounded":
        return (
            f"{profile} profile keeps coordinate Fourier disabled to preserve periodic "
            "translation-equivariance; value Fourier remains enabled for vorticity value lift"
        )
    if pde == "shallow_water":
        return f"{profile} profile keeps scalar FiLM conditioning disabled; conservative variables stay in native channels"
    return None


def _with_scalar_metadata(pde: str, cfg: dict[str, Any]) -> dict[str, Any]:
    spec = get_pde_spec(pde)
    out = deepcopy(cfg)
    out["scalar_conditioning"] = False
    out["scalar_conditioning_dim"] = 0
    out["scalar_conditioning_params"] = tuple(spec.scalar_param_names)
    return out


def _build_recommended_configs() -> dict[str, dict[str, Any]]:
    return {
        "poisson": _with_scalar_metadata("poisson", _light_smooth_config(2)),
        "heat": _with_scalar_metadata(
            "heat",
            _temporal_endpoint_base_config(
                2,
                with_value_fourier_features=False,
                with_coordinate_fourier_features=False,
            ),
        ),
        "darcy": _with_scalar_metadata(
            "darcy",
            _elliptic_static_config(
                2,
                with_value_fourier_features=False,
                with_coordinate_fourier_features=True,
            ),
        ),
        "helmholtz": _with_scalar_metadata(
            "helmholtz",
            _elliptic_static_config(
                2,
                with_value_fourier_features=True,
                with_coordinate_fourier_features=True,
            ),
        ),
        "steady_heat_conduction": _with_scalar_metadata(
            "steady_heat_conduction",
            _elliptic_static_config(
                2,
                with_value_fourier_features=False,
                with_coordinate_fourier_features=True,
            ),
        ),
        "advection_diffusion": _with_scalar_metadata(
            "advection_diffusion",
            _temporal_endpoint_base_config(
                2,
                with_value_fourier_features=False,
                with_coordinate_fourier_features=True,
            ),
        ),
        "reaction_diffusion": _with_scalar_metadata(
            "reaction_diffusion",
            _temporal_endpoint_base_config(
                4,
                with_value_fourier_features=False,
                with_coordinate_fourier_features=False,
            ),
        ),
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


MODEL_CONFIGS_RECOMMENDED = _build_recommended_configs()
MODEL_CONFIGS_LIGHT = _build_profile_configs("light")
MODEL_CONFIGS_BASE = _build_profile_configs("base")
MODEL_CONFIGS_HEAVY = _build_profile_configs("heavy")

MODEL_CONFIGS_BY_PROFILE: dict[str, dict[str, dict[str, Any]]] = {
    "recommended": MODEL_CONFIGS_RECOMMENDED,
    "light": MODEL_CONFIGS_LIGHT,
    "base": MODEL_CONFIGS_BASE,
    "heavy": MODEL_CONFIGS_HEAVY,
}

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
    _normalize_fourier_config(cfg)
    _normalize_scalar_conditioning(cfg)
    if architecture in PDE_DATA_SPECS:
        if not cfg.get("scalar_conditioning", False):
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
    return model_config_metadata_from_config(cfg)


def model_config_metadata_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    canonical_keys = MODEL_PARAMETER_KEYS | MODEL_METADATA_KEYS
    cfg = {
        key: deepcopy(value)
        for key, value in config.items()
        if key in canonical_keys
    }
    _normalize_fourier_config(cfg)
    _normalize_scalar_conditioning(cfg)
    _add_derived_fourier_metadata(cfg)
    return _jsonable_model_config(cfg)


def _normalize_fourier_config(cfg: dict[str, Any]) -> None:
    value_enabled = bool(cfg.get("with_value_fourier_features", False))
    coordinate_enabled = bool(cfg.get("with_coordinate_fourier_features", False))
    cfg["with_value_fourier_features"] = value_enabled
    cfg["with_coordinate_fourier_features"] = coordinate_enabled
    cfg.setdefault("fourier_feature_start", 6)
    cfg.setdefault("fourier_feature_stop", 8)
    cfg.setdefault("fourier_feature_step", 1)
    cfg.setdefault("coordinate_fourier_start", 0)
    cfg.setdefault("coordinate_fourier_stop", 4)
    cfg.setdefault("coordinate_fourier_step", 1)
    cfg.setdefault("coordinate_fourier_include_raw_coords", True)
    cfg.setdefault("coordinate_fourier_coord_range", "unit")
    cfg["fourier_feature_type"] = _fourier_feature_type(value_enabled, coordinate_enabled)
    cfg.setdefault(
        "fourier_feature_notes",
        "Value and coordinate Fourier features are controlled independently.",
    )


def _normalize_scalar_conditioning(cfg: dict[str, Any]) -> None:
    enabled = bool(cfg.get("scalar_conditioning", False))
    cfg["scalar_conditioning"] = enabled
    params = cfg.get("scalar_conditioning_params", ())
    if params is None:
        params = ()
    if isinstance(params, str):
        params = (params,)
    else:
        params = tuple(params)
    cfg["scalar_conditioning_params"] = params
    if enabled:
        dim = int(cfg.get("scalar_conditioning_dim") or len(params))
        cfg["scalar_conditioning_dim"] = dim
    else:
        cfg["scalar_conditioning_dim"] = int(cfg.get("scalar_conditioning_dim") or 0)


def _add_derived_fourier_metadata(cfg: dict[str, Any]) -> None:
    data_in_channels = int(cfg.get("in_channels", 0) or 0)
    value_channels = 0
    if cfg.get("with_value_fourier_features", False):
        value_channels = fourier_feature_channel_count(
            data_in_channels,
            start=int(cfg.get("fourier_feature_start", 6)),
            stop=int(cfg.get("fourier_feature_stop", 8)),
            step=int(cfg.get("fourier_feature_step", 1)),
        )
    coordinate_channels = 0
    if cfg.get("with_coordinate_fourier_features", False):
        coordinate_channels = coordinate_fourier_feature_channel_count(
            spatial_dims=2,
            start=int(cfg.get("coordinate_fourier_start", 0)),
            stop=int(cfg.get("coordinate_fourier_stop", 4)),
            step=int(cfg.get("coordinate_fourier_step", 1)),
            include_raw_coords=bool(cfg.get("coordinate_fourier_include_raw_coords", True)),
        )
    cfg["value_fourier_feature_channels"] = value_channels
    cfg["coordinate_fourier_feature_channels"] = coordinate_channels
    cfg["fourier_feature_channels"] = value_channels + coordinate_channels
    cfg["effective_in_channels"] = data_in_channels + value_channels + coordinate_channels
    cfg["coordinate_fourier_coord_range"] = str(cfg.get("coordinate_fourier_coord_range", "unit"))
    cfg["coordinate_fourier_include_raw_coords"] = bool(
        cfg.get("coordinate_fourier_include_raw_coords", True)
    )


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
    ema_decay: float = 0.999,
    ema_warmup: bool = True,
) -> Union[UNetModel, DiscreteUNetModel, EMA]:
    if model_config is not None:
        cfg = deepcopy(dict(model_config))
        if in_channels is not None:
            cfg["in_channels"] = int(in_channels)
        if out_channels is not None:
            cfg["out_channels"] = int(out_channels)
        _normalize_fourier_config(cfg)
        _normalize_scalar_conditioning(cfg)
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
    return EMA(model=model, decay=ema_decay, warmup=ema_warmup) if use_ema else model

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from copy import deepcopy
from typing import Any, Union

from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.unet import UNetModel


def _base_config(in_channels: int, model_channels: int = 128, out_channels: int | None = None) -> dict[str, Any]:
    return {
        "in_channels": in_channels,
        "model_channels": model_channels,
        "out_channels": out_channels if out_channels is not None else in_channels,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.1,
        "channel_mult": (1, 2, 4),
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    }


def _elliptic_config(in_channels: int = 2) -> dict[str, Any]:
    cfg = _base_config(in_channels)
    cfg.update(
        {
            "attention_resolutions": (32,),
            "channel_mult": (1, 2, 2),
            "use_scale_shift_norm": False,
            "use_new_attention_order": False,
        }
    )
    return cfg


MODEL_CONFIGS = {
    "darcy": _base_config(2),
    "poisson": _elliptic_config(2),
    "helmholtz": _elliptic_config(2),
    "nsnonbounded": _base_config(2),
    "burger": _base_config(1),
    "reaction_diffusion": _base_config(4),
    "shallow_water": _base_config(6),
    "heat": _base_config(2),
    "wave": _base_config(4),
    "advection_diffusion": _base_config(2),
    "steady_heat_conduction": _base_config(2),
}


def instantiate_model(
    architechture: str,
    use_ema: bool,
    in_channels: int | None = None,
    out_channels: int | None = None,
    num_classes: int | None = None,
) -> Union[UNetModel, DiscreteUNetModel, EMA]:
    if architechture not in MODEL_CONFIGS:
        raise ValueError(f"Model architecture {architechture!r} is missing its config.")

    cfg = deepcopy(MODEL_CONFIGS[architechture])
    if in_channels is not None:
        cfg["in_channels"] = int(in_channels)
    if out_channels is not None:
        cfg["out_channels"] = int(out_channels)
    if num_classes is not None:
        cfg["num_classes"] = int(num_classes)

    model = UNetModel(**cfg)
    return EMA(model=model) if use_ema else model

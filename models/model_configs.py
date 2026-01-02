# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.unet import UNetModel

MODEL_CONFIGS = {
    "darcy": {
        "in_channels": 2,
        "model_channels": 128,
        "out_channels": 2,
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
    },
    "poisson": {
        "in_channels": 2,
        "model_channels": 128,
        "out_channels": 2,
        "num_res_blocks": 4,
        "attention_resolutions": (32,),
        "dropout": 0.1,
        "channel_mult": (1, 2, 2),
        "conv_resample": True,
        "dims": 2,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": False,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
    },
    "nsnonbounded": {
        "in_channels": 2,
        "model_channels": 128,
        "out_channels": 2,
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
    },
    "burger": {
        "in_channels": 1,
        "model_channels": 128,
        "out_channels": 1,
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
    },
    "helmholtz": {
        "in_channels": 2,
        "model_channels": 128,
        "out_channels": 2,
        "num_res_blocks": 4,
        "attention_resolutions": (32,),
        "dropout": 0.1,
        "channel_mult": (1, 2, 2),
        "conv_resample": True,
        "dims": 2,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": False,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
    },
    "poisson-helmholtz": {
        "in_channels": 2,
        "model_channels": 128,
        "out_channels": 2,
        "num_res_blocks": 4,
        "attention_resolutions": (32,),
        "dropout": 0.1,
        "channel_mult": (1, 2, 2),
        "conv_resample": True,
        "dims": 2,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": False,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "with_fourier_features": False,
    },
    "reaction_diffusion": {
        "in_channels": 4,
        "model_channels": 256,
        "out_channels": 4,
        "num_res_blocks": 4,
        "attention_resolutions": (32, ),
        "dropout": 0.1,
        "channel_mult": (1, 2, 4, 8),
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "shallow_water": {
        "in_channels": 6,
        "model_channels": 128,
        "out_channels": 6,
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
    },

}



def instantiate_model(
    architechture: str, is_discrete: bool, use_ema: bool
) -> Union[UNetModel, DiscreteUNetModel]:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    if is_discrete:
        if architechture + "_discrete" in MODEL_CONFIGS:
            config = MODEL_CONFIGS[architechture + "_discrete"]
        else:
            config = MODEL_CONFIGS[architechture]
        model = DiscreteUNetModel(
            vocab_size=257,
            **config,
        )
    else:
        model = UNetModel(**MODEL_CONFIGS[architechture])

    if use_ema:
        return EMA(model=model)
    else:
        return model

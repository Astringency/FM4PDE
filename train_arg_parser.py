# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

from models.model_configs import MODEL_CONFIGS

logger = logging.getLogger(__name__)


class DatasetChoices:
    def __init__(self, choices):
        self.choices = tuple(choices)

    def __contains__(self, value):
        return all(part in self.choices for part in str(value).split("-"))

    def __iter__(self):
        return iter(self.choices)

    def __repr__(self):
        return repr(self.choices)


def get_args_parser():
    parser = argparse.ArgumentParser("PDE Network Training", add_help=False)

    # Main Training parameters
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--epochs", 
        default=921, 
        type=int
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--optimizer_betas",
        nargs="+",
        type=float,
        default=[0.9, 0.999],
        help="[beta1, beta2] for AdamW",
    )
    parser.add_argument(
        "--decay_lr",
        action="store_true",
        help="Adds a linear decay to the lr during training.",
    )
    parser.add_argument(
        "--skewed_timesteps",
        action="store_true",
        help="Use skewed timestep sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--edm_schedule",
        action="store_true",
        help="Use the alternative time discretization during sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="When evaluating, use the model Exponential Moving Average weights.",
    )
    parser.add_argument(
        "--eval_frequency",
        default=50,
        type=int,
        help="Frequency (in number of epochs) for running FID evaluation. -1 to never run evaluation.",
    )
    parser.add_argument(
        "--cfg_scale",
        default=0.2,
        type=float,
        help="Classifier-free guidance scale for generating samples.",
    )
    parser.add_argument(
        "--class_drop_prob",
        type=float,
        default=0.2,
        help="Probability to drop conditioning during training",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        help="Optional global gradient norm clipping threshold.",
    )
    parser.add_argument(
        "--normalization_eps",
        type=float,
        default=1e-6,
        help="Epsilon for channel-wise training-set standardization.",
    )
    parser.add_argument(
        "--sampling_dtype",
        default="float32",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Autocast dtype used during training on CUDA.",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        default=list(MODEL_CONFIGS.keys())[0],
        type=str,
        choices=DatasetChoices(MODEL_CONFIGS.keys()),
        help="PDE to solve.",
    )
    parser.add_argument(
        "--data_path",
        default="/large_storage/zhangxf/PDEdata/",
        type=str,
        help="data root folder with train, val and test subfolders",
    )
    parser.add_argument(
        "--data_size",
        default=5,
        type=int,
        help="Number of training shards/files to read per PDE.",
    )
    parser.add_argument(
        "--max_train_samples",
        default=None,
        type=int,
        help="Optional cap on loaded training samples per PDE for smoke tests.",
    )
    parser.add_argument(
        "--rd_init_mode_filter",
        default=None,
        choices=["grf", "iid"],
        help="Optional reaction-diffusion train-file init mode filter. Use grf or iid to avoid mixing RD train files.",
    )
    parser.add_argument(
        "--save_full_pde_params",
        action="store_true",
        help="Save full sample-aligned future PDE scalar parameters to output_dir/pde_params.pt.",
    )
    parser.add_argument(
        "--output_dir",
        default="/research_data/users/zhangxifeng/C01Python/FM4PDE/output/pretrained/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)

    # Checkpoint and Resume training
    parser.add_argument("--resume", default="", help="resume from checkpoint which is a path to the checkpoint file")
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start epoch (used when resumed from checkpoint)",
    )

    # Distributed training parameters
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Only run one batch of training and evaluation.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Compatibility flag for one-pass evaluation-style runs.",
    )
    

    return parser

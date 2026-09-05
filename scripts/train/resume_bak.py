#!/usr/bin/env python3
"""Resume a bak checkpoint using its saved training settings and current I/O.

Example: python -m scripts.train.resume_bak --checkpoint outputs/pretrained/bak/fm4poisson.pth
  --additional-epochs 10 --data_path /path/to/original/PDEdata --batch_size 1
  --model_gradient_checkpointing --output_dir outputs/train/bak_poisson
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from models.legacy_checkpoint import read_checkpoint
from train_arg_parser import get_args_parser


def resolve_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--checkpoint", required=True)
    selector.add_argument("--additional-epochs", type=int, default=10)
    selector.add_argument("--print-config", action="store_true")
    options, remaining = selector.parse_known_args(argv)
    if options.additional_epochs <= 0:
        raise ValueError("--additional-epochs must be positive")
    checkpoint = read_checkpoint(options.checkpoint)
    if checkpoint.get("model_profile") != "legacy":
        raise ValueError("resume_bak expects a legacy architecture checkpoint")
    saved = checkpoint["args"]
    saved = saved if isinstance(saved, dict) else vars(saved)
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    defaults = {key: saved[key] for key in (
        "batch_size", "accum_iter", "lr", "optimizer_betas", "dataset", "data_size",
        "seed", "skewed_timesteps", "use_ema", "class_drop_prob",
    ) if key in saved}
    defaults.update(
        resume=options.checkpoint, model_profile="auto",
        epochs=int(checkpoint["epoch"]) + 1 + options.additional_epochs,
        lr_scheduler=checkpoint.get("resolved_lr_scheduler", "linear"),
        min_lr=1e-8, sampling_dtype="float16", legacy_full_training_set=True,
        output_dir=f"outputs/train/bak_{saved['dataset']}",
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    if args.dataset != saved["dataset"] or args.use_ema:
        raise ValueError("Legacy resume must retain dataset and use_ema=False")
    if args.epochs <= checkpoint["epoch"] + 1:
        raise ValueError("--epochs is the total epoch count and must exceed the completed count")
    old_effective = int(saved["batch_size"]) * int(saved["accum_iter"]) * int(saved.get("world_size", 1))
    runtime_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not any(arg == "--accum_iter" or arg.startswith("--accum_iter=") for arg in remaining):
        args.accum_iter = math.ceil(old_effective / (args.batch_size * runtime_world_size))
    args.legacy_resume_settings = {
        "checkpoint": options.checkpoint, "completed_epochs": checkpoint["epoch"] + 1,
        "saved_effective_batch_size": old_effective,
        "runtime_effective_batch_size": args.batch_size * args.accum_iter * runtime_world_size,
        "normalizer_source": "original training pool Min-Max; legacy checkpoint has no saved statistics",
        "learning_rate_policy": "restore optimizer and original scheduler state (completed schedules stay at their final LR)",
        "validation_policy": "overlapping training diagnostic, not held-out validation",
    }
    return args, options.print_config


def main(argv=None):
    args, print_only = resolve_args(argv)
    print(json.dumps(vars(args), indent=2, default=str), flush=True)
    if print_only:
        return
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    from train import main as train_main
    train_main(args)


if __name__ == "__main__":
    main()

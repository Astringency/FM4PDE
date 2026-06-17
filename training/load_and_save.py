# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from training.distributed_mode import is_main_process


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(
    args,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    lr_schedule,
    loss_scaler,
    final=False,
    normalizer=None,
    data_shape: tuple[int, ...] | None = None,
    num_channels: int | None = None,
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    normalizer_state = normalizer.state_dict() if normalizer is not None else None
    payload = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_schedule": lr_schedule.state_dict() if lr_schedule is not None else None,
        "epoch": epoch,
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "args": args,
        "normalizer": normalizer_state,
        "data_shape": tuple(data_shape) if data_shape is not None else None,
        "num_channels": int(num_channels) if num_channels is not None else None,
        "normalization": {
            "type": "channelwise_standardization",
            "eps": getattr(normalizer, "eps", None),
        },
    }

    if loss_scaler is not None:
        checkpoint_paths = [
            output_dir / (f"fm4{args.dataset}-checkpoint-{epoch_name}.pth"),
            output_dir / f"fm4{args.dataset}-checkpoint.pth",
        ]
        for checkpoint_path in checkpoint_paths:
            save_on_master(payload, checkpoint_path)
    else:
        client_state = {"epoch": epoch, "normalizer": normalizer_state}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag=f"fm4{args.dataset}-checkpoint-{epoch_name}",
            client_state=client_state,
        )

    if final:
        if loss_scaler is not None:
            save_on_master(payload, output_dir / f"fm4{args.dataset}.pth")
        else:
            client_state = {"epoch": epoch, "normalizer": normalizer_state}
            model.save_checkpoint(
                save_dir=args.output_dir,
                tag=f"fm4{args.dataset}",
                client_state=client_state,
            )


def load_model(args, model_without_ddp, optimizer, loss_scaler, lr_schedule) -> dict[str, Any] | None:
    if not args.resume:
        return None
    if args.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model_without_ddp.load_state_dict(checkpoint["model"])
    print(f"Resume {args.dataset} checkpoint {args.resume}")
    if "normalizer" not in checkpoint or checkpoint.get("normalizer") is None:
        warnings.warn(
            "Checkpoint has no normalizer; this is a legacy checkpoint. "
            "Training will use the normalizer fitted from the current training data.",
            RuntimeWarning,
            stacklevel=2,
        )
    if (
        "optimizer" in checkpoint
        and checkpoint.get("optimizer") is not None
        and "epoch" in checkpoint
        and not getattr(args, "eval_only", False)
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_schedule" in checkpoint and checkpoint.get("lr_schedule") is not None:
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
        args.start_epoch = checkpoint["epoch"] + 1
        if "scaler" in checkpoint and checkpoint.get("scaler") is not None:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        print("With optim & sched!")
    return checkpoint

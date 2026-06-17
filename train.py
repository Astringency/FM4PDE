# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data.load import PDEloader, TensorDataset
from data.transform import PDEStandardizer
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser
from training import distributed_mode
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process() and args.output_dir:
        args_filepath = Path(args.output_dir) / f"{args.dataset}-args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w", encoding="utf-8") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)

    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    pde_names = args.dataset.split("-")
    data, label, loader_metadata = _load_training_data(pde_names, args.data_path)
    num_channels = int(data.shape[1])
    logger.info(f"Loaded data shape={tuple(data.shape)}, labels dtype={label.dtype}")

    fitted_normalizer = PDEStandardizer.fit(
        data,
        eps=args.normalization_eps,
        channel_names=[f"channel_{idx}" for idx in range(num_channels)],
        pde=args.dataset,
    )

    model_arch = pde_names[0]
    logger.info("Initializing Model")
    model = instantiate_model(
        architechture=model_arch,
        use_ema=args.use_ema,
        in_channels=num_channels,
        out_channels=num_channels,
    )
    model.to(device)

    model_without_ddp = model
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")
    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )

    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()
    checkpoint = load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )
    normalizer = _resolve_normalizer(checkpoint, fitted_normalizer, num_channels)

    if distributed_mode.is_main_process() and args.output_dir:
        normalizer.save(Path(args.output_dir) / "normalizer.pt")

    data_train = normalizer.transform(data)
    dataset_train = TensorDataset(data_train, label)

    logger.info(dataset_train)
    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()

    last_epoch = args.start_epoch - 1
    for epoch in range(args.start_epoch, args.epochs):
        last_epoch = epoch
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=args,
        )

        log_stats = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "num_channels": num_channels,
        }

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, f"{args.dataset}_log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        final_epoch = epoch == args.epochs - 1
        should_save = args.output_dir and (
            final_epoch
            or args.test_run
            or (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
        )
        if should_save:
            save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                loss_scaler=loss_scaler,
                epoch=epoch,
                final=final_epoch or args.test_run,
                normalizer=normalizer,
                data_shape=tuple(data.shape),
                num_channels=num_channels,
            )
            if final_epoch or args.test_run:
                logger.info("Final model saved.")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}; last_epoch={last_epoch}")
    if loader_metadata:
        logger.info(f"Loaded PDE metadata keys: {sorted(loader_metadata)}")


def _load_training_data(pde_names: list[str], data_path: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    dataset_list = []
    label_list = []
    channel_counts = {}
    metadata: dict[str, Any] = {}

    for pde_name in pde_names:
        logger.info(f">>> Initializing Dataset: {pde_name} <<<")
        pde_loader = PDEloader(pde_name)
        dataset, label = pde_loader.load_data(data_path)
        if dataset.ndim != 4:
            raise ValueError(f"{pde_name} loader returned non-BCHW data: {tuple(dataset.shape)}")
        channel_counts[pde_name] = int(dataset.shape[1])
        dataset_list.append(dataset)
        label_list.append(label.long())
        if pde_loader.pde_params:
            metadata[pde_name] = {
                key: value.detach().cpu() for key, value in pde_loader.pde_params.items()
            }

    if len(set(channel_counts.values())) != 1:
        raise ValueError(
            "Joint PDE training requires identical channel counts; "
            f"got {channel_counts}. Padding/masking is intentionally not enabled."
        )

    data = torch.cat(dataset_list, dim=0).to(torch.float32)
    label = torch.cat(label_list, dim=0).to(torch.long)
    return data, label, metadata


def _resolve_normalizer(
    checkpoint: dict[str, Any] | None,
    fitted_normalizer: PDEStandardizer,
    num_channels: int,
) -> PDEStandardizer:
    if checkpoint and checkpoint.get("normalizer") is not None:
        normalizer = PDEStandardizer.from_state_dict(checkpoint["normalizer"])
        _check_normalizer_channels(normalizer, num_channels, "checkpoint normalizer")
        if checkpoint.get("data_shape") is not None and int(checkpoint["data_shape"][1]) != num_channels:
            raise ValueError(
                f"Checkpoint data_shape channel count {checkpoint['data_shape'][1]} "
                f"does not match loaded training data channels {num_channels}"
            )
        return normalizer

    if checkpoint is not None:
        warnings.warn(
            "Using normalizer fitted from current training data because checkpoint has no normalizer.",
            RuntimeWarning,
            stacklevel=2,
        )
    _check_normalizer_channels(fitted_normalizer, num_channels, "fitted normalizer")
    return fitted_normalizer


def _check_normalizer_channels(normalizer: PDEStandardizer, num_channels: int, context: str) -> None:
    got = int(normalizer.mean.shape[1])
    if got != num_channels:
        raise ValueError(f"{context} has {got} channels, but training data has {num_channels}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    training_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    training_desc = (
        f"{training_time}-{args.dataset}-batch{args.batch_size}-"
        f"epoch{args.epochs}-accum{args.accum_iter}-{args.sampling_dtype}/"
    )
    args.output_dir = args.output_dir + training_desc if args.output_dir else ""
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

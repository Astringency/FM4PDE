# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import contextlib
import logging
import math
from typing import Iterable

import torch
from flow_matching.path import CondOTProbPath
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from training.grad_scaler import NativeScalerWithGradNormCount
from training.timesteps import sample_timesteps

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50
TIMESTEP_EPS = 1e-5


class MeanAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: torch.Tensor | float, count: int = 1) -> None:
        if torch.is_tensor(value):
            value = value.detach().item()
        self.total += float(value) * int(count)
        self.count += int(count)

    def compute(self) -> float:
        return self.total / max(self.count, 1)


def skewed_timestep_sample(num_samples: int, device: torch.device, eps: float = TIMESTEP_EPS) -> torch.Tensor:
    p_mean = -1.2
    p_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * p_std + p_mean).exp()
    time = 1 / (1 + sigma)
    return torch.clamp(time, min=eps, max=1.0 - eps)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    model.train(True)
    batch_loss = MeanAccumulator()
    epoch_loss = MeanAccumulator()

    accum_iter = args.accum_iter
    path = CondOTProbPath()

    for data_iter_step, batch in enumerate(data_loader):
        samples, labels, scalar_conditioning = _unpack_batch(batch)
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        if scalar_conditioning is not None:
            scalar_conditioning = scalar_conditioning.to(device, non_blocking=True).float()
        loss = _flow_matching_loss(
            model=model,
            samples=samples,
            labels=labels,
            scalar_conditioning=scalar_conditioning,
            path=path,
            device=device,
            class_drop_prob=args.class_drop_prob,
            skewed_timesteps=args.skewed_timesteps,
            sampling_dtype=getattr(args, "sampling_dtype", "float32"),
            timestep_sampling=getattr(args, 'timestep_sampling', None),
            logit_time_mean=getattr(args, 'logit_time_mean', 0.0),
            logit_time_std=getattr(args, 'logit_time_std', 1.0),
        )

        loss_value = loss.detach().item()
        batch_loss.update(loss_value, int(samples.shape[0]))
        epoch_loss.update(loss_value, int(samples.shape[0]))

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss = loss / accum_iter
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            clip_grad=getattr(args, "clip_grad", None),
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    return {"loss": epoch_loss.compute()}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
):
    was_training = model.training
    model.eval()
    epoch_loss = MeanAccumulator()
    path = CondOTProbPath()

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    rng_context = torch.random.fork_rng(devices=devices)
    rng_context.__enter__()
    torch.manual_seed(int(getattr(args, 'validation_seed', 20260919)) + rank)
    bins = torch.zeros(10, 2, dtype=torch.float64, device=device)
    channel_sums = None
    try:
        for data_iter_step, batch in enumerate(data_loader):
            samples, labels, scalar_conditioning = _unpack_batch(batch)
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            if scalar_conditioning is not None:
                scalar_conditioning = scalar_conditioning.to(device, non_blocking=True).float()
            loss, timestep, per_channel = _flow_matching_loss(
                model=model,
                samples=samples,
                labels=labels,
                scalar_conditioning=scalar_conditioning,
                path=path,
                device=device,
                class_drop_prob=0.0,
                skewed_timesteps=False,
                sampling_dtype=getattr(args, "sampling_dtype", "float32"),
                timestep_sampling='uniform',
                return_details=True,
            )
            index = (timestep * 10).long().clamp_max(9)
            bins[:, 0].scatter_add_(0, index, per_channel.mean(1).double())
            bins[:, 1].scatter_add_(0, index, torch.ones_like(timestep, dtype=torch.float64))
            if channel_sums is None:
                channel_sums = torch.zeros(per_channel.shape[1], device=device, dtype=torch.float64)
            channel_sums += per_channel.double().sum(0)
            loss_value = loss.detach().item()
            if not math.isfinite(loss_value):
                raise ValueError(f"Validation loss is {loss_value}, stopping training")
            epoch_loss.update(loss_value, int(samples.shape[0]))
            if data_iter_step % PRINT_FREQUENCY == 0:
                logger.info(
                    f"Validation epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {epoch_loss.compute()}"
                )
    finally:
        model.train(was_training)
        rng_context.__exit__(None, None, None)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        totals = torch.tensor(
            [epoch_loss.total, float(epoch_loss.count)],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(bins, op=torch.distributed.ReduceOp.SUM)
        channels = torch.tensor(0 if channel_sums is None else channel_sums.numel(), device=device)
        torch.distributed.all_reduce(channels, op=torch.distributed.ReduceOp.MAX)
        if int(channels) > 0:
            if channel_sums is None:
                channel_sums = torch.zeros(int(channels), device=device, dtype=torch.float64)
            torch.distributed.all_reduce(channel_sums, op=torch.distributed.ReduceOp.SUM)
        global_loss = float((totals[0] / totals[1].clamp_min(1.0)).cpu())
    else:
        global_loss = epoch_loss.compute()
    counts = bins[:, 1].cpu().tolist()
    means = (bins[:, 0] / bins[:, 1].clamp_min(1)).cpu().tolist()
    return {"loss": global_loss,
            'loss_by_time_bin': [v if n else None for v, n in zip(means, counts)],
            'time_bin_counts': counts,
            'loss_by_channel': (channel_sums / bins[:, 1].sum().clamp_min(1)).cpu().tolist() if channel_sums is not None else [],
            'time_distribution': 'uniform', 'validation_seed': int(getattr(args, 'validation_seed', 20260919))}


def _flow_matching_loss(
    model: torch.nn.Module,
    samples: torch.Tensor,
    labels: torch.Tensor,
    scalar_conditioning: torch.Tensor | None,
    path: CondOTProbPath,
    device: torch.device,
    class_drop_prob: float,
    skewed_timesteps: bool,
    sampling_dtype: str,
    timestep_sampling: str | None = None,
    logit_time_mean: float = 0.0,
    logit_time_std: float = 1.0,
    return_details: bool = False,
) -> torch.Tensor:
    if samples.ndim != 4:
        raise ValueError(f"Flow matching expects samples [N,C,H,W], got {tuple(samples.shape)}")

    conditioning = _conditioning_for_model(model, labels, class_drop_prob)
    if scalar_conditioning is not None:
        if scalar_conditioning.ndim != 2:
            raise ValueError(
                "scalar_conditioning batch tensor must have shape [N,K], "
                f"got {tuple(scalar_conditioning.shape)}"
            )
        if int(scalar_conditioning.shape[0]) != int(samples.shape[0]):
            raise ValueError(
                "scalar_conditioning batch size must match samples; "
                f"got {int(scalar_conditioning.shape[0])} vs {int(samples.shape[0])}"
            )
        conditioning = dict(conditioning)
        conditioning["scalar_conditioning"] = scalar_conditioning

    noise = torch.randn_like(samples)
    if skewed_timesteps and timestep_sampling not in (None, 'legacy_skewed'):
        raise ValueError('Choose either --skewed_timesteps or --timestep_sampling')
    if skewed_timesteps and timestep_sampling is None:
        t = skewed_timestep_sample(samples.shape[0], device=device)
    else:
        t = sample_timesteps(samples.shape[0], device, timestep_sampling or 'uniform',
                             logit_mean=logit_time_mean, logit_std=logit_time_std)

    path_sample = path.sample(t=t, x_0=noise, x_1=samples)
    x_t = path_sample.x_t
    u_t = path_sample.dx_t

    with _autocast_context(device, sampling_dtype):
        model_out = model(x_t, t, extra=conditioning)
        if model_out.shape != u_t.shape:
            raise ValueError(f"Model output shape {tuple(model_out.shape)} does not match target {tuple(u_t.shape)}")
        error = torch.pow(model_out - u_t, 2)
        loss = error.mean()
        if return_details:
            return loss, t.detach(), error.detach().mean((2, 3))
        return loss


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(batch, (tuple, list)):
        raise ValueError(f"Expected data loader batch tuple/list, got {type(batch).__name__}")
    if len(batch) == 2:
        samples, labels = batch
        return samples, labels, None
    if len(batch) == 3:
        samples, labels, scalar_conditioning = batch
        return samples, labels, scalar_conditioning
    raise ValueError(f"Expected data loader batch of length 2 or 3, got {len(batch)}")


def _conditioning_for_model(model: torch.nn.Module, labels: torch.Tensor, class_drop_prob: float) -> dict[str, torch.Tensor]:
    num_classes = _model_num_classes(model)
    if num_classes is None:
        return {}
    labels = labels.long()
    if bool(torch.any((labels < 0) | (labels >= int(num_classes))).detach().cpu()):
        raise ValueError(
            f"Class labels must be contiguous in [0, {int(num_classes) - 1}] for joint PDE training"
        )
    if not 0.0 <= float(class_drop_prob) <= 1.0:
        raise ValueError("class_drop_prob must be in [0, 1]")
    if class_drop_prob > 0:
        drop = torch.rand(labels.shape, device=labels.device) < float(class_drop_prob)
        labels = torch.where(drop, torch.full_like(labels, int(num_classes)), labels)
    return {"label": labels}


def _model_num_classes(model: torch.nn.Module) -> int | None:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    if isinstance(module, EMA):
        module = module.model
    return getattr(module, "num_classes", None)


def _autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = _autocast_dtype(dtype_name)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _autocast_dtype(dtype_name: str) -> torch.dtype | None:
    normalized = dtype_name.lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return None
    raise ValueError(f"Unsupported sampling_dtype={dtype_name!r}")

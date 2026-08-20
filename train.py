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
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data.load import PDEloader, TensorDataset
from data.metadata import detach_pde_params, summarize_pde_params
from data.scalar_conditioning import (
    fit_scalar_conditioning,
    normalize_scalar_conditioning_params,
    scalar_conditioning_metadata_disabled,
    scalar_conditioning_params_from_config,
    scalar_conditioning_tensor_from_metadata,
    standardize_scalar_conditioning,
)
from data.specs import get_pde_spec
from data.transform import PDEStandardizer
from models.model_configs import (
    get_model_config,
    get_model_config_metadata,
    instantiate_model,
    model_config_metadata_from_config,
)
from train_arg_parser import get_args_parser
from sampling.losses import prediction_only_pde_params
from sampling.metrics import pde_residual_norm
from sampling.pde_residuals import compute_pde_residual
from sampling.state import split_pair_state
from training import distributed_mode
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import inspect_checkpoint_architecture, load_model, save_model
from training.train_loop import train_one_epoch, validate_one_epoch

logger = logging.getLogger(__name__)

DEFAULT_VAL_RATIO = 0.1


class DistributedEvalSampler(torch.utils.data.Sampler):
    """Shard validation indices across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas


def _resolve_lr_scheduler_name(args) -> str:
    scheduler_name = getattr(args, "lr_scheduler", "warmup_cosine")
    if getattr(args, "decay_lr", False):
        if scheduler_name != "linear":
            warnings.warn(
                "--decay_lr is a legacy alias and overrides --lr_scheduler to 'linear'. "
                "Prefer --lr_scheduler linear for new runs.",
                RuntimeWarning,
                stacklevel=2,
            )
        return "linear"
    return scheduler_name


def _validate_lr_scheduler_args(args) -> tuple[float, int, float]:
    min_lr = float(getattr(args, "min_lr", 1e-6))
    base_lr = float(args.lr)
    if base_lr <= 0:
        raise ValueError(f"--lr must be positive, got {base_lr}")
    if min_lr < 0:
        raise ValueError(f"--min_lr must be non-negative, got {min_lr}")
    if min_lr > base_lr:
        raise ValueError(f"--min_lr must be <= --lr, got min_lr={min_lr}, lr={base_lr}")

    warmup_epochs = int(getattr(args, "warmup_epochs", 5))
    if warmup_epochs < 0:
        raise ValueError(f"--warmup_epochs must be non-negative, got {warmup_epochs}")

    warmup_start_factor = float(getattr(args, "warmup_start_factor", 0.1))
    if not 0 < warmup_start_factor <= 1:
        raise ValueError(
            "--warmup_start_factor must be in (0, 1], "
            f"got {warmup_start_factor}"
        )
    return min_lr, warmup_epochs, warmup_start_factor


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, args, scheduler_name: str):
    min_lr, warmup_epochs, warmup_start_factor = _validate_lr_scheduler_args(args)
    total_epochs = max(int(args.epochs), 1)

    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            total_iters=total_epochs,
            factor=1.0,
        )

    if scheduler_name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=total_epochs,
            start_factor=1.0,
            end_factor=min_lr / float(args.lr),
        )

    if scheduler_name == "warmup_cosine":
        warmup_epochs = min(warmup_epochs, total_epochs)
        if warmup_epochs == 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=min_lr,
            )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=warmup_epochs,
            start_factor=warmup_start_factor,
            end_factor=1.0,
        )
        if warmup_epochs == total_epochs:
            return warmup
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(args, "plateau_factor", 0.5)),
            patience=int(getattr(args, "plateau_patience", 10)),
            threshold=float(getattr(args, "plateau_threshold", 1e-4)),
            min_lr=min_lr,
        )

    raise ValueError(f"Unsupported lr_scheduler={scheduler_name!r}")


def _step_lr_scheduler(lr_schedule, scheduler_name: str, val_loss: float) -> None:
    if scheduler_name == "plateau":
        lr_schedule.step(float(val_loss))
    else:
        lr_schedule.step()


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
    (
        data,
        label,
        loader_metadata,
        data_val,
        label_val,
        val_loader_metadata,
        validation_metadata,
    ) = _load_training_and_validation_data(
        pde_names,
        args.data_path,
        data_size=args.data_size,
        max_train_samples=args.max_train_samples,
        rd_init_mode_filter=args.rd_init_mode_filter,
        seed=args.seed,
    )
    num_channels = int(data.shape[1])
    requested_scalar_conditioning_params = _normalize_scalar_conditioning_params(
        getattr(args, "scalar_conditioning_params", [])
    )
    logger.info(
        f"Loaded train data shape={tuple(data.shape)}, validation data shape={tuple(data_val.shape)}, "
        f"labels dtype={label.dtype}"
    )
    model_arch = pde_names[0]
    resolved_model_profile, resume_arch_meta = resolve_training_model_profile(args)
    resume_model_profile_override = bool(resume_arch_meta.get("override", False))
    model_config, model_config_metadata = _resolve_training_model_config(
        model_arch=model_arch,
        pde_names=pde_names,
        resolved_model_profile=resolved_model_profile,
        resume_arch_meta=resume_arch_meta,
        num_channels=num_channels,
        scalar_conditioning_params=requested_scalar_conditioning_params,
    )
    (
        scalar_conditioning_train,
        scalar_conditioning_val,
        scalar_conditioning_metadata,
    ) = _prepare_scalar_conditioning(
        model_config=model_config,
        pde_names=pde_names,
        train_loader_metadata=loader_metadata,
        val_loader_metadata=val_loader_metadata,
        train_sample_count=int(data.shape[0]),
        val_sample_count=int(data_val.shape[0]),
        eps=args.normalization_eps,
    )
    data_metadata = _build_data_metadata(
        args=args,
        pde_names=pde_names,
        data=data,
        label=label,
        loader_metadata=loader_metadata,
        model_config_metadata=model_config_metadata,
        requested_model_profile=args.model_profile,
        resolved_model_profile=resolved_model_profile,
        resume_architecture_metadata=resume_arch_meta,
        resume_model_profile_override=resume_model_profile_override,
        checkpoint_model_profile=resume_arch_meta.get("checkpoint_model_profile"),
        checkpoint_model_config_metadata=resume_arch_meta.get("checkpoint_model_config_metadata"),
        validation_metadata=validation_metadata,
        scalar_conditioning_metadata=scalar_conditioning_metadata,
    )
    if distributed_mode.is_main_process() and args.output_dir:
        output_dir = Path(args.output_dir)
        _write_json(output_dir / "data_metadata.json", data_metadata)
        if args.save_full_pde_params:
            torch.save(detach_pde_params(loader_metadata), output_dir / "pde_params.pt")

    fitted_normalizer = PDEStandardizer.fit(
        data,
        eps=args.normalization_eps,
        channel_names=_training_channel_names(pde_names, loader_metadata, num_channels),
        pde=args.dataset,
    )

    logger.info("Initializing Model")
    model = instantiate_model(
        architechture=model_arch,
        use_ema=args.use_ema,
        model_config=model_config,
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

    resolved_lr_scheduler = _resolve_lr_scheduler_name(args)
    args.resolved_lr_scheduler = resolved_lr_scheduler
    lr_schedule = _build_lr_scheduler(optimizer, args, resolved_lr_scheduler)

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Resolved LR scheduler: {resolved_lr_scheduler}")
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
    data_validation = normalizer.transform(data_val)
    dataset_train = TensorDataset(data_train, label, scalar_conditioning_train)
    dataset_val = TensorDataset(data_validation, label_val, scalar_conditioning_val)

    logger.info(dataset_train)
    logger.info(dataset_val)
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
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=DistributedEvalSampler(dataset_val, num_replicas=num_tasks, rank=global_rank),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
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
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=args,
        )
        val_stats = validate_one_epoch(
            model=model,
            data_loader=data_loader_val,
            device=device,
            epoch=epoch,
            args=args,
        )

        _step_lr_scheduler(lr_schedule, resolved_lr_scheduler, val_stats["loss"])

        log_stats = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "lr_scheduler": resolved_lr_scheduler,
            "num_channels": num_channels,
        }

        should_eval = args.output_dir and args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0
        if should_eval and distributed_mode.is_main_process():
            eval_stats = _run_periodic_flow_eval(
                args=args,
                model=model_without_ddp,
                normalizer=normalizer,
                pde_name=pde_names[0],
                epoch=epoch,
                num_channels=num_channels,
                resolution=int(data_val.shape[-1]),
                device=device,
                val_loader_metadata=val_loader_metadata,
                scalar_conditioning_metadata=scalar_conditioning_metadata,
            )
            log_stats.update(eval_stats)

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, f"{args.dataset}_log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        final_epoch = epoch == args.epochs - 1
        should_save = args.output_dir and (
            final_epoch
            or args.test_run
            or should_eval
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
                data_metadata=data_metadata,
                model_profile=resolved_model_profile,
                requested_model_profile=args.model_profile,
                resolved_model_profile=resolved_model_profile,
                resume_architecture_metadata=resume_arch_meta,
                resume_model_profile_override=resume_model_profile_override,
                checkpoint_model_profile=resume_arch_meta.get("checkpoint_model_profile"),
                checkpoint_model_config_metadata=resume_arch_meta.get("checkpoint_model_config_metadata"),
                model_config=model_config,
                model_config_metadata=model_config_metadata,
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


def resolve_training_model_profile(args) -> tuple[str, dict[str, Any]]:
    requested_profile = getattr(args, "model_profile", "auto")
    allow_override = bool(getattr(args, "allow_model_profile_override", False))
    resume = getattr(args, "resume", "") or ""

    if not resume:
        resolved_profile = "recommended" if requested_profile == "auto" else requested_profile
        return resolved_profile, {
            "resume": False,
            "requested_model_profile": requested_profile,
            "resolved_model_profile": resolved_profile,
            "override": False,
            "checkpoint_model_profile": None,
            "checkpoint_model_config": None,
            "checkpoint_model_config_metadata": None,
        }

    checkpoint_metadata = inspect_checkpoint_architecture(resume)
    checkpoint_profile = checkpoint_metadata.get("checkpoint_model_profile")
    has_model_config = bool(checkpoint_metadata.get("has_model_config", False))

    resume_metadata = {
        "resume": True,
        "requested_model_profile": requested_profile,
        "override": False,
        **checkpoint_metadata,
    }

    if checkpoint_profile:
        if requested_profile == "auto":
            resolved_profile = str(checkpoint_profile)
        elif requested_profile == checkpoint_profile:
            resolved_profile = requested_profile
        elif allow_override:
            resolved_profile = requested_profile
            resume_metadata["override"] = True
            warnings.warn(
                "resume checkpoint architecture/profile does not match requested model_profile; "
                f"checkpoint_model_profile={checkpoint_profile}, "
                f"requested_model_profile={requested_profile}. "
                "Proceeding because --allow_model_profile_override was set.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                "resume checkpoint architecture/profile does not match requested model_profile; "
                f"checkpoint_model_profile={checkpoint_profile}; "
                f"requested_model_profile={requested_profile}. "
                f"Use --model_profile {checkpoint_profile} or retrain. "
                "To intentionally override, pass --allow_model_profile_override."
            )
    elif has_model_config and requested_profile == "auto":
        resolved_profile = "checkpoint"
    elif requested_profile == "auto":
        raise ValueError(
            "resume checkpoint lacks architecture metadata and model_config; auto model_profile cannot infer "
            "the correct architecture. Pass an explicit --model_profile legacy_base or another correct profile."
        )
    else:
        resolved_profile = requested_profile
        resume_metadata["checkpoint_lacks_architecture_metadata"] = True

    resume_metadata["resolved_model_profile"] = resolved_profile
    return resolved_profile, resume_metadata


def _resolve_training_model_config(
    *,
    model_arch: str,
    pde_names: list[str],
    resolved_model_profile: str,
    resume_arch_meta: dict[str, Any],
    num_channels: int,
    scalar_conditioning_params: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested_scalar_params = _normalize_scalar_conditioning_params(scalar_conditioning_params or ())
    checkpoint_model_config = resume_arch_meta.get("checkpoint_model_config")
    if checkpoint_model_config and not resume_arch_meta.get("override", False):
        model_config = deepcopy(dict(checkpoint_model_config))
        _validate_or_apply_joint_conditioning(
            model_config,
            pde_names,
            checkpoint_metadata=resume_arch_meta.get("checkpoint_model_config_metadata"),
            checkpoint_path=resume_arch_meta.get("checkpoint_path"),
            is_resume=True,
        )
        _validate_checkpoint_model_config_channels(
            model_config,
            num_channels=num_channels,
            checkpoint_path=resume_arch_meta.get("checkpoint_path"),
        )
        checkpoint_metadata = resume_arch_meta.get("checkpoint_model_config_metadata")
        _validate_checkpoint_scalar_conditioning_request(
            model_config=model_config,
            model_config_metadata=checkpoint_metadata if isinstance(checkpoint_metadata, dict) else {},
            requested_scalar_params=requested_scalar_params,
            checkpoint_path=resume_arch_meta.get("checkpoint_path"),
        )
        checkpoint_metadata = resume_arch_meta.get("checkpoint_model_config_metadata")
        model_config_metadata = {
            **(checkpoint_metadata if isinstance(checkpoint_metadata, dict) else {}),
            **model_config_metadata_from_config(model_config),
        }
        model_config_metadata["model_arch"] = model_arch
        model_config_metadata["model_profile"] = resolved_model_profile
        _add_joint_model_metadata(model_config_metadata, pde_names)
        return model_config, model_config_metadata

    model_config = get_model_config(
        model_arch,
        profile=resolved_model_profile,
        in_channels=num_channels,
        out_channels=num_channels,
    )
    _validate_or_apply_joint_conditioning(
        model_config,
        pde_names,
        checkpoint_metadata=None,
        checkpoint_path=None,
        is_resume=False,
    )
    _apply_scalar_conditioning_to_model_config(model_config, requested_scalar_params)
    model_config_metadata = model_config_metadata_from_config(model_config)
    model_config_metadata["model_arch"] = model_arch
    model_config_metadata["model_profile"] = resolved_model_profile
    _add_joint_model_metadata(model_config_metadata, pde_names)
    return model_config, model_config_metadata


def _validate_checkpoint_model_config_channels(
    model_config: dict[str, Any],
    *,
    num_channels: int,
    checkpoint_path: str | None = None,
) -> None:
    mismatches = []
    for key in ("in_channels", "out_channels"):
        if key in model_config and model_config[key] is not None:
            if int(model_config[key]) != int(num_channels):
                mismatches.append(f"{key}={model_config[key]}")
    if mismatches:
        raise ValueError(
            "checkpoint architecture channel count does not match current training data; "
            f"checkpoint_path={checkpoint_path}; "
            f"checkpoint_model_config {', '.join(mismatches)}; "
            f"current_training_num_channels={num_channels}."
        )


def _validate_or_apply_joint_conditioning(
    model_config: dict[str, Any],
    pde_names: list[str],
    *,
    checkpoint_metadata: Any,
    checkpoint_path: str | None,
    is_resume: bool,
) -> None:
    if len(pde_names) <= 1:
        return
    expected_mapping = {name: index for index, name in enumerate(pde_names)}
    if is_resume:
        metadata = checkpoint_metadata if isinstance(checkpoint_metadata, dict) else {}
        saved_mapping = metadata.get("pde_label_mapping")
        if not isinstance(saved_mapping, dict) or model_config.get("num_classes") is None:
            raise ValueError(
                "Old joint checkpoint lacks the required contiguous PDE label mapping/category layer; "
                f"checkpoint_path={checkpoint_path}. Retrain the joint model."
            )
        normalized_mapping = {str(name): int(index) for name, index in saved_mapping.items()}
        if normalized_mapping != expected_mapping:
            raise ValueError(
                "Joint checkpoint PDE label mapping does not match this dataset order; "
                f"checkpoint={normalized_mapping}, requested={expected_mapping}"
            )
        if int(model_config["num_classes"]) != len(expected_mapping):
            raise ValueError("Joint checkpoint num_classes does not match its PDE label mapping")
    else:
        model_config["num_classes"] = len(expected_mapping)


def _normalize_scalar_conditioning_params(params: Any) -> tuple[str, ...]:
    return normalize_scalar_conditioning_params(params)


def _apply_scalar_conditioning_to_model_config(
    model_config: dict[str, Any],
    params: tuple[str, ...],
) -> None:
    if not params:
        model_config["scalar_conditioning"] = False
        model_config["scalar_conditioning_dim"] = 0
        return
    model_config["scalar_conditioning"] = True
    model_config["scalar_conditioning_dim"] = len(params)
    model_config["scalar_conditioning_params"] = tuple(params)


def _validate_checkpoint_scalar_conditioning_request(
    *,
    model_config: dict[str, Any],
    model_config_metadata: dict[str, Any],
    requested_scalar_params: tuple[str, ...],
    checkpoint_path: str | None,
) -> None:
    enabled = bool(
        model_config.get("scalar_conditioning", False)
        or model_config_metadata.get("scalar_conditioning", False)
    )
    if not enabled:
        if requested_scalar_params:
            raise ValueError(
                "resume checkpoint was not trained with scalar conditioning, but "
                f"--scalar_conditioning_params={list(requested_scalar_params)} was requested; "
                f"checkpoint_path={checkpoint_path}. Retrain instead, or use the existing "
                "model profile override flow with a compatible checkpoint."
            )
        model_config["scalar_conditioning"] = False
        model_config["scalar_conditioning_dim"] = 0
        return

    checkpoint_params = _scalar_conditioning_params_from_config(
        model_config,
        model_config_metadata,
    )
    if not checkpoint_params:
        raise ValueError(
            "resume checkpoint has scalar_conditioning=True but does not record "
            f"scalar_conditioning_params; checkpoint_path={checkpoint_path}."
        )
    checkpoint_dim = int(
        model_config.get("scalar_conditioning_dim")
        or model_config_metadata.get("scalar_conditioning_dim")
        or len(checkpoint_params)
    )
    if checkpoint_dim != len(checkpoint_params):
        raise ValueError(
            "resume checkpoint scalar conditioning metadata is inconsistent; "
            f"scalar_conditioning_dim={checkpoint_dim}, "
            f"scalar_conditioning_params={list(checkpoint_params)}, "
            f"checkpoint_path={checkpoint_path}."
        )
    if requested_scalar_params and requested_scalar_params != checkpoint_params:
        raise ValueError(
            "resume checkpoint scalar conditioning parameters do not match the new command; "
            f"checkpoint_scalar_conditioning_params={list(checkpoint_params)}, "
            f"requested_scalar_conditioning_params={list(requested_scalar_params)}, "
            f"checkpoint_path={checkpoint_path}. Retrain instead, or use the existing "
            "model profile override flow with a compatible checkpoint."
        )
    model_config["scalar_conditioning"] = True
    model_config["scalar_conditioning_dim"] = checkpoint_dim
    model_config["scalar_conditioning_params"] = tuple(checkpoint_params)


def _scalar_conditioning_params_from_config(
    model_config: dict[str, Any],
    model_config_metadata: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    return scalar_conditioning_params_from_config(model_config, model_config_metadata)


def _prepare_scalar_conditioning(
    *,
    model_config: dict[str, Any],
    pde_names: list[str],
    train_loader_metadata: dict[str, Any],
    val_loader_metadata: dict[str, Any],
    train_sample_count: int,
    val_sample_count: int,
    eps: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    return fit_scalar_conditioning(
        model_config=model_config,
        pde_names=pde_names,
        train_loader_metadata=train_loader_metadata,
        val_loader_metadata=val_loader_metadata,
        train_sample_count=train_sample_count,
        val_sample_count=val_sample_count,
        eps=eps,
    )


def _scalar_conditioning_metadata_disabled() -> dict[str, Any]:
    return scalar_conditioning_metadata_disabled()


def _scalar_conditioning_tensor_from_metadata(
    *,
    pde_names: list[str],
    loader_metadata: dict[str, Any],
    params: tuple[str, ...],
    expected_sample_count: int,
    split_name: str,
) -> torch.Tensor:
    return scalar_conditioning_tensor_from_metadata(
        pde_names=pde_names,
        loader_metadata=loader_metadata,
        params=params,
        expected_sample_count=expected_sample_count,
        split_name=split_name,
    )


def _build_data_metadata(
    args,
    pde_names: list[str],
    data: torch.Tensor,
    label: torch.Tensor,
    loader_metadata: dict[str, Any],
    model_config_metadata: dict[str, Any] | None = None,
    requested_model_profile: str | None = None,
    resolved_model_profile: str | None = None,
    resume_architecture_metadata: dict[str, Any] | None = None,
    resume_model_profile_override: bool = False,
    checkpoint_model_profile: str | None = None,
    checkpoint_model_config_metadata: dict[str, Any] | None = None,
    validation_metadata: dict[str, Any] | None = None,
    scalar_conditioning_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    specs = {pde: get_pde_spec(pde).to_metadata() for pde in pde_names}
    channel_names = _training_channel_names(pde_names, loader_metadata, int(data.shape[1]))
    scalar_meta = scalar_conditioning_metadata or _scalar_conditioning_metadata_disabled()
    scalar_enabled = bool(scalar_meta.get("enabled", False))
    scalar_params = (
        list(scalar_meta.get("params", []))
        if scalar_enabled
        else (model_config_metadata or {}).get("scalar_conditioning_params", [])
    )
    return {
        "dataset": args.dataset,
        "pde_names": list(pde_names),
        "pde_label_mapping": {
            pde_name: index for index, pde_name in enumerate(pde_names)
        },
        "pde_data_specs": specs,
        "data_path": args.data_path,
        "data_size": args.data_size,
        "max_train_samples": args.max_train_samples,
        "data_shape": [int(dim) for dim in data.shape],
        "num_channels": int(data.shape[1]),
        "channel_names": channel_names,
        "scalar_param_keys": {
            pde: sorted((loader_metadata.get(pde, {}) or {}).get("pde_params", {}))
            for pde in pde_names
        },
        "residual_family": {pde: specs[pde]["residual_family"] for pde in pde_names},
        "loadby": {pde: specs[pde]["default_loadby"] for pde in pde_names},
        "requested_model_profile": requested_model_profile or getattr(args, "model_profile", "auto"),
        "resolved_model_profile": resolved_model_profile or getattr(args, "model_profile", "recommended"),
        "model_profile": resolved_model_profile or getattr(args, "model_profile", "recommended"),
        "resume_architecture_metadata": resume_architecture_metadata,
        "resume_model_profile_override": bool(resume_model_profile_override),
        "checkpoint_model_profile": checkpoint_model_profile,
        "checkpoint_model_config_metadata": checkpoint_model_config_metadata,
        "model_config": model_config_metadata,
        "model_config_metadata": model_config_metadata,
        "architecture_family": (model_config_metadata or {}).get("architecture_family"),
        "attention_resolutions": (model_config_metadata or {}).get("attention_resolutions"),
        "channel_mult": (model_config_metadata or {}).get("channel_mult"),
        "dropout": (model_config_metadata or {}).get("dropout"),
        "model_channels": (model_config_metadata or {}).get("model_channels"),
        "num_res_blocks": (model_config_metadata or {}).get("num_res_blocks"),
        "with_fourier_features": (model_config_metadata or {}).get("with_fourier_features"),
        "with_value_fourier_features": (model_config_metadata or {}).get("with_value_fourier_features"),
        "with_coordinate_fourier_features": (model_config_metadata or {}).get("with_coordinate_fourier_features"),
        "value_fourier_feature_channels": (model_config_metadata or {}).get("value_fourier_feature_channels"),
        "coordinate_fourier_feature_channels": (model_config_metadata or {}).get("coordinate_fourier_feature_channels"),
        "axis_semantics": (model_config_metadata or {}).get("axis_semantics"),
        "scalar_conditioning": (model_config_metadata or {}).get("scalar_conditioning", scalar_enabled),
        "scalar_conditioning_enabled": scalar_enabled,
        "scalar_conditioning_dim": int(scalar_meta.get("dim", 0) or (model_config_metadata or {}).get("scalar_conditioning_dim", 0) or 0),
        "scalar_conditioning_params": scalar_params,
        "scalar_conditioning_mean": _jsonable(scalar_meta.get("mean", [])),
        "scalar_conditioning_std": _jsonable(scalar_meta.get("std", [])),
        "scalar_conditioning_raw_std": _jsonable(scalar_meta.get("raw_std", [])),
        "scalar_conditioning_std_was_clamped": _jsonable(scalar_meta.get("std_was_clamped", [])),
        "scalar_conditioning_normalization": scalar_meta.get("normalization"),
        "num_samples": int(data.shape[0]),
        "label_values": sorted(int(value) for value in torch.unique(label).detach().cpu().tolist()),
        "validation": _jsonable(validation_metadata or {}),
        "pde_param_summary": summarize_pde_params(loader_metadata),
        "loader_metadata": _loader_metadata_for_json(loader_metadata),
        "normalization": {
            "type": "channelwise_standardization",
            "eps": args.normalization_eps,
            "channel_names": channel_names,
        },
    }


def _build_model_config_metadata(
    model_arch: str,
    pde_names: list[str],
    profile: str,
    num_channels: int,
) -> dict[str, Any]:
    metadata = get_model_config_metadata(
        model_arch,
        profile=profile,
        in_channels=num_channels,
        out_channels=num_channels,
    )
    metadata["model_arch"] = model_arch
    metadata["model_profile"] = profile
    _add_joint_model_metadata(metadata, pde_names)
    return metadata


def _add_joint_model_metadata(metadata: dict[str, Any], pde_names: list[str]) -> None:
    if len(pde_names) > 1:
        metadata["model_arch_source"] = "first_pde_in_joint_dataset"
        metadata["joint_pde_names"] = list(pde_names)
        metadata["pde_label_mapping"] = {
            pde_name: index for index, pde_name in enumerate(pde_names)
        }
        metadata["null_pde_label"] = len(pde_names)
        metadata["joint_architecture_limitation"] = (
            "joint training uses the first PDE architecture because current joint datasets "
            "require identical channel counts and a single UNet instance"
        )
    else:
        metadata["model_arch_source"] = "single_pde"
        metadata["joint_pde_names"] = list(pde_names)


def _training_channel_names(
    pde_names: list[str],
    loader_metadata: dict[str, Any],
    num_channels: int,
) -> list[str]:
    per_pde = []
    for pde_name in pde_names:
        entry = loader_metadata.get(pde_name, {}) if isinstance(loader_metadata, dict) else {}
        names = entry.get("channel_names") if isinstance(entry, dict) else None
        if not names:
            names = list(get_pde_spec(pde_name).channel_names)
        if len(names) == num_channels:
            per_pde.append(list(names))
    if per_pde and all(names == per_pde[0] for names in per_pde):
        return per_pde[0]
    if len(pde_names) == 1:
        names = list(get_pde_spec(pde_names[0]).channel_names)
        if len(names) == num_channels:
            return names
    return [f"channel_{idx}" for idx in range(num_channels)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _load_training_data(
    pde_names: list[str],
    data_path: str,
    data_size: int = 5,
    max_train_samples: int | None = None,
    rd_init_mode_filter: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    dataset_list = []
    label_list = []
    channel_counts = {}
    metadata: dict[str, Any] = {}
    pde_label_mapping = {pde_name: index for index, pde_name in enumerate(pde_names)}

    for pde_name in pde_names:
        logger.info(f">>> Initializing Dataset: {pde_name} <<<")
        pde_loader = PDEloader(pde_name)
        load_kwargs = {"size": data_size, "max_samples": max_train_samples}
        if pde_name == "reaction_diffusion" and rd_init_mode_filter is not None:
            load_kwargs["rd_init_mode_filter"] = rd_init_mode_filter
        dataset, label = pde_loader.load_data(data_path, **load_kwargs)
        if dataset.ndim != 4:
            raise ValueError(f"{pde_name} loader returned non-BCHW data: {tuple(dataset.shape)}")
        channel_counts[pde_name] = int(dataset.shape[1])
        dataset_list.append(dataset)
        label_list.append(
            torch.full(
                (int(dataset.shape[0]),),
                int(pde_label_mapping[pde_name]),
                dtype=torch.long,
            )
        )
        loader_meta = pde_loader.metadata()
        _assert_unique_sample_identity(loader_meta, pde_name)
        if pde_loader.pde_params or loader_meta.get("extra_metadata"):
            metadata[pde_name] = loader_meta

    if len(set(channel_counts.values())) != 1:
        raise ValueError(
            "Joint PDE training requires identical channel counts; "
            f"got {channel_counts}. Padding/masking is intentionally not enabled."
        )

    data = torch.cat(dataset_list, dim=0).to(torch.float32)
    label = torch.cat(label_list, dim=0).to(torch.long)
    return data, label, metadata


def _load_training_and_validation_data(
    pde_names: list[str],
    data_path: str,
    data_size: int = 5,
    max_train_samples: int | None = None,
    rd_init_mode_filter: str | None = None,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
    dict[str, Any],
]:
    train_data_parts = []
    train_label_parts = []
    val_data_parts = []
    val_label_parts = []
    train_metadata: dict[str, Any] = {}
    val_metadata: dict[str, Any] = {}
    split_metadata: dict[str, Any] = {"val_ratio": DEFAULT_VAL_RATIO, "per_pde": {}}
    train_channel_counts = {}
    val_channel_counts = {}
    pde_label_mapping = {pde_name: index for index, pde_name in enumerate(pde_names)}

    for pde_name in pde_names:
        logger.info(f">>> Initializing Dataset: {pde_name} <<<")
        pde_loader = PDEloader(pde_name)
        load_kwargs = {"size": data_size, "max_samples": max_train_samples}
        if pde_name == "reaction_diffusion" and rd_init_mode_filter is not None:
            load_kwargs["rd_init_mode_filter"] = rd_init_mode_filter
        dataset, label = pde_loader.load_data(data_path, **load_kwargs)
        if dataset.ndim != 4:
            raise ValueError(f"{pde_name} loader returned non-BCHW data: {tuple(dataset.shape)}")
        loader_meta = pde_loader.metadata()
        _assert_unique_sample_identity(loader_meta, pde_name)

        train_idx, val_idx = _train_val_split_indices(
            int(dataset.shape[0]),
            seed=seed + int(get_pde_spec(pde_name).label_id) * 1009,
            val_ratio=DEFAULT_VAL_RATIO,
        )
        train_dataset = dataset[train_idx].contiguous()
        val_dataset = dataset[val_idx].contiguous()
        local_label = int(pde_label_mapping[pde_name])
        train_label = torch.full((len(train_idx),), local_label, dtype=torch.long)
        val_label = torch.full((len(val_idx),), local_label, dtype=torch.long)
        pde_train_meta = _slice_loader_metadata(loader_meta, train_idx)
        pde_val_meta = _slice_loader_metadata(loader_meta, val_idx)
        split_source = "deterministic_disjoint_9_1_from_train_data"

        train_channel_counts[pde_name] = int(train_dataset.shape[1])
        val_channel_counts[pde_name] = int(val_dataset.shape[1])
        train_data_parts.append(train_dataset)
        train_label_parts.append(train_label)
        val_data_parts.append(val_dataset)
        val_label_parts.append(val_label.long())
        if pde_train_meta.get("pde_params") or pde_train_meta.get("extra_metadata"):
            train_metadata[pde_name] = pde_train_meta
        if pde_val_meta.get("pde_params") or pde_val_meta.get("extra_metadata"):
            val_metadata[pde_name] = pde_val_meta
        split_metadata["per_pde"][pde_name] = {
            "source": split_source,
            "train_samples": int(train_dataset.shape[0]),
            "validation_samples": int(val_dataset.shape[0]),
            "original_loaded_samples": int(dataset.shape[0]),
        }

    if len(set(train_channel_counts.values())) != 1 or len(set(val_channel_counts.values())) != 1:
        raise ValueError(
            "Joint PDE training requires identical channel counts; "
            f"got train={train_channel_counts}, validation={val_channel_counts}."
        )
    if set(train_channel_counts.values()) != set(val_channel_counts.values()):
        raise ValueError(
            "Training and validation channel counts must match; "
            f"got train={train_channel_counts}, validation={val_channel_counts}."
        )

    train_data = torch.cat(train_data_parts, dim=0).to(torch.float32)
    train_label = torch.cat(train_label_parts, dim=0).to(torch.long)
    validation_data = torch.cat(val_data_parts, dim=0).to(torch.float32)
    validation_label = torch.cat(val_label_parts, dim=0).to(torch.long)
    split_metadata["train_samples"] = int(train_data.shape[0])
    split_metadata["validation_samples"] = int(validation_data.shape[0])
    split_metadata["pde_label_mapping"] = pde_label_mapping
    return (
        train_data,
        train_label,
        train_metadata,
        validation_data,
        validation_label,
        val_metadata,
        split_metadata,
    )


def _train_val_split_indices(
    num_samples: int,
    seed: int,
    val_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_samples < 1:
        raise ValueError("Cannot split an empty dataset")
    if num_samples == 1:
        raise ValueError(
            "Cannot create disjoint training and validation sets from one sample; "
            "provide at least two distinct samples"
        )
    val_count = max(1, int(round(float(num_samples) * float(val_ratio))))
    val_count = min(val_count, num_samples - 1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(num_samples, generator=generator)
    val_idx = permutation[:val_count].sort().values
    train_idx = permutation[val_count:].sort().values
    return train_idx, val_idx


def _assert_unique_sample_identity(metadata: dict[str, Any], pde_name: str) -> None:
    extra = metadata.get("extra_metadata", {}) if isinstance(metadata, dict) else {}
    if not isinstance(extra, dict):
        return
    for name in ("sample_id", "sample_seed"):
        values = extra.get(name)
        if values is None:
            continue
        flattened = list(values) if isinstance(values, (list, tuple)) else list(np.asarray(values).reshape(-1))
        normalized = [str(value) if name == "sample_id" else int(value) for value in flattened]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Duplicate {name} values detected in {pde_name} training data")


def _slice_loader_metadata(metadata: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    sliced = dict(metadata or {})
    count = int(indices.numel())
    index = indices.detach().cpu().long()
    if isinstance(sliced.get("pde_params"), dict):
        sliced["pde_params"] = {
            name: _slice_sample_aligned_value(value, index)
            for name, value in sliced["pde_params"].items()
        }
    extra = sliced.get("extra_metadata")
    if isinstance(extra, dict):
        sliced["extra_metadata"] = {
            name: _slice_sample_aligned_value(value, index)
            for name, value in extra.items()
        }
    sliced["num_loaded_samples"] = count
    sliced["pde_param_slices"] = []
    if isinstance(extra, dict):
        sliced.setdefault("extra_metadata", {})["num_loaded_samples"] = count
    return sliced


def _slice_sample_aligned_value(value: Any, indices: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) >= int(indices.max().item()) + 1:
            return value.index_select(0, indices.to(value.device))
        return value
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and int(value.shape[0]) >= int(indices.max().item()) + 1:
            return value[indices.numpy()]
        return value
    if isinstance(value, list):
        if len(value) >= int(indices.max().item()) + 1:
            return [value[int(idx)] for idx in indices.tolist()]
        return list(value)
    if isinstance(value, tuple):
        if len(value) >= int(indices.max().item()) + 1:
            return tuple(value[int(idx)] for idx in indices.tolist())
        return tuple(value)
    return value


def _run_periodic_flow_eval(
    args,
    model: torch.nn.Module,
    normalizer: PDEStandardizer,
    pde_name: str,
    epoch: int,
    num_channels: int,
    resolution: int,
    device: torch.device,
    val_loader_metadata: dict[str, Any],
    scalar_conditioning_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_epoch = int(epoch) + 1
    batch_size = int(getattr(args, "eval_num_samples", 1))
    num_steps = int(getattr(args, "eval_num_steps", 100))
    if batch_size < 1:
        raise ValueError("--eval_num_samples must be positive")
    if num_steps < 1:
        raise ValueError("--eval_num_steps must be positive")

    dtype = torch.float32
    pde_params = _pde_params_for_eval(
        val_loader_metadata,
        pde_name=pde_name,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    eval_residual_mode = str(getattr(args, "eval_residual_mode", "auto"))
    if eval_residual_mode == "near_endpoint_temporal":
        raise ValueError(
            "Periodic training evaluation is unconditional and therefore cannot use "
            "near_endpoint_temporal sparse observations. Use formal sampling evaluation instead."
        )
    pde_params, excluded_ground_truth_field_params = prediction_only_pde_params(
        pde_params,
        allow_sparse_near_endpoint=False,
    )
    uses_sparse_near_endpoint_exception = False
    if pde_name != "burger" and eval_residual_mode in {"full_trajectory_fd", "full_time_space"}:
        raise ValueError(
            f"Periodic generated-sample evaluation for {pde_name!r} cannot use {eval_residual_mode!r}: "
            "the model outputs only a/u endpoints. Only Burgers outputs a full predicted time-space field."
        )
    if pde_name == "burger" and eval_residual_mode in {"hermite_bridge", "endpoint_secant"}:
        raise ValueError(
            f"Periodic Burgers evaluation cannot use endpoint mode {eval_residual_mode!r}; "
            "evaluate the full model-predicted time-space field instead."
        )
    model_extra = None
    scalar_meta = scalar_conditioning_metadata or _scalar_conditioning_metadata_disabled()
    if bool(scalar_meta.get("enabled", False)):
        model_extra = {
            "scalar_conditioning": standardize_scalar_conditioning(
                pde_params=pde_params,
                metadata=scalar_meta,
                expected_sample_count=batch_size,
                source_name=f"validation pde_params for PDE {pde_name!r}",
                device=device,
                dtype=dtype,
            )
        }

    sample_standardized = _euler_flow_sample(
        model=model,
        pde_name=pde_name,
        batch_size=batch_size,
        num_channels=num_channels,
        resolution=resolution,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
        seed=int(args.seed) + 1_000_003 + eval_epoch,
        model_extra=model_extra,
        cfg_scale=float(getattr(args, "cfg_scale", 1.0)),
    )
    sample_physical = normalizer.inverse_transform(sample_standardized)
    split = split_pair_state(sample_physical, pde_name)
    residual = compute_pde_residual(
        pde_name,
        split.coef,
        split.sol,
        pde_params=pde_params,
        residual_mode=eval_residual_mode,
    )
    residual_norm = pde_residual_norm(residual.residual)

    check_dir = Path(args.output_dir) / "check_figs"
    check_dir.mkdir(parents=True, exist_ok=True)
    stem = f"epoch_{eval_epoch:04d}_{pde_name}"
    figure_path = check_dir / f"{stem}.pdf"
    _save_eval_figure(
        figure_path,
        sample_physical=sample_physical,
        residual_field=residual.residual,
        pde_name=pde_name,
        eval_epoch=eval_epoch,
        residual_norm=residual_norm,
    )
    logger.info(
        f"Eval epoch {eval_epoch}: pde={pde_name}, residual_norm={residual_norm:.6g}, "
        f"figure={figure_path}"
    )
    return {
        "eval_epoch": eval_epoch,
        "eval_pde": pde_name,
        "eval_num_steps": num_steps,
        "eval_num_samples": batch_size,
        "eval_pde_residual_norm": residual_norm,
        "eval_pde_residual_status": residual.status,
        "eval_pde_residual_mode": residual.metadata.get("resolved_residual_mode", eval_residual_mode),
        "eval_pde_field_input_sources": {"coef": "model_output", "sol": "model_output"},
        "eval_pde_uses_ground_truth_fields": uses_sparse_near_endpoint_exception,
        "eval_pde_uses_ground_truth_endpoint_fields": False,
        "eval_pde_ground_truth_field_exception": (
            "near_endpoint_temporal_sparse_observations"
            if uses_sparse_near_endpoint_exception
            else None
        ),
        "eval_pde_auxiliary_field_input_sources": (
            {
                "q_dt": "sparse_ground_truth_observations",
                "q_T_minus_dt": "sparse_ground_truth_observations",
            }
            if uses_sparse_near_endpoint_exception
            else {}
        ),
        "eval_pde_excluded_ground_truth_field_params": excluded_ground_truth_field_params,
        "eval_figure_path": str(figure_path),
    }


@torch.no_grad()
def _euler_flow_sample(
    model: torch.nn.Module,
    pde_name: str,
    batch_size: int,
    num_channels: int,
    resolution: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    model_extra: dict[str, Any] | None = None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        x = torch.randn(
            batch_size,
            num_channels,
            resolution,
            resolution,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        # Periodic evaluation samples pde_names[0], whose checkpoint-local
        # category is always zero. Single-PDE models ignore this tensor.
        label = torch.zeros((batch_size,), device=device, dtype=torch.long)
        extra = _eval_conditioning_for_model(model, label)
        if model_extra:
            extra = {**extra, **model_extra}
        grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        for step in range(num_steps):
            t = torch.full((batch_size,), float(grid[step].item()), device=device, dtype=dtype)
            step_size = grid[step + 1] - grid[step]
            velocity = _cfg_eval_velocity(model, x, t, extra, cfg_scale)
            if velocity.shape != x.shape:
                raise ValueError(f"Eval model output shape {tuple(velocity.shape)} does not match sample shape {tuple(x.shape)}")
            x = x + step_size * velocity
        return x.detach()
    finally:
        model.train(was_training)


def _eval_conditioning_for_model(model: torch.nn.Module, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    module = getattr(model, "module", model)
    if hasattr(module, "model") and hasattr(module.model, "num_classes"):
        module = module.model
    if getattr(module, "num_classes", None) is None:
        return {}
    return {"label": labels.long()}


def _cfg_eval_velocity(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    extra: dict[str, torch.Tensor],
    cfg_scale: float,
) -> torch.Tensor:
    module = getattr(model, "module", model)
    if hasattr(module, "model") and hasattr(module.model, "num_classes"):
        module = module.model
    num_classes = getattr(module, "num_classes", None)
    if num_classes is None:
        return model(x, t, extra=extra)
    if "label" not in extra:
        raise ValueError("Class-conditional periodic evaluation requires a PDE label")
    scale = float(cfg_scale)
    conditional = model(x, t, extra=extra) if scale != 0.0 else None
    if scale == 1.0:
        return conditional
    unconditional_extra = dict(extra)
    unconditional_extra["label"] = torch.full_like(extra["label"], int(num_classes))
    unconditional = model(x, t, extra=unconditional_extra)
    if scale == 0.0:
        return unconditional
    return unconditional + scale * (conditional - unconditional)


def _pde_params_for_eval(
    loader_metadata: dict[str, Any],
    pde_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    entry = loader_metadata.get(pde_name, {}) if isinstance(loader_metadata, dict) else {}
    params = entry.get("pde_params", {}) if isinstance(entry, dict) else {}
    if not isinstance(params, dict):
        return {}
    out: dict[str, Any] = {}
    for name, value in params.items():
        if name == "near_endpoint_temporal" and isinstance(value, dict):
            out[name] = {
                child_name: (
                    child_value
                    if child_name == "metadata"
                    else _eval_batch_param_value(child_value, batch_size, device, dtype)
                )
                for child_name, child_value in value.items()
            }
            continue
        selected = _eval_batch_param_value(value, batch_size, device, dtype)
        if selected is not None:
            out[name] = selected
    return out


def _eval_batch_param_value(
    value: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Any | None:
    try:
        tensor = torch.as_tensor(value, device=device)
    except (TypeError, ValueError):
        return value
    if torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.repeat(batch_size)
    elif int(tensor.shape[0]) >= batch_size:
        tensor = tensor[:batch_size]
    elif int(tensor.shape[0]) == 1:
        tensor = tensor.repeat(batch_size, *([1] * (tensor.ndim - 1)))
    else:
        return None
    return tensor


def _save_eval_figure(
    path: Path,
    sample_physical: torch.Tensor,
    residual_field: torch.Tensor | None,
    pde_name: str,
    eval_epoch: int,
    residual_norm: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sample = sample_physical[0].detach().cpu()
    panels = [(f"sample[{idx}]", sample[idx]) for idx in range(int(sample.shape[0]))]
    if residual_field is not None:
        residual_map = residual_field[0].detach().cpu().abs().mean(dim=0)
        panels.append(("abs residual mean", residual_map))
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(max(3 * cols, 4), 3.2), constrained_layout=True)
    if cols == 1:
        axes = [axes]
    for ax, (title, field) in zip(axes, panels):
        field_np = field.numpy()
        im = ax.imshow(field_np, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{pde_name} epoch {eval_epoch} | PDE residual {residual_norm:.6g}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _loader_metadata_for_json(loader_metadata: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pde_name, entry in (loader_metadata or {}).items():
        if not isinstance(entry, dict):
            continue
        pde_params = entry.get("pde_params") if isinstance(entry.get("pde_params"), dict) else entry
        out[pde_name] = {
            "pde_params_keys": sorted(pde_params),
            "pde_param_sources": _jsonable(entry.get("pde_param_sources", {})),
            "pde_param_slices": _jsonable(entry.get("pde_param_slices", [])),
            "channel_names": _jsonable(entry.get("channel_names")),
            "coef_channel_names": _jsonable(entry.get("coef_channel_names")),
            "sol_channel_names": _jsonable(entry.get("sol_channel_names")),
            "pde_data_spec": _jsonable(entry.get("pde_data_spec", get_pde_spec(pde_name).to_metadata())),
            "scalar_param_names": _jsonable(entry.get("scalar_param_names", get_pde_spec(pde_name).scalar_param_names)),
            "optional_scalar_param_names": _jsonable(entry.get("optional_scalar_param_names", [])),
            "residual_family": _jsonable(entry.get("residual_family", get_pde_spec(pde_name).residual_family)),
            "loadby": _jsonable(entry.get("loadby", get_pde_spec(pde_name).default_loadby)),
            "scalar_params_loaded": bool(entry.get("scalar_params_loaded", bool(pde_params))),
            "selected_file_format": _jsonable(entry.get("selected_file_format")),
            "selected_files": _jsonable(entry.get("selected_files", [])),
            "num_loaded_samples": _jsonable(entry.get("num_loaded_samples")),
            "extra_metadata": _jsonable(entry.get("extra_metadata", {})),
        }
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


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

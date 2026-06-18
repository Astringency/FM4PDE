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
from data.specs import get_pde_spec
from data.transform import PDEStandardizer
from models.model_configs import (
    get_model_config,
    get_model_config_metadata,
    instantiate_model,
    model_config_metadata_from_config,
)
from train_arg_parser import get_args_parser
from training import distributed_mode
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import inspect_checkpoint_architecture, load_model, save_model
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
    data, label, loader_metadata = _load_training_data(
        pde_names,
        args.data_path,
        data_size=args.data_size,
        max_train_samples=args.max_train_samples,
        rd_init_mode_filter=args.rd_init_mode_filter,
    )
    num_channels = int(data.shape[1])
    logger.info(f"Loaded data shape={tuple(data.shape)}, labels dtype={label.dtype}")
    model_arch = pde_names[0]
    resolved_model_profile, resume_arch_meta = resolve_training_model_profile(args)
    resume_model_profile_override = bool(resume_arch_meta.get("override", False))
    model_config, model_config_metadata = _resolve_training_model_config(
        model_arch=model_arch,
        pde_names=pde_names,
        resolved_model_profile=resolved_model_profile,
        resume_arch_meta=resume_arch_meta,
        num_channels=num_channels,
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_model_config = resume_arch_meta.get("checkpoint_model_config")
    if checkpoint_model_config and not resume_arch_meta.get("override", False):
        model_config = deepcopy(dict(checkpoint_model_config))
        _validate_checkpoint_model_config_channels(
            model_config,
            num_channels=num_channels,
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
    model_config_metadata = _build_model_config_metadata(
        model_arch=model_arch,
        pde_names=pde_names,
        profile=resolved_model_profile,
        num_channels=num_channels,
    )
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
) -> dict[str, Any]:
    specs = {pde: get_pde_spec(pde).to_metadata() for pde in pde_names}
    channel_names = _training_channel_names(pde_names, loader_metadata, int(data.shape[1]))
    return {
        "dataset": args.dataset,
        "pde_names": list(pde_names),
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
        "scalar_conditioning": (model_config_metadata or {}).get("scalar_conditioning", False),
        "scalar_conditioning_params": (model_config_metadata or {}).get("scalar_conditioning_params", []),
        "num_samples": int(data.shape[0]),
        "label_values": sorted(int(value) for value in torch.unique(label).detach().cpu().tolist()),
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
        label_list.append(label.long())
        loader_meta = pde_loader.metadata()
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import torch
from models.legacy_checkpoint import read_checkpoint, validate_legacy_optimizer
from training.distributed_mode import is_main_process

CHECKPOINT_SCHEMA_VERSION = 3
OPTIMIZER_RUNTIME_OPTION_KEYS = ("foreach", "fused")


def _apply_resume_optimizer_betas(optimizer, betas):
    """Intentional hyperparameter change, retaining all restored Adam state."""
    if betas is None:
        return
    if len(betas) != 2 or any(not math.isfinite(float(x)) or not 0 <= float(x) < 1 for x in betas):
        raise ValueError("--resume_optimizer_betas requires two finite values in [0, 1)")
    for group in optimizer.param_groups:
        if "betas" not in group:
            raise ValueError("--resume_optimizer_betas requires an Adam-style optimizer")
        group["betas"] = tuple(float(x) for x in betas)


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def _clone_state_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return {key: _clone_state_value(value) for key, value in state.items()}


def _is_ema_module(module) -> bool:
    from models.ema import EMA

    return isinstance(module, EMA)


def _base_model_state_dict(model_without_ddp) -> dict[str, Any]:
    if _is_ema_module(model_without_ddp):
        return _clone_state_dict(model_without_ddp.model.state_dict())
    return _clone_state_dict(model_without_ddp.state_dict())


def _ema_model_state_dict(model_without_ddp) -> dict[str, Any] | None:
    if not _is_ema_module(model_without_ddp):
        return None

    base_model = model_without_ddp.model
    state = _clone_state_dict(base_model.state_dict())
    shadow_params = list(model_without_ddp.shadow_params)
    trainable_params = [
        (name, param) for name, param in base_model.named_parameters() if param.requires_grad
    ]
    if len(shadow_params) != len(trainable_params):
        raise ValueError(
            "EMA shadow parameter count does not match trainable base model parameters: "
            f"{len(shadow_params)} vs {len(trainable_params)}"
        )
    for (name, _), shadow in zip(trainable_params, shadow_params):
        if name not in state:
            raise KeyError(f"EMA trainable parameter {name!r} is missing from base model state")
        state[name] = _clone_state_value(shadow)
    return state


def _resume_state_dict(model_without_ddp) -> dict[str, Any]:
    return _clone_state_dict(model_without_ddp.state_dict())


def _load_resume_state(model_without_ddp, checkpoint: dict[str, Any], *, add_ema=False) -> None:
    resume_state = checkpoint.get("model_for_resume")
    if not isinstance(resume_state, dict) or not resume_state:
        raise ValueError("Checkpoint has no model_for_resume state_dict")
    if add_ema:
        if not _is_ema_module(model_without_ddp):
            raise ValueError('--resume_add_ema requires --use_ema')
        if checkpoint.get('has_ema') or checkpoint.get('use_ema') or 'num_updates' in resume_state:
            raise ValueError('--resume_add_ema is only for checkpoints without EMA')
        model_without_ddp.model.load_state_dict(resume_state, strict=True)
        model_without_ddp.reset_from_model()
        return
    try:
        model_without_ddp.load_state_dict(resume_state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint model_for_resume does not match the current model. "
            "Use the same --use_ema and model configuration as the saved run."
        ) from exc


def _optimizer_state_without_ema_shadows(model, state_dict):
    """Migrate old one-group optimizers that included non-trainable EMA shadows."""
    if not _is_ema_module(model):
        return state_dict
    params = list(model.parameters())
    groups = state_dict['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(params):
        return state_dict
    frozen_ids = [pid for pid, param in zip(groups[0]['params'], params) if not param.requires_grad]
    if any(state_dict['state'].get(pid) for pid in frozen_ids):
        raise ValueError('EMA shadow parameters unexpectedly have optimizer state')
    group = dict(groups[0])
    group['params'] = [pid for pid, param in zip(group['params'], params) if param.requires_grad]
    if 'param_names' in group:
        group['param_names'] = [name for name, param in zip(group['param_names'], params) if param.requires_grad]
    return dict(state_dict, param_groups=[group],
                state={k:v for k,v in state_dict['state'].items() if k not in frozen_ids})


def _load_optimizer_state_preserving_runtime_options(
    optimizer: torch.optim.Optimizer,
    state_dict: dict[str, Any],
) -> None:
    """Load checkpoint state without reverting execution-only optimizer options.

    AdamW stores ``foreach`` and ``fused`` in its parameter groups. An older
    checkpoint therefore carries the implementation selected by the old run and
    would otherwise overwrite the implementation requested for the resumed run.
    Shallow-copy only the group dictionaries so large optimizer tensors are not
    duplicated in host memory during resume.
    """

    state_to_load = dict(state_dict)
    saved_groups = state_dict.get("param_groups")
    if isinstance(saved_groups, list):
        adapted_groups = []
        for index, saved_group in enumerate(saved_groups):
            adapted_group = dict(saved_group)
            if index < len(optimizer.param_groups):
                current_group = optimizer.param_groups[index]
                for key in OPTIMIZER_RUNTIME_OPTION_KEYS:
                    if key in current_group:
                        adapted_group[key] = current_group[key]
            adapted_groups.append(adapted_group)
        state_to_load["param_groups"] = adapted_groups
    optimizer.load_state_dict(state_to_load)


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
    data_metadata: dict[str, Any] | None = None,
    model_profile: str | None = None,
    requested_model_profile: str | None = None,
    resolved_model_profile: str | None = None,
    resume_architecture_metadata: dict[str, Any] | None = None,
    resume_model_profile_override: bool = False,
    checkpoint_model_profile: str | None = None,
    checkpoint_model_config_metadata: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    model_config_metadata: dict[str, Any] | None = None,
):
    if normalizer is None:
        raise ValueError("Saving a checkpoint requires a fitted normalizer")
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError("Saving a checkpoint requires model_config")
    if not isinstance(model_config_metadata, dict) or not model_config_metadata:
        raise ValueError("Saving a checkpoint requires model_config_metadata")
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    normalizer_state = normalizer.state_dict() if normalizer is not None else None
    base_model_state = _base_model_state_dict(model_without_ddp)
    ema_model_state = _ema_model_state_dict(model_without_ddp)
    resume_state = _resume_state_dict(model_without_ddp)
    has_ema = ema_model_state is not None
    lr_scheduler_metadata = {
        "lr_scheduler": getattr(args, "lr_scheduler", None),
        "resolved_lr_scheduler": getattr(
            args,
            "resolved_lr_scheduler",
            getattr(args, "lr_scheduler", None),
        ),
        "min_lr": getattr(args, "min_lr", None),
        "warmup_epochs": getattr(args, "warmup_epochs", None),
        "warmup_start_factor": getattr(args, "warmup_start_factor", None),
        "plateau_factor": getattr(args, "plateau_factor", None),
        "plateau_patience": getattr(args, "plateau_patience", None),
        "plateau_threshold": getattr(args, "plateau_threshold", None),
    }
    payload = {
        "model": base_model_state,
        "model_ema": ema_model_state,
        "model_for_resume": resume_state,
        "use_ema": bool(getattr(args, "use_ema", False) or has_ema),
        "has_ema": bool(has_ema),
        "ema_decay": model_without_ddp.decay if has_ema else None,
        "ema_warmup": model_without_ddp.warmup if has_ema else None,
        "inference_weight": "ema" if has_ema else "raw",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_schedule": lr_schedule.state_dict() if lr_schedule is not None else None,
        "epoch": epoch,
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "args": args,
        "normalizer": normalizer_state,
        "data_shape": tuple(data_shape) if data_shape is not None else None,
        "num_channels": int(num_channels) if num_channels is not None else None,
        "normalization": {
            "type": getattr(normalizer, "normalization_type", "channelwise_standardization"),
            "eps": getattr(normalizer, "eps", None),
        },
        "data_metadata": data_metadata,
        "model_profile": model_profile or getattr(args, "model_profile", None),
        "requested_model_profile": requested_model_profile or getattr(args, "model_profile", None),
        "resolved_model_profile": resolved_model_profile or model_profile or getattr(args, "model_profile", None),
        "resume_architecture_metadata": resume_architecture_metadata,
        "resume_model_profile_override": bool(resume_model_profile_override),
        "checkpoint_model_profile": checkpoint_model_profile,
        "checkpoint_model_config_metadata": checkpoint_model_config_metadata,
        "model_config": model_config,
        "model_config_metadata": model_config_metadata,
        **lr_scheduler_metadata,
    }

    if loss_scaler is not None:
        checkpoint_paths = [
            output_dir / (f"fm4{args.dataset}-checkpoint-{epoch_name}.pth"),
            output_dir / f"fm4{args.dataset}-checkpoint.pth",
        ]
        for checkpoint_path in checkpoint_paths:
            save_on_master(payload, checkpoint_path)
    else:
        client_state = {
            "epoch": epoch,
            "normalizer": normalizer_state,
            "data_metadata": data_metadata,
            "model_profile": model_profile or getattr(args, "model_profile", None),
            "requested_model_profile": requested_model_profile or getattr(args, "model_profile", None),
            "resolved_model_profile": resolved_model_profile or model_profile or getattr(args, "model_profile", None),
            "resume_architecture_metadata": resume_architecture_metadata,
            "resume_model_profile_override": bool(resume_model_profile_override),
            "checkpoint_model_profile": checkpoint_model_profile,
            "checkpoint_model_config_metadata": checkpoint_model_config_metadata,
            "model_config": model_config,
            "model_config_metadata": model_config_metadata,
            **lr_scheduler_metadata,
        }
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag=f"fm4{args.dataset}-checkpoint-{epoch_name}",
            client_state=client_state,
        )

    if final:
        if loss_scaler is not None:
            save_on_master(payload, output_dir / f"fm4{args.dataset}.pth")
        else:
            client_state = {
                "epoch": epoch,
                "normalizer": normalizer_state,
                "data_metadata": data_metadata,
                "model_profile": model_profile or getattr(args, "model_profile", None),
                "requested_model_profile": requested_model_profile or getattr(args, "model_profile", None),
                "resolved_model_profile": resolved_model_profile or model_profile or getattr(args, "model_profile", None),
                "resume_architecture_metadata": resume_architecture_metadata,
                "resume_model_profile_override": bool(resume_model_profile_override),
                "checkpoint_model_profile": checkpoint_model_profile,
                "checkpoint_model_config_metadata": checkpoint_model_config_metadata,
                "model_config": model_config,
                "model_config_metadata": model_config_metadata,
                **lr_scheduler_metadata,
            }
            model.save_checkpoint(
                save_dir=args.output_dir,
                tag=f"fm4{args.dataset}",
                client_state=client_state,
            )


def inspect_checkpoint_architecture(path: str | Path) -> dict[str, Any]:
    path_str = str(path)
    result: dict[str, Any] = {
        "checkpoint_path": path_str,
        "has_checkpoint": False,
        "has_model_config": False,
        "checkpoint_model_profile": None,
        "checkpoint_model_config": None,
        "checkpoint_model_config_metadata": None,
        "checkpoint_num_channels": None,
        "checkpoint_schema_version": None,
    }

    if path_str.startswith("https"):
        payload = torch.hub.load_state_dict_from_url(
            path_str, map_location="cpu", check_hash=True
        )
        result["has_checkpoint"] = True
    else:
        checkpoint_path = Path(path_str)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        result["checkpoint_path"] = str(checkpoint_path)
        result["has_checkpoint"] = True
        payload = read_checkpoint(checkpoint_path)

    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {result['checkpoint_path']}")
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}, "
            f"got {payload.get('checkpoint_schema_version')!r}: {result['checkpoint_path']}"
        )

    model_config = payload.get("model_config")
    model_config_metadata = payload.get("model_config_metadata")
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(f"Checkpoint has no model_config: {result['checkpoint_path']}")
    if not isinstance(model_config_metadata, dict) or not model_config_metadata:
        raise ValueError(f"Checkpoint has no model_config_metadata: {result['checkpoint_path']}")

    result["has_model_config"] = bool(model_config)
    result["checkpoint_model_config"] = model_config
    result["checkpoint_model_config_metadata"] = model_config_metadata
    result["checkpoint_model_profile"] = (
        payload.get("model_profile")
        or (model_config or {}).get("architecture_profile")
        or (model_config_metadata or {}).get("architecture_profile")
    )
    if not result["checkpoint_model_profile"]:
        raise ValueError(
            f"Checkpoint has no model_profile or architecture_profile: {result['checkpoint_path']}"
        )
    result["checkpoint_num_channels"] = payload.get("num_channels")
    result["checkpoint_schema_version"] = payload.get("checkpoint_schema_version")
    result["legacy_compatibility"] = payload.get("legacy_compatibility")
    return result


def load_model(args, model_without_ddp, optimizer, loss_scaler, lr_schedule) -> dict[str, Any] | None:
    if not args.resume:
        if getattr(args, 'resume_add_ema', False):
            raise ValueError('--resume_add_ema requires --resume')
        return None
    if args.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = read_checkpoint(args.resume, args.dataset)
    if checkpoint.get("legacy_compatibility"):
        legacy_base = model_without_ddp.model if _is_ema_module(model_without_ddp) else model_without_ddp
        validate_legacy_optimizer(legacy_base, checkpoint)
        expected = checkpoint["resolved_lr_scheduler"]
        actual = getattr(args, "resolved_lr_scheduler", getattr(args, "lr_scheduler", None))
        if actual != expected and not getattr(args, "resume_reset_lr_schedule", False):
            raise ValueError(f"Legacy resume requires --lr_scheduler {expected}; use scripts/train/resume_bak.py")
    _load_resume_state(model_without_ddp, checkpoint, add_ema=getattr(args, 'resume_add_ema', False))
    if _is_ema_module(model_without_ddp) and checkpoint.get('ema_decay') is not None:
        decay = float(checkpoint['ema_decay'])
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError('Invalid saved EMA decay')
        model_without_ddp.decay = decay
    if _is_ema_module(model_without_ddp):
        # Old EMA checkpoints predate this metadata and used warmup.
        if checkpoint.get('has_ema') or checkpoint.get('use_ema'):
            model_without_ddp.warmup = bool(checkpoint.get('ema_warmup', True))
    print(f"Resume {args.dataset} checkpoint {args.resume}")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}, "
            f"got {checkpoint.get('checkpoint_schema_version')!r}"
        )
    if not isinstance(checkpoint.get("normalizer"), dict):
        raise ValueError("Checkpoint has no saved normalizer")
    if (
        "optimizer" in checkpoint
        and checkpoint.get("optimizer") is not None
        and "epoch" in checkpoint
    ):
        _load_optimizer_state_preserving_runtime_options(
            optimizer, _optimizer_state_without_ema_shadows(model_without_ddp, checkpoint["optimizer"])
        )
        _apply_resume_optimizer_betas(optimizer, getattr(args, "resume_optimizer_betas", None))
        args.effective_optimizer_betas = [list(group["betas"]) for group in optimizer.param_groups]
        print(f"Effective resumed optimizer betas: {args.effective_optimizer_betas}")
        print(f"Effective resumed optimizer learning rates: {[group['lr'] for group in optimizer.param_groups]}")
        if checkpoint.get("lr_schedule") is not None and not getattr(args, "resume_reset_lr_schedule", False):
            try:
                lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            except Exception as exc:
                warnings.warn(
                    "Could not load checkpoint LR scheduler state into the current scheduler; "
                    "continuing with the current --lr_scheduler initialization. "
                    f"checkpoint_lr_scheduler={checkpoint.get('resolved_lr_scheduler') or checkpoint.get('lr_scheduler')}, "
                    f"current_lr_scheduler={getattr(args, 'resolved_lr_scheduler', getattr(args, 'lr_scheduler', None))}. "
                    f"Original error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        args.start_epoch = checkpoint["epoch"] + 1
        if "scaler" in checkpoint and checkpoint.get("scaler") is not None:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        print("With optim & sched!")
    return checkpoint

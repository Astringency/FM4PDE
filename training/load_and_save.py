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


def _looks_like_ema_state_dict(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    return (
        "num_updates" in state
        or any(str(key).startswith("shadow_params.") for key in state)
        or any(str(key).startswith("model.") for key in state)
    )


def _strip_model_prefix_state(state: dict[str, Any]) -> dict[str, Any]:
    stripped = {
        str(key)[len("model.") :]: value
        for key, value in state.items()
        if str(key).startswith("model.")
    }
    if not stripped:
        raise ValueError("EMA wrapper state_dict has no model.* keys to load into a plain model")
    return _clone_state_dict(stripped)


def _sync_ema_shadow_params(model_without_ddp) -> None:
    shadow_params = list(model_without_ddp.shadow_params)
    trainable_params = [p for p in model_without_ddp.model.parameters() if p.requires_grad]
    if len(shadow_params) != len(trainable_params):
        raise ValueError(
            "Cannot initialize EMA shadow parameters from plain checkpoint because the "
            f"shadow/trainable counts differ: {len(shadow_params)} vs {len(trainable_params)}"
        )
    for shadow, param in zip(shadow_params, trainable_params):
        shadow.data.copy_(param.detach().data)


def _load_resume_state(model_without_ddp, checkpoint: dict[str, Any]) -> None:
    if "model_for_resume" in checkpoint:
        resume_state = checkpoint["model_for_resume"]
    else:
        resume_state = checkpoint["model"]

    try:
        model_without_ddp.load_state_dict(resume_state)
        return
    except RuntimeError as exc:
        first_error = exc

    if _is_ema_module(model_without_ddp):
        if isinstance(resume_state, dict) and not _looks_like_ema_state_dict(resume_state):
            model_without_ddp.model.load_state_dict(resume_state)
            _sync_ema_shadow_params(model_without_ddp)
            warnings.warn(
                "Loaded a plain model checkpoint into an EMA wrapper and initialized "
                "EMA shadow parameters from the loaded model weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
    elif _looks_like_ema_state_dict(resume_state):
        try:
            model_without_ddp.load_state_dict(_strip_model_prefix_state(resume_state))
            warnings.warn(
                "Checkpoint model state is an EMA wrapper state_dict. Loaded the inner "
                "raw model.* weights into the plain model. Resume with --use_ema to "
                "continue EMA training state.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Checkpoint contains an EMA wrapper state_dict, but it could not be "
                "loaded into the current plain model. Resume with --use_ema or use a "
                "checkpoint whose model/model_for_resume state matches this model."
            ) from exc

    if (
        not _is_ema_module(model_without_ddp)
        and "model" in checkpoint
        and checkpoint["model"] is not resume_state
        and isinstance(checkpoint["model"], dict)
        and not _looks_like_ema_state_dict(checkpoint["model"])
    ):
        model_without_ddp.load_state_dict(checkpoint["model"])
        warnings.warn(
            "model_for_resume did not match the current plain model; loaded checkpoint['model'] "
            "raw weights instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    raise RuntimeError(
        "Could not load checkpoint model state into the current model. If this is an "
        "EMA checkpoint, set --use_ema to resume EMA training, or load it through the "
        "sampling checkpoint loader to extract plain weights."
    ) from first_error


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
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    normalizer_state = normalizer.state_dict() if normalizer is not None else None
    base_model_state = _base_model_state_dict(model_without_ddp)
    ema_model_state = _ema_model_state_dict(model_without_ddp)
    resume_state = _resume_state_dict(model_without_ddp)
    has_ema = ema_model_state is not None
    payload = {
        "model": base_model_state,
        "model_ema": ema_model_state,
        "model_for_resume": resume_state,
        "use_ema": bool(getattr(args, "use_ema", False) or has_ema),
        "has_ema": bool(has_ema),
        "inference_weight": "ema" if has_ema else "raw",
        "checkpoint_schema_version": 2,
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
        "data_metadata": data_metadata,
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
            }
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
    _load_resume_state(model_without_ddp, checkpoint)
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

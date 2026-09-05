"""Compatibility for the five unversioned FM4PDE_bak pretrained checkpoints.

No import from the backup checkout is needed at runtime. All legacy networks
have one attention head and no Fourier features, so the old and current QKV
orders are identical. Weight names AND parameter order are retained.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

LEGACY_PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded", "burger")


def legacy_model_config(pde: str) -> dict[str, Any]:
    if pde not in LEGACY_PDES:
        raise ValueError(f"Unsupported legacy PDE: {pde!r}")
    small = pde in {"poisson", "helmholtz"}
    return dict(
        in_channels=1 if pde == "burger" else 2,
        out_channels=1 if pde == "burger" else 2,
        model_channels=128, num_res_blocks=4,
        attention_resolutions=(32,) if small else (2,),
        dropout=0.1, channel_mult=(1, 2, 2) if small else (1, 2, 4),
        conv_resample=True, dims=2, num_classes=None, use_checkpoint=False,
        num_heads=1, num_head_channels=-1, use_scale_shift_norm=not small,
        resblock_updown=False, with_value_fourier_features=False,
        with_coordinate_fourier_features=False, scalar_conditioning=False,
        architecture_family="fm4pde_bak_unet", architecture_profile="legacy",
        axis_semantics="time_space" if pde == "burger" else "spatial_2d",
        notes="FM4PDE_bak: single-head QKV orders are mathematically identical",
    )


def legacy_sampling_normalizer(pde: str):
    from data.transform import LegacyAffineNormalizer

    # Physical = old_inverse((latent + 1) / 2), NOT old_inverse(latent).
    affine = {
        "poisson": ([1.25, 0.5 / 36.5], [1.25, 0.5 / 36.5]),
        "helmholtz": ([0.01515, 0.0004], [2.14645, 0.0273]),
        "darcy": ([10.0, 1.4 / 115], [2.5, 0.5 / 115]),
        "nsnonbounded": ([0.8, 0.8], [0.8, 0.8]),
        "burger": ([0.7075], [0.7075]),
    }
    mean, std = affine[pde]
    return LegacyAffineNormalizer(torch.tensor(mean), torch.tensor(std), eps=0.0, pde=pde)


def adapt_legacy_checkpoint(payload: Any, pde: str | None = None) -> Any:
    """Adapt only recognized unversioned checkpoints; leave versioned validation strict."""
    if not isinstance(payload, dict) or "checkpoint_schema_version" in payload:
        return payload
    required = {"model", "optimizer", "lr_schedule", "epoch", "scaler", "args"}
    if not required.issubset(payload):
        return payload
    saved = payload["args"]
    saved = saved if isinstance(saved, dict) else vars(saved)
    dataset = saved.get("dataset")
    if dataset not in LEGACY_PDES:
        raise ValueError(f"Unsupported unversioned checkpoint dataset={dataset!r}")
    if pde is not None and pde != dataset:
        raise ValueError(f"Legacy checkpoint dataset={dataset!r} does not match requested PDE={pde!r}")
    if saved.get("use_ema", False):
        raise ValueError("Legacy EMA checkpoints require an explicit EMA conversion")
    cfg = legacy_model_config(dataset)
    from models.model_configs import model_config_metadata_from_config

    result = dict(payload)
    result.update(
        checkpoint_schema_version=3, model_config=cfg,
        model_config_metadata=model_config_metadata_from_config(cfg),
        model_profile="legacy", model_for_resume=payload["model"],
        normalizer=legacy_sampling_normalizer(dataset).state_dict(),
        normalization={"type": "legacy_affine", "source": "FM4PDE_bak/data/transform_old.py"},
        num_channels=cfg["in_channels"], use_ema=False, has_ema=False,
        resolved_lr_scheduler="linear" if saved.get("decay_lr") else "constant",
        legacy_compatibility={
            "source_schema": "unversioned_fm4pde_bak",
            "dataset": dataset,
            "sampling_normalization": "transform_old.inverse((latent+1)/2)",
            "training_normalization": "refit_minmax_from_original_training_data",
            "training_statistics_saved": False,
            "single_head_attention_order_equivalent": True,
        },
    )
    return result


def read_checkpoint(path: str | Path, pde: str | None = None) -> dict[str, Any]:
    # Local checkpoints include argparse.Namespace. mmap avoids materializing
    # multi-GB optimizer buffers for architecture inspection and inference.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return adapt_legacy_checkpoint(payload, pde)


def validate_legacy_optimizer(model, checkpoint: dict[str, Any]) -> None:
    """Adam matches saved IDs by order, so check names/order and moment shapes first."""
    names = [name for name, _ in model.named_parameters()]
    if list(checkpoint["model"]) != names:
        raise ValueError("Legacy model parameter order differs; refusing unsafe optimizer restoration")
    opt = checkpoint.get("optimizer")
    if not isinstance(opt, dict):
        raise ValueError("Legacy checkpoint has no optimizer state for resume")
    ids = [pid for group in opt["param_groups"] for pid in group["params"]]
    if len(ids) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Legacy optimizer parameter count/order is invalid")
    for (name, param), pid in zip(model.named_parameters(), ids):
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = opt["state"].get(pid, {}).get(key)
            if value is not None and value.shape != param.shape:
                raise ValueError(f"Legacy optimizer {key} shape mismatch for {name}")

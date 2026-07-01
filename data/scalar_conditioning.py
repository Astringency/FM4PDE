from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def normalize_scalar_conditioning_params(
    params: Any,
    *,
    option_name: str = "--scalar_conditioning_params",
) -> tuple[str, ...]:
    if params is None:
        return ()
    if isinstance(params, str):
        values = (params,)
    else:
        values = tuple(params)
    normalized = tuple(str(value) for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{option_name} cannot contain empty parameter names")
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if duplicates:
        raise ValueError(
            f"{option_name} contains duplicate parameter names: {duplicates}"
        )
    return normalized


def scalar_conditioning_metadata_disabled() -> dict[str, Any]:
    return {
        "enabled": False,
        "params": [],
        "dim": 0,
        "mean": [],
        "std": [],
        "raw_std": [],
        "std_was_clamped": [],
        "normalization": None,
    }


def scalar_conditioning_params_from_config(
    model_config: Mapping[str, Any],
    model_config_metadata: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    metadata = model_config_metadata or {}
    params = model_config.get("scalar_conditioning_params")
    if params is None or params == ():
        params = metadata.get("scalar_conditioning_params", ())
    return normalize_scalar_conditioning_params(params)


def fit_scalar_conditioning(
    *,
    model_config: Mapping[str, Any],
    pde_names: Sequence[str],
    train_loader_metadata: Mapping[str, Any],
    val_loader_metadata: Mapping[str, Any],
    train_sample_count: int,
    val_sample_count: int,
    eps: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    enabled = bool(model_config.get("scalar_conditioning", False))
    params = scalar_conditioning_params_from_config(model_config)
    if not enabled:
        return None, None, scalar_conditioning_metadata_disabled()
    if not params:
        raise ValueError("scalar_conditioning=True requires scalar_conditioning_params")
    expected_dim = int(model_config.get("scalar_conditioning_dim") or len(params))
    if expected_dim != len(params):
        raise ValueError(
            "scalar_conditioning_dim must match scalar_conditioning_params; "
            f"got dim={expected_dim}, params={list(params)}"
        )

    train_raw = scalar_conditioning_tensor_from_metadata(
        pde_names=pde_names,
        loader_metadata=train_loader_metadata,
        params=params,
        expected_sample_count=train_sample_count,
        split_name="training",
    )
    val_raw = scalar_conditioning_tensor_from_metadata(
        pde_names=pde_names,
        loader_metadata=val_loader_metadata,
        params=params,
        expected_sample_count=val_sample_count,
        split_name="validation",
    )
    mean = train_raw.mean(dim=0, keepdim=True)
    raw_std = train_raw.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(raw_std < float(eps), torch.ones_like(raw_std), raw_std)
    train_standardized = (train_raw - mean) / std
    val_standardized = (val_raw - mean) / std
    metadata = {
        "enabled": True,
        "params": list(params),
        "dim": len(params),
        "mean": [float(value) for value in mean.view(-1).tolist()],
        "std": [float(value) for value in std.view(-1).tolist()],
        "raw_std": [float(value) for value in raw_std.view(-1).tolist()],
        "std_was_clamped": [bool(value) for value in (raw_std < float(eps)).view(-1).tolist()],
        "normalization": "train_mean_std",
    }
    return train_standardized.contiguous(), val_standardized.contiguous(), metadata


def scalar_conditioning_tensor_from_metadata(
    *,
    pde_names: Sequence[str],
    loader_metadata: Mapping[str, Any],
    params: Sequence[str],
    expected_sample_count: int,
    split_name: str,
) -> torch.Tensor:
    param_names = normalize_scalar_conditioning_params(params)
    parts: list[torch.Tensor] = []
    missing_pdes = []
    for pde_name in pde_names:
        entry = loader_metadata.get(pde_name, {}) if isinstance(loader_metadata, Mapping) else {}
        pde_params = _pde_params_from_metadata_entry(entry)
        if not pde_params:
            missing_pdes.append(str(pde_name))
            continue
        parts.append(
            scalar_conditioning_tensor_from_pde_params(
                pde_params=pde_params,
                params=param_names,
                expected_sample_count=None,
                source_name=f"{split_name} loader metadata for PDE {pde_name!r}",
                allow_scalar_broadcast=False,
            )
        )

    if missing_pdes:
        raise ValueError(
            f"Scalar conditioning was requested, but {split_name} loader metadata has no "
            f"pde_params for PDE(s): {missing_pdes}"
        )
    if not parts:
        raise ValueError(f"Scalar conditioning was requested, but no {split_name} scalar tensors were built")
    tensor = torch.cat(parts, dim=0).to(torch.float32)
    if int(tensor.shape[0]) != int(expected_sample_count):
        raise ValueError(
            f"Built {split_name} scalar conditioning tensor with {int(tensor.shape[0])} samples, "
            f"but data has {int(expected_sample_count)} samples"
        )
    return tensor


def scalar_conditioning_tensor_from_pde_params(
    *,
    pde_params: Mapping[str, Any],
    params: Sequence[str],
    expected_sample_count: int | None,
    source_name: str,
    allow_scalar_broadcast: bool = True,
) -> torch.Tensor:
    param_names = normalize_scalar_conditioning_params(params)
    columns = []
    for param in param_names:
        if param not in pde_params:
            raise ValueError(
                f"Requested scalar conditioning parameter {param!r} is missing from "
                f"{source_name}; available={sorted(str(key) for key in pde_params)}"
            )
        try:
            tensor = torch.as_tensor(pde_params[param], dtype=torch.float32).detach().cpu()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Scalar conditioning parameter {param!r} from {source_name} must be numeric"
            ) from exc
        if tensor.ndim == 0:
            if expected_sample_count is not None and allow_scalar_broadcast:
                tensor = tensor.repeat(int(expected_sample_count))
            else:
                tensor = tensor.reshape(1)
        if tensor.ndim != 1:
            raise ValueError(
                f"Scalar conditioning parameter {param!r} from {source_name} "
                f"must be sample-aligned [N], got shape={tuple(tensor.shape)}"
            )
        columns.append(tensor)
    lengths = {int(column.shape[0]) for column in columns}
    if len(lengths) != 1:
        raise ValueError(
            f"Scalar conditioning parameters from {source_name} have inconsistent "
            f"sample counts: {sorted(lengths)}"
        )
    tensor = torch.stack(columns, dim=1).to(torch.float32)
    if expected_sample_count is not None and int(tensor.shape[0]) != int(expected_sample_count):
        raise ValueError(
            f"Built scalar conditioning tensor from {source_name} with {int(tensor.shape[0])} samples, "
            f"but expected {int(expected_sample_count)}"
        )
    return tensor


def scalar_conditioning_metadata_from_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or not payload:
        return scalar_conditioning_metadata_disabled()
    data_metadata = _mapping_or_empty(payload.get("data_metadata"))
    model_config = _mapping_or_empty(payload.get("model_config"))
    model_config_metadata = _mapping_or_empty(payload.get("model_config_metadata"))
    selected_model_config_metadata = _mapping_or_empty(payload.get("selected_model_config_metadata"))

    enabled = any(
        bool(value)
        for value in (
            data_metadata.get("scalar_conditioning_enabled"),
            data_metadata.get("scalar_conditioning"),
            model_config.get("scalar_conditioning"),
            model_config_metadata.get("scalar_conditioning"),
            selected_model_config_metadata.get("scalar_conditioning"),
        )
    )
    if not enabled:
        return scalar_conditioning_metadata_disabled()

    params = _first_present(
        data_metadata.get("scalar_conditioning_params"),
        model_config.get("scalar_conditioning_params"),
        model_config_metadata.get("scalar_conditioning_params"),
        selected_model_config_metadata.get("scalar_conditioning_params"),
    )
    param_names = normalize_scalar_conditioning_params(params)
    if not param_names:
        raise ValueError("scalar-conditioned checkpoint does not record scalar_conditioning_params")

    dim_value = _first_present(
        data_metadata.get("scalar_conditioning_dim"),
        model_config.get("scalar_conditioning_dim"),
        model_config_metadata.get("scalar_conditioning_dim"),
        selected_model_config_metadata.get("scalar_conditioning_dim"),
        len(param_names),
    )
    dim = int(dim_value)
    if dim != len(param_names):
        raise ValueError(
            "scalar-conditioned checkpoint metadata is inconsistent; "
            f"scalar_conditioning_dim={dim}, scalar_conditioning_params={list(param_names)}"
        )

    mean_value = _first_present(
        data_metadata.get("scalar_conditioning_mean"),
        payload.get("scalar_conditioning_mean"),
    )
    std_value = _first_present(
        data_metadata.get("scalar_conditioning_std"),
        payload.get("scalar_conditioning_std"),
    )
    mean, std = _validate_standardization_stats(
        mean_value=mean_value,
        std_value=std_value,
        dim=dim,
        context="scalar-conditioned checkpoint",
    )
    raw_std = data_metadata.get("scalar_conditioning_raw_std", [])
    std_was_clamped = data_metadata.get("scalar_conditioning_std_was_clamped", [])
    normalization = data_metadata.get("scalar_conditioning_normalization") or "train_mean_std"
    return {
        "enabled": True,
        "params": list(param_names),
        "dim": dim,
        "mean": [float(value) for value in mean.view(-1).tolist()],
        "std": [float(value) for value in std.view(-1).tolist()],
        "raw_std": _jsonable_list(raw_std),
        "std_was_clamped": _jsonable_list(std_was_clamped),
        "normalization": normalization,
    }


def standardize_scalar_conditioning(
    *,
    pde_params: Mapping[str, Any],
    metadata: Mapping[str, Any],
    expected_sample_count: int,
    source_name: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not bool(metadata.get("enabled", False)):
        raise ValueError("Cannot standardize scalar conditioning with disabled metadata")
    params = normalize_scalar_conditioning_params(metadata.get("params", ()))
    dim = int(metadata.get("dim") or len(params))
    if dim != len(params):
        raise ValueError(
            "scalar conditioning metadata is inconsistent; "
            f"dim={dim}, params={list(params)}"
        )
    mean, std = _validate_standardization_stats(
        mean_value=metadata.get("mean"),
        std_value=metadata.get("std"),
        dim=dim,
        context="scalar conditioning metadata",
    )
    raw = scalar_conditioning_tensor_from_pde_params(
        pde_params=pde_params,
        params=params,
        expected_sample_count=expected_sample_count,
        source_name=source_name,
        allow_scalar_broadcast=True,
    )
    standardized = (raw - mean) / std
    if device is not None or dtype is not None:
        to_kwargs: dict[str, Any] = {}
        if device is not None:
            to_kwargs["device"] = device
        if dtype is not None:
            to_kwargs["dtype"] = dtype
        standardized = standardized.to(**to_kwargs)
    return standardized.contiguous()


def _pde_params_from_metadata_entry(entry: Any) -> Mapping[str, Any]:
    if isinstance(entry, Mapping) and isinstance(entry.get("pde_params"), Mapping):
        return entry["pde_params"]
    if isinstance(entry, Mapping):
        return entry
    return {}


def _validate_standardization_stats(
    *,
    mean_value: Any,
    std_value: Any,
    dim: int,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mean_value is None or std_value is None:
        raise ValueError(f"{context} is missing scalar conditioning mean/std")
    try:
        mean = torch.as_tensor(mean_value, dtype=torch.float32).reshape(1, -1)
        std = torch.as_tensor(std_value, dtype=torch.float32).reshape(1, -1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} scalar conditioning mean/std must be numeric") from exc
    if int(mean.shape[1]) != int(dim) or int(std.shape[1]) != int(dim):
        raise ValueError(
            f"{context} scalar conditioning stats must have length {dim}; "
            f"got mean={int(mean.shape[1])}, std={int(std.shape[1])}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"{context} scalar conditioning mean/std must be finite")
    if torch.any(std <= 0):
        raise ValueError(f"{context} scalar conditioning std must be positive")
    return mean, std


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _jsonable_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]

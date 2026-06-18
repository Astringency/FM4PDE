from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


@dataclass
class SplitState:
    coef: Any
    sol: Any


CHANNEL_SPLITS: dict[str, tuple[int, int]] = {
    "darcy": (1, 1),
    "poisson": (1, 1),
    "helmholtz": (1, 1),
    "nsnonbounded": (1, 1),
    "reaction_diffusion": (2, 2),
    "shallow_water": (3, 3),
    "heat": (1, 1),
    "wave": (2, 2),
    "advection_diffusion": (1, 1),
    "steady_heat_conduction": (1, 1),
}


def split_pair_state(x: Any, pde: str, img_channels: int | None = None) -> SplitState:
    """Split a physical pair state into coefficient/source and solution tensors."""
    if getattr(x, "ndim", None) != 4:
        raise ValueError(f"Expected BCHW state, got shape={getattr(x, 'shape', None)}")
    channels = int(x.shape[1])
    if pde == "burger":
        if channels != 1:
            raise ValueError(f"Burgers expects one channel, got {channels}")
        return SplitState(coef=x[:, 0:1], sol=x[:, 0:1])
    expected = CHANNEL_SPLITS.get(pde)
    if expected is not None:
        total = expected[0] + expected[1]
        if channels != total:
            raise ValueError(
                f"{pde} expects {total} FM channels ({expected[0]} coefficient/source + "
                f"{expected[1]} solution) under the current ablation channel definition, got {channels}. "
                "If this checkpoint or data was trained with scalar PDE parameters materialized as "
                "constant fields, retrain it using sample-level pde_params metadata instead."
            )
        return SplitState(coef=x[:, : expected[0]], sol=x[:, expected[0] :])
    if img_channels is not None and img_channels > 0 and channels != img_channels:
        raise ValueError(f"img_channels={img_channels} does not match state channels={channels}")
    if channels % 2 != 0:
        raise ValueError(f"Cannot split odd channel state for {pde}: channels={channels}")
    half = channels // 2
    return SplitState(coef=x[:, :half], sol=x[:, half:])


def compose_pair_state(a: Any, u: Any) -> Any:
    import torch

    _assert_bchw(a, "coef")
    _assert_bchw(u, "sol")
    if a.shape[0] != u.shape[0] or a.shape[-2:] != u.shape[-2:]:
        raise ValueError(f"Cannot compose tensors with shapes coef={a.shape}, sol={u.shape}")
    return torch.cat([a, u], dim=1)


def standardized_to_physical_state(
    x_standardized: Any,
    pde: str,
    img_channels: int | None = None,
    normalizer: Any | None = None,
    legacy_minmax: bool = False,
) -> SplitState:
    """Inverse-transform a full standardized pair state, then split it into physical fields."""
    _assert_bchw(x_standardized, "model_state")
    if normalizer is not None:
        physical_pair = normalizer.inverse_transform(x_standardized)
        return split_pair_state(physical_pair, pde, img_channels)
    if legacy_minmax:
        warnings.warn(
            "legacy_minmax=True uses the old [-1,1] -> [0,1] and transform_old path. "
            "This is only for explicit legacy checkpoints.",
            RuntimeWarning,
            stacklevel=2,
        )
        unit_pair = (x_standardized + 1.0) / 2.0
        split = split_pair_state(unit_pair, pde, img_channels)
        return SplitState(*_legacy_inverse_transform(split.coef, split.sol, pde))
    warnings.warn(
        "No normalizer was provided; treating standardized model state as physical identity.",
        RuntimeWarning,
        stacklevel=2,
    )
    return split_pair_state(x_standardized, pde, img_channels)


def inverse_transform_state(
    a_raw: Any,
    u_raw: Any,
    pde: str,
    transformer: Any | None = None,
    legacy_minmax: bool = False,
) -> tuple[Any, Any]:
    """Compatibility wrapper; new code should use standardized_to_physical_state on full pair tensors."""
    _assert_bchw(a_raw, "coef")
    _assert_bchw(u_raw, "sol")
    if transformer is not None:
        if hasattr(transformer, "inverse_transform_sample"):
            return transformer.inverse_transform_sample(a_raw, u_raw)
        if hasattr(transformer, "inverse_transform"):
            return transformer.inverse_transform(a_raw, u_raw)
        raise TypeError(f"Unsupported transformer type: {type(transformer)!r}")
    if legacy_minmax:
        return _legacy_inverse_transform(a_raw, u_raw, pde)
    return a_raw, u_raw


def _legacy_inverse_transform(a_raw: Any, u_raw: Any, pde: str) -> tuple[Any, Any]:
    try:
        from data.transform_old import PDEtransform as OldPDEtransform

        return OldPDEtransform(pde).inverse_transform(a_raw, u_raw)
    except Exception:
        return a_raw, u_raw


def _assert_bchw(x: Any, name: str) -> None:
    if getattr(x, "ndim", None) != 4:
        raise ValueError(f"{name} must be BCHW, got shape={getattr(x, 'shape', None)}")

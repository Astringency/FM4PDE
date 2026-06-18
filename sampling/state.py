from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data.specs import get_pde_spec


@dataclass
class SplitState:
    coef: Any
    sol: Any


def split_pair_state(x: Any, pde: str, img_channels: int | None = None) -> SplitState:
    """Split a physical pair state into coefficient/source and solution tensors."""
    if getattr(x, "ndim", None) != 4:
        raise ValueError(f"Expected BCHW state, got shape={getattr(x, 'shape', None)}")
    channels = int(x.shape[1])
    if pde == "burger":
        if channels != 1:
            raise ValueError(f"Burgers expects one channel, got {channels}")
        return SplitState(coef=x[:, 0:1], sol=x[:, 0:1])
    spec = get_pde_spec(pde)
    expected = (spec.coef_channels, spec.sol_channels)
    total = expected[0] + expected[1]
    if channels != total:
        raise ValueError(
            f"{pde} expects {total} FM channels ({expected[0]} coefficient/source + "
            f"{expected[1]} solution) under the current data spec, got {channels}. "
            "Scalar PDE parameters must be passed as sample-level pde_params metadata, not FM channels."
        )
    return SplitState(coef=x[:, : expected[0]], sol=x[:, expected[0] :])


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
) -> SplitState:
    """Inverse-transform a full standardized pair state, then split it into physical fields."""
    _assert_bchw(x_standardized, "model_state")
    if normalizer is not None:
        physical_pair = normalizer.inverse_transform(x_standardized)
        return split_pair_state(physical_pair, pde, img_channels)
    raise ValueError(
        "No PDEStandardizer normalizer was provided for sampling. "
        "Use dry_run to create an explicit identity normalizer, or use a checkpoint with a saved normalizer."
    )


def _assert_bchw(x: Any, name: str) -> None:
    if getattr(x, "ndim", None) != 4:
        raise ValueError(f"{name} must be BCHW, got shape={getattr(x, 'shape', None)}")

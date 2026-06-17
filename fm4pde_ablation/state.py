from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SplitState:
    coef: Any
    sol: Any


def split_pair_state(x: Any, pde: str, img_channels: int | None = None) -> SplitState:
    """Split a raw normalized pair state into coefficient/source and solution tensors."""
    if getattr(x, "ndim", None) != 4:
        raise ValueError(f"Expected BCHW state, got shape={getattr(x, 'shape', None)}")
    channels = int(x.shape[1])
    if pde == "burger":
        if channels != 1:
            raise ValueError(f"Burgers expects one channel, got {channels}")
        return SplitState(coef=x[:, 0:1], sol=x[:, 0:1])
    expected = {
        "reaction_diffusion": (2, 2),
        "shallow_water": (3, 3),
    }.get(pde)
    if expected is not None:
        total = expected[0] + expected[1]
        if channels != total:
            raise ValueError(f"{pde} expects {total} channels, got {channels}")
        return SplitState(coef=x[:, : expected[0]], sol=x[:, expected[0] :])
    if img_channels is not None and channels != img_channels:
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


def inverse_transform_state(a_raw: Any, u_raw: Any, pde: str, transformer: Any | None = None) -> tuple[Any, Any]:
    """Map normalized model state to physical PDE fields."""
    _assert_bchw(a_raw, "coef")
    _assert_bchw(u_raw, "sol")
    if transformer is not None:
        if hasattr(transformer, "inverse_transform_sample"):
            return transformer.inverse_transform_sample(a_raw, u_raw)
        if hasattr(transformer, "inverse_transform"):
            return transformer.inverse_transform(a_raw, u_raw)
        raise TypeError(f"Unsupported transformer type: {type(transformer)!r}")
    try:
        from data.transform_old import PDEtransform as OldPDEtransform

        return OldPDEtransform(pde).inverse_transform(a_raw, u_raw)
    except Exception:
        return a_raw, u_raw


def raw_to_unit_interval(x: Any) -> Any:
    return (x + 1.0) / 2.0


def unit_interval_to_raw(x: Any) -> Any:
    return x * 2.0 - 1.0


def _assert_bchw(x: Any, name: str) -> None:
    if getattr(x, "ndim", None) != 4:
        raise ValueError(f"{name} must be BCHW, got shape={getattr(x, 'shape', None)}")

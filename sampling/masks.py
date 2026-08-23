from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class PairMasks:
    coef: Any
    sol: Any
    metadata: dict[str, Any]


def make_mask(
    shape: tuple[int, ...] | Any,
    num_obs: int,
    mode: str,
    seed: int,
    device: str | Any = "cpu",
    dtype: Any | None = None,
    num_sensor_columns: int | None = None,
) -> Any:
    import torch

    b, c, h, w = _normalize_shape(shape)
    dtype = dtype or torch.float32
    target = torch.device(device if isinstance(device, str) else device)
    if target.type == "cuda" and not torch.cuda.is_available():
        target = torch.device("cpu")
    mask = torch.zeros((b, c, h, w), dtype=dtype, device=target)
    if mode != "sensor_column" and num_obs <= 0:
        return mask
    spatial_count = h * w
    k = min(int(num_obs), spatial_count)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    if mode == "random":
        idx = torch.randperm(spatial_count, generator=gen)[:k]
        mask.view(b, c, -1)[:, :, idx] = 1
    elif mode == "per_sample_random":
        for batch in range(b):
            idx = torch.randperm(spatial_count, generator=gen)[:k]
            mask[batch].view(c, -1)[:, idx] = 1
    elif mode == "fixed":
        mask.view(b, c, -1)[:, :, :k] = 1
    elif mode == "grid":
        coords = _grid_indices(h, w, k)
        for y, x in coords:
            mask[:, :, y, x] = 1
    elif mode == "sensor_column":
        if num_sensor_columns is None or int(num_sensor_columns) <= 0:
            raise ValueError("sensor_column requires an explicit positive num_sensor_columns")
        columns = int(num_sensor_columns)
        if columns > w:
            raise ValueError(
                f"num_sensor_columns={columns} exceeds spatial width={w}; "
                "sensor columns are complete columns, not individual points"
            )
        idx = torch.randperm(w, generator=gen)[:columns]
        mask[:, :, :, idx] = 1
    else:
        raise ValueError(f"Unknown sensor mode: {mode}")
    return mask


def make_pair_masks(
    coef_shape: tuple[int, ...] | Any,
    sol_shape: tuple[int, ...] | Any,
    num_obs: int,
    mode: str,
    shared_mask: bool,
    seed: int,
    device: str | Any = "cpu",
    dtype: Any | None = None,
    num_sensor_columns: int | None = None,
) -> PairMasks:
    coef_norm = _normalize_shape(coef_shape)
    sol_norm = _normalize_shape(sol_shape)
    coef_mask = make_mask(
        coef_norm, num_obs, mode, seed, device, dtype,
        num_sensor_columns=num_sensor_columns,
    )
    if shared_mask and coef_norm[-2:] == sol_norm[-2:]:
        import torch

        sol_mask = coef_mask
        if coef_norm[1] != sol_norm[1]:
            spatial = coef_mask[:, :1]
            sol_mask = spatial.repeat(1, sol_norm[1], 1, 1)
        else:
            sol_mask = coef_mask.clone()
    else:
        sol_mask = make_mask(
            sol_norm, num_obs, mode, seed + 1, device, dtype,
            num_sensor_columns=num_sensor_columns,
        )
    return PairMasks(
        coef=coef_mask,
        sol=sol_mask,
        metadata={
            "num_obs": num_obs,
            "num_sensor_columns": num_sensor_columns,
            "sensor_mode": mode,
            "shared_mask": shared_mask,
            "mask_seed_coef": seed,
            "mask_seed_sol": seed if shared_mask else seed + 1,
        },
    )


def residual_region_mask(
    region: str,
    coef_mask: Any,
    sol_mask: Any,
    residual_shape: tuple[int, ...],
    *,
    task: str = "both",
) -> Any | None:
    import torch

    if region == "full":
        return None
    b, c, h, w = _normalize_shape(residual_shape)
    device = coef_mask.device
    dtype = coef_mask.dtype
    if region == "boundary_excluded":
        mask = torch.ones((b, c, h, w), dtype=dtype, device=device)
        if h > 2 and w > 2:
            mask[..., 0, :] = 0
            mask[..., -1, :] = 0
            mask[..., :, 0] = 0
            mask[..., :, -1] = 0
        return mask
    active_coef = task in {"forward", "both"}
    active_sol = task in {"inverse", "both"}
    if region == "coef_obs":
        if not active_coef:
            raise ValueError(f"coef_obs is not active for task={task!r}")
        base = coef_mask
    elif region == "sol_obs":
        if not active_sol:
            raise ValueError(f"sol_obs is not active for task={task!r}")
        base = sol_mask
    elif region == "active_obs_union":
        active_masks = []
        if active_coef:
            active_masks.append(_single_channel(coef_mask))
        if active_sol:
            active_masks.append(_single_channel(sol_mask))
        if not active_masks:
            raise ValueError("active_obs_union is undefined for task='unconditional'")
        base = active_masks[0]
        for candidate in active_masks[1:]:
            base = torch.maximum(base, candidate)
    else:
        raise ValueError(f"Unknown residual region: {region}")
    base = _single_channel(base)
    if base.shape[-2:] != (h, w):
        raise ValueError(f"Residual mask shape mismatch: residual={(h, w)}, mask={base.shape[-2:]}")
    return base.repeat(1, c, 1, 1) if base.shape[1] != c else base


def _single_channel(mask: Any) -> Any:
    return mask[:, :1] if mask.shape[1] != 1 else mask


def _normalize_shape(shape: tuple[int, ...] | Any) -> tuple[int, int, int, int]:
    if hasattr(shape, "shape"):
        shape = tuple(shape.shape)
    if len(shape) == 2:
        return (1, 1, int(shape[0]), int(shape[1]))
    if len(shape) == 3:
        return (1, int(shape[0]), int(shape[1]), int(shape[2]))
    if len(shape) == 4:
        return tuple(int(v) for v in shape)  # type: ignore[return-value]
    raise ValueError(f"Expected HW, CHW or BCHW shape, got {shape}")


def _grid_indices(h: int, w: int, k: int) -> list[tuple[int, int]]:
    if k >= h * w:
        return [(y, x) for y in range(h) for x in range(w)]
    side = max(1, int(math.sqrt(k)))
    ys = [round(v) for v in _linspace_int(0, h - 1, side)]
    xs = [round(v) for v in _linspace_int(0, w - 1, side)]
    coords = [(min(h - 1, y), min(w - 1, x)) for y in ys for x in xs]
    if len(coords) < k:
        for y in range(h):
            for x in range(w):
                if (y, x) not in coords:
                    coords.append((y, x))
                if len(coords) >= k:
                    break
            if len(coords) >= k:
                break
    return coords[:k]


def _linspace_int(start: int, end: int, count: int) -> list[float]:
    if count == 1:
        return [(start + end) / 2]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]

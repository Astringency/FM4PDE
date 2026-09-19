"""Differentiable physical measurement operators, independent of the network.

Operators return readings on the original anchor grid. A binary mask selects
the actual sensors; dense unselected values are never observations. Box windows
are entirely inside the domain, with no padding or periodic continuation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PointObservation:
    def __call__(self, field):
        return field

    def validate_mask(self, mask):
        return None

    def metadata(self):
        return {"kind": "point", "window_size": 1}


@dataclass(frozen=True)
class BoxAverageObservation:
    window_size: int

    def __post_init__(self):
        if not isinstance(self.window_size, int) or self.window_size < 1:
            raise ValueError("window_size must be a positive integer")

    def __call__(self, field):
        import torch.nn.functional as F
        k = self.window_size
        if field.ndim != 4 or min(field.shape[-2:]) < k:
            raise ValueError("Box average requires BCHW fields at least as large as the window")
        if k == 1:
            return field
        left = k // 2
        right = k - 1 - left
        means = F.avg_pool2d(field, kernel_size=k, stride=1)
        return F.pad(means, (left, right, left, right))

    def validate_mask(self, mask):
        import torch
        h, w = mask.shape[-2:]
        k = self.window_size
        if min(h, w) < k:
            raise ValueError("Window exceeds field dimensions")
        left, right = k // 2, k - 1 - k // 2
        valid = torch.zeros_like(mask, dtype=torch.bool)
        valid[..., left:h-right, left:w-right] = True
        if bool(((mask != 0) & ~valid).any()):
            raise ValueError("Every averaging window must lie entirely inside the domain")

    def metadata(self):
        return {"kind": "box_average", "window_size": self.window_size,
                "anchor_offsets_inclusive": [-(self.window_size // 2),
                    self.window_size - 1 - self.window_size // 2],
                "boundary_policy": "valid_windows_only", "normalization": "window_area"}

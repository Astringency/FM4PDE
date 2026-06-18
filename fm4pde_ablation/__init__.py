"""Backward-compatible aliases for the renamed :mod:`sampling` package."""

from __future__ import annotations

import importlib
import sys

_MODULES = (
    "config",
    "data",
    "guidance",
    "logging",
    "losses",
    "masks",
    "metrics",
    "model_io",
    "noise",
    "pde_residuals",
    "registry",
    "sampler_wrappers",
    "state",
    "time_grid",
)

for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"sampling.{_name}")

AblationConfig = sys.modules[f"{__name__}.config"].AblationConfig

__all__ = ["AblationConfig"]

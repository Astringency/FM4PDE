from __future__ import annotations

from typing import Any

import torch


def summarize_pde_params(loader_metadata: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for pde_name, entry in (loader_metadata or {}).items():
        params = _extract_pde_params(entry)
        if not params:
            continue
        pde_summary: dict[str, Any] = {}
        for param_name, value in params.items():
            tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1)
            count = int(tensor.numel())
            if count == 0:
                pde_summary[param_name] = {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "first": None,
                    "last": None,
                }
                continue
            pde_summary[param_name] = {
                "count": count,
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std(unbiased=False).item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "first": float(tensor[0].item()),
                "last": float(tensor[-1].item()),
            }
        if pde_summary:
            summary[pde_name] = pde_summary
    return summary


def detach_pde_params(loader_metadata: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        pde_name: {
            param_name: torch.as_tensor(value).detach().cpu()
            for param_name, value in params.items()
        }
        for pde_name, entry in (loader_metadata or {}).items()
        if (params := _extract_pde_params(entry))
    }


def _extract_pde_params(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict) and isinstance(entry.get("pde_params"), dict):
        return entry["pde_params"]
    if isinstance(entry, dict):
        return entry
    return {}

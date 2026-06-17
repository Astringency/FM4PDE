from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def relative_l2(pred: Any, target: Any, eps: float = 1e-12) -> float:
    import torch

    return float((torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target).clamp_min(eps)).detach().cpu())


def obs_relative_l2(pred: Any, target: Any, mask: Any, eps: float = 1e-12) -> float:
    import torch

    diff = (pred - target) * mask
    denom = torch.linalg.vector_norm(target * mask).clamp_min(eps)
    return float((torch.linalg.vector_norm(diff) / denom).detach().cpu())


def pde_residual_norm(residual: Any | None, eps: float = 1e-12) -> float:
    import torch

    if residual is None:
        return 0.0
    return float((torch.linalg.vector_norm(residual) / max(residual.numel(), 1)).detach().cpu())


def step_metrics(
    step: int,
    step_output: Any,
    losses: Any,
    gradient: Any | None,
    phys_state: Any,
    ground_truth: Any,
    masks: Any,
    wall_time: float,
) -> dict[str, Any]:
    row = {
        "step": step,
        "t": _scalar(step_output.t),
        "t_next": _scalar(step_output.t_next),
        "phase": step_output.phase,
        "loss_state": step_output.loss_state,
        "wall_time": wall_time,
        "L_obs_a": _scalar(losses.L_obs_a),
        "L_obs_u": _scalar(losses.L_obs_u),
        "L_pde": _scalar(losses.L_pde),
        "clean_L_obs_a": _scalar(losses.clean_L_obs_a),
        "clean_L_obs_u": _scalar(losses.clean_L_obs_u),
        "rel_l2_a": relative_l2(phys_state.coef, ground_truth.coef),
        "rel_l2_u": relative_l2(phys_state.sol, ground_truth.sol),
        "obs_rel_l2_a": obs_relative_l2(phys_state.coef, ground_truth.coef, masks.coef),
        "obs_rel_l2_u": obs_relative_l2(phys_state.sol, ground_truth.sol, masks.sol),
        "pde_residual_norm": pde_residual_norm(losses.pde_residual),
        "pde_residual_status": losses.pde_residual_status,
    }
    if gradient is not None:
        row.update(
            {
                "clip_scale": gradient.clip_scale,
                "grad_norm_obs_a": gradient.grad_norm_obs_a,
                "grad_norm_obs_u": gradient.grad_norm_obs_u,
                "grad_norm_pde": gradient.grad_norm_pde,
                "grad_norm_total": gradient.grad_norm_total,
            }
        )
    else:
        row.update(
            {
                "clip_scale": 1.0,
                "grad_norm_obs_a": 0.0,
                "grad_norm_obs_u": 0.0,
                "grad_norm_pde": 0.0,
                "grad_norm_total": 0.0,
            }
        )
    return row


def final_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    last = rows[-1].copy()
    last["num_recorded_steps"] = len(rows)
    return last


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _scalar(value: Any) -> float:
    try:
        return float(value.detach().cpu())
    except Exception:
        return float(value)

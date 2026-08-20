from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_from_result_pt(
    result_pt_path: str | Path,
    output_path: str | Path | None = None,
    *,
    title: str | None = None,
    cmap: str = "RdBu_r",
    dpi: int = 120,
    sample_index: int = 0,
) -> plt.Figure:
    """Plot ground truth, sparse observations, and prediction from result.pt.

    Creates a 2×4 figure:
      Row 1: Coefficient GT | Sparse Obs | Pred | Diff
      Row 2: Solution  GT | Sparse Obs | Pred | Diff

    For single-channel PDEs (burger), coef == sol, so only one row is shown.
    Batched result files display the sample selected by ``sample_index``.
    """
    data = torch.load(str(result_pt_path), map_location="cpu", weights_only=False)

    coef_gt = _select_field(data["coef_ground_truth"], sample_index)
    sol_gt = _select_field(data["sol_ground_truth"], sample_index)
    coef_pred = _select_field(data["coef_final"], sample_index)
    sol_pred = _select_field(data["sol_final"], sample_index)
    masks = data.get("masks", {})
    coef_mask = _select_optional_mask(masks, "coef", sample_index, coef_gt.shape)
    sol_mask = _select_optional_mask(masks, "sol", sample_index, sol_gt.shape)
    metrics = data.get("metrics", {})
    config = data.get("config", {})

    # Detect if coef and sol are the same field (burger-like single channel)
    same_field = np.array_equal(coef_gt.shape, sol_gt.shape) and np.allclose(coef_gt, sol_gt)

    if same_field:
        nrows = 1
    else:
        nrows = 2

    fig, axes = plt.subplots(nrows, 4, figsize=(24, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    zeta_a = metrics.get("zeta_obs_a_t", 1)
    zeta_u = metrics.get("zeta_obs_u_t", 1)

    if same_field:
        coef_mask = np.maximum(coef_mask, sol_mask)

    _plot_field_row(axes[0], coef_gt, coef_pred, coef_mask, "Coefficient",
                    metrics.get("rel_l2_a", 0), cmap, zeta=zeta_a)

    if not same_field:
        _plot_field_row(axes[1], sol_gt, sol_pred, sol_mask, "Solution",
                        metrics.get("rel_l2_u", 0), cmap, zeta=zeta_u)

    # Title
    if title is None:
        pde = config.get("pde", "?") if isinstance(config, dict) else getattr(config, "pde", "?")
        task = config.get("task", "?") if isinstance(config, dict) else getattr(config, "task", "?")
        title = f"{str(pde).upper()} / {task}"

    # ---- sampler phase label ----
    phase = config.get("sampler_phase", "?") if isinstance(config, dict) else getattr(config, "sampler_phase", "?")
    ratio = config.get("switch_ratio", 0.5) if isinstance(config, dict) else getattr(config, "switch_ratio", 0.5)
    phase_label = {
        "stochastic": "pure s",
        "deterministic": "pure d",
        "hybrid_d2s": f"d2s({ratio})",
        "hybrid_s2d": f"s2d({ratio})",
    }.get(str(phase), str(phase))

    l_pde = metrics.get("L_pde", 0)
    wall = metrics.get("wall_clock_time", 0)
    fig.suptitle(
        f"{title} | {phase_label} | L_pde={l_pde:.2e} | {wall:.1f}s",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_result_side_by_side(
    result_pt_paths: list[str | Path],
    output_path: str | Path,
    *,
    labels: list[str] | None = None,
    title: str | None = None,
    cmap: str = "RdBu_r",
    dpi: int = 120,
) -> plt.Figure:
    """Plot predictions from multiple result.pt files side by side for comparison.

    Layout: one row per field (coefficient / solution), columns = ground truth + one per result file.
    """
    n_results = len(result_pt_paths)
    all_data = [torch.load(str(p), map_location="cpu", weights_only=False) for p in result_pt_paths]

    if labels is None:
        labels = [f"Result {i+1}" for i in range(n_results)]

    coef_gt = _to_numpy(all_data[0]["coef_ground_truth"]).squeeze()
    sol_gt = _to_numpy(all_data[0]["sol_ground_truth"]).squeeze()
    if coef_gt.ndim == 1:
        coef_gt = coef_gt[np.newaxis]
    if sol_gt.ndim == 1:
        sol_gt = sol_gt[np.newaxis]

    same_field = np.array_equal(coef_gt.shape, sol_gt.shape) and np.allclose(coef_gt, sol_gt)
    nrows = 1 if same_field else 2
    ncols = 1 + n_results

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    # Coefficient row
    _plot_single(axes[0, 0], coef_gt, "Coefficient GT", cmap)
    for i, d in enumerate(all_data):
        pred = _to_numpy(d["coef_final"]).squeeze()
        if pred.ndim == 1:
            pred = pred[np.newaxis]
        _plot_single(axes[0, 1 + i], pred, f"Coeff {labels[i]}", cmap,
                     vmin=coef_gt.min(), vmax=coef_gt.max())

    # Solution row
    if not same_field:
        _plot_single(axes[1, 0], sol_gt, "Solution GT", cmap)
        for i, d in enumerate(all_data):
            pred = _to_numpy(d["sol_final"]).squeeze()
            if pred.ndim == 1:
                pred = pred[np.newaxis]
            _plot_single(axes[1, 1 + i], pred, f"Sol {labels[i]}", cmap,
                         vmin=sol_gt.min(), vmax=sol_gt.max())

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── helpers ────────────────────────────────────────────────────────────────────

def _plot_field_row(
    axes: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    label: str,
    rel_l2: float,
    cmap: str,
    *,
    zeta: float = 1.0,
) -> None:
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())

    _plot_single(axes[0], gt, f"{label} GT", cmap, vmin, vmax)
    _plot_sparse(axes[1], gt, mask, f"{label} Sparse Obs", cmap, vmin, vmax)
    _plot_single(axes[2], pred, f"{label} Pred (ζ={zeta:.1e})", cmap, vmin, vmax)

    diff = pred - gt
    vmax_diff = max(abs(diff.min()), abs(diff.max())) + 1e-12
    im = axes[3].imshow(diff, cmap=cmap, aspect="auto", vmin=-vmax_diff, vmax=vmax_diff)
    axes[3].set_title(f"{label} Diff (rel_l2={rel_l2:.4f})")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)


def _plot_sparse(
    ax: plt.Axes,
    data: np.ndarray,
    mask: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    sparse = np.ma.masked_where(mask <= 0, data)
    sparse_cmap = matplotlib.colormaps.get_cmap(cmap).copy()
    sparse_cmap.set_bad(color="white")
    ax.set_facecolor("white")
    im = ax.imshow(
        sparse,
        cmap=sparse_cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_single(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _select_field(x: Any, sample_index: int) -> np.ndarray:
    array = _to_numpy(x)
    if array.ndim == 4:
        if not -array.shape[0] <= sample_index < array.shape[0]:
            raise IndexError(
                f"sample_index={sample_index} is out of range for batch size {array.shape[0]}"
            )
        array = array[sample_index]
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"Expected a single 2D field, got shape {array.shape}")
    return array


def _select_optional_mask(
    masks: Any,
    key: str,
    sample_index: int,
    field_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(masks, dict) or key not in masks:
        return np.zeros(field_shape, dtype=bool)
    mask = _select_field(masks[key], sample_index)
    if mask.shape != field_shape:
        raise ValueError(
            f"{key} mask shape {mask.shape} does not match field shape {field_shape}"
        )
    return mask

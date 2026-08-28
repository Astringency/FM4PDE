#!/usr/bin/env python3
"""Compare Poisson FM4PDE and baseline predictions in boundary-aware frequency bands.

The source field is expanded in an orthonormal 2-D DCT-II basis.  The Poisson
solution has homogeneous Dirichlet boundary conditions, so its interior is
expanded in an orthonormal 2-D DST-I basis.  This avoids the artificial edge
discontinuity introduced by a periodic FFT.

The baseline artifact writer stores a view into the full evaluation batch in
each per-sample ``.pt`` file.  ``load_baseline_predictions`` reads only the
prediction storage from the first artifact of each batch, rather than loading
the duplicated targets, coordinates, and full data tensors.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pickle
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import dctn, dstn, idctn, idstn
from scipy.io import whosmat


FM_ROOT_DEFAULT = Path("/home/tat512/C01Python/FM4PDE/outputs/MAIN1000_100/poisson")
BASELINE_ROOT_DEFAULT = Path("/home/tat512/share/outputs/FM4PDEbaseline/runs/main_results")
DATA_DEFAULT = Path("/home/tat512/share/PDEdata/poisson/poisson_test_10000-128-128.mat")
OUTPUT_DEFAULT = Path(
    "/home/tat512/C01Python/FM4PDE/outputs/MAIN1000_100/poisson_frequency_analysis"
)

TASK_FIELDS = {
    "forward": ("solution",),
    "inverse": ("source",),
    "both": ("source", "solution"),
}
TASK_TITLES = {
    ("forward", "solution"): "Sparse forward: solution $\\phi$",
    ("inverse", "source"): "Sparse inverse: source $f$",
    ("both", "source"): "Joint reconstruction: source $f$",
    ("both", "solution"): "Joint reconstruction: solution $\\phi$",
}

# The entries are deliberately limited to completed, sparse-observation runs.
# Dense full-field input runs are not comparable to FM4PDE's 500 observations.
BASELINE_RUNS = {
    "forward": {
        "RecFNO": (
            "task_group=sparse_forward_main_amortized/pde=poisson/baseline=recfno/seed=1/"
            "run=sparse_forward_main_amortized_recfno_poisson_s1_e6249c04289c"
        ),
        "VoronoiCNN": (
            "task_group=sparse_forward_main_amortized/pde=poisson/baseline=voronoicnn/seed=1/"
            "run=sparse_forward_main_amortized_voronoicnn_poisson_s1_073026a2fde7"
        ),
    },
    "inverse": {
        "RecFNO": (
            "task_group=sparse_inverse_main_amortized/pde=poisson/baseline=recfno/seed=1/"
            "run=sparse_inverse_main_amortized_recfno_poisson_s1_cf9ad2b4f377"
        ),
        "Senseiver": (
            "task_group=sparse_inverse_main_amortized/pde=poisson/baseline=senseiver/seed=1/"
            "run=sparse_inverse_main_amortized_senseiver_poisson_s1_1e5026430e44"
        ),
        "VoronoiCNN": (
            "task_group=sparse_inverse_main_amortized/pde=poisson/baseline=voronoicnn/seed=1/"
            "run=sparse_inverse_main_amortized_voronoicnn_poisson_s1_895e5a941753"
        ),
        "PINN-Sparse": (
            "task_group=sparse_inverse_main/pde=poisson/baseline=pinn_sparse/seed=1/"
            "run=sparse_inverse_main_pinn_sparse_poisson_s1_50103d020a42"
        ),
        "PDE-Opt": (
            "task_group=sparse_inverse_main/pde=poisson/baseline=pde_opt/seed=1/"
            "run=sparse_inverse_main_pde_opt_poisson_s1_fc73ee99e798"
        ),
        "PC-BNN": (
            "task_group=sparse_inverse_main/pde=poisson/baseline=pc_bnn/seed=1/"
            "run=sparse_inverse_main_pc_bnn_poisson_s1_854dc9ab119a"
        ),
    },
    "both": {
        "RecFNO": (
            "task_group=sparse_solution_main_amortized/pde=poisson/baseline=recfno/seed=1/"
            "run=sparse_solution_main_amortized_recfno_poisson_s1_292f8b0ce329"
        ),
        "Senseiver": (
            "task_group=sparse_solution_main_amortized/pde=poisson/baseline=senseiver/seed=1/"
            "run=sparse_solution_main_amortized_senseiver_poisson_s1_c24463d6af8d"
        ),
        "VoronoiCNN": (
            "task_group=sparse_solution_main_amortized/pde=poisson/baseline=voronoicnn/seed=1/"
            "run=sparse_solution_main_amortized_voronoicnn_poisson_s1_f1a0e74d9f39"
        ),
    },
}


@dataclass(frozen=True)
class StorageDescriptor:
    key: str
    dtype: np.dtype
    numel: int


@dataclass(frozen=True)
class TensorDescriptor:
    storage: StorageDescriptor
    offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


class StorageType:
    def __init__(self, dtype: np.dtype):
        self.dtype = np.dtype(dtype)


STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "BFloat16Storage": np.uint16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


def _rebuild_tensor(
    storage: StorageDescriptor,
    storage_offset: int,
    size: Iterable[int],
    stride: Iterable[int],
    *unused: Any,
) -> TensorDescriptor:
    return TensorDescriptor(
        storage=storage,
        offset=int(storage_offset),
        size=tuple(int(v) for v in size),
        stride=tuple(int(v) for v in stride),
    )


class DescriptorUnpickler(pickle.Unpickler):
    """Unpickle a torch archive without materializing any tensor storage."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name in STORAGE_DTYPES:
            return StorageType(STORAGE_DTYPES[name])
        if module == "torch._utils" and name in {
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
            "_rebuild_tensor_v3",
        }:
            return _rebuild_tensor
        return super().find_class(module, name)

    def persistent_load(self, saved_id: Any) -> StorageDescriptor:
        if not isinstance(saved_id, tuple) or saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {saved_id!r}")
        _, storage_type, key, _location, numel, *rest = saved_id
        dtype = getattr(storage_type, "dtype", None)
        if dtype is None:
            raise pickle.UnpicklingError(f"Unknown storage type in {saved_id!r}")
        return StorageDescriptor(str(key), np.dtype(dtype), int(numel))


def read_pt_descriptors(path: Path) -> tuple[dict[str, Any], str]:
    with zipfile.ZipFile(path, "r") as archive:
        pickle_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        prefix = pickle_name.rsplit("/", 1)[0]
        payload = DescriptorUnpickler(io.BytesIO(archive.read(pickle_name))).load()
    return payload, prefix


def read_storage(path: Path, prefix: str, storage: StorageDescriptor) -> np.ndarray:
    with zipfile.ZipFile(path, "r") as archive:
        raw = archive.read(f"{prefix}/data/{storage.key}")
    values = np.frombuffer(raw, dtype=storage.dtype)
    if values.size < storage.numel:
        raise ValueError(f"Short storage in {path}: {values.size} < {storage.numel}")
    return values[: storage.numel]


def tensor_from_storage(values: np.ndarray, desc: TensorDescriptor) -> np.ndarray:
    byte_strides = tuple(step * values.dtype.itemsize for step in desc.stride)
    view = np.ndarray(
        shape=desc.size,
        dtype=values.dtype,
        buffer=values,
        offset=desc.offset * values.dtype.itemsize,
        strides=byte_strides,
    )
    return np.array(view, copy=True)


def find_sample_dir(run_dir: Path) -> Path:
    candidates = sorted(p for p in run_dir.rglob("*_samples") if p.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one sample directory below {run_dir}, found {candidates}")
    return candidates[0]


def load_baseline_predictions(run_dir: Path, n_expected: int = 1000) -> tuple[np.ndarray, dict[str, Any]]:
    """Read only unique prediction batches from per-sample torch archives."""

    sample_dir = find_sample_dir(run_dir)
    files = sorted(sample_dir.glob("sample_*.pt"))
    if len(files) != n_expected:
        raise RuntimeError(f"{sample_dir}: expected {n_expected} samples, found {len(files)}")

    chunks: list[np.ndarray] = []
    ordinal = 0
    first_metadata: dict[str, Any] = {}
    while ordinal < n_expected:
        path = sample_dir / f"sample_{ordinal:06d}.pt"
        payload, prefix = read_pt_descriptors(path)
        if ordinal == 0:
            first_metadata = payload.get("metadata", {})
        desc = payload["prediction"]
        if not isinstance(desc, TensorDescriptor):
            raise TypeError(f"Unexpected prediction descriptor in {path}: {type(desc)}")
        sample_numel = math.prod(desc.size)
        if desc.offset != 0 or desc.storage.numel % sample_numel != 0:
            # Conservative fallback for an artifact that does not point at the
            # first item of a contiguous evaluation batch.
            values = read_storage(path, prefix, desc.storage)
            chunks.append(tensor_from_storage(values, desc)[None, ...])
            ordinal += 1
            continue

        batch_size = desc.storage.numel // sample_numel
        values = read_storage(path, prefix, desc.storage)
        # Do not assume channel-first contiguous storage.  RecFNO/Senseiver's
        # joint outputs are channel-interleaved (sample view stride
        # ``(1, 256, 2)``), whereas VoronoiCNN is contiguous CHW.  Reconstruct
        # every sample using the serialized view stride.
        batch = np.stack(
            [
                tensor_from_storage(
                    values,
                    TensorDescriptor(
                        storage=desc.storage,
                        offset=item * sample_numel,
                        size=desc.size,
                        stride=desc.stride,
                    ),
                )
                for item in range(batch_size)
            ],
            axis=0,
        ).astype(np.float64, copy=False)
        take = min(batch_size, n_expected - ordinal)
        chunks.append(batch[:take])
        ordinal += take
        if ordinal % 100 == 0 or ordinal == n_expected:
            print(f"    loaded {ordinal:4d}/{n_expected} predictions from {sample_dir.name}", flush=True)

    predictions = np.concatenate(chunks, axis=0)
    return predictions, first_metadata


def load_fm_task(task_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for path in task_dir.rglob("result.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.append((int(payload["config"]["offset"]), payload))
    records.sort(key=lambda item: item[0])
    offsets = [offset for offset, _ in records]
    if offsets != list(range(0, 1000, 50)):
        raise RuntimeError(f"Unexpected FM4PDE offsets for {task_dir}: {offsets}")

    key_map = {
        "source_pred": "coef_final",
        "solution_pred": "sol_final",
        "source_truth": "coef_ground_truth",
        "solution_truth": "sol_ground_truth",
    }
    arrays: dict[str, np.ndarray] = {}
    for output_key, payload_key in key_map.items():
        tensors = [payload[payload_key].detach().cpu().numpy() for _, payload in records]
        array = np.concatenate(tensors, axis=0)
        arrays[output_key] = np.asarray(array[:, 0], dtype=np.float64)
    for output_key, mask_key in (("source_mask", "coef"), ("solution_mask", "sol")):
        tensors = [payload["masks"][mask_key].detach().cpu().numpy() for _, payload in records]
        array = np.concatenate(tensors, axis=0)
        arrays[output_key] = np.asarray(array[:, 0], dtype=np.float64)

    sample_ids = [
        str(item)
        for _, payload in records
        for item in payload["ground_truth_metadata"]["sample_ids"]
    ]
    if sample_ids != [str(i) for i in range(1000)]:
        raise RuntimeError(f"Unexpected FM4PDE sample ids in {task_dir}")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError(f"Non-finite values found in {task_dir}")
    return arrays, records[0][1]["config"]


def transform_fields(fields: np.ndarray, field: str) -> tuple[np.ndarray, np.ndarray]:
    if field == "source":
        coefficients = dctn(fields, type=2, norm="ortho", axes=(-2, -1), workers=-1)
        indices = np.arange(fields.shape[-1], dtype=np.float64)
    elif field == "solution":
        interior = fields[:, 1:-1, 1:-1]
        coefficients = dstn(interior, type=1, norm="ortho", axes=(-2, -1), workers=-1)
        indices = np.arange(1, fields.shape[-1] - 1, dtype=np.float64)
    else:
        raise ValueError(field)
    radius = np.hypot(indices[:, None], indices[None, :])
    return coefficients, radius


def inverse_highpass(coefficients: np.ndarray, radius: np.ndarray, field: str, size: int = 128) -> np.ndarray:
    filtered = np.array(coefficients, copy=True)
    filtered[..., radius <= 32.0] = 0.0
    if field == "source":
        return idctn(filtered, type=2, norm="ortho", axes=(-2, -1), workers=-1)
    interior = idstn(filtered, type=1, norm="ortho", axes=(-2, -1), workers=-1)
    result = np.zeros((filtered.shape[0], size, size), dtype=interior.dtype)
    result[:, 1:-1, 1:-1] = interior
    return result


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def bootstrap_median_ci(values: np.ndarray, seed: int, repeats: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    medians = np.median(values[indices], axis=1)
    return quantile(medians, 0.025), quantile(medians, 0.975)


def evaluate_method(
    *,
    task: str,
    field: str,
    method: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    metrics: list[dict[str, Any]],
    radial_rows: list[dict[str, Any]],
) -> np.ndarray:
    prediction_coeff, radius = transform_fields(prediction, field)
    truth_coeff, truth_radius = transform_fields(truth, field)
    if not np.array_equal(radius, truth_radius):
        raise AssertionError("Prediction and truth frequency grids differ")

    bands = {
        "low (r<=8)": radius <= 8.0,
        "mid (8<r<=32)": (radius > 8.0) & (radius <= 32.0),
        "high (r>32)": radius > 32.0,
        "all": np.ones_like(radius, dtype=bool),
    }
    total_truth_energy = np.sum(truth_coeff**2, axis=(-2, -1))
    eps = np.finfo(np.float64).tiny
    for band, mask in bands.items():
        pred_band = prediction_coeff[..., mask]
        truth_band = truth_coeff[..., mask]
        error_energy = np.sum((pred_band - truth_band) ** 2, axis=-1)
        truth_energy = np.sum(truth_band**2, axis=-1)
        pred_energy = np.sum(pred_band**2, axis=-1)
        dot = np.sum(pred_band * truth_band, axis=-1)
        rel_error = np.sqrt(error_energy / np.maximum(truth_energy, eps))
        energy_ratio = pred_energy / np.maximum(truth_energy, eps)
        cosine = dot / np.sqrt(np.maximum(pred_energy * truth_energy, eps))
        truth_fraction = truth_energy / np.maximum(total_truth_energy, eps)
        for sample_id in range(prediction.shape[0]):
            metrics.append(
                {
                    "task": task,
                    "field": field,
                    "method": method,
                    "sample_id": sample_id,
                    "band": band,
                    "relative_error": float(rel_error[sample_id]),
                    "energy_ratio": float(energy_ratio[sample_id]),
                    "spectral_cosine": float(cosine[sample_id]),
                    "truth_energy_fraction": float(truth_fraction[sample_id]),
                }
            )

    max_shell = 64
    for shell in range(max_shell + 1):
        mask = (radius >= shell) & (radius < shell + 1)
        if not mask.any():
            continue
        pred_shell = prediction_coeff[..., mask]
        truth_shell = truth_coeff[..., mask]
        denominator = np.sum(truth_shell**2, axis=-1)
        numerator = np.sum((pred_shell - truth_shell) ** 2, axis=-1)
        valid = denominator > np.maximum(total_truth_energy * 1e-14, eps)
        if not np.any(valid):
            continue
        shell_error = np.sqrt(numerator[valid] / denominator[valid])
        radial_rows.append(
            {
                "task": task,
                "field": field,
                "method": method,
                "radius": shell + 0.5,
                "relative_error_median": quantile(shell_error, 0.5),
                "relative_error_q25": quantile(shell_error, 0.25),
                "relative_error_q75": quantile(shell_error, 0.75),
                "valid_samples": int(valid.sum()),
            }
        )
    return radius


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[(row["task"], row["field"], row["method"], row["band"])].append(row)

    summary: list[dict[str, Any]] = []
    for group_index, (key, rows) in enumerate(sorted(grouped.items())):
        task, field, method, band = key
        rel = np.array([row["relative_error"] for row in rows])
        ratio = np.array([row["energy_ratio"] for row in rows])
        cosine = np.array([row["spectral_cosine"] for row in rows])
        fraction = np.array([row["truth_energy_fraction"] for row in rows])
        ci_low, ci_high = bootstrap_median_ci(rel, seed=20260828 + group_index)
        summary.append(
            {
                "task": task,
                "field": field,
                "method": method,
                "band": band,
                "n": len(rows),
                "relative_error_median": quantile(rel, 0.5),
                "relative_error_q25": quantile(rel, 0.25),
                "relative_error_q75": quantile(rel, 0.75),
                "relative_error_median_ci95_low": ci_low,
                "relative_error_median_ci95_high": ci_high,
                "energy_ratio_median": quantile(ratio, 0.5),
                "energy_ratio_q25": quantile(ratio, 0.25),
                "energy_ratio_q75": quantile(ratio, 0.75),
                "spectral_cosine_median": quantile(cosine, 0.5),
                "truth_energy_fraction_median": quantile(fraction, 0.5),
            }
        )
    return summary


def summary_lookup(summary: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (row["task"], row["field"], row["method"], row["band"]): row for row in summary
    }


def method_order(summary: list[dict[str, Any]], task: str, field: str) -> list[str]:
    methods = {row["method"] for row in summary if row["task"] == task and row["field"] == field}
    preferred = ["FM4PDE", "RecFNO", "Senseiver", "VoronoiCNN", "PINN-Sparse", "PDE-Opt", "PC-BNN"]
    return [method for method in preferred if method in methods]


def plot_band_metric(
    summary: list[dict[str, Any]], output: Path, metric: str, ylabel: str, filename: str
) -> None:
    lookup = summary_lookup(summary)
    bands = ["low (r<=8)", "mid (8<r<=32)", "high (r>32)"]
    panels = list(TASK_TITLES)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for ax, (task, field) in zip(axes.flat, panels):
        methods = method_order(summary, task, field)
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))
        for color, method in zip(colors, methods):
            rows = [lookup[(task, field, method, band)] for band in bands]
            y = np.array([float(row[f"{metric}_median"]) for row in rows])
            q25 = np.array([float(row[f"{metric}_q25"]) for row in rows])
            q75 = np.array([float(row[f"{metric}_q75"]) for row in rows])
            ax.errorbar(
                np.arange(3),
                y,
                yerr=np.vstack((np.maximum(y - q25, 0), np.maximum(q75 - y, 0))),
                marker="o",
                markersize=4.5,
                linewidth=1.6,
                capsize=2,
                color=color,
                label=method,
            )
        ax.set_title(TASK_TITLES[(task, field)])
        ax.set_xticks(np.arange(3), ["low\n$r\\leq8$", "mid\n$8<r\\leq32$", "high\n$r>32$"])
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.22)
        ax.set_ylabel(ylabel)
        if metric == "energy_ratio":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.legend(fontsize=8, ncol=1, loc="best")
    fig.suptitle(
        "Poisson frequency-band comparison (median and interquartile range, n=1000)",
        fontsize=15,
    )
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def plot_radial(radial_rows: list[dict[str, Any]], output: Path) -> None:
    panels = list(TASK_TITLES)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for ax, (task, field) in zip(axes.flat, panels):
        methods = []
        for method in ["FM4PDE", "RecFNO", "Senseiver", "VoronoiCNN", "PINN-Sparse", "PDE-Opt", "PC-BNN"]:
            if any(
                row["task"] == task and row["field"] == field and row["method"] == method
                for row in radial_rows
            ):
                methods.append(method)
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))
        for color, method in zip(colors, methods):
            rows = sorted(
                (
                    row
                    for row in radial_rows
                    if row["task"] == task and row["field"] == field and row["method"] == method
                ),
                key=lambda row: row["radius"],
            )
            ax.plot(
                [row["radius"] for row in rows],
                [row["relative_error_median"] for row in rows],
                linewidth=1.5,
                color=color,
                label=method,
            )
        ax.axvline(8, color="0.25", linestyle=":", linewidth=1)
        ax.axvline(32, color="0.25", linestyle=":", linewidth=1)
        ax.set_title(TASK_TITLES[(task, field)])
        ax.set_xlim(0, 64)
        ax.set_yscale("log")
        ax.set_xlabel("radial mode $r$")
        ax.set_ylabel("shell relative error")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Poisson radial spectral error (median across test samples)", fontsize=15)
    fig.savefig(output / "poisson_radial_relative_error.png", dpi=180)
    plt.close(fig)


def plot_joint_example(
    examples: dict[tuple[str, str], np.ndarray],
    truths: dict[str, np.ndarray],
    sample_id: int,
    output: Path,
) -> None:
    methods = ["truth", "FM4PDE", "RecFNO", "Senseiver", "VoronoiCNN"]
    row_specs = [
        ("source", False, "source $f$"),
        ("source", True, "source high-pass ($r>32$)"),
        ("solution", False, "solution $\\phi$"),
        ("solution", True, "solution high-pass ($r>32$)"),
    ]
    fig, axes = plt.subplots(4, 5, figsize=(15, 11.3), constrained_layout=True)
    for row_index, (field, highpass, row_title) in enumerate(row_specs):
        truth = truths[field][sample_id : sample_id + 1]
        all_fields: list[np.ndarray] = []
        for method in methods:
            values = truth if method == "truth" else examples[(method, field)][None, ...]
            if highpass:
                coeff, radius = transform_fields(values, field)
                values = inverse_highpass(coeff, radius, field)
            all_fields.append(values[0])
        scale = max(float(np.quantile(np.abs(all_fields[0]), 0.995)), np.finfo(float).eps)
        for col_index, (method, values) in enumerate(zip(methods, all_fields)):
            ax = axes[row_index, col_index]
            image = ax.imshow(values, cmap="RdBu_r", vmin=-scale, vmax=scale, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title("Ground truth" if method == "truth" else method)
            if col_index == 0:
                ax.set_ylabel(row_title)
        fig.colorbar(image, ax=axes[row_index, :], fraction=0.012, pad=0.01)
    fig.suptitle(f"Joint Poisson reconstruction, representative sample {sample_id}", fontsize=15)
    fig.savefig(output / "poisson_joint_representative_sample.png", dpi=180)
    plt.close(fig)


def validate_first_baseline_truth(
    run_dir: Path, fm_arrays: dict[str, np.ndarray], task: str
) -> dict[str, Any]:
    sample_dir = find_sample_dir(run_dir)
    path = sample_dir / "sample_000000.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    target = payload["target_fields"].detach().cpu().numpy()
    full = payload["full_tensor"].detach().cpu().numpy()
    if target.ndim == 4:
        target = target[0]
    if full.ndim == 4:
        full = full[0]
    target_fields = TASK_FIELDS[task]
    expected_target = np.stack(
        [fm_arrays[f"{field}_truth"][0] for field in target_fields], axis=0
    )
    return {
        "sample0_target_max_abs_difference": float(np.max(np.abs(target - expected_target))),
        "sample0_full_source_max_abs_difference": float(
            np.max(np.abs(full[0] - fm_arrays["source_truth"][0]))
        ),
        "sample0_full_solution_max_abs_difference": float(
            np.max(np.abs(full[1] - fm_arrays["solution_truth"][0]))
        ),
        "baseline_global_sample_id": payload.get("global_sample_id"),
        "baseline_metadata": payload.get("metadata", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm-root", type=Path, default=FM_ROOT_DEFAULT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT_DEFAULT)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data_inventory = [
        {"name": name, "shape": list(shape), "dtype": dtype}
        for name, shape, dtype in whosmat(args.data)
    ]
    metrics: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    quality: dict[str, Any] = {
        "source_data": str(args.data),
        "source_data_variables": data_inventory,
        "band_definition": {
            "low": "radial mode r <= 8",
            "mid": "8 < radial mode r <= 32",
            "high": "radial mode r > 32",
        },
        "transforms": {
            "source": "orthonormal 2-D DCT-II on the 128x128 field",
            "solution": "orthonormal 2-D DST-I on the 126x126 interior (zero Dirichlet boundary)",
        },
        "tasks": {},
        "limitations": [
            "FM4PDE uses mask_seed=0 while baseline runs use sensor_seed=1; results share truth samples but not identical observation locations.",
            "Uncertainty intervals quantify variation across 1000 test samples, not variation across training seeds.",
            "Only completed sparse-observation baseline runs are included; dense full-field-input runs are excluded.",
        ],
    }
    joint_examples: dict[tuple[str, str], np.ndarray] = {}
    joint_truths: dict[str, np.ndarray] = {}
    representative_sample = 0

    cross_task_truth: dict[str, np.ndarray] | None = None
    for task in ("forward", "inverse", "both"):
        print(f"Loading FM4PDE task={task}", flush=True)
        fm_arrays, fm_config = load_fm_task(args.fm_root / task)
        if cross_task_truth is None:
            cross_task_truth = {
                "source": fm_arrays["source_truth"].copy(),
                "solution": fm_arrays["solution_truth"].copy(),
            }
        else:
            for field in ("source", "solution"):
                if not np.array_equal(cross_task_truth[field], fm_arrays[f"{field}_truth"]):
                    raise RuntimeError(f"Ground truth differs across FM4PDE tasks for {field}")

        task_quality: dict[str, Any] = {
            "fm_result_batches": 20,
            "fm_samples": int(fm_arrays["source_truth"].shape[0]),
            "fm_grid": list(fm_arrays["source_truth"].shape[1:]),
            "fm_num_obs": int(fm_config["num_obs"]),
            "fm_mask_seed": int(fm_config["mask_seed"]),
            "fm_noise_level": float(fm_config["noise_level"]),
            "baselines": {},
        }

        for field in TASK_FIELDS[task]:
            truth = fm_arrays[f"{field}_truth"]
            print(f"  evaluating FM4PDE field={field}", flush=True)
            evaluate_method(
                task=task,
                field=field,
                method="FM4PDE",
                prediction=fm_arrays[f"{field}_pred"],
                truth=truth,
                metrics=metrics,
                radial_rows=radial_rows,
            )

        if task == "both":
            # Choose a typical FM4PDE joint sample according to the sum of the
            # two full-field spectral relative errors.
            fm_all = [
                row
                for row in metrics
                if row["task"] == "both" and row["method"] == "FM4PDE" and row["band"] == "all"
            ]
            per_sample = defaultdict(float)
            for row in fm_all:
                per_sample[int(row["sample_id"])] += float(row["relative_error"])
            median_score = np.median(list(per_sample.values()))
            representative_sample = min(per_sample, key=lambda i: abs(per_sample[i] - median_score))
            joint_truths = {
                "source": fm_arrays["source_truth"],
                "solution": fm_arrays["solution_truth"],
            }
            for field in TASK_FIELDS[task]:
                joint_examples[("FM4PDE", field)] = fm_arrays[f"{field}_pred"][representative_sample]

        baseline_runs = BASELINE_RUNS[task]
        first_method = next(iter(baseline_runs))
        validation = validate_first_baseline_truth(
            args.baseline_root / baseline_runs[first_method], fm_arrays, task
        )
        task_quality["sample0_truth_validation"] = validation

        # Quantify the known observation mismatch on the first comparable run.
        first_sample = torch.load(
            find_sample_dir(args.baseline_root / baseline_runs[first_method]) / "sample_000000.pt",
            map_location="cpu",
            weights_only=False,
        )
        baseline_mask = first_sample["mask"].detach().cpu().numpy()
        if baseline_mask.ndim == 4:
            baseline_mask = baseline_mask[0]
        fm_mask = np.stack(
            [fm_arrays[f"{field}_mask"][0] for field in TASK_FIELDS[task]], axis=0
        )
        overlap = int(np.logical_and(baseline_mask != 0, fm_mask != 0).sum())
        task_quality["sample0_observation_check"] = {
            "baseline_observed_entries": int(np.count_nonzero(baseline_mask)),
            "fm_observed_entries": int(np.count_nonzero(fm_mask)),
            "overlap_entries": overlap,
            "masks_identical": bool(np.array_equal(baseline_mask != 0, fm_mask != 0)),
        }
        del first_sample

        for method, relative_run in baseline_runs.items():
            run_dir = args.baseline_root / relative_run
            print(f"  loading baseline {method} ({task})", flush=True)
            predictions, metadata = load_baseline_predictions(run_dir)
            expected_channels = len(TASK_FIELDS[task])
            if predictions.shape != (1000, expected_channels, 128, 128):
                raise RuntimeError(f"Unexpected shape {predictions.shape} for {run_dir}")
            if not np.isfinite(predictions).all():
                raise RuntimeError(f"Non-finite predictions in {run_dir}")
            task_quality["baselines"][method] = {
                "run_dir": str(run_dir),
                "samples": int(predictions.shape[0]),
                "shape": list(predictions.shape[1:]),
                "metadata": metadata,
            }
            for channel, field in enumerate(TASK_FIELDS[task]):
                truth = fm_arrays[f"{field}_truth"]
                print(f"    evaluating field={field}", flush=True)
                evaluate_method(
                    task=task,
                    field=field,
                    method=method,
                    prediction=predictions[:, channel],
                    truth=truth,
                    metrics=metrics,
                    radial_rows=radial_rows,
                )
                if task == "both":
                    joint_examples[(method, field)] = predictions[representative_sample, channel].copy()
            del predictions

        quality["tasks"][task] = task_quality
        del fm_arrays

    summary = summarize_metrics(metrics)
    write_csv(args.output / "poisson_frequency_metrics_per_sample.csv", metrics)
    write_csv(args.output / "poisson_frequency_summary.csv", summary)
    write_csv(args.output / "poisson_radial_summary.csv", radial_rows)

    plot_band_metric(
        summary,
        args.output,
        metric="relative_error",
        ylabel="band relative error",
        filename="poisson_band_relative_error.png",
    )
    plot_band_metric(
        summary,
        args.output,
        metric="energy_ratio",
        ylabel="predicted / true spectral energy",
        filename="poisson_band_energy_ratio.png",
    )
    plot_radial(radial_rows, args.output)
    plot_joint_example(joint_examples, joint_truths, representative_sample, args.output)

    quality["representative_joint_sample"] = representative_sample
    quality["metrics_rows"] = len(metrics)
    quality["summary_rows"] = len(summary)
    with (args.output / "data_quality_report.json").open("w") as handle:
        json.dump(quality, handle, indent=2, ensure_ascii=False, default=str)
    print(f"Analysis written to {args.output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate and diagnose unconditional Poisson samples from an FM4PDE checkpoint.

The sampler intentionally matches ``train._euler_flow_sample``: standard normal
initial states, a uniform time grid, deterministic Euler integration, no sparse
observations, and no PDE-gradient guidance.  Generated and real fields are then
compared with boundary-aware radial spectra and spatial roughness statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from scipy.fft import dctn, dstn
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TRAIN_DIR = ROOT / (
    "outputs/pretrained/formal/poisson/"
    "260831-122417-poisson-batch32-epoch300-accum1-bfloat16"
)
DEFAULT_CHECKPOINT = DEFAULT_TRAIN_DIR / "fm4poisson-checkpoint-99.pth"
DEFAULT_OUTPUT = ROOT / "outputs/artifacts/poisson_unconditional_distribution_epoch100"
DEFAULT_REAL_CACHE = ROOT / "outputs/analysis/poisson_real_cache"

DATASETS = ("generated", "id", "smooth", "rough")
REAL_DATASETS = ("id", "smooth", "rough")
FIELDS = ("source", "solution")
FIELD_FILES = {"source": "f", "solution": "phi"}
COLORS = {
    "generated": "#E07A24",
    "id": "#3568A8",
    "smooth": "#7A8B3A",
    "rough": "#B85C7A",
    "equal_mixture": "#5D6470",
}
METRIC_COLUMNS = (
    "mean",
    "std",
    "variance",
    "gradient_energy",
    "normalized_gradient_energy",
    "correlation_length_cells",
    "high_frequency_ratio_r_gt_16",
    "high_frequency_ratio_r_gt_32",
    "high_frequency_ratio_r_gt_48",
    "high_frequency_ratio_r_gt_64",
    "psd_band_r_le_4",
    "psd_band_4_8",
    "psd_band_8_16",
    "psd_band_16_32",
    "psd_band_32_64",
    "psd_band_r_gt_64",
)
CLASSIFIER_FEATURES = (
    "std",
    "normalized_gradient_energy",
    "correlation_length_cells",
    "psd_band_r_le_4",
    "psd_band_4_8",
    "psd_band_8_16",
    "psd_band_16_32",
    "psd_band_32_64",
    "psd_band_r_gt_64",
)
DISTANCE_METRICS = (
    "std",
    "gradient_energy",
    "normalized_gradient_energy",
    "correlation_length_cells",
    "high_frequency_ratio_r_gt_32",
)
PSD_BANDS = (
    ("psd_band_r_le_4", 0, 4),
    ("psd_band_4_8", 4, 8),
    ("psd_band_8_16", 8, 16),
    ("psd_band_16_32", 16, 32),
    ("psd_band_32_64", 32, 64),
    ("psd_band_r_gt_64", 64, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--real-cache", type=Path, default=DEFAULT_REAL_CACHE)
    parser.add_argument("--stage", choices=("all", "generate", "analyze"), default="all")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--real-batch-size", type=int, default=64)
    parser.add_argument("--real-samples-per-type", type=int, default=10000)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_000_103)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def generate_samples(args: argparse.Namespace) -> Path:
    import torch

    from sampling.model_io import load_fm4pde_checkpoint_bundle

    output_path = args.output / "generated_samples_physical.npy"
    manifest_path = args.output / "generation_manifest.json"
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output_path.is_file() and manifest_path.is_file() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = [args.num_samples, 2, 128, 128]
        existing = np.load(output_path, mmap_mode="r")
        if list(existing.shape) == expected and manifest.get("complete") is True:
            print(f"REUSE {output_path} shape={existing.shape}", flush=True)
            return output_path
        raise RuntimeError(
            f"Incomplete or incompatible generated artifact at {output_path}; use --force"
        )

    if args.num_samples < 1 or args.batch_size < 1 or args.num_steps < 1:
        raise ValueError("num-samples, batch-size and num-steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    args.output.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".partial.npy")
    if partial.exists():
        partial.unlink()

    print(f"LOAD checkpoint={checkpoint}", flush=True)
    model, normalizer, payload = load_fm4pde_checkpoint_bundle(
        str(checkpoint), "poisson", device=device, wrap=False, model_profile="recommended"
    )
    model.eval()
    model_config = payload["model_config"]
    if int(model_config.get("in_channels", -1)) != 2:
        raise ValueError(f"Expected two Poisson channels, got {model_config.get('in_channels')}")
    if model_config.get("num_classes") is not None:
        raise ValueError("This script expects the current single-PDE unconditional checkpoint")

    samples = open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=(args.num_samples, 2, 128, 128),
    )
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    grid = torch.linspace(0.0, 1.0, args.num_steps + 1, device=device, dtype=torch.float32)
    started = time.time()
    max_allocated = 0

    with torch.inference_mode():
        for start in range(0, args.num_samples, args.batch_size):
            stop = min(start + args.batch_size, args.num_samples)
            batch = stop - start
            x = torch.randn(
                batch,
                2,
                128,
                128,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            for step in range(args.num_steps):
                t = torch.full(
                    (batch,), float(grid[step].item()), device=device, dtype=torch.float32
                )
                velocity = model(x, t, extra={})
                if velocity.shape != x.shape:
                    raise ValueError(
                        f"Model output {tuple(velocity.shape)} does not match {tuple(x.shape)}"
                    )
                x = x + (grid[step + 1] - grid[step]) * velocity
            physical = normalizer.inverse_transform(x).float().cpu().numpy()
            samples[start:stop] = physical
            samples.flush()
            if device.type == "cuda":
                max_allocated = max(max_allocated, int(torch.cuda.max_memory_allocated(device)))
            elapsed = time.time() - started
            rate = stop / max(elapsed, 1e-9)
            eta = (args.num_samples - stop) / max(rate, 1e-9)
            print(
                f"GENERATE {stop:4d}/{args.num_samples} elapsed={elapsed:.1f}s "
                f"rate={rate:.2f}/s eta={eta:.1f}s",
                flush=True,
            )

    del samples
    partial.replace(output_path)
    manifest = {
        "complete": True,
        "checkpoint": str(checkpoint),
        "checkpoint_slimmed_from": payload.get("slimmed_from"),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch_zero_based": int(payload.get("epoch", -1)),
        "reported_training_epoch": int(payload.get("epoch", -1)) + 1,
        "selected_inference_weight": payload.get("selected_inference_weight"),
        "has_ema": bool(payload.get("has_ema", False)),
        "model_profile": payload.get("selected_model_profile"),
        "model_config": model_config,
        "normalizer": normalizer.to_json_dict(),
        "sampler": {
            "kind": "unconditional deterministic flow",
            "initial_distribution": "standard normal in standardized state space",
            "time_grid": "uniform [0, 1]",
            "step_method": "Euler",
            "guidance": "none",
            "num_steps": args.num_steps,
            "dtype": "float32",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
        },
        "output": str(output_path.resolve()),
        "shape": [args.num_samples, 2, 128, 128],
        "duration_seconds": time.time() - started,
        "max_cuda_memory_allocated_bytes": max_allocated,
    }
    write_json(manifest_path, manifest)
    print(f"WRITE {output_path} ({output_path.stat().st_size} bytes)", flush=True)
    return output_path


def radial_geometry(field: str, size: int = 128) -> tuple[np.ndarray, int]:
    if field == "source":
        indices = np.arange(size, dtype=np.float64)
    elif field == "solution":
        indices = np.arange(1, size - 1, dtype=np.float64)
    else:
        raise ValueError(field)
    shells = np.floor(np.hypot(indices[:, None], indices[None, :])).astype(np.int16)
    return shells, int(shells.max())


def correlation_length(fields: np.ndarray, max_lag: int = 64) -> np.ndarray:
    centered = fields - fields.mean(axis=(-2, -1), keepdims=True)
    batch, height, width = centered.shape
    fx = np.fft.rfft(centered, n=2 * width, axis=-1)
    corr_x = np.fft.irfft(fx * fx.conj(), n=2 * width, axis=-1)[..., : max_lag + 1]
    corr_x = corr_x.sum(axis=1)
    fy = np.fft.rfft(centered, n=2 * height, axis=-2)
    corr_y = np.fft.irfft(fy * fy.conj(), n=2 * height, axis=-2)[:, : max_lag + 1]
    corr_y = corr_y.sum(axis=-1)
    lags = np.arange(max_lag + 1)
    corr_x /= (height * np.maximum(width - lags, 1))[None, :]
    corr_y /= (width * np.maximum(height - lags, 1))[None, :]
    acf = 0.5 * (corr_x + corr_y)
    acf /= np.maximum(acf[:, :1], np.finfo(np.float64).tiny)
    threshold = math.exp(-1.0)
    result = np.full(batch, float(max_lag), dtype=np.float64)
    for item in range(batch):
        crossings = np.flatnonzero(acf[item, 1:] <= threshold)
        if crossings.size == 0:
            continue
        upper = int(crossings[0] + 1)
        lower = upper - 1
        y0, y1 = float(acf[item, lower]), float(acf[item, upper])
        if y1 == y0:
            result[item] = float(upper)
        else:
            result[item] = lower + (threshold - y0) / (y1 - y0)
    return result


def compute_batch_metrics(fields: np.ndarray, field: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    values = np.asarray(fields, dtype=np.float64)
    means = values.mean(axis=(-2, -1))
    centered = values - means[:, None, None]
    variance = np.mean(centered**2, axis=(-2, -1))
    dx = np.diff(values, axis=-1)
    dy = np.diff(values, axis=-2)
    gradient = 0.5 * (np.mean(dx**2, axis=(-2, -1)) + np.mean(dy**2, axis=(-2, -1)))
    normalized_gradient = gradient / np.maximum(variance, np.finfo(np.float64).tiny)
    lengths = correlation_length(values)

    shells, max_shell = radial_geometry(field, values.shape[-1])
    if field == "source":
        coefficients = dctn(centered, type=2, norm="ortho", axes=(-2, -1), workers=-1)
    else:
        interior = values[:, 1:-1, 1:-1]
        coefficients = dstn(interior, type=1, norm="ortho", axes=(-2, -1), workers=-1)
    power = np.asarray(coefficients**2, dtype=np.float64)
    shell_flat = shells.ravel()
    radial = np.stack(
        [np.bincount(shell_flat, weights=item.ravel(), minlength=max_shell + 1) for item in power],
        axis=0,
    )
    total = np.maximum(radial.sum(axis=1), np.finfo(np.float64).tiny)
    radial /= total[:, None]

    metrics: dict[str, np.ndarray] = {
        "mean": means,
        "std": np.sqrt(variance),
        "variance": variance,
        "gradient_energy": gradient,
        "normalized_gradient_energy": normalized_gradient,
        "correlation_length_cells": lengths,
    }
    radius = np.arange(radial.shape[1])
    for cutoff in (16, 32, 48, 64):
        metrics[f"high_frequency_ratio_r_gt_{cutoff}"] = radial[:, radius > cutoff].sum(axis=1)
    for name, lower, upper in PSD_BANDS:
        if upper is None:
            mask = radius > lower
        elif lower == 0:
            mask = radius <= upper
        else:
            mask = (radius > lower) & (radius <= upper)
        metrics[name] = radial[:, mask].sum(axis=1)
    return metrics, radial


def allocate_metric_table(sample_count: int) -> dict[str, np.ndarray]:
    return {name: np.empty(sample_count, dtype=np.float64) for name in METRIC_COLUMNS}


def collect_metrics(
    source: np.ndarray,
    solution: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    dataset: str,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for field, array in (("source", source), ("solution", solution)):
        table = allocate_metric_table(indices.size)
        _, max_shell = radial_geometry(field)
        radial = np.empty((indices.size, max_shell + 1), dtype=np.float32)
        for start in range(0, indices.size, batch_size):
            stop = min(start + batch_size, indices.size)
            batch_indices = indices[start:stop]
            batch_values = np.asarray(array[batch_indices], dtype=np.float64)
            batch_metrics, batch_radial = compute_batch_metrics(batch_values, field)
            for name in METRIC_COLUMNS:
                table[name][start:stop] = batch_metrics[name]
            radial[start:stop] = batch_radial.astype(np.float32)
            if stop == indices.size or stop % max(batch_size * 10, 1) == 0:
                print(f"METRICS dataset={dataset:<9s} field={field:<8s} {stop}/{indices.size}", flush=True)
        table["radial_psd"] = radial
        result[field] = table
    return result


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.quantile(finite, 0.50)),
        "q75": float(np.quantile(finite, 0.75)),
        "q95": float(np.quantile(finite, 0.95)),
    }


def log_feature_matrix(
    table: dict[str, np.ndarray], feature_names: Iterable[str] = CLASSIFIER_FEATURES
) -> np.ndarray:
    eps = 1e-14
    return np.column_stack(
        [np.log10(np.maximum(np.asarray(table[name]), eps)) for name in feature_names]
    )


def weighted_knn_labels(
    tree: cKDTree,
    train_labels: np.ndarray,
    features: np.ndarray,
    k: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    distances, neighbors = tree.query(features, k=k, workers=-1)
    distances = np.atleast_2d(distances)
    neighbors = np.atleast_2d(neighbors)
    weights = 1.0 / np.maximum(distances, 1e-12)
    votes = np.stack(
        [np.sum(weights * (train_labels[neighbors] == class_id), axis=1) for class_id in range(3)],
        axis=1,
    )
    labels = np.argmax(votes, axis=1)
    density_score = np.mean(distances[:, : min(5, k)], axis=1)
    return labels, density_score


def classify_generated(
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
    field: str,
    seed: int = 20260831,
    feature_names: tuple[str, ...] = CLASSIFIER_FEATURES,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    for class_id, dataset in enumerate(REAL_DATASETS):
        features = log_feature_matrix(all_metrics[dataset][field], feature_names)
        order = rng.permutation(features.shape[0])
        split = int(round(0.70 * features.shape[0]))
        train_parts.append(features[order[:split]])
        train_labels.append(np.full(split, class_id, dtype=np.int8))
        test_parts.append(features[order[split:]])
        test_labels.append(np.full(features.shape[0] - split, class_id, dtype=np.int8))
    train = np.concatenate(train_parts)
    labels_train = np.concatenate(train_labels)
    test = np.concatenate(test_parts)
    labels_test = np.concatenate(test_labels)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    train_z = (train - mean) / std
    test_z = (test - mean) / std
    generated_z = (log_feature_matrix(all_metrics["generated"][field], feature_names) - mean) / std
    tree = cKDTree(train_z)
    predicted_test, test_density = weighted_knn_labels(tree, labels_train, test_z)
    predicted_generated, generated_density = weighted_knn_labels(tree, labels_train, generated_z)
    confusion = np.zeros((3, 3), dtype=int)
    for truth, prediction in zip(labels_test, predicted_test):
        confusion[int(truth), int(prediction)] += 1
    thresholds = {
        dataset: float(np.quantile(test_density[labels_test == class_id], 0.99))
        for class_id, dataset in enumerate(REAL_DATASETS)
    }
    threshold_array = np.asarray([thresholds[REAL_DATASETS[int(label)]] for label in predicted_generated])
    ood = generated_density > threshold_array
    counts_all = {dataset: int(np.sum(predicted_generated == class_id)) for class_id, dataset in enumerate(REAL_DATASETS)}
    counts_in = {
        dataset: int(np.sum((predicted_generated == class_id) & ~ood))
        for class_id, dataset in enumerate(REAL_DATASETS)
    }
    in_count = int(np.sum(~ood))
    return {
        "features": list(feature_names),
        "classifier": "15-nearest-neighbor on standardized log-features",
        "ood_score": "mean distance to five nearest real-training feature vectors",
        "ood_threshold": "class-specific 99th percentile on held-out real samples",
        "real_holdout_accuracy": float(np.mean(predicted_test == labels_test)),
        "real_holdout_confusion_rows_true_columns_predicted": confusion.tolist(),
        "real_holdout_samples": int(labels_test.size),
        "ood_thresholds": thresholds,
        "generated_assignment_counts_before_ood": counts_all,
        "generated_samples": int(predicted_generated.size),
        "generated_assignment_fractions_before_ood": {
            key: value / predicted_generated.size for key, value in counts_all.items()
        },
        "generated_ood_count": int(np.sum(ood)),
        "generated_ood_fraction": float(np.mean(ood)),
        "generated_in_distribution_count": in_count,
        "generated_in_distribution_assignment_counts": counts_in,
        "generated_in_distribution_assignment_fractions": {
            key: (value / in_count if in_count else None) for key, value in counts_in.items()
        },
        "generated_labels": predicted_generated,
        "generated_ood": ood,
        "generated_density_score": generated_density,
    }


def reference_table(
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]], field: str, reference: str
) -> dict[str, np.ndarray]:
    if reference != "equal_mixture":
        return all_metrics[reference][field]
    return {
        name: np.concatenate([all_metrics[dataset][field][name] for dataset in REAL_DATASETS])
        for name in (*METRIC_COLUMNS, "radial_psd")
    }


def distribution_distances(
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]], field: str
) -> dict[str, Any]:
    generated = all_metrics["generated"][field]
    pooled = reference_table(all_metrics, field, "equal_mixture")
    references = (*REAL_DATASETS, "equal_mixture")
    result: dict[str, Any] = {}
    max_shell = generated["radial_psd"].shape[1] - 1
    shells = np.arange(max_shell + 1, dtype=float)
    generated_psd = np.asarray(generated["radial_psd"], dtype=float).mean(axis=0)
    generated_psd /= generated_psd.sum()
    for reference in references:
        target = reference_table(all_metrics, field, reference)
        scalar: dict[str, float] = {}
        for metric in DISTANCE_METRICS:
            gen_values = np.log10(np.maximum(generated[metric], 1e-14))
            target_values = np.log10(np.maximum(target[metric], 1e-14))
            scale = float(np.std(np.log10(np.maximum(pooled[metric], 1e-14))))
            scalar[metric] = float(wasserstein_distance(gen_values, target_values) / max(scale, 1e-12))
        target_psd = np.asarray(target["radial_psd"], dtype=float).mean(axis=0)
        target_psd /= target_psd.sum()
        midpoint = 0.5 * (generated_psd + target_psd)
        eps = np.finfo(float).tiny
        js = 0.5 * np.sum(generated_psd * np.log(np.maximum(generated_psd, eps) / np.maximum(midpoint, eps)))
        js += 0.5 * np.sum(target_psd * np.log(np.maximum(target_psd, eps) / np.maximum(midpoint, eps)))
        result[reference] = {
            "scalar_log_wasserstein_in_pooled_std_units": scalar,
            "scalar_distance_mean": float(np.mean(list(scalar.values()))),
            "radial_psd_wasserstein_normalized_radius": float(
                wasserstein_distance(shells, shells, u_weights=generated_psd, v_weights=target_psd)
                / max(max_shell, 1)
            ),
            "radial_psd_jensen_shannon_divergence": float(js),
        }
    return result


def write_per_sample_csv(
    path: Path, all_metrics: dict[str, dict[str, dict[str, np.ndarray]]]
) -> None:
    fields = ["dataset", "field", "sample_id", *METRIC_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for field in FIELDS:
                table = all_metrics[dataset][field]
                for sample_id in range(table["std"].size):
                    row = {"dataset": dataset, "field": field, "sample_id": sample_id}
                    row.update({name: float(table[name][sample_id]) for name in METRIC_COLUMNS})
                    writer.writerow(row)


def save_metrics_npz(path: Path, all_metrics: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        for field in FIELDS:
            for name, values in all_metrics[dataset][field].items():
                arrays[f"{dataset}__{field}__{name}"] = np.asarray(values)
    np.savez_compressed(path, **arrays)


def physical_consistency_metrics(
    source: np.ndarray, solution: np.ndarray, batch_size: int
) -> dict[str, np.ndarray]:
    sample_count = int(source.shape[0])
    result = {
        "poisson_residual_rms": np.empty(sample_count, dtype=np.float64),
        "poisson_relative_residual": np.empty(sample_count, dtype=np.float64),
        "solution_boundary_rms": np.empty(sample_count, dtype=np.float64),
        "solution_boundary_rms_over_std": np.empty(sample_count, dtype=np.float64),
    }
    h = 1.0 / 127.0
    eps = np.finfo(np.float64).tiny
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        f = np.asarray(source[start:stop], dtype=np.float64)
        phi = np.asarray(solution[start:stop], dtype=np.float64)
        laplacian = (
            phi[:, 2:, 1:-1]
            + phi[:, :-2, 1:-1]
            + phi[:, 1:-1, 2:]
            + phi[:, 1:-1, :-2]
            - 4.0 * phi[:, 1:-1, 1:-1]
        ) / (h**2)
        residual = laplacian - f[:, 1:-1, 1:-1]
        residual_rms = np.sqrt(np.mean(residual**2, axis=(-2, -1)))
        source_rms = np.sqrt(np.mean(f[:, 1:-1, 1:-1] ** 2, axis=(-2, -1)))
        boundary = np.concatenate(
            [phi[:, 0, :], phi[:, -1, :], phi[:, 1:-1, 0], phi[:, 1:-1, -1]],
            axis=1,
        )
        boundary_rms = np.sqrt(np.mean(boundary**2, axis=1))
        solution_std = np.std(phi, axis=(-2, -1))
        result["poisson_residual_rms"][start:stop] = residual_rms
        result["poisson_relative_residual"][start:stop] = residual_rms / np.maximum(source_rms, eps)
        result["solution_boundary_rms"][start:stop] = boundary_rms
        result["solution_boundary_rms_over_std"][start:stop] = boundary_rms / np.maximum(solution_std, eps)
    return result


def write_physical_consistency_csv(
    path: Path, consistency: dict[str, dict[str, np.ndarray]]
) -> None:
    metric_names = list(next(iter(consistency.values())))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "sample_id", *metric_names])
        writer.writeheader()
        for dataset in DATASETS:
            table = consistency[dataset]
            for sample_id in range(table[metric_names[0]].size):
                row = {"dataset": dataset, "sample_id": sample_id}
                row.update({name: float(table[name][sample_id]) for name in metric_names})
                writer.writerow(row)


def setup_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#D9DDE3", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def plot_radial_psd(
    output: Path,
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
    field: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 6.1), constrained_layout=True)
    for dataset in DATASETS:
        values = np.asarray(all_metrics[dataset][field]["radial_psd"], dtype=float)
        median = np.median(values, axis=0)
        q10, q90 = np.quantile(values, [0.10, 0.90], axis=0)
        radius = np.arange(values.shape[1])
        valid = radius <= 96
        ax.plot(radius[valid], np.maximum(median[valid], 1e-12), color=COLORS[dataset], label=dataset, lw=2)
        ax.fill_between(
            radius[valid],
            np.maximum(q10[valid], 1e-12),
            np.maximum(q90[valid], 1e-12),
            color=COLORS[dataset],
            alpha=0.09,
            linewidth=0,
        )
    ax.axvline(32, color="#33383F", linestyle="--", linewidth=1.1, label="HF cutoff r=32")
    ax.set_yscale("log")
    ax.set_xlim(0, 96)
    ax.set_xlabel("Radial spectral mode r")
    ax.set_ylabel("Per-sample normalized spectral energy")
    title_field = "source f (DCT-II)" if field == "source" else "solution phi (DST-I interior)"
    ax.set_title(f"Radial PSD — {title_field}\nMedian with 10th–90th percentile band")
    ax.legend(ncol=2, frameon=False)
    setup_axes(ax)
    fig.savefig(output / f"radial_psd_{field}.png", dpi=190)
    plt.close(fig)


def plot_metric_distributions(
    output: Path,
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]],
    field: str,
) -> None:
    specs = (
        ("high_frequency_ratio_r_gt_32", "High-frequency energy ratio (r > 32)", True),
        ("correlation_length_cells", "Correlation length (grid cells, 1/e)", False),
        ("gradient_energy", "Gradient energy", True),
        ("normalized_gradient_energy", "Gradient energy / variance", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    for ax, (metric, title, log_scale) in zip(axes.flat, specs):
        values = [all_metrics[dataset][field][metric] for dataset in DATASETS]
        box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.62)
        for patch, dataset in zip(box["boxes"], DATASETS):
            patch.set_facecolor(COLORS[dataset])
            patch.set_alpha(0.28)
            patch.set_edgecolor(COLORS[dataset])
        for median in box["medians"]:
            median.set_color("#20242A")
            median.set_linewidth(1.5)
        ax.set_xticks(range(1, 5), DATASETS, rotation=15)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        setup_axes(ax)
    fig.suptitle(
        f"Spatial and spectral statistics — {field}\n"
        "Generated n=1000; each real category uses the selected reference sample count",
        fontsize=14,
    )
    fig.savefig(output / f"metric_distributions_{field}.png", dpi=190)
    plt.close(fig)


def plot_assignment_coverage(output: Path, classifications: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    for ax, field in zip(axes, FIELDS):
        item = classifications[field]
        generated_count = int(item["generated_samples"])
        in_counts = item["generated_in_distribution_assignment_counts"]
        values = [in_counts[name] / generated_count for name in REAL_DATASETS]
        values.append(item["generated_ood_fraction"])
        labels = [*REAL_DATASETS, "OOD"]
        colors = [COLORS[name] for name in REAL_DATASETS] + ["#5D6470"]
        bars = ax.bar(labels, values, color=colors, edgecolor="#33383F", linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.1%}", ha="center", fontsize=9)
        ax.set_ylim(0, max(0.55, max(values) * 1.18))
        ax.set_ylabel("Fraction of all generated samples")
        ax.set_title(f"{field}: nearest real mode and OOD")
        setup_axes(ax)
    fig.suptitle("Generated-sample mode coverage (kNN feature-space diagnostic)", fontsize=14)
    fig.savefig(output / "mode_coverage_and_ood.png", dpi=190)
    plt.close(fig)


def plot_distance_bars(output: Path, distances: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.1), constrained_layout=True)
    refs = (*REAL_DATASETS, "equal_mixture")
    colors = [COLORS[ref] for ref in refs]
    for column, field in enumerate(FIELDS):
        scalar = [distances[field][ref]["scalar_distance_mean"] for ref in refs]
        psd = [distances[field][ref]["radial_psd_wasserstein_normalized_radius"] for ref in refs]
        labels = [name.replace("equal_", "") for name in refs]
        for row, (values, ylabel, suffix) in enumerate(
            (
                (scalar, "Mean scalar W1 (pooled-real std units)", "scalar statistics"),
                (psd, "Radial PSD W1 (normalized radius)", "radial PSD"),
            )
        ):
            ax = axes[row, column]
            ax.bar(labels, values, color=colors, edgecolor="#33383F", linewidth=0.6)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{field}: {suffix}")
            ax.tick_params(axis="x", rotation=15)
            setup_axes(ax)
    fig.suptitle("Generated distribution distance to real references", fontsize=14)
    fig.savefig(output / "distribution_distances.png", dpi=190)
    plt.close(fig)


def plot_physical_consistency(
    output: Path, consistency: dict[str, dict[str, np.ndarray]]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)
    specs = (
        ("poisson_relative_residual", "Relative Poisson residual ||Δφ−f|| / ||f||", "log"),
        ("solution_boundary_rms_over_std", "Boundary RMS / solution spatial std", "linear"),
    )
    for ax, (metric, title, scale) in zip(axes, specs):
        values = [consistency[dataset][metric] for dataset in DATASETS]
        box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.62)
        for patch, dataset in zip(box["boxes"], DATASETS):
            patch.set_facecolor(COLORS[dataset])
            patch.set_alpha(0.28)
            patch.set_edgecolor(COLORS[dataset])
        ax.set_xticks(range(1, 5), DATASETS, rotation=15)
        if scale == "log":
            ax.set_yscale("log")
        ax.set_title(title)
        setup_axes(ax)
    fig.suptitle("Joint-field physical consistency", fontsize=14)
    fig.savefig(output / "physical_consistency.png", dpi=190)
    plt.close(fig)


def plot_sample_montage(
    output: Path,
    generated: np.ndarray,
    real_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    rng = np.random.default_rng(260831)
    indices = {"generated": rng.integers(0, generated.shape[0], size=3)}
    for dataset in REAL_DATASETS:
        indices[dataset] = rng.integers(0, real_arrays[dataset][0].shape[0], size=3)
    fig, axes = plt.subplots(4, 6, figsize=(13.2, 8.7), constrained_layout=True)
    for row, dataset in enumerate(DATASETS):
        if dataset == "generated":
            source = generated[:, 0]
            solution = generated[:, 1]
        else:
            source, solution = real_arrays[dataset]
        chosen = indices[dataset]
        source_scale = float(np.quantile(np.abs(source[chosen]), 0.995))
        solution_scale = float(np.quantile(np.abs(solution[chosen]), 0.995))
        for col, sample_id in enumerate(chosen):
            for offset, (array, scale, title) in enumerate(
                ((source, source_scale, "f"), (solution, solution_scale, "phi"))
            ):
                ax = axes[row, 2 * col + offset]
                ax.imshow(array[sample_id], cmap="RdBu_r", origin="lower", vmin=-scale, vmax=scale)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"sample {col + 1}: {title}")
                if col == 0 and offset == 0:
                    ax.set_ylabel(dataset)
    fig.suptitle("Unconditional FM4PDE samples and real-reference examples", fontsize=14)
    fig.savefig(output / "sample_montage.png", dpi=180)
    plt.close(fig)


def clean_classification(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not isinstance(value, np.ndarray)}


def make_markdown_summary(summary: dict[str, Any]) -> str:
    def format_percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.1%}"

    consistency = summary["physical_consistency_summary"]
    generated_residual = consistency["generated"]["poisson_relative_residual"]["median"]
    real_residuals = {
        dataset: consistency[dataset]["poisson_relative_residual"]["median"]
        for dataset in REAL_DATASETS
    }
    lines = [
        "# FM4PDE Poisson 无指导生成分布诊断",
        "",
        "## 结论摘要",
        "",
        "- **完整的 `(f, φ)` 生成对目前不接近 id/smooth/rough 中任何一类。** "
        "source 边缘分布最接近三类等权混合，但 solution 高频尾部和 Poisson 配对关系明显偏离真实数据。",
    ]
    for field in FIELDS:
        cls = summary["classification"][field]
        distance = summary["distribution_distances"][field]
        nearest_scalar = min(distance, key=lambda key: distance[key]["scalar_distance_mean"])
        nearest_psd = min(
            distance,
            key=lambda key: distance[key]["radial_psd_wasserstein_normalized_radius"],
        )
        fractions = cls["generated_in_distribution_assignment_fractions"]
        raw_fractions = cls["generated_assignment_fractions_before_ood"]
        core_ood = summary["classification_core_metric_sensitivity"][field][
            "generated_ood_fraction"
        ]
        lines.append(
            f"- **{field}**：OOD {cls['generated_ood_fraction']:.1%}；"
            f"仅用四个核心指标时为 {core_ood:.1%}。"
            f"标注 OOD 前的 id/smooth/rough 最近类占比为 "
            f"{raw_fractions['id']:.1%}/{raw_fractions['smooth']:.1%}/"
            f"{raw_fractions['rough']:.1%}；"
            f"非 OOD 样本的 id/smooth/rough 占比为 "
            f"{format_percent(fractions['id'])}/{format_percent(fractions['smooth'])}/"
            f"{format_percent(fractions['rough'])}；"
            f"标量分布最近 `{nearest_scalar}`，径向 PSD 最近 `{nearest_psd}`。"
        )
    lines.extend(
        [
            "",
            "## 联合场物理一致性",
            "",
            f"- 生成样本的 `||Δφ−f||/||f||` 中位数为 **{generated_residual:.3g}**；"
            f"真实 id/smooth/rough 分别为 "
            f"{real_residuals['id']:.3g}/{real_residuals['smooth']:.3g}/{real_residuals['rough']:.3g}。",
            "- 因此 source 边缘分布接近训练混合，并不表示生成的 `(f, φ)` 是有效 Poisson 配对。",
            "",
            "## 实验定义",
            "",
            f"- checkpoint：`{summary['generation']['checkpoint']}`",
            f"- 生成：{summary['generation']['sampler']['num_samples']} 个样本，"
            f"{summary['generation']['sampler']['num_steps']} 步 Euler，"
            f"seed={summary['generation']['sampler']['seed']}，无观测和 PDE 梯度指导。",
            f"- 真实参考：id/smooth/rough 各 {summary['real_samples_per_type']} 个样本。",
            "- 径向 PSD：source 使用去均值后的正交 DCT-II；solution 使用零 Dirichlet 边界兼容的内点 DST-I。",
            "- 高频能量比：径向模式 r>32 的能量占全部谱能量的比例；另保存 r>16/48/64 敏感性结果。",
            "- 相关长度：横纵两个方向的无偏线性自相关平均后，首次降到 1/e 的插值网格距离。",
            "- 梯度能量：相邻网格一阶差分平方的横纵平均；同时报告除以场方差后的归一化值。",
            "- OOD：真实样本 70/30 拆分，在所列粗糙度与 PSD 特征上做 15-NN；阈值为各真实类别留出集最近邻距离的 99 分位。",
            "",
            "## 分类可靠性",
            "",
        ]
    )
    for field in FIELDS:
        cls = summary["classification"][field]
        lines.append(
            f"- {field} 的真实留出集分类准确率：{cls['real_holdout_accuracy']:.2%} "
            f"(n={cls['real_holdout_samples']})。"
        )
    lines.extend(
        [
            "",
            "## 重要限制",
            "",
            "- 这是单个训练 checkpoint、单个采样 seed 的分布诊断；1000 个生成样本可描述该次生成分布，但不能替代多训练 seed 稳定性分析。",
            "- `equal_mixture` 是三类真实样本等权合并。若模型正确覆盖三种模式，它可能离任何单一类别都不最近，却应接近该混合参考。",
            "- OOD 判定只覆盖本报告的频谱/空间统计，不等价于完整图像分布的感知质量判定。",
            "",
            "详细数字见 `summary.json`、`metrics_per_sample.csv` 和 `metrics_arrays.npz`。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace, generated_path: Path) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    generated = np.load(generated_path, mmap_mode="r")
    if generated.shape != (args.num_samples, 2, 128, 128):
        raise ValueError(f"Unexpected generated shape: {generated.shape}")
    if not np.isfinite(generated).all():
        raise ValueError("Generated samples contain non-finite values")
    generated_indices = np.arange(args.num_samples)
    all_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "generated": collect_metrics(
            generated[:, 0], generated[:, 1], generated_indices, args.real_batch_size, "generated"
        )
    }
    real_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dataset in REAL_DATASETS:
        source_path = args.real_cache / f"{dataset}_f.npy"
        solution_path = args.real_cache / f"{dataset}_phi.npy"
        if not source_path.is_file() or not solution_path.is_file():
            raise FileNotFoundError(f"Missing cached real arrays for {dataset}: {args.real_cache}")
        source = np.load(source_path, mmap_mode="r")
        solution = np.load(solution_path, mmap_mode="r")
        if source.shape != (10000, 128, 128) or solution.shape != source.shape:
            raise ValueError(f"Unexpected real shape for {dataset}: {source.shape}, {solution.shape}")
        count = min(args.real_samples_per_type, source.shape[0])
        indices = np.arange(count, dtype=int)
        real_arrays[dataset] = (source, solution)
        all_metrics[dataset] = collect_metrics(
            source, solution, indices, args.real_batch_size, dataset
        )

    consistency = {
        "generated": physical_consistency_metrics(
            generated[:, 0], generated[:, 1], args.real_batch_size
        )
    }
    for dataset in REAL_DATASETS:
        source, solution = real_arrays[dataset]
        count = min(args.real_samples_per_type, source.shape[0])
        consistency[dataset] = physical_consistency_metrics(
            source[:count], solution[:count], args.real_batch_size
        )

    classifications = {
        field: classify_generated(all_metrics, field, seed=20260831 + index)
        for index, field in enumerate(FIELDS)
    }
    core_features = (
        "std",
        "gradient_energy",
        "correlation_length_cells",
        "high_frequency_ratio_r_gt_32",
    )
    classification_sensitivity = {
        field: clean_classification(
            classify_generated(
                all_metrics,
                field,
                seed=20260831 + index,
                feature_names=core_features,
            )
        )
        for index, field in enumerate(FIELDS)
    }
    distances = {field: distribution_distances(all_metrics, field) for field in FIELDS}
    metric_summary = {
        field: {
            dataset: {name: quantile_summary(all_metrics[dataset][field][name]) for name in METRIC_COLUMNS}
            for dataset in DATASETS
        }
        for field in FIELDS
    }

    write_per_sample_csv(args.output / "metrics_per_sample.csv", all_metrics)
    write_physical_consistency_csv(
        args.output / "physical_consistency_per_sample.csv", consistency
    )
    save_metrics_npz(args.output / "metrics_arrays.npz", all_metrics)
    plot_radial_psd(args.output, all_metrics, "source")
    plot_radial_psd(args.output, all_metrics, "solution")
    plot_metric_distributions(args.output, all_metrics, "source")
    plot_metric_distributions(args.output, all_metrics, "solution")
    plot_assignment_coverage(args.output, classifications)
    plot_distance_bars(args.output, distances)
    plot_physical_consistency(args.output, consistency)
    plot_sample_montage(args.output, generated, real_arrays)

    generation = json.loads((args.output / "generation_manifest.json").read_text(encoding="utf-8"))
    if not generation.get("checkpoint_slimmed_from"):
        import torch

        checkpoint_payload = torch.load(
            generation["checkpoint"], map_location="cpu", weights_only=False, mmap=True
        )
        generation["checkpoint_slimmed_from"] = checkpoint_payload.get("slimmed_from")
        del checkpoint_payload
    cache_manifest_path = args.real_cache.parent / "experiment_manifest.json"
    cache_manifest = (
        json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest_path.is_file()
        else None
    )
    summary = {
        "generation": generation,
        "real_cache": str(args.real_cache.resolve()),
        "real_cache_manifest": str(cache_manifest_path.resolve()) if cache_manifest else None,
        "real_source_files": cache_manifest.get("source_files") if cache_manifest else None,
        "real_samples_per_type": min(args.real_samples_per_type, 10000),
        "metric_definitions": {
            "radial_psd_source": "per-sample normalized energy in radial shells of mean-centered orthonormal DCT-II",
            "radial_psd_solution": "per-sample normalized energy in radial shells of orthonormal DST-I on the 126x126 interior",
            "high_frequency_ratio": "fraction of total spectral energy with radial shell r above the named cutoff",
            "correlation_length_cells": "first interpolated 1/e crossing of mean x/y unbiased linear autocorrelation, maximum lag 64",
            "gradient_energy": "0.5 * (mean(diff_x^2) + mean(diff_y^2))",
            "normalized_gradient_energy": "gradient_energy / spatial_variance",
        },
        "metric_summary": metric_summary,
        "classification": {field: clean_classification(classifications[field]) for field in FIELDS},
        "classification_core_metric_sensitivity": classification_sensitivity,
        "distribution_distances": distances,
        "physical_consistency_summary": {
            dataset: {
                name: quantile_summary(values) for name, values in consistency[dataset].items()
            }
            for dataset in DATASETS
        },
        "data_quality": {
            "generated_shape": list(generated.shape),
            "generated_finite": True,
            "real_shapes": {dataset: list(real_arrays[dataset][0].shape) for dataset in REAL_DATASETS},
            "reference_files": {
                dataset: {
                    field: str((args.real_cache / f"{dataset}_{FIELD_FILES[field]}.npy").resolve())
                    for field in FIELDS
                }
                for dataset in REAL_DATASETS
            },
        },
    }
    write_json(args.output / "summary.json", summary)
    (args.output / "summary.md").write_text(make_markdown_summary(summary), encoding="utf-8")
    print(f"ANALYSIS complete: {args.output}", flush=True)


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.real_cache = args.real_cache.resolve()
    generated_path = args.output / "generated_samples_physical.npy"
    if args.stage in {"all", "generate"}:
        generated_path = generate_samples(args)
    if args.stage in {"all", "analyze"}:
        if not generated_path.is_file():
            raise FileNotFoundError(
                f"Generated samples do not exist: {generated_path}; run --stage generate first"
            )
        analyze(args, generated_path)


if __name__ == "__main__":
    main()

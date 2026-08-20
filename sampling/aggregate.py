from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sampling.config import load_yaml_file


GROUP_DIMENSION_KEYS = [
    "pde",
    "task",
    "ablation_group",
    "guidance_components",
    "loss_state",
    "sampler_phase",
    "switch_ratio",
    "guidance_schedule",
    "obs_decay",
    "obs_decay_start_ratio",
    "polynomial_power",
    "cosine_mode",
    "sensor_mode",
    "num_obs",
    "num_sensor_columns",
    "noise_level",
    "zeta_obs_a",
    "zeta_obs_u",
    "zeta_pde",
    "zeta_ratio_name",
    "num_steps",
    "time_grid",
    "step_method",
    "clip_mode",
    "clip_threshold",
    "pde_residual_region",
    "residual_mode",
    "resolved_residual_mode",
    "gradient_target",
    "stochastic_guidance_time",
    "cfg_scale",
    "ns_operator_mode",
    "hermite_collocation_times",
    "hermite_num_collocation",
    "hermite_include_integral_residual",
    "hermite_integral_weight",
    "enforce_boundary_conditions",
    "enforce_initial_conditions",
    "boundary_condition_mode",
    "initial_condition_mode",
    "bc_weight",
    "ic_weight",
    "endpoint_bc_weight",
    "boundary_residual_normalization",
    "allow_unknown_boundary_conditions",
    "legacy_ignore_boundary",
]

GROUP_KEYS = ["ablation_family", "ablation_group_key", *GROUP_DIMENSION_KEYS]

SUMMARY_METRICS = [
    "rel_l2_a",
    "rel_l2_u",
    "obs_rel_l2_a",
    "obs_rel_l2_u",
    "clean_L_obs_a",
    "clean_L_obs_u",
    "L_pde",
    "pde_residual_norm",
    "interior_residual_norm",
    "boundary_residual_norm",
    "initial_residual_norm",
    "wall_clock_time",
]

CURVE_METRICS = [
    "rel_l2_a",
    "rel_l2_u",
    "obs_rel_l2_a",
    "obs_rel_l2_u",
    "clean_L_obs_a",
    "clean_L_obs_u",
    "L_pde",
    "pde_residual_norm",
    "interior_residual_norm",
    "boundary_residual_norm",
    "initial_residual_norm",
]

SAMPLE_METRICS = [
    "rel_l2_a",
    "rel_l2_u",
    "obs_rel_l2_a",
    "obs_rel_l2_u",
    "pde_residual_norm",
]


def aggregate_root(root: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(root)
    output_dir = Path(output_dir) if output_dir is not None else root
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, curve_rows, sample_rows, sample_weighted_rows = _collect_rows(root)
    raw_path = output_dir / "summary_all_raw.csv"
    grouped_path = output_dir / "summary_all_grouped.csv"
    run_grouped_path = output_dir / "summary_run_seed_grouped.csv"
    sample_raw_path = output_dir / "metrics_per_sample_all.csv"
    curves_path = output_dir / "curves_grouped.csv"

    _write_csv(raw_path, raw_rows)
    _write_csv(sample_raw_path, sample_rows)
    _write_csv(grouped_path, _aggregate_rows(sample_weighted_rows, SAMPLE_METRICS, GROUP_KEYS))
    _write_csv(run_grouped_path, _aggregate_rows(raw_rows, SUMMARY_METRICS, GROUP_KEYS))
    _write_csv(curves_path, _aggregate_rows(curve_rows, CURVE_METRICS, GROUP_KEYS + ["step"]))
    return {
        "raw": raw_path,
        "sample_raw": sample_raw_path,
        "grouped": grouped_path,
        "run_seed_grouped": run_grouped_path,
        "curves": curves_path,
    }


def _collect_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = []
    curve_rows = []
    sample_rows = []
    sample_weighted_rows = []
    for metrics_path in sorted(root.rglob("metrics_final.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        config = _read_config(run_dir / "resolved_config.yaml")
        row = _merge_config_metrics(config, metrics)
        row["run_dir"] = str(run_dir)
        row["metrics_path"] = str(metrics_path)
        raw_rows.append(row)
        if _is_successful_run(metrics):
            curve_rows.extend(_read_curve_rows(run_dir, config))
            run_sample_rows = _read_sample_rows(run_dir, config)
            if run_sample_rows:
                sample_rows.extend(run_sample_rows)
                sample_weighted_rows.extend(run_sample_rows)
            else:
                # Legacy fallback: one run-level mean counts as one observation.
                sample_weighted_rows.append(row)
    return raw_rows, curve_rows, sample_rows, sample_weighted_rows


def _read_sample_rows(run_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    path = run_dir / "metrics_per_sample.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for sample in csv.DictReader(handle):
            row = _merge_config_metrics(config, sample)
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def _merge_config_metrics(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    extra = config.get("extra") if isinstance(config.get("extra"), dict) else {}
    row = {}
    for key in GROUP_DIMENSION_KEYS:
        if key in config:
            row[key] = config[key]
        elif key in extra:
            row[key] = extra[key]
        elif key in metrics:
            row[key] = metrics[key]
        else:
            row[key] = ""
    row["ablation_name"] = config.get("ablation_name", extra.get("ablation_name", metrics.get("ablation_name", "")))
    row["ablation_family"] = _ablation_family(row)
    row["ablation_group_key"] = _ablation_group_key(row)
    for key in ("sample_seed", "mask_seed", "noise_seed", "offset", "test_index", "batch_size"):
        row[key] = config.get(key, extra.get(key, ""))
    for key, value in metrics.items():
        row[key] = value
    return row


def _read_curve_rows(run_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    path = run_dir / "metrics_step.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            metrics = json.loads(line)
            rows.append(_merge_config_metrics(config, metrics))
    return rows


def _aggregate_rows(rows: list[dict[str, Any]], metrics: list[str], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not _is_successful_run(row):
            continue
        grouped[tuple(normalize_group_value(row.get(key, "")) for key in group_keys)].append(row)

    out = []
    for key_values, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row = dict(zip(group_keys, key_values))
        for metric in metrics:
            values = [_to_float(item.get(metric)) for item in group_rows]
            values = [value for value in values if value is not None and math.isfinite(value)]
            row.update(_stats(metric, values))
        out.append(row)
    return out


def _is_successful_run(row: dict[str, Any]) -> bool:
    """Accept successful and legacy status-less artifacts, but not failed PDE evaluations."""
    return row.get("status", "ok") == "ok" and row.get("pde_residual_status") != "error"


def _stats(name: str, values: list[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {
            f"{name}_mean": "",
            f"{name}_std": "",
            f"{name}_n": 0,
            f"{name}_sem": "",
            f"{name}_ci95": "",
            f"{name}_median": "",
            f"{name}_min": "",
            f"{name}_max": "",
        }
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else 0.0
    return {
        f"{name}_mean": mean,
        f"{name}_std": std,
        f"{name}_n": n,
        f"{name}_sem": sem,
        f"{name}_ci95": 1.96 * sem,
        f"{name}_median": statistics.median(values),
        f"{name}_min": min(values),
        f"{name}_max": max(values),
    }


def _to_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ablation_family(row: dict[str, Any]) -> str:
    return _stable_join((row.get("pde", ""), row.get("task", ""), row.get("ablation_group", "")))


def _ablation_group_key(row: dict[str, Any]) -> str:
    return _stable_join(f"{key}={normalize_group_value(row.get(key, ''))}" for key in GROUP_DIMENSION_KEYS)


def _stable_join(values: Any) -> str:
    return "|".join(str(value) for value in values)


def normalize_group_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_yaml_file(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate FM4PDE ablation metrics.")
    parser.add_argument("root", nargs="?", default="outputs/ablations", help="Root directory containing run folders.")
    parser.add_argument("--output-dir", default=None, help="Directory for summary CSV files. Defaults to root.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_arg_parser().parse_args(argv)
    outputs = aggregate_root(args.root, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


if __name__ == "__main__":
    main()

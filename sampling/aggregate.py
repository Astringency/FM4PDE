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
    "test_type",
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
    "pde_guidance_start_ratio",
    "pde_guidance_ramp_ratio",
    "num_steps",
    "time_grid",
    "step_method",
    "deterministic_endpoint_mode",
    "deterministic_rollout_checkpoint",
    "deterministic_bt_mode",
    "deterministic_guidance_coeff",
    "deterministic_bt_max_scale",
    "deterministic_guidance_start_ratio",
    "deterministic_guidance_ramp_ratio",
    "deterministic_correction_max_rms",
    "deterministic_numerical_guard",
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
    "boundary_condition_mode",
    "bc_weight",
    "endpoint_bc_weight",
    "boundary_residual_normalization",
    "allow_unknown_boundary_conditions",
]

GROUP_KEYS = ["ablation_family", "ablation_group_key", *GROUP_DIMENSION_KEYS]

SUMMARY_METRICS = [
    "rel_l2_a",
    "rel_l2_u",
    "obs_rel_l2_a",
    "obs_rel_l2_u",
    "L_obs_a",
    "L_obs_u",
    "clean_L_obs_a",
    "clean_L_obs_u",
    "L_pde",
    "pde_residual_norm",
    "interior_residual_norm",
    "boundary_residual_norm",
    "wall_clock_time",
]

CURVE_METRICS = [
    "pde_guidance_factor",
    "zeta_pde_t",
    "guidance_update_scale",
    "deterministic_guidance_factor",
    "guidance_correction_norm",
    "guidance_correction_rms",
    "correction_clip_scale",
    "nonfinite_correction_samples",
    "endpoint_model_evaluations",
    "rel_l2_a",
    "rel_l2_u",
    "obs_rel_l2_a",
    "obs_rel_l2_u",
    "L_obs_a",
    "L_obs_u",
    "clean_L_obs_a",
    "clean_L_obs_u",
    "L_pde",
    "pde_residual_norm",
    "interior_residual_norm",
    "boundary_residual_norm",
]

SAMPLE_METRICS = [
    "rel_l2_a",
    "rel_l2_u",
    "obs_rel_l2_a",
    "obs_rel_l2_u",
    "pde_residual_norm",
]

# Stable, explicit reporting contract for every selected ablation run.  Keep
# coefficient and solution metrics separate: task-specific rankings may use a
# derived score, but that score must not replace the underlying measurements.
ABLATION_REPORT_COLUMNS = [
    "pde",
    "task",
    "test_type",
    "data_path",
    "ablation_group",
    "ablation_name",
    "sample_seed",
    "offset",
    "batch_size",
    "guidance_components",
    "loss_state",
    "sampler_phase",
    "switch_ratio",
    "time_grid",
    "num_steps",
    "step_method",
    "deterministic_endpoint_mode",
    "deterministic_bt_mode",
    "deterministic_guidance_coeff",
    "deterministic_bt_max_scale",
    "deterministic_guidance_start_ratio",
    "deterministic_guidance_ramp_ratio",
    "deterministic_correction_max_rms",
    "deterministic_numerical_guard",
    "sensor_mode",
    "num_obs",
    "noise_level",
    "residual_mode",
    "resolved_residual_mode",
    "rel_l2_a",
    "rel_l2_u",
    "L_obs_a",
    "L_obs_u",
    "L_pde",
    "run_dir",
    "metrics_path",
]


def collect_ablation_report_rows(root: str | Path) -> list[dict[str, Any]]:
    """Collect the latest analysis-ready row for every ablation run identity."""
    raw_rows, _, _, _ = _collect_rows(Path(root))
    latest_rows, _ = _select_latest_analysis_rows(raw_rows)
    return _ablation_report_rows(latest_rows)


def aggregate_root(root: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(root)
    output_dir = Path(output_dir) if output_dir is not None else root
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, curve_rows, sample_rows, sample_weighted_rows = _collect_rows(root)
    latest_rows, excluded_rows = _select_latest_analysis_rows(raw_rows)
    selected_run_dirs = {str(row.get("run_dir", "")) for row in latest_rows}
    latest_curve_rows = [row for row in curve_rows if str(row.get("run_dir", "")) in selected_run_dirs]
    latest_sample_rows = [row for row in sample_rows if str(row.get("run_dir", "")) in selected_run_dirs]
    latest_sample_weighted_rows = [
        row for row in sample_weighted_rows if str(row.get("run_dir", "")) in selected_run_dirs
    ]

    raw_path = output_dir / "summary_all_raw.csv"
    grouped_path = output_dir / "summary_all_grouped.csv"
    run_grouped_path = output_dir / "summary_run_seed_grouped.csv"
    sample_raw_path = output_dir / "metrics_per_sample_all.csv"
    curves_path = output_dir / "curves_grouped.csv"
    latest_path = output_dir / "summary_latest_unique.csv"
    latest_grouped_path = output_dir / "summary_latest_grouped.csv"
    latest_run_grouped_path = output_dir / "summary_latest_run_seed_grouped.csv"
    latest_sample_path = output_dir / "metrics_per_sample_latest_unique.csv"
    latest_curves_path = output_dir / "curves_latest_grouped.csv"
    excluded_path = output_dir / "summary_excluded_runs.csv"
    report_metrics_path = output_dir / "ablation_report_metrics.csv"

    _write_csv(raw_path, raw_rows)
    _write_csv(sample_raw_path, sample_rows)
    _write_csv(grouped_path, _aggregate_rows(sample_weighted_rows, SAMPLE_METRICS, GROUP_KEYS))
    _write_csv(run_grouped_path, _aggregate_rows(raw_rows, SUMMARY_METRICS, GROUP_KEYS))
    _write_csv(curves_path, _aggregate_rows(curve_rows, CURVE_METRICS, GROUP_KEYS + ["step"]))
    _write_csv(latest_path, latest_rows)
    _write_csv(latest_sample_path, latest_sample_rows)
    _write_csv(latest_grouped_path, _aggregate_rows(latest_sample_weighted_rows, SAMPLE_METRICS, GROUP_KEYS))
    _write_csv(latest_run_grouped_path, _aggregate_rows(latest_rows, SUMMARY_METRICS, GROUP_KEYS))
    _write_csv(latest_curves_path, _aggregate_rows(latest_curve_rows, CURVE_METRICS, GROUP_KEYS + ["step"]))
    _write_csv(excluded_path, excluded_rows)
    _write_csv(
        report_metrics_path,
        _ablation_report_rows(latest_rows),
        fieldnames=ABLATION_REPORT_COLUMNS,
    )
    return {
        "raw": raw_path,
        "sample_raw": sample_raw_path,
        "grouped": grouped_path,
        "run_seed_grouped": run_grouped_path,
        "curves": curves_path,
        "latest": latest_path,
        "latest_sample_raw": latest_sample_path,
        "latest_grouped": latest_grouped_path,
        "latest_run_seed_grouped": latest_run_grouped_path,
        "latest_curves": latest_curves_path,
        "excluded": excluded_path,
        "report_metrics": report_metrics_path,
    }


def _select_latest_analysis_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one current, analysis-ready run for every ablation identity.

    Historical raw outputs remain available in ``summary_all_raw.csv``.  The
    current snapshot is keyed by PDE, task, test distribution, and ablation name
    so repeated anchor configurations in different distributions are preserved.
    """
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_ablation_run_key(row)].append(row)

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        ready_rows = [row for row in group_rows if _is_analysis_ready_run(row)]
        current = max(ready_rows, key=_row_recency) if ready_rows else None
        if current is not None:
            selected.append(current)

        for row in group_rows:
            if row is current:
                continue
            if current is None:
                reason = "no_analysis_ready_run"
            elif _is_analysis_ready_run(row):
                reason = "superseded_by_newer_successful_run"
            else:
                reason = "not_analysis_ready"
            excluded.append(
                {
                    "pde": key[0],
                    "task": key[1],
                    "test_type": key[2],
                    "ablation_name": key[3],
                    "exclusion_reason": reason,
                    "excluded_run_dir": row.get("run_dir", ""),
                    "excluded_metrics_path": row.get("metrics_path", ""),
                    "selected_run_dir": current.get("run_dir", "") if current is not None else "",
                    "selected_metrics_path": current.get("metrics_path", "") if current is not None else "",
                    "status": row.get("status", ""),
                    "pde_residual_status": row.get("pde_residual_status", ""),
                    "clip_threshold": row.get("clip_threshold", ""),
                }
            )

    selected.sort(key=lambda row: _ablation_run_key(row))
    excluded.sort(
        key=lambda row: (
            str(row.get("pde", "")),
            str(row.get("task", "")),
            str(row.get("ablation_name", "")),
            str(row.get("excluded_run_dir", "")),
        )
    )
    return selected, excluded


def _ablation_run_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    name = str(row.get("ablation_name", ""))
    if not name:
        name = str(row.get("run_dir", row.get("metrics_path", "")))
    return (
        str(row.get("pde", "")),
        str(row.get("task", "")),
        str(row.get("test_type", "")),
        name,
    )


def _row_recency(row: dict[str, Any]) -> tuple[int, str]:
    metrics_path = Path(str(row.get("metrics_path", "")))
    try:
        modified_ns = metrics_path.stat().st_mtime_ns
    except OSError:
        modified_ns = 0
    return modified_ns, str(row.get("run_dir", metrics_path))


def _is_analysis_ready_run(row: dict[str, Any]) -> bool:
    if not _is_successful_run(row):
        return False
    for metric in ("rel_l2_a", "rel_l2_u", "L_obs_a", "L_obs_u", "L_pde"):
        value = _to_float(row.get(metric))
        if value is None or not math.isfinite(value):
            return False
    return True


def _ablation_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project selected runs onto the stakeholder-facing metric contract."""
    return [
        {column: row.get(column, "") for column in ABLATION_REPORT_COLUMNS}
        for row in sorted(rows, key=_ablation_run_key)
    ]


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
            curve_rows.extend(_read_curve_rows(run_dir, config, metrics))
            run_sample_rows = _read_sample_rows(run_dir, config, metrics)
            if run_sample_rows:
                sample_rows.extend(run_sample_rows)
                sample_weighted_rows.extend(run_sample_rows)
    return raw_rows, curve_rows, sample_rows, sample_weighted_rows


def _read_sample_rows(
    run_dir: Path,
    config: dict[str, Any],
    run_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    path = run_dir / "metrics_per_sample.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for sample in csv.DictReader(handle):
            row = _merge_config_metrics(config, sample)
            row["run_dir"] = str(run_dir)
            row.setdefault("status", run_metrics.get("status", ""))
            row.setdefault("pde_residual_status", run_metrics.get("pde_residual_status", ""))
            rows.append(row)
    return rows


def _merge_config_metrics(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row = {}
    for key in GROUP_DIMENSION_KEYS:
        if key in config:
            row[key] = config[key]
        elif key in metrics:
            row[key] = metrics[key]
        else:
            row[key] = ""
    row["ablation_name"] = config.get("ablation_name", metrics.get("ablation_name", ""))
    row["ablation_family"] = _ablation_family(row)
    row["ablation_group_key"] = _ablation_group_key(row)
    for key in (
        "data_path",
        "sample_seed",
        "mask_seed",
        "noise_seed",
        "offset",
        "batch_size",
    ):
        row[key] = config.get(key, "")
    for key, value in metrics.items():
        row[key] = value
    return row


def _read_curve_rows(
    run_dir: Path,
    config: dict[str, Any],
    run_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    path = run_dir / "metrics_step.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            metrics = json.loads(line)
            row = _merge_config_metrics(config, metrics)
            row["run_dir"] = str(run_dir)
            row.setdefault("status", run_metrics.get("status", ""))
            row.setdefault("pde_residual_status", run_metrics.get("pde_residual_status", ""))
            rows.append(row)
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
    return row.get("status") == "ok" and row.get("pde_residual_status") != "error"


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
            f"{name}_p90": "",
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
        f"{name}_p90": _percentile(values, 0.9),
        f"{name}_min": min(values),
        f"{name}_max": max(values),
    }


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile on the sorted observations."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = fieldnames or sorted({key for row in rows for key in row})
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

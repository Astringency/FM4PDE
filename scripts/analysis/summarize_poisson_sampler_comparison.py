#!/usr/bin/env python3
"""Validate and export the 36-run Poisson sampler comparison."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sampling.aggregate import collect_ablation_report_rows  # noqa: E402


GROUP = "poisson_sampler_comparison"
DISTRIBUTIONS = ("id", "smooth", "rough")
TASKS = ("both", "forward", "inverse")
SAMPLERS = ("stochastic", "deterministic", "hybrid_d2s", "hybrid_s2d")
OUTPUT_COLUMNS = (
    "DIST",
    "TASK",
    "SAMPLER",
    "SWITCH RATIO",
    "rel L2(a)",
    "rel L2(u)",
    "pde L",
    "obs L(a)",
    "obs L(u)",
    "max correction RMS",
    "min correction clip scale",
    "max nonfinite corrections",
    "Remark",
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _curve_diagnostics(run_dir: Path) -> tuple[float | None, float | None, float | None]:
    path = run_dir / "metrics_step.jsonl"
    correction_rms: list[float] = []
    clip_scales: list[float] = []
    nonfinite_counts: list[float] = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for key, target in (
                    ("guidance_correction_rms", correction_rms),
                    ("correction_clip_scale", clip_scales),
                    ("nonfinite_correction_samples", nonfinite_counts),
                ):
                    value = _number(record.get(key))
                    if value is not None:
                        target.append(value)
    return (
        max(correction_rms) if correction_rms else None,
        min(clip_scales) if clip_scales else None,
        max(nonfinite_counts) if nonfinite_counts else None,
    )


def collect_comparison(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    selected = [
        row
        for row in collect_ablation_report_rows(root)
        if row.get("ablation_group") == GROUP and row.get("pde") == "poisson"
    ]
    by_key = {
        (str(row.get("test_type")), str(row.get("task")), str(row.get("sampler_phase"))): row
        for row in selected
    }
    expected = set(itertools.product(DISTRIBUTIONS, TASKS, SAMPLERS))
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    if missing or extra or len(selected) != len(by_key):
        raise RuntimeError(
            "Incomplete or duplicated Poisson sampler comparison: "
            f"selected={len(selected)}, unique={len(by_key)}, missing={missing}, extra={extra}"
        )

    output: list[dict[str, Any]] = []
    for key in itertools.product(DISTRIBUTIONS, TASKS, SAMPLERS):
        row = by_key[key]
        if str(row.get("batch_size")) != "1" or str(row.get("offset")) != "0":
            raise RuntimeError(f"Expected batch_size=1 and offset=0 for {key}, got {row}")
        if str(row.get("sample_seed")) != "0":
            raise RuntimeError(f"Expected sample_seed=0 for {key}, got {row.get('sample_seed')}")
        metrics = [row.get(name) for name in ("rel_l2_a", "rel_l2_u", "L_pde", "L_obs_a", "L_obs_u")]
        if any(_number(value) is None for value in metrics):
            raise RuntimeError(f"Non-finite or missing final metric for {key}: {metrics}")
        run_dir = Path(str(row.get("run_dir")))
        max_rms, min_clip, max_nonfinite = _curve_diagnostics(run_dir)
        output.append(
            {
                "DIST": key[0].upper() if key[0] == "id" else key[0].title(),
                "TASK": key[1],
                "SAMPLER": key[2],
                "SWITCH RATIO": row.get("switch_ratio"),
                "rel L2(a)": metrics[0],
                "rel L2(u)": metrics[1],
                "pde L": metrics[2],
                "obs L(a)": metrics[3],
                "obs L(u)": metrics[4],
                "max correction RMS": max_rms,
                "min correction clip scale": min_clip,
                "max nonfinite corrections": max_nonfinite,
                "Remark": str(run_dir),
            }
        )
    return output


def write_comparison(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ablations/poisson_sampler_comparison"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.root / "poisson_sampler_comparison.csv"
    rows = collect_comparison(args.root)
    path = write_comparison(rows, output)
    print(f"Validated {len(rows)} Poisson sampler comparisons: {path}")
    return path


if __name__ == "__main__":
    main()

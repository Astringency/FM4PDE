#!/usr/bin/env python3
"""Build reproducible MAIN1000/tuned1 tables and a six-PDE tuning snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TEST_TYPES = ("id", "rough", "smooth")
VARIANTS = ("baseline", "tuned1")
TASK_ORDER = {"forward": 0, "inverse": 1, "both": 2}
PDE_ORDER = {
    "burger": 0,
    "poisson": 1,
    "helmholtz": 2,
    "darcy": 3,
    "nsnonbounded": 4,
}
SIX_PDES = (
    "advection_diffusion",
    "reaction_diffusion",
    "steady_heat_conduction",
    "heat",
    "shallow_water",
    "wave",
)
STANDARD_CANDIDATES = {
    "advection_diffusion": 12,
    "reaction_diffusion": 12,
    "steady_heat_conduction": 11,
    "heat": 12,
    "shallow_water": 12,
    "wave": 12,
}

RESULT_FIELDS = (
    "test_type",
    "variant",
    "pde",
    "task",
    "sensor_mode",
    "n",
    "rel_l2_a_mean",
    "rel_l2_a_median",
    "rel_l2_a_p90",
    "rel_l2_a_std",
    "rel_l2_u_mean",
    "rel_l2_u_median",
    "rel_l2_u_p90",
    "rel_l2_u_std",
    "pde_residual_norm_mean",
    "pde_residual_norm_median",
    "pde_residual_norm_p90",
    "zeta_obs_a",
    "zeta_obs_u",
    "zeta_pde",
    "clip_mode",
    "clip_threshold",
    "sampler_phase",
    "num_steps",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], field: str) -> float:
    return float(row[field])


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore")
        if selected:
            writer.writeheader()
            writer.writerows(rows)


def _primary_metric(task: str) -> str:
    if task == "forward":
        return "rel_l2_u"
    if task == "inverse":
        return "rel_l2_a"
    return "max_rel_l2_a_u"


def _primary_value(row: dict[str, Any]) -> float:
    a = _float(row, "rel_l2_a_mean")
    u = _float(row, "rel_l2_u_mean")
    if row["task"] == "forward":
        return u
    if row["task"] == "inverse":
        return a
    return max(a, u)


def _pct_improvement(baseline: float, tuned: float) -> float:
    return (baseline - tuned) / baseline if baseline else math.nan


def _intended_groups(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep full MAIN1000 groups and exclude small smooth pilot bundles."""
    full = [row for row in rows if int(float(row["rel_l2_a_n"])) >= 1000]
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in full:
        grouped[(row["pde"], row["task"], row["sensor_mode"])].append(row)
    selected = []
    for key, candidates in grouped.items():
        if len(candidates) != 1:
            raise RuntimeError(f"Ambiguous full MAIN1000 groups for {key}: {len(candidates)}")
        selected.append(candidates[0])
    return sorted(
        selected,
        key=lambda row: (
            PDE_ORDER.get(row["pde"], 99),
            TASK_ORDER.get(row["task"], 99),
            row["sensor_mode"],
        ),
    )


def collect_results(outputs_root: Path) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for test_type in TEST_TYPES:
        for variant in VARIANTS:
            suffix = "" if variant == "baseline" else "_tuned1"
            source = (
                outputs_root
                / f"MAIN1000_100_TEST_{test_type}{suffix}"
                / "summary_all_grouped.csv"
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            for row in _intended_groups(_read_csv(source)):
                result: dict[str, Any] = {
                    "test_type": test_type,
                    "variant": variant,
                    "pde": row["pde"],
                    "task": row["task"],
                    "sensor_mode": row["sensor_mode"],
                    "n": int(float(row["rel_l2_a_n"])),
                }
                for field in RESULT_FIELDS:
                    if field in result:
                        continue
                    source_field = f"{field}"
                    result[field] = row.get(source_field, "")
                collected.append(result)
    return sorted(
        collected,
        key=lambda row: (
            TEST_TYPES.index(str(row["test_type"])),
            PDE_ORDER.get(str(row["pde"]), 99),
            TASK_ORDER.get(str(row["task"]), 99),
            str(row["sensor_mode"]),
            VARIANTS.index(str(row["variant"])),
        ),
    )


def build_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {
        (
            row["test_type"],
            row["variant"],
            row["pde"],
            row["task"],
            row["sensor_mode"],
        ): row
        for row in results
    }
    comparisons: list[dict[str, Any]] = []
    for baseline in (row for row in results if row["variant"] == "baseline"):
        key = (
            baseline["test_type"],
            "tuned1",
            baseline["pde"],
            baseline["task"],
            baseline["sensor_mode"],
        )
        tuned = keyed.get(key)
        row: dict[str, Any] = {
            "test_type": baseline["test_type"],
            "pde": baseline["pde"],
            "task": baseline["task"],
            "sensor_mode": baseline["sensor_mode"],
            "primary_metric": _primary_metric(str(baseline["task"])),
            "baseline_n": baseline["n"],
            "baseline_rel_l2_a_mean": _float(baseline, "rel_l2_a_mean"),
            "baseline_rel_l2_a_p90": _float(baseline, "rel_l2_a_p90"),
            "baseline_rel_l2_u_mean": _float(baseline, "rel_l2_u_mean"),
            "baseline_rel_l2_u_p90": _float(baseline, "rel_l2_u_p90"),
            "baseline_pde_residual_mean": _float(baseline, "pde_residual_norm_mean"),
            "baseline_primary_mean": _primary_value(baseline),
            "tuned1_available": tuned is not None,
            "tuned1_n": tuned["n"] if tuned else None,
            "tuned1_rel_l2_a_mean": _float(tuned, "rel_l2_a_mean") if tuned else None,
            "tuned1_rel_l2_a_p90": _float(tuned, "rel_l2_a_p90") if tuned else None,
            "tuned1_rel_l2_u_mean": _float(tuned, "rel_l2_u_mean") if tuned else None,
            "tuned1_rel_l2_u_p90": _float(tuned, "rel_l2_u_p90") if tuned else None,
            "tuned1_pde_residual_mean": (
                _float(tuned, "pde_residual_norm_mean") if tuned else None
            ),
            "tuned1_primary_mean": _primary_value(tuned) if tuned else None,
        }
        if tuned:
            row.update(
                {
                    "primary_improvement": _pct_improvement(
                        row["baseline_primary_mean"], row["tuned1_primary_mean"]
                    ),
                    "rel_l2_a_improvement": _pct_improvement(
                        row["baseline_rel_l2_a_mean"], row["tuned1_rel_l2_a_mean"]
                    ),
                    "rel_l2_u_improvement": _pct_improvement(
                        row["baseline_rel_l2_u_mean"], row["tuned1_rel_l2_u_mean"]
                    ),
                    "pde_residual_improvement": _pct_improvement(
                        row["baseline_pde_residual_mean"],
                        row["tuned1_pde_residual_mean"],
                    ),
                }
            )
        else:
            row.update(
                {
                    "primary_improvement": None,
                    "rel_l2_a_improvement": None,
                    "rel_l2_u_improvement": None,
                    "pde_residual_improvement": None,
                }
            )
        comparisons.append(row)
    return comparisons


def tuning_progress(tuning_root: Path) -> dict[str, Any]:
    statuses = []
    for path in sorted((tuning_root / "runs" / "tune" / "standard").rglob("job.json")):
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    per_pde = []
    for pde in SIX_PDES:
        pde_rows = [row for row in statuses if row.get("pde") == pde]
        counts = Counter(str(row.get("status", "unknown")) for row in pde_rows)
        expected = STANDARD_CANDIDATES[pde] * 3 * 3 * 2
        per_pde.append(
            {
                "pde": pde,
                "expected_tune_jobs": expected,
                "recorded_jobs": len(pde_rows),
                "ok_jobs": counts.get("ok", 0),
                "failed_jobs": counts.get("failed", 0),
                "completion_rate": len(pde_rows) / expected,
            }
        )
    log_paths = sorted((tuning_root / "launcher_logs").glob("*.log"))
    latest_log_ns = max((path.stat().st_mtime_ns for path in log_paths), default=0)
    return {
        "snapshot_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": "standard",
        "phase": "tune",
        "expected_tune_jobs": sum(row["expected_tune_jobs"] for row in per_pde),
        "expected_holdout_jobs_at_most": 216,
        "recorded_jobs": len(statuses),
        "ok_jobs": sum(row["ok_jobs"] for row in per_pde),
        "failed_jobs": sum(row["failed_jobs"] for row in per_pde),
        "completion_rate": (
            len(statuses) / sum(row["expected_tune_jobs"] for row in per_pde)
        ),
        "latest_worker_log_mtime": (
            datetime.fromtimestamp(latest_log_ns / 1e9).astimezone().isoformat(timespec="seconds")
            if latest_log_ns
            else None
        ),
        "checkpoints_ready": sum(
            (tuning_root / "checkpoints" / f"{pde}_inference.pth").is_file()
            for pde in SIX_PDES
        ),
        "per_pde": per_pde,
    }


def summary_payload(
    results: list[dict[str, Any]], comparisons: list[dict[str, Any]], progress: dict[str, Any]
) -> dict[str, Any]:
    tuned = [row for row in comparisons if row["tuned1_available"]]
    improvements = [float(row["primary_improvement"]) for row in tuned]
    residual_regressions = [
        row for row in tuned if float(row["pde_residual_improvement"]) < 0
    ]
    return {
        "result_rows": len(results),
        "baseline_cells": sum(row["variant"] == "baseline" for row in results),
        "tuned1_cells": sum(row["variant"] == "tuned1" for row in results),
        "all_selected_groups_have_n_1000": all(int(row["n"]) == 1000 for row in results),
        "matched_tuned1_cells": len(tuned),
        "primary_improvement_wins": sum(value > 0 for value in improvements),
        "primary_improvement_median": statistics.median(improvements),
        "primary_improvement_min": min(improvements),
        "primary_improvement_max": max(improvements),
        "pde_residual_regression_cells": len(residual_regressions),
        "worst_pde_residual_improvement": min(
            (float(row["pde_residual_improvement"]) for row in residual_regressions),
            default=0.0,
        ),
        "smooth_inverse_baseline_parameter_mismatch": [
            "darcy/inverse",
            "helmholtz/inverse",
            "nsnonbounded/inverse",
            "poisson/inverse",
        ],
        "tuning_progress": progress,
    }


def build_artifact(
    results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    matched = [row for row in comparisons if row["tuned1_available"]]
    chart_rows = []
    for row in matched:
        chart_rows.append(
            {
                **row,
                "comparison_label": (
                    f"{row['test_type']} | {row['pde']} | {row['task']}"
                ),
            }
        )
    chart_rows.sort(key=lambda row: float(row["primary_improvement"]))

    progress = summary["tuning_progress"]
    win_count = summary["primary_improvement_wins"]
    matched_count = summary["matched_tuned1_cells"]
    median_improvement = summary["primary_improvement_median"]
    residual_regressions = summary["pde_residual_regression_cells"]
    progress_pct = progress["completion_rate"]
    technical_summary = (
        "## 技术摘要\n\n"
        f"六个结果目录中的目标配置均有 **1,000 个有效样本**。baseline 共 "
        f"{summary['baseline_cells']} 个 TEST/PDE/task/sensor 单元，tuned1 共 "
        f"{summary['tuned1_cells']} 个单元。可直接配对的 {matched_count} 个单元中，"
        f"tuned1 的任务主指标全部改善，中位改善为 **{median_improvement:.1%}**。\n\n"
        f"不过，{residual_regressions}/{matched_count} 个配对单元的 PDE residual 均值上升；"
        "因此 tuned1 应解释为端点重建误差更优，而不是物理残差全面更优。"
    )
    findings = (
        "## tuned1 在所有可比单元改善任务主指标\n\n"
        "下图使用任务相关主指标：forward 看 `rel_l2_u`，inverse 看 `rel_l2_a`，"
        "both 看两者均值中的较大值。横条表示相对 baseline 的均值改善率，正值越大越好。"
        "每个条目两侧均为 1,000 样本聚合。"
    )
    limitations = (
        "## 参数口径与稳健性限制\n\n"
        "smooth baseline 的四个 inverse 任务使用的 zeta 参数与 id/rough baseline 不一致："
        "`darcy`、`helmholtz`、`nsnonbounded`、`poisson` 均受影响。故 smooth 上的 tuned1 增益"
        "不能完全归因于 TEST 分布；它同时包含 baseline 参数版本差异。\n\n"
        f"另有 {residual_regressions} 个 tuned1 单元出现 PDE residual 回退，最差为 "
        f"{abs(summary['worst_pde_residual_improvement']):.1%} 上升。比较结果是描述性的均值对比，"
        "没有利用逐样本配对置信区间，因此不应表述为显著性结论。"
    )
    progress_text = (
        "## six_pde_sampling 正在筛选阶段\n\n"
        f"截至 {progress['snapshot_time']}，6/6 个瘦身 checkpoint 已准备；standard tune 已记录 "
        f"{progress['recorded_jobs']}/{progress['expected_tune_jobs']} 个作业"
        f"（{progress_pct:.1%}），其中 {progress['ok_jobs']} 成功、"
        f"{progress['failed_jobs']} 失败。当前仅前两个 PDE 的 `both` 任务有记录；"
        "尚未进入 winner 选择和 holdout。失败项来自 reaction_diffusion baseline 的 PDE loss "
        "NaN/Inf，参数候选本身已有成功作业，因此这是调参器需要识别的 baseline 不稳定性。"
    )

    source_results = {
        "id": "main1000_results",
        "label": "MAIN1000 and tuned1 combined aggregates",
        "path": "outputs/sampling_results_analysis/main1000_tuned1_summary/all_results.csv",
        "query": {
            "language": "python",
            "description": "Select full n=1000 configuration groups and compare task-aware means.",
            "sql": (
                "SELECT * FROM read_csv_auto("
                "'outputs/sampling_results_analysis/main1000_tuned1_summary/comparison.csv') "
                "ORDER BY test_type, pde, task, sensor_mode"
            ),
            "tables_used": [
                f"outputs/MAIN1000_100_TEST_{test_type}{suffix}/summary_all_grouped.csv"
                for test_type in TEST_TYPES
                for suffix in ("", "_tuned1")
            ],
            "filters": [
                "rel_l2_a_n >= 1000",
                "small smooth pilot bundles excluded",
                "matched on test_type, pde, task, and sensor_mode",
            ],
            "metric_definitions": [
                "forward primary = mean rel_l2_u",
                "inverse primary = mean rel_l2_a",
                "both primary = max(mean rel_l2_a, mean rel_l2_u)",
                "improvement = (baseline - tuned1) / baseline",
            ],
            "executed_at": generated_at,
        },
    }
    source_progress = {
        "id": "six_pde_progress",
        "label": "six_pde_sampling job status snapshot",
        "path": "outputs/sampling_results_analysis/main1000_tuned1_summary/six_pde_progress.json",
        "query": {
            "language": "python",
            "description": "Count standard-profile job.json records by status and PDE.",
            "sql": (
                "SELECT * FROM read_json_auto("
                "'outputs/sampling_results_analysis/main1000_tuned1_summary/six_pde_progress.json')"
            ),
            "tables_used": [
                "outputs/tuning/six_pde_sampling/runs/tune/standard/**/job.json"
            ],
            "executed_at": generated_at,
        },
    }

    comparison_columns = [
        {"field": "pde", "label": "PDE", "type": "text"},
        {"field": "task", "label": "Task", "type": "text"},
        {"field": "sensor_mode", "label": "Sensor", "type": "text"},
        {"field": "baseline_primary_mean", "label": "Base primary", "format": "number"},
        {"field": "tuned1_primary_mean", "label": "Tuned primary", "format": "number"},
        {
            "field": "primary_improvement",
            "label": "Primary improvement",
            "format": "percent",
            "movement": True,
        },
        {"field": "baseline_pde_residual_mean", "label": "Base PDE residual", "format": "number"},
        {"field": "tuned1_pde_residual_mean", "label": "Tuned PDE residual", "format": "number"},
    ]
    tables = []
    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": "# MAIN1000 与 tuned1 采样结果汇总"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": technical_summary,
            "sourceId": "main1000_results",
        },
        {
            "id": "tuned_findings",
            "type": "markdown",
            "body": findings,
            "sourceId": "main1000_results",
        },
        {"id": "tuned_chart_block", "type": "chart", "chartId": "tuned_improvement"},
    ]
    test_interpretations = {
        "id": "ID 下 6/6 个 tuned1 单元改善任务主指标，inverse 任务收益最明显。",
        "rough": "rough 下 tuned1 仍为 6/6 改善，但幅度整体小于 ID，说明 OOD 高频结构更难。",
        "smooth": "smooth 下 6/6 个 tuned1 单元改善；四个 inverse baseline 参数版本不一致，需结合限制说明解读。",
    }
    for test_type in TEST_TYPES:
        table_id = f"comparison_{test_type}"
        tables.append(
            {
                "id": table_id,
                "title": f"{test_type.upper()}：PDE × task 汇总",
                "subtitle": "每个目标配置 1,000 样本；空 tuned1 表示该组合未运行 tuned1。",
                "showDescription": True,
                "dataset": table_id,
                "defaultSort": {"field": "pde", "direction": "asc"},
                "density": "dense",
                "sourceId": "main1000_results",
                "layout": "full",
                "columns": comparison_columns,
            }
        )
        blocks.extend(
            [
                {
                    "id": f"section_{test_type}",
                    "type": "markdown",
                    "body": f"## {test_type.upper()} 结果\n\n{test_interpretations[test_type]}",
                    "sourceId": "main1000_results",
                },
                {"id": f"block_{table_id}", "type": "table", "tableId": table_id},
            ]
        )
    blocks.extend(
        [
            {
                "id": "scope_definitions",
                "type": "markdown",
                "body": (
                    "## 数据范围与指标定义\n\n"
                    "结果来自六个 `summary_all_grouped.csv`，只保留 n≥1,000 的完整目标配置组；"
                    "smooth 中 n=1–4 的 pilot 参数组被排除。表中 `rel L2(a)` 是系数场误差，"
                    "`rel L2(u)` 是解场误差，PDE residual 是生成样本的方程残差范数。"
                ),
                "sourceId": "main1000_results",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## 汇总方法\n\n"
                    "每个 TEST 先按完整配置组读取 1,000 样本聚合，再按 TEST/PDE/task/sensor 精确匹配 tuned1。"
                    "未重新平均 `summary_latest_grouped.csv`，因为该文件只代表最新 batch；"
                    "本报告使用覆盖全部 batch 的 `summary_all_grouped.csv`。"
                ),
                "sourceId": "main1000_results",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": limitations,
                "sourceId": "main1000_results",
            },
            {
                "id": "tuning_progress_section",
                "type": "markdown",
                "body": progress_text,
                "sourceId": "six_pde_progress",
            },
            {"id": "tuning_progress_table_block", "type": "table", "tableId": "tuning_progress"},
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 建议的下一步\n\n"
                    "- tuned1 正式结论应同时报告任务主误差与 PDE residual，尤其关注 rough inverse。\n"
                    "- 若要严格比较三种 TEST 的 tuned1 增益，应使用同一 baseline 参数版本重跑 smooth inverse。\n"
                    "- six_pde_sampling 可继续运行；成功 job 会断点跳过，reaction_diffusion baseline NaN/Inf 会作为失败记录保留。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## 后续问题\n\n"
                    "是否需要把 tuned1 的逐样本差值进一步做 paired bootstrap 置信区间？"
                    "这会判断当前均值改善是否由少量极端样本驱动。"
                ),
            },
        ]
    )
    tables.append(
        {
            "id": "tuning_progress",
            "title": "six_pde_sampling tune 作业进度",
            "subtitle": "standard profile；holdout 尚未开始。",
            "showDescription": True,
            "dataset": "tuning_progress",
            "defaultSort": {"field": "completion_rate", "direction": "desc"},
            "density": "spacious",
            "sourceId": "six_pde_progress",
            "layout": "full",
            "columns": [
                {"field": "pde", "label": "PDE", "type": "text"},
                {"field": "expected_tune_jobs", "label": "Expected", "format": "number"},
                {"field": "recorded_jobs", "label": "Recorded", "format": "number"},
                {"field": "ok_jobs", "label": "OK", "format": "number"},
                {"field": "failed_jobs", "label": "Failed", "format": "number"},
                {"field": "completion_rate", "label": "Completion", "format": "percent"},
            ],
        }
    )

    datasets: dict[str, list[dict[str, Any]]] = {
        "tuned_comparisons": chart_rows,
        "tuning_progress": progress["per_pde"],
    }
    for test_type in TEST_TYPES:
        datasets[f"comparison_{test_type}"] = [
            row for row in comparisons if row["test_type"] == test_type
        ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "MAIN1000 与 tuned1 采样结果汇总",
        "description": "Baseline/tuned1 results by TEST, PDE, and task, plus six-PDE tuning progress.",
        "generatedAt": generated_at,
        "blocks": blocks,
        "charts": [
            {
                "id": "tuned_improvement",
                "title": "tuned1 任务主指标相对改善",
                "subtitle": "18 个匹配单元；正值表示 tuned1 的均值误差低于 baseline。",
                "showDescription": True,
                "intent": "comparison",
                "question": "How much did tuned1 improve the task-primary mean error in each matched cell?",
                "rationale": "A sorted horizontal bar chart supports exact category comparison with long TEST/PDE/task labels.",
                "type": "horizontalBar",
                "dataset": "tuned_comparisons",
                "sourceId": "main1000_results",
                "encodings": {
                    "x": {"field": "comparison_label", "type": "nominal", "label": "TEST | PDE | task"},
                    "y": {
                        "field": "primary_improvement",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Primary improvement",
                    },
                    "tooltip": [
                        {"field": "baseline_primary_mean", "type": "quantitative", "label": "Baseline"},
                        {"field": "tuned1_primary_mean", "type": "quantitative", "label": "Tuned1"},
                        {"field": "primary_improvement", "type": "quantitative", "format": "percent", "label": "Improvement"},
                        {"field": "baseline_n", "type": "quantitative", "label": "Baseline n"},
                        {"field": "tuned1_n", "type": "quantitative", "label": "Tuned1 n"},
                    ],
                },
                "valueFormat": "percent",
                "layout": "full",
                "maxRows": 18,
                "palette": {"kind": "sequential", "name": "blue"},
                "labels": {"values": "all"},
                "settings": {"sort": "ascending", "showValues": True},
                "comparisonContext": {
                    "baseline": "MAIN1000 baseline",
                    "denominator": "baseline task-primary mean relative L2 error",
                    "grain": "TEST × PDE × task × sensor_mode",
                    "normalization": "(baseline - tuned1) / baseline",
                    "semanticFamily": "relative error improvement",
                    "unit": "fraction",
                },
            }
        ],
        "tables": tables,
        "sources": [source_results, source_progress],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": [source_results, source_progress],
        "package_info": {"artifact_name": "main1000_tuned1_summary"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sampling_results_analysis/main1000_tuned1_summary"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(args.outputs_root)
    comparisons = build_comparisons(results)
    progress = tuning_progress(args.outputs_root / "tuning" / "six_pde_sampling")
    summary = summary_payload(results, comparisons, progress)

    _write_csv(output_dir / "all_results.csv", results, list(RESULT_FIELDS))
    _write_csv(output_dir / "comparison.csv", comparisons)
    (output_dir / "six_pde_progress.json").write_text(
        json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

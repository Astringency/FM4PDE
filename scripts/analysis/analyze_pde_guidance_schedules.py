#!/usr/bin/env python3
"""Validate paired sampling runs and build source-backed report input."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tuning.compare_pde_guidance_schedules import SCHEDULES, summarize, write_csv, write_json

LABELS = {"late_hard": "① 后20步直接开启", "late_ramp": "② 后段线性渐增", "always_on": "③ 全程开启"}
PDE_LABELS = {"poisson": "Poisson", "helmholtz": "Helmholtz", "darcy": "Darcy", "nsnonbounded": "NS", "burger": "Burger"}

SUMMARY_SQL = """SELECT stage, pde, task, schedule, zeta_pde, COUNT(*) AS n,
    SUM(CASE WHEN status <> 'ok' OR primary_error IS NULL THEN 1 ELSE 0 END) AS failed,
    CASE WHEN COUNT(primary_error) = COUNT(*) THEN AVG(primary_error) END AS primary_error_mean,
    CASE WHEN COUNT(rel_l2_a) = COUNT(*) THEN AVG(rel_l2_a) END AS rel_l2_a_mean,
    CASE WHEN COUNT(rel_l2_u) = COUNT(*) THEN AVG(rel_l2_u) END AS rel_l2_u_mean,
    CASE WHEN COUNT(pde_residual_norm) = COUNT(*) THEN AVG(pde_residual_norm) END AS pde_residual_norm_mean
FROM observations
GROUP BY stage, pde, task, schedule, zeta_pde
ORDER BY stage, pde, task, schedule, zeta_pde"""


def independent_sql_check(root, rows, summaries):
    database = root / "analysis.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE IF EXISTS observations")
        connection.execute("CREATE TABLE observations (stage TEXT, pde TEXT, task TEXT, schedule TEXT, zeta_pde REAL, sample_id INTEGER, status TEXT, primary_error REAL, rel_l2_a REAL, rel_l2_u REAL, pde_residual_norm REAL)")
        fields = ["stage", "pde", "task", "schedule", "zeta_pde", "sample_id", "status", "primary_error", "rel_l2_a", "rel_l2_u", "pde_residual_norm"]
        values = []
        for row in rows:
            values.append([None if row[field] == "" else float(row[field]) if field in {"zeta_pde", "primary_error", "rel_l2_a", "rel_l2_u", "pde_residual_norm"} else int(row[field]) if field == "sample_id" else row[field] for field in fields])
        connection.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)
        connection.row_factory = sqlite3.Row
        checked = [dict(row) for row in connection.execute(SUMMARY_SQL)]
    key = lambda row: tuple(row[field] for field in ("stage", "pde", "task", "schedule", "zeta_pde"))
    reference = {key(row): row for row in summaries}
    for row in checked:
        expected = reference[key(row)]
        for field in ("primary_error_mean", "rel_l2_a_mean", "rel_l2_u_mean", "pde_residual_norm_mean"):
            if row[field] is None and expected[field] is None:
                continue
            if row[field] is None or expected[field] is None or not math.isclose(row[field], expected[field], rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f"Independent SQL aggregation mismatch: {key(row)}, {field}")
    (root / "summary_source.sql").write_text(SUMMARY_SQL + ";\n")
    write_csv(root / "sql_summary.csv", checked)
    return checked


def bootstrap_ratio(candidate, baseline):
    rng = random.Random(93017)
    n = len(candidate)
    ratios = []
    for _ in range(5000):
        indices = rng.choices(range(n), k=n)
        ratios.append(sum(candidate[i] for i in indices) / sum(baseline[i] for i in indices))
    ratios.sort()
    return ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]


def validate_pairing(root, protocol):
    import torch
    references = {}
    checked = 0
    schedule_counts = defaultdict(set)
    for receipt in sorted((root / "runs").glob("*/*/*/*/*/receipt.json")):
        record = json.loads(receipt.read_text())
        first = record["rows"][0]
        if first["status"] != "ok":
            continue
        result = torch.load(Path(first["run_dir"]) / "result.pt", map_location="cpu", weights_only=False)
        cfg = result["config"]
        if cfg.get("pde_guidance_reduction", "mse") != protocol.get("pde_guidance_reduction", "mse"):
            raise ValueError(f"Wrong PDE loss reduction: {receipt}")
        key = (first["stage"], first["pde"], first["task"], tuple(record["sample_ids"]))
        digest = hashlib.sha256()
        for value in [result["coef_ground_truth"], result["sol_ground_truth"], result["masks"]["coef"], result["masks"]["sol"]]:
            digest.update(value.contiguous().numpy().tobytes())
        digest.update(json.dumps({k: cfg[k] for k in ("sample_seed", "mask_seed", "batch_size", "time_grid", "num_steps", "sampler_phase", "zeta_obs_a", "zeta_obs_u", "clip_threshold")}, sort_keys=True).encode())
        fingerprint = digest.hexdigest()
        if key in references and references[key] != fingerprint:
            raise ValueError(f"Unpaired masks/targets/seeds/settings: {receipt}")
        references[key] = fingerprint
        with (Path(first["run_dir"]) / "curves.csv").open() as handle:
            curves = list(csv.DictReader(handle))
        if len(curves) != 100:
            raise ValueError(f"Incomplete step curve: {receipt}")
        active_steps = [int(r["step"]) for r in curves if float(r["zeta_pde_t"]) > 0]
        expected = list(range({"late_hard": 80, "late_ramp": 81, "always_on": 0}[first["schedule"]], 100))
        if active_steps != expected:
            raise ValueError(f"Unexpected PDE gate: {receipt}: {active_steps}")
        bound = float(protocol.get("clip_threshold", 1e10)) * math.sqrt(len(record["sample_ids"]))
        if any(math.isfinite(float(r["grad_norm_total"])) and float(r["grad_norm_total"]) > bound * 1.00001 for r in curves):
            raise ValueError(f"Gradient exceeds per-sample clipping bound: {receipt}")
        schedule_counts[first["schedule"]].add(len(active_steps))
        checked += 1
    return dict(checked_runs=checked, paired_batches=len(references),
                active_steps={k: sorted(v) for k, v in schedule_counts.items()},
                tune_holdout_disjoint=not set(protocol["tune_sample_ids"]) & set(protocol["holdout_sample_ids"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skip-pairing-audit", action="store_true")
    parser.add_argument("--unclipped-reference", type=Path)
    args = parser.parse_args()
    root = args.root
    protocol = json.loads((root / "protocol.json").read_text())
    summaries = summarize(root)
    selected = json.loads((root / "selected.json").read_text())
    complete = (root / "complete.json").exists()
    with (root / "per_sample.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    sql_rows = independent_sql_check(root, rows, summaries)
    sql_groups = len(sql_rows)
    sql_lookup = {tuple(row[field] for field in ("stage", "pde", "task", "schedule", "zeta_pde")): row for row in sql_rows}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["pde"], row["task"], row["schedule"], float(row["zeta_pde"]))].append(row)
    comparisons, cells = [], []
    for pde, task in protocol["cells"]:
        key = f"{pde}/{task}"
        base_zeta = protocol["configs"][key]["zeta_pde"]
        baseline = groups.get(("holdout", pde, task, "late_hard", base_zeta), [])
        expected_n = len(protocol["holdout_sample_ids"])
        incomplete = len(baseline) != expected_n or key not in selected or any(
            zeta is not None and len(groups.get(("holdout", pde, task, schedule, zeta), [])) != expected_n
            for schedule, zeta in selected.get(key, {}).items())
        if incomplete:
            if complete:
                raise ValueError(f"Experiment marked complete but holdout missing: {key}")
            continue
        baseline = sorted(baseline, key=lambda r: int(r["sample_id"]))
        baseline_valid = all(r["primary_error"] and r["status"] == "ok" for r in baseline)
        baseline_errors = [float(r["primary_error"]) for r in baseline] if baseline_valid else []
        baseline_mean = sql_lookup[("holdout", pde, task, "late_hard", base_zeta)]["primary_error_mean"] if baseline_valid else None
        local = []
        for schedule in SCHEDULES:
            zeta = selected.get(key, {}).get(schedule)
            group = groups.get(("holdout", pde, task, schedule, zeta), []) if zeta is not None else []
            group = sorted(group, key=lambda r: int(r["sample_id"]))
            if group and [r["sample_id"] for r in group] != [r["sample_id"] for r in baseline]:
                raise ValueError(f"Holdout sample pairing mismatch: {key}/{schedule}")
            finite = bool(group) and all(r["primary_error"] and r["status"] == "ok" for r in group)
            values = [float(r["primary_error"]) for r in group] if finite else []
            sql_mean = sql_lookup[("holdout", pde, task, schedule, zeta)]["primary_error_mean"] if values else None
            ratio = sql_mean / baseline_mean if values and baseline_errors else None
            low, high = bootstrap_ratio(values, baseline_errors) if ratio is not None else (None, None)
            out = dict(pde=pde, task=task, schedule=schedule, schedule_label=LABELS[schedule], zeta_pde=zeta,
                       n=len(group), status="ok" if finite else "failed_or_unavailable",
                       primary_error_mean=sql_mean,
                       primary_error_median=statistics.median(values) if values else None,
                       baseline_zeta_pde=base_zeta, baseline_error_mean=baseline_mean,
                       ratio_to_baseline=ratio, ratio_ci95_low=low, ratio_ci95_high=high,
                       paired_wins=sum(a < b for a, b in zip(values, baseline_errors)) if ratio is not None else None)
            for field in ("rel_l2_a", "rel_l2_u", "pde_residual_norm"):
                numbers = [float(r[field]) for r in group if r.get(field)]
                out[field + "_mean"] = statistics.mean(numbers) if group and len(numbers) == len(group) else None
            comparisons.append(out)
            local.append(out)
        valid = [r for r in local if r["primary_error_mean"] is not None]
        best = min(valid, key=lambda r: r["primary_error_mean"]) if valid else None
        cell = dict(pde=pde, task=task, winner=best["schedule"] if best else None,
                    winner_zeta=best["zeta_pde"] if best else None,
                    winner_mean=best["primary_error_mean"] if best else None,
                    winner_ratio=best["ratio_to_baseline"] if best else None)
        cell["practical_tie"] = len(valid) == 3 and max(r["primary_error_mean"] for r in valid) / min(r["primary_error_mean"] for r in valid) < 1.01
        for row in local:
            cell[row["schedule"] + "_zeta"] = row["zeta_pde"]
            cell[row["schedule"] + "_mean"] = row["primary_error_mean"]
            cell[row["schedule"] + "_pde_residual"] = row["pde_residual_norm_mean"]
        cells.append(cell)
    write_csv(root / "holdout_comparison.csv", comparisons)
    write_csv(root / "winners.csv", cells)
    audit = validate_pairing(root, protocol) if not args.skip_pairing_audit else {"pairing_audit": "not_run"}
    audit.update(complete=complete, completed_cells=len(cells), expected_cells=len(protocol["cells"]),
                 independently_recomputed_groups=sql_groups,
                 confidence="exploratory_small_sample; bootstrap intervals are not multiplicity adjusted")
    write_json(root / "validation.json", audit)
    clipping_diagnostic = []
    if args.unclipped_reference:
        with (args.unclipped_reference / "per_sample.csv").open() as handle:
            old_rows = list(csv.DictReader(handle))
        old_index = {(r["pde"], r["task"], r["schedule"], float(r["zeta_pde"]), r["sample_id"]): r
                     for r in old_rows if r["stage"] == "tune"}
        for row in rows:
            if row["stage"] != "tune" or row["schedule"] != "always_on" or float(row["zeta_pde"]) != 10:
                continue
            key = (row["pde"], row["task"], row["schedule"], float(row["zeta_pde"]), row["sample_id"])
            old = old_index.get(key)
            if old:
                clipping_diagnostic.append(dict(pde=row["pde"], task=row["task"], sample_id=int(row["sample_id"]),
                    unclipped_primary=float(old["primary_error"]) if old["primary_error"] else None,
                    clipped_primary=float(row["primary_error"]) if row["primary_error"] else None))
        write_csv(root / "clipping_diagnostic.csv", clipping_diagnostic)
    wins = Counter(r["winner"] for r in cells if r["winner"] and not r["practical_tie"])
    ties = sum(r["practical_tie"] for r in cells)
    print(json.dumps(dict(winner_counts=wins, validation=audit), indent=2, ensure_ascii=False))
    if not cells:
        return
    title = "PDE Guidance Schedule Comparison"
    now = datetime.now(timezone.utc).isoformat()
    source = {"id": "paired_experiment", "label": "PDE guidance paired experiment", "path": "holdout_comparison.csv",
              "query": {"language": "sql", "engine": "sqlite", "sql": SUMMARY_SQL, "description": "Independently aggregate per_sample.csv in analysis.sqlite and verify Python statistics. Tune selects zeta separately per schedule; holdout ratios use the late-hard original-zeta baseline with the same clipping. Paired bootstrap intervals are computed by the companion Python analysis.",
                        "tables_used": ["observations"], "executed_at": now,
                        "filters": {"test_type": "id", "num_steps": 100, "sampler_phase": "stochastic", "sensor_mode": "random", "num_obs": 500}}}
    blocks, tables, charts = [], [], []
    def prose(id, heading, body, sourced=False):
        block = dict(id=id, type="markdown", body=f"{heading}\n\n{body}")
        if sourced:
            block["sourceId"] = source["id"]
        blocks.append(block)
    prose("title", "# " + title, "")
    ranking = "；".join(f"{LABELS[s]}在 {wins[s]} 组中取得最低均值" for s in SCHEDULES)
    prose("summary", "## 小样本结果", f"已完成 {len(cells)}/{len(protocol['cells'])} 个 PDE/任务组合。其中 {ties} 组三种方式的复核均值差距不足 1%，描述为近似持平；其余组中，{ranking}。这里每种方式的 zeta_pde 均先在筛选集选择，然后在独立样本上比较。\n\n**这些结果用于筛选下一轮实验，不能据此认定一种方式普遍最优。** 需要同时看逐组误差、PDE 残差和失败情况。1% 是描述性阈值，不代表统计显著性。", True)
    prose("definitions", "## 比较范围与误差定义", f"采用当前 main 模型与任务配置，固定 100 步均匀随机采样、500 个随机观测点、无观测噪声。每组 {len(protocol['tune_sample_ids'])} 个筛选样本、{len(protocol['holdout_sample_ids'])} 个独立复核样本。\n\n主指标为逐样本相对 L2 重建误差：forward 看解 u，inverse 看系数 a，both 对 a、u 的相对误差等权平均；Burger 看完整解场。所有误差越低越好。相对同裁剪基线的比值为 1 表示持平，小于 1 表示改善。", True)
    if clipping_diagnostic:
        improved = [r for r in clipping_diagnostic if r["unclipped_primary"] is not None and r["clipped_primary"] is not None and r["unclipped_primary"] > 100 * r["clipped_primary"]]
        prose("clipping", "## 有效裁剪控制了已观察到的失稳", "Poisson both 的全程开启、zeta=10 使用相同目标、掩码与种子复测，仅将裁剪阈值从 1e10 改为 50。" +
              " ".join(f"样本 {r['sample_id']} 的主误差从 {r['unclipped_primary']:.4g} 降至 {r['clipped_primary']:.4g}。" for r in improved) +
              "\n\n这验证了限制梯度可以控制已观察到的失稳；仍需在统一裁剪后比较三种启用时机。该诊断的逐样本结果保存在 clipping_diagnostic.csv。")
    for task in ["both", "forward", "inverse"]:
        subset = [r for r in cells if r["task"] == task]
        if not subset:
            continue
        statements = []
        for row in subset:
            if row["winner"]:
                statements.append(f"- **{PDE_LABELS[row['pde']]}**：" + ("三种方式近似持平（均值差距不足 1%）；最低均值为" if row["practical_tie"] else "最低均值为") + f"{LABELS[row['winner']]}，zeta={row['winner_zeta']:g}，复核均值 {row['winner_mean']:.5g}" + (f"，为同裁剪基线的 {row['winner_ratio']:.3f} 倍。" if row['winner_ratio'] is not None else "。"))
        prose(f"findings_{task}", f"## {task}：逐方程比较", "\n".join(statements) + "\n\n下表列出各方式在筛选集选出的权重与独立复核误差。不同方式可以选择不同权重；固定同一权重的完整筛选结果保存在 summary.csv。", True)
        columns = [{"field": "pde", "label": "PDE", "type": "text"}]
        for schedule in SCHEDULES:
            columns += [{"field": schedule + "_zeta", "label": LABELS[schedule] + " zeta", "format": "number"},
                        {"field": schedule + "_mean", "label": LABELS[schedule] + " 误差", "format": "number"}]
        tables.append(dict(id=f"table_{task}", title=f"{task} 独立复核结果", dataset=f"cells_{task}", sourceId=source["id"],
                           defaultSort={"field": "pde", "direction": "asc"}, columns=columns))
        blocks.append(dict(id=f"table_block_{task}", type="table", tableId=f"table_{task}"))
        prose(f"physics_note_{task}", "### PDE 残差作为辅助指标", "下表比较同一批复核预测的 PDE 残差 RMS，越低越好；仅比较同一方程内的三种方式，不同方程的残差尺度不可直接比较。这里的权重按重建误差选出，因此不一定是各方式的残差最优权重。", True)
        tables.append(dict(id=f"physics_{task}", title=f"{task} PDE 残差 RMS", dataset=f"cells_{task}", sourceId=source["id"],
                           defaultSort={"field": "pde", "direction": "asc"},
                           columns=[{"field": "pde", "label": "PDE", "type": "text"}] +
                           [{"field": s + "_pde_residual", "label": LABELS[s], "format": "number"} for s in SCHEDULES]))
        blocks.append(dict(id=f"physics_block_{task}", type="table", tableId=f"physics_{task}"))
        # A shared relative scale makes task-specific PDE error levels comparable.
        chart_rows = [dict(r, pde_label=PDE_LABELS[r["pde"]]) for r in comparisons if r["task"] == task and r["ratio_to_baseline"] is not None and r["ratio_to_baseline"] < 5]
        if len({r["pde"] for r in chart_rows}) >= 4:
            charts.append(dict(id=f"chart_{task}", title=f"{task} 重建误差相对同裁剪基线", dataset=f"chart_{task}", type="bar", sourceId=source["id"],
                               settings={"groupMode": "grouped"}, palette={"kind": "categorical"},
                               referenceLines=[{"axis": "y", "value": 1, "label": "同裁剪基线"}],
                               encodings={"x": {"field": "pde_label", "type": "nominal", "label": "PDE"},
                                          "y": {"field": "ratio_to_baseline", "type": "quantitative", "label": "误差 / 同裁剪基线"},
                                          "color": {"field": "schedule_label", "type": "nominal", "label": "PDE 启用方式"},
                                          "tooltip": [{"field": "zeta_pde", "type": "quantitative", "label": "zeta_pde"}]}))
            prose(f"chart_note_{task}", "### 相同基线下的误差比值", "图中小于 1 表示优于同裁剪基线。失败或误差超过基线 5 倍的方案不进入此图，完整状态与数值保留在结果表和 CSV 中；这类方案不应视为缺少实验。", True)
            blocks.append(dict(id=f"chart_block_{task}", type="chart", chartId=f"chart_{task}"))
    prose("method", "## 配对实验与门控验证", "① start=0.8、ramp=0：第 81–100 步直接使用完整 PDE 权重。② start=0.8、ramp=0.1：第 81 步权重仍为零，第 82 步起逐渐增加，t=0.9 时达到完整权重。③ start=0、ramp=0：全部 100 步使用 PDE Guide。\n\n每种方式测试当前权重及 0.1、1、10（重复值去重）。所有候选使用 global_norm=50 的逐样本总梯度裁剪；基线为相同裁剪下的方案①加原 main PDE 权重。除 PDE 时机和权重外，同组候选保持模型、观测权重、梯度裁剪、目标样本、随机种子、批量和观测掩码一致。真实目标来自历史测试运行保存的 ground-truth 张量；历史预测未用于本轮指标。样本 ID 在查看结果前由固定随机种子选定。", True)
    if (root / "remote_cells.json").exists():
        remote_cells = json.loads((root / "remote_cells.json").read_text())
        prose("execution_hosts", "### 每个任务组合保持执行环境一致", "本地计算与 server197 分担完整的 PDE/任务组合。同一组合的所有候选在同一环境上计算，避免在一个配对比较中混入不同 GPU 或 PyTorch 版本。服务器完成的组合为：" + "、".join(remote_cells) + "。执行来源保存在 remote_cells.json 中。")
    prose("limits", "## 不确定性与稳健性", "当前只覆盖 ID 测试集、随机观测和 stochastic 采样。NS 使用当前 main 的 endpoint_secant 残差，它是端点近似一致性指标，不能作为完整时间轨迹满足 NS 方程的证明。筛选集与复核集互不重叠，但样本量较小；复核集内的三种方式排名仍有选择偏差。holdout_comparison.csv 给出相对基线的配对 bootstrap 95% 区间和逐样本胜出数；区间未做多重比较校正。\n\nPDE 残差降低不等于重建更准确，both 的平均误差也可能掩盖 a、u 之间的取舍。增大 PDE 权重可能引起数值发散或通过全局梯度裁剪改变观测更新，应结合分量误差、失败率与曲线解释。未将失败或非有限值作为零误差参与排名。", True)
    prose("next", "## 下一轮应验证什么", "保留各 PDE/任务独立的候选结果。优先扩大复核样本量，并在 smooth、rough 分布上重复胜出候选与同裁剪基线的配对比较，再决定是否修改默认配置。若不同启用方式差距落在小样本波动范围内，暂不更换 main。")
    datasets = {"holdout_comparison": comparisons}
    for task in ("both", "forward", "inverse"):
        datasets[f"cells_{task}"] = [r for r in cells if r["task"] == task]
        datasets[f"chart_{task}"] = [dict(r, pde_label=PDE_LABELS[r["pde"]]) for r in comparisons if r["task"] == task and r["ratio_to_baseline"] is not None and r["ratio_to_baseline"] < 5]
    sources = [source]
    if clipping_diagnostic:
        sources.append({"id": "clipping_diagnostic", "label": "Paired gradient clipping check", "path": "clipping_diagnostic.csv"})
        next(b for b in blocks if b["id"] == "clipping")["sourceId"] = "clipping_diagnostic"
    artifact = dict(surface="report", manifest=dict(version=1, surface="report", title=title, generatedAt=now,
                    blocks=blocks, cards=[], charts=charts, tables=tables, sources=sources),
                    snapshot=dict(version=1, generatedAt=now, status="ready" if complete else "partial", datasets=datasets,
                                  accessIssues=[] if complete else [{"id": "running", "message": "实验尚未全部完成，当前结果为部分结果。"}]), sources=sources)
    write_json(root / "artifact.json", artifact)
    write_json(root / "report_notes.json", dict(audience="technical", delivery="html", chart_contract={
        "family": "grouped bar", "question": "How do the three tuned schedules compare to the clipped late-hard baseline for each PDE and task?",
        "grain": "one held-out mean per PDE/task/schedule", "palette": "shared categorical blue, gold, orange; labels distinguish schedules",
        "repeated_family_reason": "Each task asks the same paired relative-error comparison; raw task metrics differ.",
        "omissions": "Small-sample fixed-zeta screening uses exact tables/CSV; no misleading three-point trend."},
        structure="title, technical summary, definitions, per-task findings, methodology, robustness, next steps; further questions integrated into next steps"))


if __name__ == "__main__":
    main()

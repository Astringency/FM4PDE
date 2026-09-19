"""Summarize completed optimizer screens without hiding pending work."""
import argparse
import csv
import json
from pathlib import Path

from experiments.optimizer_diagnostics.study import PDES, write


def report(root):
    rows=[]
    summary=[]
    pending=[]
    for pde in PDES:
        run=root/"runs"/pde
        if not (run/"complete.json").exists() or not (run/"audit.json").exists():
            pending.append(pde)
            continue
        complete=json.loads((run/"complete.json").read_text())
        protocol=json.loads((run/"protocol.json").read_text())
        for config in protocol["configs"]:
            result=json.loads((run/config["name"]/"result.json").read_text())
            probe=json.loads((run/config["name"]/"layers_step_0001.json").read_text())["summary"]
            delta=result["paired_development"]
            rows.append(dict(pde=pde,candidate=config["name"],lr=config["lr"],beta1=config["betas"][0],
                beta2=config["betas"][1],updates=result["steps"],seconds=result["seconds"],
                validation=result["development"]["mean"],development_change_pct=delta["relative_change_pct"],
                paired_ci_low=delta["ci95"][0],paired_ci_high=delta["ci95"][1],**probe))
        c=complete["confirmations"]
        summary.append(dict(pde=pde,selected=c["selected_resume"]["config"],
            selected_vs_original=c["selected_resume"]["paired_original"],
            lr_control_vs_original=c["lr_control"]["paired_original"],
            selected_vs_lr_control=c["selected_vs_lr_control"],
            gpu=protocol["gpu"],host=protocol["host"],microbatch=protocol["microbatch"],
            peak_gib=complete["peak_allocated_bytes"]/2**30))
    out=root/"report"
    out.mkdir(exist_ok=True)
    write(out/"summary.json",dict(status="complete" if not pending else "partial",pending=pending,models=summary))
    if rows:
        with (out/"candidates.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines=["# Adam 参数与梯度诊断（2026-09-19）","",
        "状态："+("五类筛选、checkpoint 与参数核验完成。" if not pending else "部分完成，等待："+", ".join(pending)),"",
        "每个候选从同一原模型与 Adam 一、二阶矩、step 恢复；使用相同样本顺序、噪声、时间和 dropout 随机流。FP32/TF32，有效 batch 64。",
        "每类筛选使用首个原训练分片内的 4096 个训练输入、256 个开发输入、512 个确认输入，保持原 50000 样本的拆分身份。六月 checkpoint 的更早预训练可能见过验证池；NS 九月模型有原始固定拆分。",
        "开发集用于选择候选，确认集只比较冻结选择与学习率对照。逐输入先平均四个时间区间/随机重复，再计算配对正态近似区间；未进行多重比较校正。128 次更新不足以证明收敛或模型能力上限。","",
        "| PDE | 确认集选择 | 相比原模型 MSE | 相比仅改 LR 对照 MSE | microbatch | 峰值 GiB |",
        "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['pde']} | {r['selected']['name']} | {r['selected_vs_original']['relative_change_pct']:+.3f}% | {r['selected_vs_lr_control']['relative_change_pct']:+.3f}% | {r['microbatch']} | {r['peak_gib']:.2f} |")
    lines.extend(["","## 原学习率下的实际更新","",
        "| PDE | 原 LR | 梯度范数 | 参数相对更新 | 未改变参数元素比例 | Adam v / 当前 g² |",
        "|---|---:|---:|---:|---:|---:|"])
    for r in rows:
        if r["candidate"]=="source_lr":
            lines.append(f"| {r['pde']} | {r['lr']:.3g} | {r['grad_norm']:.4g} | {r['relative_update']:.4g} | {100*r['update_zero_fraction']:.3f}% | {r['moment_to_current_g2']:.3g} |")
    lines.extend(["","原 LR 对照为末次 checkpoint 记录的 LR 保持常数。梯度非零不能证明还可有效降低期望损失；随机时间、噪声与 dropout 也会产生非零梯度。v/g² 是一批的聚合诊断，不能单独据此断言历史二阶矩异常。",
        "降低 beta2 后继承的旧二阶矩仍逐步衰减；beta2=0.99 在 128 次更新后保留约 27.6% 的初始二阶矩贡献。不能把这组短筛选当作已完成适应。",
        "本报告的主要终点是匹配的 FM 验证损失，未据此声称条件采样或 OOD 能力提升。",
        "","可复核文件：`runs/<pde>/protocol.json`、`result.json`（各候选目录）、`layers_step_*.json`、`frozen_gradient_probe.json`、`complete.json`、`audit.json`；`selected_resume.pth` 与 `lr_control.pth` 保留完整优化器状态。"])
    (out/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(pending=pending,completed=len(summary))),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    report(p.parse_args().root)

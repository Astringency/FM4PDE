"""Build the final readout from verified optimizer and physical sampling evidence."""
import argparse
import csv
import json
from pathlib import Path
from experiments.optimizer_diagnostics.study import sha,write


def main(root):
    fm=json.loads((root/'report/summary.json').read_text())
    sample=json.loads((root/'hard_sampling/finalization.json').read_text())
    four=json.loads((root/'hard_sampling/darcy/four_seed_summary.json').read_text())
    assert fm['status']=='complete' and sample['state']=='complete' and four['status']=='verified'
    assert json.loads((root/'continuation/nsnonbounded/audit.json').read_text())['status']=='verified'
    with (root/'report/candidates.csv').open() as stream: rows=list(csv.DictReader(stream))
    lines=['# 续训、梯度与五类 PDE 采样结果','',
        '本轮完成了五类 PDE 的优化器筛选，并在训练期间并行检查每类固定的 16 个困难样本。NS 的两组续训进一步达到累计 512 次更新。结果不支持统一采用较大学习率或降低 beta2 来获得稳定的大幅采样改善，也不能据此断言已达到模型能力上限。','',
        '## 采样结果','',
        '下表均为解场物理空间相对 L2 误差；正改善率表示误差下降。每类使用两个种子（0、1），先按输入平均，再对 16 个输入做配对 bootstrap。选样来自原模型历史误差；收益分母为相同 batch、观测、采样设置和全部噪声重跑的原模型。','',
        '| PDE | 候选 | 原误差 | 新误差 | 相对改善 | 改善例数 | ≥10% / ≥20% | 差值 95% 区间（百分点） |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for model in sample['models']:
        for variant in model['variants']:
            d=variant['fields']['u'];ci=d['paired_difference_ci95']
            lines.append(f"| {model['pde']} | {variant['variant']} | {100*d['original_mean']:.3f}% | {100*d['candidate_mean']:.3f}% | {d['aggregate_improvement_pct']:+.3f}% | {d['improved']}/16 | {d['improved_ge10']} / {d['improved_ge20']} | [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] |")
    lines.extend(['','## Darcy 的四种子复核','',
        '最初两个种子方向不一致，因此在相同 16 例上追加了独立 seeds 2、3。以下保留全部四个种子，不仅报告最有利的一次。这个追加实验由初步结果触发，仍是困难样本上的探索性复核。','',
        '| 候选 | 原误差 | 新误差 | 相对改善 | 改善 / ≥10% / ≥20% | 四个种子的相对改善 | 95% 区间（百分点） |',
        '|---|---:|---:|---:|---:|---|---|'])
    for v in four['variants']:
        d=v['fields']['u'];ci=d['paired_difference_ci95'];seeds=' / '.join(f'{x:+.2f}%' for x in d['seed_improvement_pct'])
        lines.append(f"| {v['variant']} | {100*d['original_mean']:.3f}% | {100*d['candidate_mean']:.3f}% | {d['aggregate_improvement_pct']:+.3f}% | {d['improved']} / {d['improved_ge10']} / {d['improved_ge20']} | {seeds} | [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] |")
    lines.extend(['','## Betas 实际如何调整','',
        '原 betas 为 (0.9, 0.999)。独立对照分别只降低 beta1 到 0.8，或只降低 beta2 到 0.99；两者都使用 LR=1e-5，并保留原 Adam 一阶矩、二阶矩和 step。还比较了原 LR、1e-5、3e-5。每个候选 128 次更新，样本顺序、噪声、时间、dropout 和有效 batch=64 一致。',
        '五类开发集选中的干预均为 (0.9, 0.99)，但“优于其他干预”不代表优于原模型。降低 beta2 后，原二阶矩贡献在 128 步仍约为 27.6%，512 步约为 0.58%；NS 的更长对照依然没有带来稳定的解场采样收益。','',
        '## 梯度与真实更新','',
        '下表为原学习率下实际测到的一次有效 batch 更新。梯度存在；Helmholtz、Darcy、Burgers 的原 LR 约为 1e-8，约四分之一参数元素在该步的 FP32 权重上没有发生可表示的变化。非零梯度也可能主要来自时间、噪声和 dropout，并不保证更大 LR 会改善期望误差。','',
        '| PDE | 原 LR | 梯度范数 | 参数相对更新 | 未改变参数元素比例 | 缺失梯度张量数 |',
        '|---|---:|---:|---:|---:|---:|'])
    for r in rows:
        if r['candidate']=='source_lr':
            lines.append(f"| {r['pde']} | {float(r['lr']):.3g} | {float(r['grad_norm']):.4g} | {float(r['relative_update']):.4g} | {100*float(r['update_zero_fraction']):.2f}% | {r['missing_grad_tensors']} |")
    lines.extend(['','## 结果适用范围与复核','',
        '128 步筛选使用原训练拆分中的 4096 个输入；开发集 256、确认集 512，保留原拆分身份。NS 后续从完整 45000 个训练输入池采样，两个分支均新增 384 次更新，原归一化统计保持一致。六月 checkpoint 更早的预训练可能见过验证池，不能把 FM 确认结果视为完全独立的总体泛化结论。',
        '原 NS 训练使用 BF16，本轮各对照统一 FP32/TF32；相对原模型的变化可能包含精度差异，而同 LR 的 beta 对照精度相同。保存参数还确认原训练全局有效 batch=64、未启用梯度裁剪及偏斜时间采样。',
        '80 个固定困难样本不能代表所有测试样本或 OOD；区间未作多重比较校正。历史 NS 采样存在明显 batch 数值敏感性，因此历史误差只用于选样，所有收益都相对当前相同 batch 的原模型计算。',
        '独立审计从保存预测重新计算 CPU float64 误差，逐例核对输入、mask、全部 101 次噪声、模型与预测 SHA256；训练审计检查实际优化器步数、betas/LR、有限权重和矩状态。每类全部 16 例的真值、预测、绝对误差均保留，图中每例使用一致色标。',
        '', '详细 FM 筛选见 `README.md`；采样汇总见 `SAMPLING.md`；逐例数据和图位于主结果目录的 `hard_sampling/`，Darcy 四种子结果见 `hard_sampling/darcy/FOUR_SEEDS.md`。',
        '', '主结果目录（server197）：`'+str(root)+'`。保存的续训 checkpoint 包含完整优化器状态；原模型、筛选指标和采样失败的对照预测均保留，便于后续选择训练方案。'])
    (root/'report/CONCLUSION.md').write_text('\n'.join(lines)+'\n')
    write(root/'report/conclusion_evidence.json',dict(verified=True,sources={str(p.relative_to(root)):sha(p) for p in [root/'report/summary.json',root/'hard_sampling/finalization.json',root/'hard_sampling/darcy/four_seed_summary.json',root/'continuation/nsnonbounded/audit.json',root/'report/candidates.csv']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)

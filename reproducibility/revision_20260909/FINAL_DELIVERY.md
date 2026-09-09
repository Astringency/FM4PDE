# 最终存储位置与复现说明

本次结果及复现材料集中存储于 **server197（192.168.191.197）** 的
`/research_data/users/zhangxifeng/C01Python/FM4PDE/`。原始来源目录保留。

| 内容 | FM4PDE 下的目录 |
| --- | --- |
| 主实验、Burgers 实验及采样时间 | `outputs/main/` |
| 消融、条件样本平均及 NS 补充实验 | `outputs/ablations/` |
| 源版本、环境、复现步骤及校验记录 | `reproducibility/revision_20260909/` |
| 最终论文、图表数据、补充材料及审稿文件 | `reproducibility/revision_20260909/paper_assets_final_20260909/` |

新增归档放在各结果目录的 `revision_20260909/` 下，原有结果保留原位置。
[最终归档清单](ARCHIVE_COMPLETED_67.md)登记了 67 项材料、194,202 个文件、
94,486,625,572 字节，全部通过独立文件校验。已原地核对的历史结果不重复计入该总量。
[最终服务器交付记录](verification_runs/final_delivery_20260909/verification_complete.json)
记录实际安装的 Git 版本、文件比对结果及源码恢复检查。

## 复现入口

先阅读[实验复现步骤](STUDY_RECIPES.md)和[复现范围](REPRODUCTION_COVERAGE.md)。
各实验分别列出实际源码、输入、观测掩码、模型、配置、种子、批次和运行环境。
已有代码仍按 `data/`、`models/`、`training/`、`sampling/`、`plot/`、`scripts/`
及 `configs/` 组织；原在仓库外的必要程序见 `legacy_scripts/`，比较方法的
指定源码及历史保存在 `source_snapshots/` 中。

在 server197 上可使用现有解释器
`/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python`。以下命令中的
`python` 指该解释器或已经激活的对应环境。源码归档可直接检查：

```bash
python reproducibility/revision_20260909/source_snapshots.py verify
```

60 个指定源码版本、70 个源码及历史归档均已核对；独立恢复不依赖原仓库。
如需重建论文，在具有记录中 TeX 依赖的环境中使用新的输出目录：

```bash
python reproducibility/revision_20260909/build_reproduction_paper.py \
  --output reproducibility/revision_20260909/verification_runs/new_paper_build
```

该命令恢复文稿、编译至交叉引用稳定，并核对四份 PDF 的完整文本。
实际恢复检查得到 190、190、54、13 页，全文与封存版本一致。
较早的 `paper_assets/` 另行保留；当前使用 `paper_assets_final_20260909/`。

## 已完成的结果复算

- [原有消融与三次预测平均](RELOCATION_SERVER197.md)：完整结果从集中归档重算，表格数值和排名保持一致。
- [新 NS 主实验](NS_ARCHIVED_RECOMPUTATION.md)：15,000 个样本、450 个实际批次已从归档重新核对。
- [Poisson 条件样本平均](K_ARCHIVED_RECOMPUTATION.md)：96 个样本池全部重导出，480 行统计及 672 个数组一致；合并统计的 2,361 项比较通过。
- [基线](BASELINE_FROZEN_REPLAY.md)及[DiffusionPDE](DIFFUSION_INPUT_REPLAY.md)：保留实际输入、模型对应关系、原始预测与已执行的数值检查。

历史 VIVID 的六个拟合模型未保存神经网络权重；原预测和统计结果可复核，
重新推理需重新训练。原 DiffusionPDE 评估未记录 Git HEAD，其保留源码不能
被称为已恢复的原始提交身份。各项限制在复现范围中逐一列明；不同环境或硬件
下的重新训练、随机轨迹和计时不作逐位相同的承诺。

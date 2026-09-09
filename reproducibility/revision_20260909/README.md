# FM4PDE：2026-09-09 修订实验复现

本目录集中保存本次修订的代码版本、环境记录、实验入口和结果归档说明。
论文采用的数值由对应实验的冻结输入、实际预测和逐例统计确定。

集中存储位置及最终核验记录见[最终存储与复现说明](FINAL_DELIVERY.md)。

## 存储位置

统一结果库位于 **server197**：

```text
/research_data/users/zhangxifeng/C01Python/FM4PDE/
├── outputs/
│   ├── main/
│   │   ├── …                         # 原有主实验，原地保留
│   │   └── revision_20260909/         # 本次新增主实验、Burgers和采样时间
│   └── ablations/
│       ├── …                         # 原有消融实验，原地保留
│       └── revision_20260909/         # 本次消融、样本平均和NS补充实验
└── reproducibility/revision_20260909/
    ├── source_snapshots/              # 源代码及Git历史的完整归档
    ├── environments/                  # 实际使用环境的版本记录
    ├── paper_assets_final_20260909/   # 权威论文封存：源码、图表和完整核验记录
    ├── paper_assets/                  # 较早封存版本，保留原状
    └── legacy_scripts/                # 原先在仓库外的必要统计和核验代码
```

每个研究分别保存原始结果、冻结输入、模型对应关系、配置、运行记录及汇总。
分布在不同服务器上的分片按原来源保留，避免同名文件相互覆盖。
原生产目录保留；复制成功以文件校验结果为准。
各研究当前是否已经归档，见 [归档清单](archive_inventory.json) 和
[归档说明](ARCHIVING.md)。正在运行或尚未通过完整核验的研究不得标记为完成。
全部 67 项材料已于 2026-09-09 19:38:49（UTC+8）完成复制与独立校验，
共 194,202 个计划文件、94,486,625,572 字节；见
[完整归档记录](ARCHIVE_COMPLETED_67.md) 和 `archive_completed_67_entries.json`。
此前的 64 项记录保留为历史证据。Poisson 样本数实验的三项归档另有精确清单
`archive_inventory.conditional_scaling_final.json`。这些完成记录不替代下述数值复算、
模型推理或环境恢复的各自证据。

## 代码入口

- [实验复现步骤](STUDY_RECIPES.md)：按主实验和消融实验列出采样、核验、统计及绘图命令。
- [复现范围](REPRODUCTION_COVERAGE.md)：逐项说明可复算的统计、可重新推理的模型，以及历史训练记录的缺项。
- [基线评估入口](BASELINE_FROZEN_REPLAY.md)：用原程序版本、权重和评估数据重建观测与预测。
- [DiffusionPDE 评估输入](DIFFUSION_INPUT_REPLAY.md)：从归档数据生成原采样程序可读取的评估文件及对应配置。
- [外部脚本清单](external_script_inventory.json)：补入仓库的脚本、原位置及内容校验值。
- [源版本清单](source_repositories.json)：正式运行和必要辅助计算使用的版本及用途。
- `source_snapshots.py`：保存、验证和恢复指定版本的源码；同时保留 Git 历史。
- `collect_environments.py`：只读记录既有服务器的包版本，不安装或更改环境。
- `paper_assets.py`：保存并逐文件核对论文及图表来源；主实验和消融共用的权威论文材料存于 `paper_assets_final_20260909/`。

现有 `plot/`、`sampling/`、`models/`、`data/`、`training/`、`scripts/` 和
`configs/` 保留各自职责。已经在仓库内的代码不再复制一套。
外部比较方法的源版本保存在本目录的源码归档中，包括 DiffusionPDE 和
FM4PDEbaseline；其原有许可证和署名随原代码保留。

server197 默认 shell 没有 `python`。在 FM4PDE 根目录先指定已实际测试的解释器：

```bash
FM_PYTHON=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
```

以下 CPU 核验与恢复命令使用该解释器。历史推理和计时仍须恢复各自记录的环境。核验源码归档：

```bash
"$FM_PYTHON" reproducibility/revision_20260909/source_snapshots.py verify
```

该命令核对文件内容和源版本清单，并按 `source_restore_coverage.json`
记录的 Git bundle 分别建立临时独立仓库，重新检查全部指定提交和源码树。
检查不读取原仓库；缺少指定版本或记录的 Git 历史时会报错，其他旧版
bundle 的存在不能代替它。临时仓库在检查后清理。
源码内容与原 Git 对象的逐文件比对记录见 `source_validation.json`。
2026-09-09 的完整核对覆盖 60 个版本（49 个 FM4PDE、3 个 DiffusionPDE、
8 个 FM4PDEbaseline），27,029 个文件的内容和执行权限均与 Git 对象一致；
70 个附带归档和三个独立 bundle 恢复也通过。`source_restore_coverage.json` 的 SHA256 为
`dd6228c136845aeb2836293fc92ff008aeba3ad8659e3cb0361a0e07fa4079a1`。
这些是已登记的实验和核验版本；原 Diffusion 历史运行没有记录的 HEAD 仍不据此补认。

例如，把计时所用 DiffusionPDE 版本恢复为本仓库内的可运行目录：

```bash
"$FM_PYTHON" reproducibility/revision_20260909/source_snapshots.py restore \
  --archive reproducibility/revision_20260909/source_snapshots/DiffusionPDE-151e721b9991.tar.gz \
  --destination reproducibility/revision_20260909/dependencies/DiffusionPDE
```

随后把该目录传给相关运行器的 `--diffusion-root`。恢复操作要求目标目录
尚不存在，且先核对源码归档和 Git 历史的 SHA-256。恢复通过 Git bundle
建立独立仓库并检出指定提交，核对真实 HEAD 和源码树；运行器不会误读上级
FM4PDE 仓库的版本。恢复的源目录保留其版本记录。
同样可恢复指定 FM4PDE 生产版本；新计算应使用新的输出目录。
源码归档及恢复后的依赖目录作为附带文件保存，不递归写入自身的 Git 历史。

## 复现条件

先复核已保存预测的统计结果，再按实验步骤重新采样。两者的输入要求不同：
前者需要原始预测、物理真值、观测掩码和配置；后者还需要匹配的模型权重、
源代码版本以及原运行环境。

主实验使用的 44,121,218 参数 NS 模型与原十一类 PDE 消融使用的
387,487,682 参数 NS 模型分别登记。当前默认配置不替代保存的实际配置。
NS 主实验允许 TF32，受控采样时间实验关闭 TF32。Poisson 样本数实验的
每个任务、输入和全部 K 值在同一设备上完成；误差来自同一个 1000 样本池
的前 K 个样本，较小 K 的运行另行提供实测时间。

随机种子、样本编号、观测掩码及批次划分均属于实验设置。
不同 PyTorch/CUDA 环境即使使用同一个整数种子，也不保证产生同一条随机轨迹。
需要逐例对照时，应恢复对应分片的环境、设备分配与批次设置。
计时只在原研究定义的计时边界和硬件条件下比较。

`environments/*.json` 是整理时只读采集的安装版本；原始结果中的
`environment*.json`、配置和运行记录说明每次实验实际采用的设置。
`*.requirements.txt` 是安装包清单，不能替代 CUDA、驱动和硬件记录。
原有 `environment.yml` 不应被当作所有历史实验共同使用的环境。

论文归档完成后，在 FM4PDE 根目录核验并重新编译四份文稿：

```bash
"$FM_PYTHON" reproducibility/revision_20260909/paper_assets.py verify
"$FM_PYTHON" reproducibility/revision_20260909/build_reproduction_paper.py \
  --output reproducibility/revision_20260909/verification_runs/paper_rebuild
```

工具默认读取 `paper_assets_final_20260909/`，要求整个输出目录尚不存在，
并恢复到其 `paper/` 子目录。它使用 `bibtex8 -8`，编译到交叉引用稳定，
核对四份 PDF 的提取文本，并再次校验原封存文件不变。编译需要论文原有的
TeX 包、`pdflatex`、`bibtex8`、`pdftotext` 和 `pdfinfo`。图表的数值重算按
`STUDY_RECIPES.md` 使用归档原始结果执行，重新编译本身不代替数值核验。
其中 K 的 compact 数值复算使用已恢复论文的 `source_data/conditional_scaling_0909`
和 K 归档的 `local_audit/{server197,server216}`，无需启动原生产 collector。
完整的 96 池原始结果及 480 条 compact 统计已从实际 server197 归档复算通过；
命令、逐项差异和原文件不变记录见 [K 归档复算](K_ARCHIVED_RECOMPUTATION.md)。
`paper_assets_final_20260909/` 的 final 标记说明论文整合与排版核验完成；服务器结果迁移
是否完成仍以各研究的归档校验记录为准。

冻结的评估输入和选定权重随对应研究保存。训练原数据及未采用的历史模型
不整体复制；其原位置、已核对的数据生成信息及训练记录按来源清单保留。
这些评估复现步骤不声称重新训练会逐位得到相同的模型权重。

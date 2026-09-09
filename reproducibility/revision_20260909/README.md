# FM4PDE：2026-09-09 修订实验复现

本目录集中保存本次修订的代码版本、环境记录、实验入口和结果归档说明。
论文采用的数值由对应实验的冻结输入、实际预测和逐例统计确定。

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
    └── legacy_scripts/                # 原先在仓库外的必要统计和核验代码
```

每个研究分别保存原始结果、冻结输入、模型对应关系、配置、运行记录及汇总。
分布在不同服务器上的分片按原来源保留，避免同名文件相互覆盖。
原生产目录保留；复制成功以文件校验结果为准。
各研究当前是否已经归档，见 [归档清单](archive_inventory.json) 和
[归档说明](ARCHIVING.md)。正在运行或尚未通过完整核验的研究不得标记为完成。

## 代码入口

- [实验复现步骤](STUDY_RECIPES.md)：按主实验和消融实验列出采样、核验、统计及绘图命令。
- [外部脚本清单](external_script_inventory.json)：补入仓库的脚本、原位置及内容校验值。
- [源版本清单](source_repositories.json)：正式运行和必要辅助计算使用的版本及用途。
- `source_snapshots.py`：保存、验证和恢复指定版本的源码；同时保留 Git 历史。
- `collect_environments.py`：只读记录既有服务器的包版本，不安装或更改环境。

现有 `plot/`、`sampling/`、`models/`、`data/`、`training/`、`scripts/` 和
`configs/` 保留各自职责。已经在仓库内的代码不再复制一套。
外部比较方法的源版本保存在本目录的源码归档中，包括 DiffusionPDE 和
FM4PDEbaseline；其原有许可证和署名随原代码保留。

在 FM4PDE 根目录核验源码归档：

```bash
python reproducibility/revision_20260909/source_snapshots.py verify
```

例如，把计时所用 DiffusionPDE 版本恢复为本仓库内的可运行目录：

```bash
python reproducibility/revision_20260909/source_snapshots.py restore \
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

冻结的评估输入和选定权重随对应研究保存。训练原数据及未采用的历史模型
不整体复制；其原位置、已核对的数据生成信息及训练记录按来源清单保留。
这些评估复现步骤不声称重新训练会逐位得到相同的模型权重。

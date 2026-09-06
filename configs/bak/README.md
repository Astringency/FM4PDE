# 旧 pretrained 的兼容配置

本目录独立于 `configs/main/`。使用 `outputs/pretrained/bak/` 的旧权重，不修改原 checkpoint。
支持 Poisson、Helmholtz、Darcy、NS 和单通道 Burgers 的旧权重加载、采样与恢复训练。

## 采样参数

四个 PDE 都提供 `both/forward/inverse` 配置。`test_type` 可切换 `id/smooth/rough`。
默认 500 个稀疏观测、stochastic、100 次更新。zeta 来自 `FM4PDE_bak/configs` 中对应 sparse 注释，
不是当前文件中启用的 full 数值；没有单独的 Rough zeta，三个分布使用同一组预先确定的参数。

| PDE | both: a / u / PDE | inverse: u | 旧 observation 乘数 |
|---|---:|---:|---:|
| Poisson | 60000 / 5000000 / 1 | 5000000 | 1 |
| Helmholtz | 200000 / 20000000 / 1 | 20000000 | 1 |
| Darcy | 100000 / 200000 / 1 | 50000000 | 0.1 |
| NS | 30000 / 300000 / 10 | 300000 | 1 |
| Burgers | 0 / 320000 / 10 | —（单通道，使用 both） | 1 |

forward 只启用 a 观测，inverse 只启用 u 观测，保留各自 PDE 项。
Darcy 乘数复现旧 `sample.py` 默认 `--lr_decay=True`：每一步都乘 `obsguide_decay=0.1`，不是逐步累乘。
设 `legacy_obs_multiplier=1` 可运行关闭该开关的版本，但那是另一套采样参数。

旧损失为 `||masked residual||₂ / observed_count`，不同于当前默认 MSE；旧 PDE 项也使用 L2 / 网格数。
`guidance_operator: legacy` 单独复现旧残差，包括 Darcy 未除网格间距、NS 仅为空间导数之和的旧代理项。
**评估指标仍使用当前物理算子**，历史代理项不冒充正确的 NS 残差。
在 latent 坐标里，旧采样的更新系数等价于 `0.2 * (1 - t_next)`。

适配保留当前推理的 `eval()` 模式。旧脚本没有显式关闭 Dropout；旧 `num_steps=100` 建立 100 个时间点，
实际更新 99 次。这里统一使用 100 次更新，且使用当前掩码生成器以支持配对比较。因此它不是旧脚本逐位复现。

```bash
python -m sampling.runner --config configs/bak/inverse/poisson.yaml \
  --override test_type=rough --override device=cuda:0 \
  --override data_path=/path/to/poisson_test_10000-128-128_rough.mat
```

`model_gradient_checkpointing: true` 降低梯度引导采样的显存占用。五份权重均以严格名称/形状匹配加载，
不会忽略缺失权重或回退到当前推荐网络。模型使用单个 attention head，因此旧/新 QKV 顺序完全等价。

## 恢复训练

```bash
python -m scripts.train.resume_bak \
  --checkpoint outputs/pretrained/bak/fm4poisson.pth \
  --additional-epochs 10 --data_path /path/to/original/PDEdata \
  --batch_size 1 --model_gradient_checkpointing \
  --output_dir outputs/train/bak_poisson
```

加 `--print-config` 只检查解析后的设置。启动器读取 checkpoint 的网络与训练参数，恢复模型、Adam、
scheduler、GradScaler 和已完成 epoch；适配生成的新 checkpoint 使用当前 schema，之后可以继续采样/恢复。
额外 epoch 是新增数量；直接使用 `train.py --epochs` 则必须填大于已完成数量的总 epoch。

| PDE | 已完成 epochs | 旧 batch / accumulation / GPU 数 | 旧有效 batch |
|---|---:|---:|---:|
| Poisson | 2000 | 32 / 128 / 4 | 16384 |
| Helmholtz | 400 | 10 / 256 / 4 | 10240 |
| Darcy | 500 | 10 / 128 / 4 | 5120 |
| NS | 500 | 10 / 64 / 4 | 2560 |

除非显式传 `--accum_iter`，启动器根据当前进程数和 batch 调整 accumulation，以尽量保留有效 batch。
沿用旧 LinearLR / 最低 LR=1e-8。已结束的 scheduler 保持末尾学习率，**不会擅自把 LR 重置到 1e-4**。
旧训练源码实际上无条件使用 CUDA FP16 autocast，所以启动器默认 `sampling_dtype=float16`。
旧训练使用全部载入数据；启动器保留该行为，当前验证子集会与训练集重叠，日志仅是训练诊断，不能称为独立验证。

**归一化分两条路径：**旧采样使用 `transform_old.inverse((z+1)/2)` 的固定仿射尺度；旧训练源码对原始训练池
做 Min-Max 再映射到 [-1,1]。旧 checkpoint 没有保存训练 Min/Max，恢复时必须从载入的原始训练池重算并保存。
只有相同的原始文件和样本数量才可能恢复相同尺度；不能仅靠 checkpoint 证明数据尺度完全一致。
本机原始 `/large_storage/...` 数据路径不存在，因此未启动完整恢复训练；兼容测试验证了优化器下一步和保存/重载。

## 与 outputs/main 比较

在双 A100 80GB 上，从项目根目录运行（激活已有项目环境）：

```bash
# 先每组 20 例、Rough/inverse 每组 100 例。
BAK_OUTPUT=outputs/bak_comparison_pilot SAMPLES=20 PRIORITY_SAMPLES=100 \
  bash scripts/run_bak_comparison_a100.sh

# 全量：36 组，每组 1000 例，共 36000 次重建，100 步。
bash scripts/run_bak_comparison_a100.sh
```

默认 `python`、逻辑 GPU 0/1、每个进程 batch=4（只合并同一历史 batch 的样本以保留噪声配对）。
可设置 `BAK_PYTHON`、`BAK_GPU0/1`、`BAK_BATCH_SIZE`。默认启用梯度 checkpoint；若显存仍不足，
降低 `BAK_BATCH_SIZE` 后用相同命令续跑。若已有失败记录，处理原因后增加 `--retry-failed`。
两个进程分别处理 Poisson+Darcy、Helmholtz+NS，均先跑 Rough/inverse，写独立结果文件，最后统一统计。
终端默认显示各 GPU 的样本进度条、当前采样步数和 u 误差；完整日志保存在 `worker0.log` / `worker1.log`。
重定向终端输出时每 30 秒打印一次进度，可用 `BAK_PROGRESS=0` 关闭。首次运行先统一建立 manifest，worker 不会互相覆盖结果。

服务器需要本项目修改后的代码、`configs/bak/`、四个旧 `.pth`，以及 `outputs/main/MAIN1000_100_TEST_{id,smooth,rough}`
和对应 `_tuned1` 目录中的 `metrics_per_sample_all.csv` 与被选运行的 `result.pt`。
不需要复制 `FULL/DET` 目录，不需要当前正式 checkpoint（基线已经保存在历史结果里），也不依赖 `FM4PDE_bak` 源码目录。
使用 `--prepare-only` 可先检查 36 组基线齐备并生成准确的运行来源清单。
该步骤同时检查数据文件存在，并输出 `required_assets.txt`，列出需要复制到服务器、保持相对路径的历史结果及权重。
先复制修改后的项目代码，再按这份清单复制数据即可；文件列表不包括无关的全观测和 deterministic 实验。

```bash
python -m scripts.compare_bak --output outputs/bak_comparison \
  --samples 20 --priority-samples 100 --device cuda:0
```

矩阵为 4 PDE × 3 分布 × 3 任务，Rough/inverse 优先。样本 ID 用固定种子随机抽取，不按历史误差筛选。
每组优先选择完整的 1000 样本 tuned1 结果，否则使用原始完整主实验；强制核对 checkpoint 为六月/七月原先
5 万数据版本，排除 8 月 31 日重训 Poisson。

脚本直接读取历史 `result.pt` 的真值、PDE 参数和掩码，并校验汇总 CSV 与历史预测误差一致，所以比较不需要原 MAT 文件。
复用历史 seed、原 batch 大小和对应行生成初始/bridge 噪声；历史没有保存 RNG tensor/state，无法保证跨设备逐位一致。
比较对象是“旧权重 + 旧 guidance 配置”与“当前权重 + 当前配置”的完整系统，不能单独归因于权重质量。

产物为 `manifest.json`、`protocol.json`、逐样本 `paired_results.jsonl`、`summary.csv`，没有 HTML。
均值差定义为 bak − current，负值表示 bak 更好；报告配对 bootstrap 95% CI、Wilcoxon 和 72 个指标的 Holm 校正。
失败样本保留并停止任务，不会悄悄从统计里删掉。相同命令可续跑；改样本数/步数应使用新的输出目录。
采样 YAML 的 SHA256 写入协议文件，参数改变后禁止把结果混入已有比较目录。
`--priority-only` 只执行 Rough/inverse，`--pde`、`--limit` 可分批运行，`--prepare-only` 只建立清单。

## Burgers 补充实验

```bash
# 双 A100：3 分布 × 2 观测方式 × 1000 例，共 6000 次重建。
bash scripts/run_bak_comparison_a100_burgers.sh

# 小规模试跑：ID/Smooth 各 20 例，Rough 各 100 例。
BAK_OUTPUT=outputs/bak_comparison_burgers_pilot SAMPLES=20 PRIORITY_SAMPLES=100 \
  bash scripts/run_bak_comparison_a100_burgers.sh

# 仅检查完整实验所需文件并生成 required_assets.txt，不使用 GPU。
python -m scripts.compare_bak --suite burgers --samples 1000 --priority-samples 1000 \
  --output outputs/bak_comparison_burgers_full --prepare-only
```

独立输出目录默认为 `outputs/bak_comparison_burgers_full/`，不混入四 PDE 的已有结果。
GPU 0 运行 random，GPU 1 运行 sensor_column；都按 Rough → ID → Smooth 顺序。
支持与四 PDE 脚本相同的环境变量、进度条和断点续跑。再次运行相同命令跳过成功样本；修复失败原因后加 `--retry-failed`。
`--priority-only` 在 Burgers 中选择两种观测方式的 Rough；`--sensor-mode random` 或 `sensor_column` 可进一步筛选。

配置为 `configs/bak/both/burger.yaml`，权重文件名为 **`outputs/pretrained/bak/fm4burgers.pth`**（带 s）。
Burgers 预测一个完整的时空场，a/u 指向同一通道，因此只设 `both`，汇总只统计 `rel_l2_u`，对 6 个组做 Holm 校正。
续跑和统计的样本标识包含观测方式，相同 sample_id 的 random/column 两次重建不会互相覆盖。

旧 zeta 使用 `0 / 320000 / 10`。旧 YAML 的 500 个时间点改为 **100 次更新**，与当前已保存的 n100 结果比较。
历史基线实际是 random 500 点和 **5 列（640 点）**；脚本从 `result.pt` 复用相同真值和掩码，并核对列数。
它不会使用目前 `configs/main/both/burger.yaml` 的 16 列，也不会把基线换成旧 YAML 的 20 列。
单独使用 bak YAML 时 `num_sensor_columns: 20` 保留旧默认值；比较脚本会覆写为历史运行的列数。

旧 Burgers 残差是未除 dt/dx 的中心差分 `u_t + u*u_x - 0.01*u_xx`，保留零填充边界。
旧列观测还有两个需要保留的缩放：观测损失为 `||masked residual||₂ / 500`，
PDE 损失为 `||residual||₂ / (列数 × 128)`；random 的 PDE 分母则是 `128 × 128`。
这些只影响 legacy guidance，评估仍使用当前物理算子。该比较反映两套完整系统，不能单独归因于 checkpoint。

服务器需要新脚本、适配代码、上述 Burgers YAML/权重，以及 `required_assets.txt` 中的历史 CSV 和 `result.pt`。
与四 PDE 比较一样，无需原 MAT、当前正式 checkpoint 或旧项目源码，也不生成 HTML。

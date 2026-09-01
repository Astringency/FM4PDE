# 六个 PDE 的采样参数调优

该流程覆盖以下六个 PDE：

- `advection_diffusion`
- `reaction_diffusion`
- `steady_heat_conduction`
- `heat`
- `shallow_water`
- `wave`

并分别调优 `forward`、`inverse`、`both`。它不会修改 `configs/main`；通过留出验证的参数会写成独立的推荐 YAML。推荐 YAML 会记录本次运行的 `PDE_DATA_ROOT` 和瘦身推理 checkpoint 路径，因此可直接交给正式 sweep。

## 一键运行

先检查作业规模。该命令不读取数据、不加载 checkpoint，也不使用 GPU：

```bash
PLAN_ONLY=true PROFILE=quick \
  bash scripts/tuning/run_six_pde_sampling_tuning.sh
```

服务器有两张 GPU 时，推荐从 `standard` 开始：

```bash
PDE_DATA_ROOT=/path/to/PDEdata \
DEVICE_LIST="cuda:0 cuda:1" \
MAX_PARALLEL_TASKS=2 \
PROFILE=standard \
nohup bash scripts/tuning/run_six_pde_sampling_tuning.sh \
  > six_pde_tuning.log 2>&1 &
```

只有一张 GPU 时：

```bash
PDE_DATA_ROOT=/path/to/PDEdata \
DEVICE_LIST="cuda:0" \
MAX_PARALLEL_TASKS=1 \
PROFILE=standard \
bash scripts/tuning/run_six_pde_sampling_tuning.sh
```

相同命令可以直接重跑。默认 `RESUME=true`，签名和产物均匹配的成功作业会跳过；失败或配置已经变化的作业会重新执行。

## 调优内容

每个 PDE、每类任务都会保留当前主配置作为严格配对的 `baseline`，并比较以下参数族：

- PDE 尺度化的 `zeta_obs_a`、`zeta_obs_u`、`zeta_pde`
- stochastic、deterministic、hybrid sampler
- uniform、geometric、cosine 时间网格
- Euler、midpoint 积分
- temporal PDE 的 Hermite bridge 与 near-endpoint residual
- guidance 启动/ramp 和梯度裁剪阈值（`thorough`）

筛选使用相同样本、相同初始噪声和相同观测 mask 做配对比较；不同 offset 会确定性地改变 seed，以避免所有样本重复使用同一份噪声和 mask。选择指标按任务确定：

- `forward`：`rel_l2_u`
- `inverse`：`rel_l2_a`
- `both`：每个样本上 `max(rel_l2_a, rel_l2_u)`

排序分数为 `0.5 × mean + 0.3 × p90 + 0.2 × max`，同时限制辅助误差和 PDE residual 的退化。筛选胜者还会在完全不重叠的 offsets 上与 baseline 复验；未通过复验时，推荐配置自动回退到 baseline。

## Profile

| Profile | 分布 | 筛选 offsets | 留出 offsets | 每作业 batch | 用途 |
|---|---|---:|---:|---:|---|
| `quick` | ID | 0 | 1000 | 1 | 冒烟检查，不能作为最终结论 |
| `standard` | ID/smooth/rough | 0, 1000 | 3000, 5000 | 2 | 默认推荐 |
| `thorough` | ID/smooth/rough | 0, 1000, 2000 | 4000, 6000, 8000 | 2 | 扩大候选网格和复验样本 |

默认 `standard` 计划是 1278 个筛选作业、至多 216 个留出作业；每个作业包含 2 个样本。作业很多，但单个 PDE 的 checkpoint 只会在对应 worker 中加载一次。

`shallow_water` 和 `wave` 的模型明显更大，准备阶段会为其推理 checkpoint 启用 gradient checkpointing。显存仍不足时，设置 `BATCH_SIZE=1`；这会改变分析签名，应在整个筛选和留出阶段保持一致。

可以先只跑一个 PDE 或一个任务：

```bash
PDE_LIST="heat wave" TASK_LIST="inverse" PROFILE=quick \
PDE_DATA_ROOT=/path/to/PDEdata DEVICE_LIST="cuda:0" \
bash scripts/tuning/run_six_pde_sampling_tuning.sh
```

也可覆盖 profile 的采样点：

```bash
TEST_TYPE_LIST="id smooth rough" \
TUNE_OFFSET_LIST="0 1000" \
HOLDOUT_OFFSET_LIST="4000 6000" \
BATCH_SIZE=2 NUM_STEPS=100 \
bash scripts/tuning/run_six_pde_sampling_tuning.sh
```

注意：筛选 offsets 与留出 offsets 必须互不重叠，并且都应满足数据集索引范围。

## 产物

默认输出根目录为 `outputs/tuning/six_pde_sampling`：

```text
outputs/tuning/six_pde_sampling/
├── checkpoints/                         # 推理用瘦身 checkpoint
├── runs/tune/<profile>/...              # 筛选作业、日志和逐样本指标
├── runs/holdout/<profile>/...           # 留出复验
├── selected_params_<profile>.csv        # 筛选胜者
├── holdout_comparison_<profile>.csv     # 胜者与 baseline 的配对比较
├── recommended_params_<profile>.csv     # 最终推荐及验证状态
└── recommended_configs/<profile>/
    ├── forward/*.yaml
    ├── inverse/*.yaml
    └── both/*.yaml
```

查看某个 worker 的实时进展：

```bash
tail -f outputs/tuning/six_pde_sampling/launcher_logs/heat_all_standard.log
```

## 使用推荐参数正式采样

调优完成后，可直接把推荐配置目录交给原有 sweep：

```bash
CONFIG_DIR=outputs/tuning/six_pde_sampling/recommended_configs/standard \
PDE_LIST="advection_diffusion reaction_diffusion steady_heat_conduction heat shallow_water wave" \
TASK_LIST="forward inverse both" \
OUTPUT_DIR=outputs/SIX_PDE_RECOMMENDED \
bash scripts/sample/run_sample_sweep.sh
```

建议先检查 `recommended_params_standard.csv` 中的 `holdout_validated`、`validation_reason` 和 holdout ratio，再开始大规模正式采样。

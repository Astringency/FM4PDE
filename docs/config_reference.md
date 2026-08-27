# 采样配置文件说明

本文档说明 `configs/main/<task>/` 主采样配置和 `configs/ablations/` 消融覆盖中各字段的含义。

---

## 一、基础设置

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `pde` | `str` | `poisson` | PDE 方程名。可选: `poisson`, `helmholtz`, `darcy`, `nsnonbounded`, `burger`, `heat`, `wave`, `reaction_diffusion`, `shallow_water`, `advection_diffusion`, `steady_heat_conduction` |
| `task` | `str` | `forward` | 任务类型：`forward`（已知系数推断解）、`inverse`（已知解推断系数）、`both`（同时推断系数和解）、`unconditional`（无观测无条件生成） |
| `data_path` | `str` | `""` | 测试数据文件路径（`.mat` 或 `.h5`） |
| `loadby` | `str` | `""` | 数据加载方式，如 `scipy`（.mat）或 `h5py`（.h5） |
| `coef_name` | `str` | `""` | 数据文件中系数变量的名称，如 `f_data` |
| `solution_name` | `str` | `""` | 数据文件中解变量的名称，如 `phi_data` |
| `allow_synthetic_data` | `bool` | `true` | 是否允许使用合成数据（无 ground truth pde_params） |
| `offset` | `int` | `0` | 从数据文件中取第几个样本（多样本数据文件时使用） |

---

## 二、模型与 Checkpoint

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `checkpoint_path` | `str` | `outputs/pretrained/fm4poisson.pth` | 预训练模型权重文件路径 |
| `model_profile` | `str` | `recommended` | 模型规格。可选: `recommended`, `light`, `base`, `heavy` |
| `img_channels` | `int` | `2` | 图像总通道数（系数通道 + 解通道） |
| `img_resolution` | `int` | `128` | 网格分辨率（正方形） |

---

## 三、引导（Guidance）—— 核心控制参数

引导机制通过施加额外约束（观测数据匹配、物理方程残差）来控制生成方向，类似分类器引导（classifier guidance）。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `guidance_components` | `str` | `obs_pde` | 引导组件选择：`noguide`（无引导）、`obs_only`（仅观测约束）、`pde_only`（仅 PDE 残差）、`obs_pde`（观测 + PDE）、`coef_obs_only`（仅系数观测）、`sol_obs_only`（仅解观测）、`both_obs`（系数和解观测分别约束） |
| `zeta_obs_a` | `float` | `1.0` | 系数观测引导强度，越大越强制生成结果在观测点上拟合系数数据 |
| `zeta_obs_u` | `float` | `1.0` | 解观测引导强度，越大越强制生成结果在观测点上拟合解数据 |
| `zeta_pde` | `float` | `1.0` | PDE 残差引导强度，越大越强制生成结果满足物理方程 |
| `pde_guidance_start_ratio` | `float` | `0.0` | PDE guidance 开始启用的归一化 flow time；此前仅保留 observation guidance。`0.0` 保持从第一步启用的旧行为 |
| `pde_guidance_ramp_ratio` | `float` | `0.0` | PDE guidance 从 0 线性增长到完整 `zeta_pde` 所占的 flow-time 区间；`0.0` 表示在 start 位置直接开启。要求与 start 之和不大于 1 |
| `guidance_schedule` | `str` | `constant` | 引导强度随时间变化策略：<br>`constant` — 全程恒定不变<br>`delta` — 集中在采样末期施加<br>`bt` — 与时间步长 `b_t` 相关<br>`cosine` — 余弦衰减/增长<br>`polynomial` — 多项式调度<br>`obs_decay` — 观测权重逐渐衰减 |
| `gradient_target` | `str` | `current_state_chain_rule` | 梯度计算方式：`current_state_chain_rule`（链式法则通过当前状态）、`loss_state_direct`（直接对 loss_state 求导）、`next_state_direct`（仅允许与 `loss_state=x_next` 配合）；启用但断图会直接报错 |
| `stochastic_guidance_coeff` | `float` | `0.1` | 随机阶段引入的额外噪声系数 |
| `stochastic_guidance_time` | `str` | `t` | 随机引导作用时间，`t` 表示在整个随机阶段有效 |
| `clip_mode` | `str` | `global_norm` | 梯度裁剪方式：`none`（不裁剪）、`global_norm`（每个样本内对总梯度做全局范数裁剪）、`per_component_norm`（每个样本内逐引导分量裁剪）；batch 中不同样本不会共享裁剪范数 |
| `clip_threshold` | `float` | `1e10` | 梯度裁剪阈值（`clip_mode != none` 时生效） |
| `pde_residual_region` | `str` | `full` | PDE 内部残差区域：`full`、`boundary_excluded`、`coef_obs`、`sol_obs`、`active_obs_union`。观测区域按 task 判断有效侧 |
| `obs_decay` | `float` | `1.0` | obs_decay 调度模式的衰减系数 |
| `obs_decay_start_ratio` | `float` | `1.0` | obs_decay 调度开始衰减的时间比例 |
| `polynomial_power` | `float` | `2.0` | polynomial 调度模式的幂次 |
| `cosine_mode` | `str` | `decay` | cosine 调度模式：`decay`（衰减）|

例如，前 50% 仅使用 observation guidance，随后用 10% 的 flow time 将 PDE guidance 线性增至完整权重：

```yaml
guidance_components: obs_pde
pde_guidance_start_ratio: 0.5
pde_guidance_ramp_ratio: 0.1
```

PDE gate 只作用于 `zeta_pde`，不会改变 `zeta_obs_a` 或 `zeta_obs_u`；其因子还会与现有 `guidance_schedule` 因子相乘。

---

## 四、采样器（Sampler）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `sampler_phase` | `str` | `stochastic` | 采样阶段模式：`stochastic`（纯随机 / SDE）、`deterministic`（纯确定性 / ODE）、`hybrid_d2s`（先确定性后随机）、`hybrid_s2d`（先随机后确定性） |
| `switch_ratio` | `float` | `0.5` | hybrid 模式的阶段切换比例（0.0~1.0，如 0.5 = 一半步数确定性一半随机） |
| `loss_state` | `str` | `endpoint` | 计算 loss 时所处的状态表示：`xt`（当前状态）、`x_next`（下一步状态）、`endpoint`（端点预测 x̂₀） |
| `time_grid` | `str` | `uniform` | 时间网格分布：`uniform`（均匀线性分布）、`geometric`（等比级数，末端密集） |
| `time_grid_eta` | `float` | `0.4` | geometric 网格的中心偏移参数 |
| `num_steps` | `int` | `100` | 采样总步数（对应 `run_sample.sh` 中的 `NUM_STEPS` 环境变量） |
| `step_method` | `str` | `euler` | ODE/SDE 积分方法：`euler`（Euler-Maruyama）、`midpoint`（中点法） |

---

## 五、传感器 & 观测（Sensor & Observation）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `num_obs` | `int` | `500` | 观测点总数量（每个 sample 中可见的数据点个数） |
| `num_sensor_columns` | `int` 或 `null` | `null` | `sensor_column` 专用的完整列数；该模式必须显式给出正整数且不能超过网格宽度，不复用 `num_obs` |
| `sensor_mode` | `str` | `random` | 传感器分布模式：`random`（随机撒点）、`fixed`（固定位置）、`grid`（规则网格）、`sensor_column`（按列）、`per_sample_random`（每个样本独立随机） |
| `shared_mask` | `bool` | `false` | 系数 coef 和解 sol 是否共用同一组观测点掩码 |
| `mask_seed` | `int` | `0` | 观测点掩码生成的随机种子 |
| `noise_level` | `float` | `0.0` | 观测噪声标准差（同时作用于 coef 和 sol） |
| `noise_level_coef` | `float` 或 `null` | `null` | 单独指定系数的噪声标准差（覆盖 `noise_level`） |
| `noise_level_sol` | `float` 或 `null` | `null` | 单独指定解的噪声标准差（覆盖 `noise_level`） |
| `noise_seed` | `int` | `0` | 噪声生成随机种子（coef 使用 `noise_seed`，sol 使用 `noise_seed + 1`） |

---

## 六、PDE 残差模式

PDE 残差通过计算生成轨迹上物理方程的约束来指导采样。不同的 PDE 类型（稳态/时变）支持不同的残差模式。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `residual_mode` | `str` | `auto` | 残差计算模式：<br>`auto` — 根据 PDE 自动选择（稳态→`static`，端点时变 PDE→`hermite_bridge`，Burgers→完整时空残差）<br>`hermite_bridge` — 仅从模型预测的两个端点构造 Hermite bridge<br>`endpoint_secant` — 仅从模型预测端点计算割线近似<br>`full_trajectory_fd` / `full_time_space` — 仅适用于模型直接输出完整时空场的 Burgers<br>`near_endpoint_temporal` — 仅 Heat、Wave、Advection-Diffusion、Reaction-Diffusion、Shallow-Water、NS 可用；`q0/qT` 使用模型预测，`q(dt)` 复用 coef/q0 mask，`q(T-dt)` 复用 sol/qT mask。这是 PDE loss 唯一允许真实场观测作为辅助输入的例外<br>`disabled` — 禁用 PDE 残差 |
| `hermite_collocation_times` | `list[float]` | `[0.25, 0.5, 0.75]` | hermite_bridge 模式的时间配点（归一化时间 t∈[0,1] 内的取值） |
| `hermite_num_collocation` | `int` | `0` | 自动等距配点数（0=使用 `hermite_collocation_times`） |
| `hermite_include_integral_residual` | `bool` | `true` | 是否包含时间积分残差项 |
| `hermite_integral_weight` | `float` | `1.0` | 积分残差权重 |
| `ns_operator_mode` | `str` | `generator_dealiased` | NS 默认按数据生成器对非线性项和 forcing 做 2/3 去混叠；`continuous_spectral` 保留连续谱诊断版本 |
| `cfg_scale` | `float` | `1.0` | 联合 PDE checkpoint 的标准 CFG 系数：`0` 无条件、`1` 条件，公式为 `v_uncond+s(v_cond-v_uncond)` |
| `save_per_sample_curves` | `bool` | `false` | 是否保存逐 step、逐 sample 的 `metrics_step_per_sample.csv`；最终逐 sample 指标总是写入 `metrics_per_sample.csv` |

## 七、边界条件

部分 PDE 需要额外的边界条件约束。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enforce_boundary_conditions` | `bool` | `true` | 是否施加边界条件（BC）损失 |
| `boundary_condition_mode` | `str` | `auto` | 边界条件类型：`auto`（按 PDE/数据元数据解析）、`dirichlet_zero`（零 Dirichlet）、`neumann_zero`（零 Neumann）、`periodic`（周期边界）、`mixed`（混合）、`wall`（壁面/固壁）、`open`（开放式/流出）、`none`（无边界） |
| `bc_weight` | `float` | `1.0` | 边界条件损失权重 |
| `endpoint_bc_weight` | `float` | `1.0` | 端点处边界条件权重 |
| `boundary_residual_normalization` | `str` | `sqrt_grid_over_mask` | 边界残差归一化方式：`sqrt_grid_over_mask`、`mean`、`mask_mean` |
| `allow_unknown_boundary_conditions` | `bool` | `false` | 是否允许 checkpoint 中边界条件类型未知（按 `auto` 回退） |

---

## 八、输出 & 运行控制

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `output_dir` | `str` | `outputs/ablations` | 结果输出根目录。运行结果将存到 `<output_dir>/<pde>/<task>/<ablation_name>/<timestamp>/` |
| `batch_size` | `int` | `1` | 同时并行采样的独立样本数 |
| `sample_seed` | `int` | `42` | 采样随机种子（控制初始噪声及整体随机性，对应 `SAMPLE_SEED` 环境变量） |
| `device` | `str` | `cpu` | 计算设备：`cpu` 或 `cuda` |
| `dtype` | `str` | `float32` | 计算精度 |
| `save_intermediate` | `bool` | `false` | 是否将所有中间步的完整张量状态保存到 `result.pt`（显著增大文件体积） |
| `save_plots` | `bool` | `false` | 采样后是否保存 ground truth vs prediction 的可视化对比图到 `figures/` 目录 |
| `dry_run` | `bool` | `false` | 仅校验配置不加载模型推理 |
| `empty_cache_each_step` | `bool` | `false` | 每步清空 CUDA 缓存（显存紧张时使用） |
| `ablation_name` | `str` | `""` | 自定义运行名称（留空则由系统根据参数自动生成） |
| `extra` | `dict` | `{}` | 额外自定义参数字典，如 `det_guidance_stoch_scale: true` 等非标准配置项 |

---

## 文件组织结构

```
configs/
├── main/                          # 按任务存放主采样配置
│   ├── both/                     # 十个非 Burger 方程 + Burger
│   │   ├── poisson.yaml
│   │   ├── burger.yaml
│   │   └── ...
│   ├── forward/                  # 十个非 Burger 方程
│   └── inverse/                  # 十个非 Burger 方程（含调参结果）
│
└── ablations/
    ├── formal_suite.yaml          # 正式消融公共覆盖和测试集路径
    ├── all_internal_ablation_grid.yaml # 11 种 PDE 的正式消融网格（1041 jobs）
    ├── all_ablation_grid.yaml     # Poisson 聚焦网格 + 时间 PDE residual（111 jobs）
    ├── smoke.yaml                 # 冒烟测试
    └── main_*.yaml                # 新方案的分主题消融配置
```

### 使用方式

```bash
# 按 PDE 和实验组查看正式消融计划
PDE_LIST="poisson heat" PLAN_ONLY=true \
  bash scripts/run_ablations.sh guidance_components time_grid_by_sampler

# 单次采样（通过环境变量覆盖配置）
PDE=poisson TASK=both bash scripts/sample/run_sample.sh

# 指定采样步数和观测噪声
PDE=helmholtz TASK=inverse NUM_STEPS=1000 NUM_OBS=500 NOISE_LEVEL=0.05 \
  bash scripts/sample/run_sample.sh

# Poisson/Helmholtz/Darcy/NS：三种 task，每个实验采样 1000 个样本
OUTPUT_DIR=outputs/MAIN1000 \
NUM_SAMPLES=1000 \
PDE_LIST="poisson helmholtz darcy nsnonbounded" \
TASK_LIST="forward inverse both" \
SAMPLER_LIST="stochastic" \
PARALLEL=true \
MAX_PARALLEL_TASKS=2 \
DEVICE_LIST="cuda:0 cuda:1" \
RESUME=true \
AGGREGATE=false \
  bash scripts/sample/run_sample_sweep.sh

# Burgers：仅 both，分别运行 random 和 sensor_column 两种观测模式
OUTPUT_DIR=outputs/MAIN1000 \
NUM_SAMPLES=1000 \
SAMPLER_LIST="stochastic" \
BURGER_SENSOR_MODE_LIST="random sensor_column" \
PARALLEL=true \
MAX_PARALLEL_TASKS=2 \
DEVICE_LIST="cuda:0 cuda:1" \
RESUME=true \
AGGREGATE=true \
  bash scripts/sample/run_sample_sweep_burger.sh

# 只查看计划、已有样本和待运行分片，不启动采样
OUTPUT_DIR=outputs/MAIN1000 PLAN_ONLY=true \
  bash scripts/sample/run_sample_sweep.sh

OUTPUT_DIR=outputs/MAIN1000 PLAN_ONLY=true \
  bash scripts/sample/run_sample_sweep_burger.sh

# 直接调用 Python 模块
python -m sampling.runner \
  --config configs/main/both/poisson.yaml \
  --override num_steps=1000 \
  --override noise_level=0.01 \
  --vis
```

正式消融组为：

- 指导组成：四种组成与 `both/forward/inverse` 的 12 个组合。
- 损失状态：`xt/x_next/endpoint` 分别在确定性、随机采样下比较。
- 采样阶段：确定性、随机，以及 D→S/S→D 在 `0.2/0.5/0.8` 的切换。
- 时间离散：网格、步数、积分方法三个独立组，各自跨四种采样阶段。
- 观测设置：稀疏度、布局、噪声三个独立组，固定 `both + stochastic`。
- 时间 residual：六种时间 PDE 上比较 `endpoint_secant`、`hermite_bridge`、
  `near_endpoint_temporal`，统一使用 500 个观测点。
- 稳定性：固定 `offset=0` 和 `noise_seed=0`，配对使用五组 sample/mask seeds。

`sampling.sweep` 支持重复传入 `--pde`、`--group` 和 `--override key=value`。
不存在的过滤值或零任务选择会在运行采样前报错。

正式消融首先按每个展开任务读取 `configs/main/<task>/<pde>.yaml`；Burger 在
`forward/inverse` 交叉任务消融中显式回退到唯一的 `configs/main/both/burger.yaml`。
随后依次应用 `formal_suite.yaml` 公共/PDE 覆盖、组内 `fixed`、`matrix`、
`conditional_overrides` 和命令行 `--override`。因此 zeta 默认只有 `configs/main`
一处来源，测试数据路径等消融专属设置则集中在 `formal_suite.yaml`。

网格组可使用 `conditional_overrides` 命名映射为特定 PDE/变体声明经过校准的参数。每条规则包含
`when` 和 `set` 两个映射；规则在 `fixed + matrix` 展开之后应用，而命令行
`--override` 最后应用、优先级最高。Poisson 的正式网格仅对
`loss_state=x_next + sampler_phase=stochastic` 使用强 zeta profile，其余变体与
`configs/main/both/poisson.yaml` 保持一致，避免为其他变体套用专用强引导。

`run_sample_sweep.sh` 以 PDE × task × sensor mode 为独立调度任务。`PARALLEL=false` 时这些任务串行运行；
`PARALLEL=true` 时最多同时运行 `MAX_PARALLEL_TASKS` 个任务，设备按 `DEVICE_LIST` 轮转分配。
不同 sensor mode 可以并行；同一 PDE × task × sensor mode 内的 sampler 和 offset 分片保持串行。
`SENSOR_MODE_LIST` 设置多个空格分隔值时，每个值形成独立实验、并行任务和恢复记录。
终端会实时显示当前 sensor mode、sampler、已完成样本数、offset/batch、采样 step、相对误差和总体进度。
`run_sample_sweep_burger.sh` 固定运行 `burger / both`，默认
`BURGER_SENSOR_MODE_LIST="random sensor_column"`，并强制覆盖调用环境中可能残留的通用
`SENSOR_MODE_LIST`。其中 `sensor_column` 使用 `configs/main/both/burger.yaml` 的 `num_sensor_columns`。
在两张 GPU 上设置 `PARALLEL=true MAX_PARALLEL_TASKS=2 DEVICE_LIST="cuda:0 cuda:1"` 时，
两个 Burgers sensor mode 会各占一个 worker 并行执行。

通常应令 `MAX_PARALLEL_TASKS` 不大于 `DEVICE_LIST` 中的独立设备数。若并发槽位多于设备数，
多个任务会共享同一设备，脚本会给出警告，并可能因显存不足失败。

`RESUME=true` 默认开启。恢复只接受与当前完整采样配置匹配、且同时具有
`resolved_config.yaml`、成功的 `metrics_final.json` 和 `result.pt` 的样本范围；中断中的 batch
不会被标记为完成。状态和独立 batch 日志保存在
`<output_dir>/.sample_sweeps/<configuration-fingerprint>/`。相同命令重新启动会组合历史成功产物
与完成标记，仅对缺失 offset 重新分片。改变模型、数据路径、采样器、`sensor_mode` 或引导配置会产生
新的配置指纹，不会复用不匹配的结果。

### 输出目录结构

```
<output_dir>/<pde>/<task>/<ablation_name>/<YYYYmmdd-HHMMSS>/
├── run_metadata.json      # 运行元数据（配置、设备、checkpoint 等）
├── resolved_config.yaml   # 解析后的完整配置快照
├── masks.pt               # 观测点掩码张量
├── metrics_step.jsonl     # 每步指标（JSONL 格式）
├── metrics_final.json     # 最终汇总指标
├── curves.csv             # 所有步骤指标表
├── summary.csv            # 最终汇总单行
├── result.pt              # 最终结果（coef/sol 预测值、ground truth、配置等）
└── figures/               # --vis 时的可视化对比图

<output_dir>/.sample_sweeps/<configuration-fingerprint>/
├── manifest.json          # 本次 sweep 的配置与实验清单
├── completed/             # 成功 batch 的原子完成标记
├── progress/              # 正在运行的 step 进度文件
├── logs/                  # 按 PDE/task/sensor_mode/sampler/offset 保存的独立日志
└── sweep.log              # 调度、恢复和失败事件日志
```

当 `batch_size > 1` 时，`rel_l2_a`、`rel_l2_u` 及观测区域相对误差均为“逐样本计算相对 L2，再对样本取算术平均”。每个样本的原始值同时保存在对应的 `*_per_sample` 字段中。`pde_residual_norm` 同样是逐样本 RMS 的平均；没有可归属于生成结果的 PDE residual 时记录为空值，而不是伪造为 0。PDE 参数也逐样本广播（包括 Burgers 的 `nu`、`T/trajectory_dt` 和 `domain_length`），不会从 batch 第一个样本复制到其他样本。

# 采样配置文件说明

本文档说明 `configs/ablations/base/` 下采样实验 YAML 配置文件中各字段的含义。

---

## 一、基础设置

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `pde` | `str` | `poisson` | PDE 方程名。可选: `poisson`, `helmholtz`, `darcy`, `nsnonbounded`, `burger`, `heat`, `wave`, `reaction_diffusion`, `shallow_water`, `advection_diffusion`, `steady_heat_conduction` |
| `task` | `str` | `forward` | 任务类型：`forward`（已知系数推断解）、`inverse`（已知解推断系数）、`both`（同时推断系数和解）、`unconditional`（无观测无条件生成） |
| `data_config_path` | `str` | `configs/main/poisson.yaml` | 指向 `configs/main/` 下数据格式配置文件 |
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
| `model_profile` | `str` | `recommended` | 模型规格。可选: `recommended`, `light`, `base`, `heavy`, `legacy_base` |
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
| `guidance_schedule` | `str` | `constant` | 引导强度随时间变化策略：<br>`constant` — 全程恒定不变<br>`delta` — 集中在采样末期施加<br>`bt` — 与时间步长 `b_t` 相关<br>`cosine` — 余弦衰减/增长<br>`polynomial` — 多项式调度<br>`obs_decay` — 观测权重逐渐衰减 |
| `gradient_target` | `str` | `current_state_chain_rule` | 梯度计算方式：`current_state_chain_rule`（链式法则通过当前状态）、`loss_state_direct`（直接对 loss_state 求导）、`next_state_direct`（直接对下一步求导） |
| `stochastic_guidance_coeff` | `float` | `0.1` | 随机阶段引入的额外噪声系数 |
| `stochastic_guidance_time` | `str` | `t` | 随机引导作用时间，`t` 表示在整个随机阶段有效 |
| `clip_mode` | `str` | `global_norm` | 梯度裁剪方式：`none`（不裁剪）、`global_norm`（全局范数裁剪）、`per_component_norm`（逐引导分量裁剪）、`per_sample_norm`（逐样本裁剪） |
| `clip_threshold` | `float` | `1e10` | 梯度裁剪阈值（`clip_mode != none` 时生效） |
| `pde_residual_region` | `str` | `full` | PDE 残差计算空间区域：`full`（全域网格）、`observed`（仅观测点区域）、`boundary_excluded`（排除边界点的内部区域）、`union_obs`（观测点并集区域） |
| `obs_decay` | `float` | `1.0` | obs_decay 调度模式的衰减系数 |
| `obs_decay_start_ratio` | `float` | `1.0` | obs_decay 调度开始衰减的时间比例 |
| `polynomial_power` | `float` | `2.0` | polynomial 调度模式的幂次 |
| `cosine_mode` | `str` | `decay` | cosine 调度模式：`decay`（衰减）|

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
| `residual_mode` | `str` | `auto` | 残差计算模式：<br>`auto` — 根据 PDE 自动选择（稳态→`static`，时变→`hermite_bridge`）<br>`hermite_bridge` — 用 Hermite 插值在配点时刻计算时间导数（时变 PDE 推荐）<br>`endpoint_secant` — 用端点割线近似时间导数（最低成本）<br>`near_endpoint_temporal` — 利用近端点额外观测计算时间导数<br>`full_trajectory_fd` — 对整个轨迹做有限差分（最高精度但最慢）<br>`full_time_space` — 时空联合残差<br>`disabled` — 禁用 PDE 残差 |
| `hermite_collocation_times` | `list[float]` | `[0.25, 0.5, 0.75]` | hermite_bridge 模式的时间配点（归一化时间 t∈[0,1] 内的取值） |
| `hermite_num_collocation` | `int` | `0` | 自动等距配点数（0=使用 `hermite_collocation_times`） |
| `hermite_include_integral_residual` | `bool` | `true` | 是否包含时间积分残差项 |
| `hermite_integral_weight` | `float` | `1.0` | 积分残差权重 |
| `num_near_endpoint_obs` | `int` | `0` | near_endpoint_temporal 模式下近端点的额外时间观测点数 |
| `near_endpoint_sensor_mode` | `str` | `random` | 额外时间观测点的传感器模式 |
| `near_endpoint_mask_seed` | `int` | `0` | 额外时间观测点掩码种子 |
| `near_endpoint_shared_mask` | `bool` | `true` | 额外时间观测点是否共享掩码 |

---

## 七、边界条件 & 初始条件

部分 PDE 需要额外的边界和初始条件约束。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enforce_boundary_conditions` | `bool` | `true` | 是否施加边界条件（BC）损失 |
| `enforce_initial_conditions` | `bool` | `true` | 是否施加初始条件（IC）损失 |
| `boundary_condition_mode` | `str` | `auto` | 边界条件类型：`auto`（从 checkpoint 自动检测）、`dirichlet_zero`（零 Dirichlet）、`neumann_zero`（零 Neumann）、`periodic`（周期边界）、`mixed`（混合）、`wall`（壁面/固壁）、`open`（开放式/流出）、`none`（无边界）、`legacy_ignore`（旧版忽略） |
| `initial_condition_mode` | `str` | `auto` | 初始条件类型：`auto`（自动检测）、`endpoint_initial`（端点值作为初值）、`observed_initial`（观测值作为初值）、`trajectory_initial`（轨迹初值）、`none`（无初始条件）、`legacy_ignore`（旧版忽略） |
| `bc_weight` | `float` | `1.0` | 边界条件损失权重 |
| `ic_weight` | `float` | `1.0` | 初始条件损失权重 |
| `endpoint_bc_weight` | `float` | `1.0` | 端点处边界条件权重 |
| `boundary_residual_normalization` | `str` | `sqrt_grid_over_mask` | 边界残差归一化方式：`sqrt_grid_over_mask`、`mean`、`mask_mean` |
| `allow_unknown_boundary_conditions` | `bool` | `false` | 是否允许 checkpoint 中边界条件类型未知（按 `auto` 回退） |
| `legacy_ignore_boundary` | `bool` | `false` | 旧版兼容开关：忽略所有边界条件约束 |

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
├── main/                          # 数据格式定义（路径、变量名、加载方式）
│   ├── poisson.yaml
│   ├── helmholtz.yaml
│   └── ...
│
└── ablations/
    ├── base/                      # 单次采样实验配置文件
    │   ├── poisson_forward.yaml   #   PDE + task 组合
    │   ├── poisson_inverse.yaml
    │   ├── poisson_both.yaml
    │   └── ...
    │
    ├── all_ablation_grid.yaml     # 消融实验网格定义（批量生成多组实验）
    ├── smoke.yaml                 # 冒烟测试
    └── main_*.yaml                # 分主题消融实验配置
```

### 使用方式

```bash
# 单次采样（通过环境变量覆盖配置）
PDE=poisson TASK=both bash scripts/sample/run_sample.sh

# 指定采样步数和观测噪声
PDE=helmholtz TASK=inverse NUM_STEPS=1000 NUM_OBS=500 NOISE_LEVEL=0.05 \
  bash scripts/sample/run_sample.sh

# 直接调用 Python 模块
python -m sampling.runner \
  --config configs/ablations/base/poisson_both.yaml \
  --override num_steps=1000 \
  --override noise_level=0.01 \
  --vis
```

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
```

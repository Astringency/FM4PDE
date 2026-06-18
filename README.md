# FM4PDE: Flow Matching for Partial Differential Equations

FM4PDE 是一个用 Flow Matching 生成、补全和反演 PDE 解的实验代码库。当前工程覆盖 11 类 PDE，包含数据生成、训练数据读入、连续 U-Net 速度场训练、带观测和 PDE residual guidance 的采样、评估指标记录，以及内部消融实验网格。

环境使用 Conda 管理：

```bash
conda env create -f environment.yml
conda activate fm4pde
```

主要入口文件：

```text
train.py                         # 模型训练入口
python -m sampling.runner         # 单次采样/评估入口
python -m sampling.sweep          # 消融实验网格入口
python -m sampling.aggregate      # 结果聚合入口
```

## 1. 数据生成

数据生成代码分为静态 PDE、PDEBench/时间依赖 PDE、以及新加入的 future PDE HDF5 生成器。更完整的数据格式说明见 `data/DataGen/pde_data_generation_summary.md`。

| PDE | 生成入口 | 默认输出与说明 |
| --- | --- | --- |
| Darcy | `data/DataGen/static/generate_darcy.m`，统一脚本 `data/DataGen/static/gen_pde.sh` | MATLAB v7.3/HDF5 `.mat`，字段 `thresh_a_data`、`thresh_p_data` |
| Poisson | `data/DataGen/static/generate_poisson.m`，统一脚本 `data/DataGen/static/gen_pde.sh` | `.mat`，字段 `f_data`、`phi_data` |
| Helmholtz | `data/DataGen/static/generate_inhom_helmholtz.m`，统一脚本 `data/DataGen/static/gen_pde.sh` | `.mat`，字段 `f_data`、`psi_data`，默认 `k=1` |
| Burgers | `data/DataGen/static/gen_burgers1.m` 和 `burgers1.m` | `.mat`，字段 `output`，二维数组表示一维 Burgers 的时空图 |
| Non-bounded Navier-Stokes | `data/DataGen/time_dependent/gen_nbns.py` | `.mat`，字段包含 `w0`、`w`、速度和时间信息 |
| Reaction-Diffusion | `data/DataGen/time_dependent/gen_rd.py`，脚本 `data/DataGen/time_dependent/run_reaction_diffusion.sh` | 新 HDF5 格式 `reaction_diffusion_{grf|iid}_...h5`，每个 sample group 含 `data=[T,H,W,2]` 和元数据 |
| Shallow Water | `data/DataGen/time_dependent/gen_swe.py` | HDF5，每个 sample group 含 `data/h`、`data/hu`、`data/hv` |
| Heat | `data/DataGen/python/generate_future_pdes.py --pde heat` | HDF5，`input_data=[u0]`、`output_data=[uT]`，`alpha` 存为标量参数 |
| Wave | `data/DataGen/python/generate_future_pdes.py --pde wave` | HDF5，`input_data=[u0,v0]`、`output_data=[uT,vT]`，`c` 存为标量参数 |
| Advection-Diffusion | `data/DataGen/python/generate_future_pdes.py --pde advection_diffusion` | HDF5，`input_data=[u0]`、`output_data=[uT]`，`b_x,b_y,kappa` 存为标量参数 |
| Steady Heat Conduction | `data/DataGen/python/generate_future_pdes.py --pde steady_heat_conduction` | HDF5，`input_data=[f]`、`output_data=[u]`，保存边界温度、热源和 Picard 诊断量 |

future PDE 的正式生成脚本：

```bash
bash data/DataGen/run_generate_future_pdes_50k_10k_fulltraj.sh
```

默认生成 `heat`、`wave`、`advection_diffusion`、`steady_heat_conduction`，训练集 50000 个样本，5 个 shard，每个 10000；测试集 10000；分辨率 `128 x 128`；保存 11 个时间节点和 `full_trajectory`。小规模快速检查可用：

```bash
python data/DataGen/python/generate_future_pdes.py \
  --pde all \
  --out-root /tmp/fm4pde_future_pdes \
  --quick-test \
  --overwrite
```

## 2. 数据读入

训练读入由 `data/load.py::PDEloader` 完成。它统一返回 `data=[N,C,H,W]` 和 PDE label，其中 `C` 是模型状态通道数。采样/评估读入由 `sampling/data.py::load_ground_truth` 完成，会根据 YAML 中的 `loadby`、`coef`、`solution` 等字段读取测试样本，并拆成 coefficient 侧和 solution 侧。

| PDE | 训练模型通道 | 读入函数/格式 | 备注 |
| --- | --- | --- | --- |
| `darcy` | `[a,p]`，2 通道 | `_darcy_load`，HDF5 `.mat` | label 0 |
| `poisson` | `[f,phi]`，2 通道 | `_poisson_load`，SciPy `.mat` | label 1 |
| `helmholtz` | `[f,psi]`，2 通道 | `_helmholtz_load`，SciPy `.mat` | label 2 |
| `nsnonbounded` | `[w0,wT]`，2 通道 | `_nsnonbounded_load`，HDF5 `.mat` | PDE guidance 当前禁用 |
| `burger` | `[u]`，1 通道 | `_burger_load`，SciPy `.mat` | `128 x 128` 表示时空图 |
| `reaction_diffusion` | `[u0,v0,uT,vT]`，4 通道 | `_reaction_diffusion_load`，HDF5 group | 支持 `rd_init_mode_filter=grf/iid` |
| `shallow_water` | `[h0,hu0,hv0,hT,huT,hvT]`，6 通道 | `_shallow_water_load`，HDF5 group | 守恒变量 |
| `heat` | `[u0,uT]`，2 通道 | `_future_h5_load` | `alpha,T,dt` 进 metadata |
| `wave` | `[u0,v0,uT,vT]`，4 通道 | `_future_h5_load` | `c,T,dt` 进 metadata |
| `advection_diffusion` | `[u0,uT]`，2 通道 | `_future_h5_load` | `b_x,b_y,kappa,T,dt` 进 metadata |
| `steady_heat_conduction` | `[f,u]`，2 通道 | `_future_h5_load` | `u_D`、热源和求解诊断量进 metadata |

标量 PDE 参数默认不是 Flow Matching 通道。训练时它们进入 `output_dir/data_metadata.json` 和 checkpoint 的 `data_metadata`；采样时进入 `PDEGroundTruth.pde_params`，用于 PDE residual。

归一化由 `data/transform.py::PDEStandardizer` 负责。训练集按通道统计：

```text
mean = data.mean(dim=(0, 2, 3), keepdim=True)
std  = data.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
z    = (data - mean) / std
```

训练会保存 `normalizer.pt` 和 `normalization.json`。采样必须使用 checkpoint 中保存的 normalizer，不能从测试样本重新估计统计量。

## 3. 模型训练

训练入口是 `train.py`，参数定义在 `train_arg_parser.py`。示例脚本：

```bash
bash scripts/train/run_train.sh
```

单个 PDE 训练示例：

```bash
python train.py \
  --dataset heat \
  --data_path /large_storage/zhangxf/PDEdata/ \
  --output_dir outputs/pretrained/ \
  --epochs 500 \
  --batch_size 32
```

训练流程：

1. `PDEloader` 读取一个或多个 PDE 的训练 shard，拼接成 `[N,C,H,W]`。
2. `PDEStandardizer` 用训练集拟合逐通道均值和标准差。
3. `models/model_configs.py::instantiate_model` 根据 PDE 名称实例化 U-Net。
4. 使用 CondOT probability path：`x0` 是高斯噪声，`x1` 是标准化后的 PDE 样本，在随机 `t` 上构造 `x_t` 和目标速度 `dx_t`。
5. U-Net 预测速度场，损失为 `mean((v_theta(x_t,t) - dx_t)^2)`。
6. 优化器为 AdamW，可选线性学习率衰减、梯度累积、混合精度、EMA、DDP 和 checkpoint resume。

不同方程的模型都来自 `models/unet.py::UNetModel`。默认架构是 guided-diffusion 风格的二维连续 U-Net：`model_channels=128`、time embedding 维度 `512`、`num_res_blocks=4`、`dropout=0.1`、`num_heads=1`、`conv_resample=True`、无类别条件。base 配置使用 `channel_mult=(1,2,4)`，宽度为 `128/256/512`，在 downsample rate `2` 处使用 attention；Poisson 和 Helmholtz 使用椭圆方程轻量配置 `channel_mult=(1,2,2)`，宽度为 `128/256/256`，`attention_resolutions=(32,)` 在当前三层 U-Net 中实际不触发 attention。

参数量由 `python count_param.py` 统计，未包 EMA 时总参数和可训练参数相同：

| PDE | 模型通道 C | 基本架构 | 参数量 |
| --- | ---: | --- | ---: |
| `burger` | 1 | base U-Net，`channel_mult=(1,2,4)` | 98,805,121 |
| `darcy` | 2 | base U-Net，`channel_mult=(1,2,4)` | 98,807,426 |
| `nsnonbounded` | 2 | base U-Net，`channel_mult=(1,2,4)` | 98,807,426 |
| `heat` | 2 | base U-Net，`channel_mult=(1,2,4)` | 98,807,426 |
| `advection_diffusion` | 2 | base U-Net，`channel_mult=(1,2,4)` | 98,807,426 |
| `steady_heat_conduction` | 2 | base U-Net，`channel_mult=(1,2,4)` | 98,807,426 |
| `poisson` | 2 | elliptic U-Net，`channel_mult=(1,2,2)` | 39,711,234 |
| `helmholtz` | 2 | elliptic U-Net，`channel_mult=(1,2,2)` | 39,711,234 |
| `reaction_diffusion` | 4 | base U-Net，`channel_mult=(1,2,4)` | 98,812,036 |
| `wave` | 4 | base U-Net，`channel_mult=(1,2,4)` | 98,812,036 |
| `shallow_water` | 6 | base U-Net，`channel_mult=(1,2,4)` | 98,816,646 |

checkpoint 会写入模型权重、优化器、学习率调度器、normalizer、数据 shape、通道数和 PDE metadata。采样侧的 `sampling/model_io.py` 会优先读取 EMA 权重；旧 EMA wrapper checkpoint 也有兼容转换逻辑。

## 4. 采样生成

正式入口：

```bash
python -m sampling.runner \
  --config configs/heat.yaml \
  --override checkpoint_path=outputs/pretrained/fm4heat.pth
```

旧接口 `sample.py` 和 `python -m sampling.legacy` 只负责把旧参数映射到 `AblationConfig`，随后仍调用 `sampling.runner`。

采样过程：

1. `sampling.config.load_config` 读取 YAML。既支持旧格式 `configs/*.yaml`，也支持消融格式 `configs/ablations/*.yaml`。
2. `sampling.data.load_ground_truth` 读取测试样本，得到 coefficient、solution、完整 pair、通道名和标量 PDE 参数。
3. `sampling.masks.make_pair_masks` 根据 `num_obs`、`sensor_mode`、`shared_mask` 和随机种子生成观测 mask。
4. `sampling.noise.add_observation_noise` 可对 coefficient/solution 观测加入噪声。
5. `sampling.model_io.load_fm4pde_checkpoint_bundle` 读取 checkpoint、模型权重和训练 normalizer。
6. 初始状态 `x_next` 从标准高斯采样，时间网格由 `sampling.time_grid.make_time_grid` 生成，支持 `uniform`、`geometric`、`cosine`。
7. 每一步调用 `sampling.sampler_wrappers.sampler_step`。确定性相位执行 `x_{t+dt}=x_t+v_theta dt`；随机相位先估计 endpoint `x_1=x_t+(1-t)v_theta`，再用新噪声构造 `x_{t_next}=(1-t_next)x0+t_next*x_1`。支持 `euler` 和 `midpoint`。
8. 如果开启 guidance，`sampling.losses.compute_guidance_losses` 在物理空间计算观测损失和 PDE residual，`sampling.guidance.compute_guidance_gradient` 求梯度并按 `zeta_obs_a/zeta_obs_u/zeta_pde`、schedule 和 clipping 更新状态。
9. 每步写入指标，最后保存生成结果、mask、metadata 和可选中间状态。

主要采样控制项：

| 配置项 | 可选值/含义 |
| --- | --- |
| `task` | `forward`、`inverse`、`both`、`unconditional` |
| `guidance_components` | `noguide`、`obs_only`、`pde_only`、`obs_pde`、`coef_obs_only`、`sol_obs_only`、`both_obs` |
| `loss_state` | `xt`、`x_next`、`endpoint` |
| `sampler_phase` | `deterministic`、`stochastic`、`hybrid_d2s`、`hybrid_s2d` |
| `guidance_schedule` | `constant`、`delta`、`bt`、`cosine`、`polynomial`、`obs_decay` |
| `clip_mode` | `none`、`global_norm`、`per_component_norm` |
| `pde_residual_region` | `full`、`observed`、`boundary_excluded`、`union_obs` |

PDE residual 状态：

| 状态 | PDE |
| --- | --- |
| reliable | Darcy、Poisson、Helmholtz |
| approximate | Burgers、Reaction-Diffusion、Shallow Water、Heat、Wave、Advection-Diffusion、Steady Heat Conduction |
| disabled | `nsnonbounded` |

端点式时间依赖 PDE 默认 `residual_mode: auto`，解析为 `hermite_bridge`。两时间层近似 residual 可通过 `endpoint_secant` 或 `legacy_endpoint_secant` 做消融；近端时间观测 residual 使用 `near_endpoint_temporal`。细节见 `docs/time_dependent_residuals.md`。

## 5. 模型评估

采样运行同时完成评估，不需要单独的 eval 脚本。评估逻辑集中在 `sampling/metrics.py`、`sampling/losses.py` 和 `sampling/pde_residuals.py`。

每一步记录：

```text
t, t_next, phase, loss_state, wall_time
L_obs_a, L_obs_u, L_pde
clean_L_obs_a, clean_L_obs_u
rel_l2_a, rel_l2_u
obs_rel_l2_a, obs_rel_l2_u
pde_residual_norm
gradient norms, clip_scale, actual zeta values
```

每次运行输出目录形如：

```text
outputs/ablations/{pde}/{task}/{ablation_name}/{timestamp}/
  resolved_config.yaml
  run_metadata.json
  metrics_step.jsonl
  metrics_final.json
  curves.csv
  summary.csv
  result.pt
  masks.pt
  legacy_results.pkl
```

`result.pt` 保存最终生成的 coefficient/solution、真值、mask、PDE 参数、normalizer、checkpoint metadata、完整 config 和可选中间状态。可视化辅助函数在 `plot/plot.py::plot_eval_pde`。

聚合结果：

```bash
python -m sampling.aggregate outputs/ablations --output-dir outputs/ablations
# 或
bash scripts/ablations/aggregate_results.sh outputs/ablations outputs/ablations
```

聚合会生成：

```text
summary_all_raw.csv       # 每次运行一行，保留 seed、offset、batch_size
summary_all_grouped.csv   # 按消融维度聚合均值、标准差、SEM、CI95
curves_grouped.csv        # 按 step 聚合曲线指标
```

## 6. 消融实验

消融实验使用 `sampling.sweep` 扩展 YAML 网格，逐个调用 `sampling.runner`。兼容模块 `fm4pde_ablation/sweep.py` 只是转发到 `sampling.sweep`。

基础配置位于 `configs/ablations/base/*.yaml`，覆盖 Darcy、Poisson、Helmholtz、Burgers、Reaction-Diffusion、Shallow Water、Heat、Wave、Advection-Diffusion、Steady Heat Conduction。冒烟测试配置为 `configs/ablations/smoke.yaml`：

```bash
python -m sampling.runner --config configs/ablations/smoke.yaml --dry-run
bash scripts/ablations/smoke.sh
```

主要消融配置和运行脚本：

| 消融内容 | 配置文件 | 运行脚本/命令 | 变量 |
| --- | --- | --- | --- |
| guidance 组成 | `configs/ablations/main_guidance_components.yaml` | `scripts/ablations/run_guidance_components.sh` | `noguide`、`obs_only`、`pde_only`、`obs_pde`、`coef_obs_only`、`sol_obs_only` |
| loss state | `configs/ablations/main_loss_state.yaml` | `scripts/ablations/run_loss_state.sh` | `xt`、`x_next`、`endpoint` |
| sampler phase | `configs/ablations/main_sampler_phase.yaml` | `scripts/ablations/run_sampler_phase.sh` | deterministic、stochastic、hybrid d2s/s2d 和 `switch_ratio` |
| guidance schedule | `configs/ablations/main_guidance_schedule.yaml` | `scripts/ablations/run_guidance_schedule.sh` | `constant`、`delta`、`bt`、`cosine`、`polynomial`、`obs_decay` |
| 梯度裁剪 | `configs/ablations/main_clipping.yaml` | `scripts/ablations/run_clipping.sh` | `none`、不同阈值的 `global_norm`、`per_component_norm` |
| PDE residual 区域 | `configs/ablations/main_pde_residual_region.yaml` | `scripts/ablations/run_pde_residual_region.sh` | `full`、`observed`、`boundary_excluded`、`union_obs` |
| 传感器和噪声 | `configs/ablations/main_sensor_noise.yaml` | `scripts/ablations/run_sensor_noise.sh` | `num_obs`、`sensor_mode`、`noise_level` |
| 时间网格和步数 | `configs/ablations/main_steps_timegrid.yaml` | `scripts/ablations/run_steps_timegrid.sh` | `time_grid`、`num_steps`、`step_method` |
| PDE guidance 权重 | `configs/ablations/main_zeta_sensitivity.yaml` | `scripts/ablations/run_zeta_sensitivity.sh` | `zeta_pde` 从 0 到 100，跨多个 PDE base config |

更完整的内部网格在 `configs/ablations/all_internal_ablation_grid.yaml`，可列出或按 group 运行：

```bash
python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --list

python -m sampling.sweep \
  --grid configs/ablations/all_internal_ablation_grid.yaml \
  --group time_dependent_residual_mode
```

`all_internal_ablation_grid.yaml` 还包含这些 group：

| group | 说明 |
| --- | --- |
| `guidance_components` | guidance 组成 |
| `loss_state` | guidance loss 使用的状态 |
| `sampler_phase` | 确定性、随机、混合采样 |
| `sensor_sparsity` | 观测点数量 |
| `sensor_mode` | mask 采样方式 |
| `noise_robustness` | 观测噪声强度 |
| `zeta_sensitivity` | PDE residual 权重 |
| `time_grid_steps` | 时间网格、步数、Euler/Midpoint |
| `clipping` | guidance 梯度裁剪 |
| `residual_region` | PDE residual 计算区域 |
| `time_dependent_residual_mode` | `endpoint_secant` 与 `hermite_bridge` |
| `near_endpoint_temporal_residual` | 近端时间观测 residual |
| `task` | forward、inverse、both、unconditional |
| `statistics_seed_offset` | seed、mask、noise、offset 重复实验 |
| `ns_observation_only` | NS 在 PDE residual disabled 条件下的观测消融 |

结果收集和聚合：

```bash
bash scripts/ablations/collect_results.sh outputs/ablations outputs/ablations/summary_all.csv
bash scripts/ablations/aggregate_results.sh outputs/ablations outputs/ablations
```

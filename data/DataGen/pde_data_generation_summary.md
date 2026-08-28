# FM4PDE 数据生成与读入格式

本文只记录当前 FM4PDE 正式使用的 11 个 PDE 数据集：公式、初边值条件、生成代码设置、磁盘格式、FM4PDE 读入格式、物理量，以及训练/采样时的数据变换方式。本文以数据生成代码为唯一真值；若公式的通常写法、说明文档或文件名与生成代码不一致，以生成器实际组装的离散方程和文件 attrs/datasets 为准。

## 通用约定

- 默认空间分辨率为 `128 x 128`，除 Burgers 外均表示二维空间网格。
- 训练 loader 输出 BCHW 张量 `[N,C,H,W]`，标签由 `data/load.py` 中的 PDE 顺序给出。
- 对于 `pair_h5` 数据，磁盘中的 scalar 物理参数默认不展开成模型通道，而是进入 `PDEloader.pde_params`、训练 `data_metadata` 和采样 `pde_params`。
- 采样端会把模型状态反标准化回物理量后再计算 observation loss 和 PDE residual。
- 随样本变化的物理参数（如 `alpha,b_x,b_y,kappa`）必须从样本 dataset/attrs 读取；缺失时不得用任意常数静默替代。
- 参数解析优先级为：样本 dataset > sample/group attrs > root attrs > 显式 `generator_profile`。`split=train|test` 本身不能唯一确定历史生成器版本。

## 五个核心方程的统一生成入口

Poisson、Helmholtz、Darcy、non-bounded Navier-Stokes 和 Burgers 统一通过下面的脚本生成，不再通过修改或取消 MATLAB 注释来切换分布：

```bash
# 默认生成五个方程，每个方程 10000 个样本、空间分辨率 128
bash data/DataGen/static/gen_pde.sh train
bash data/DataGen/static/gen_pde.sh easytest
bash data/DataGen/static/gen_pde.sh hardtest

# 只生成指定方程，并覆盖已有目标文件
bash data/DataGen/static/gen_pde.sh hardtest \
  --pdes poisson,helmholtz,darcy \
  --samples 10000 \
  --overwrite

# NS 和 Burgers 的小规模粗糙测试集
bash data/DataGen/static/gen_pde.sh hardtest \
  --pdes nsnonbounded,burgers \
  --samples 1000 \
  --device cuda:0
```

三类数据使用的 GRF 参数为：

| 类型 | Poisson / Helmholtz / Darcy `(alpha,tau)` | NS `(alpha,tau)` | Burgers `(gamma,tau)` | 默认 seed offset |
|---|---:|---:|---:|---:|
| `train` | `(2.0,3.0)` | `(2.5,7.0)` | `(2.5,7.0)` | `0` |
| `easytest`（平滑） | `(3.0,4.0)` | `(3.0,6.5)` | `(3.0,6.5)` | `10000000` |
| `hardtest`（粗糙） | `(1.5,5.0)` | `(1.5,5.0)` | `(1.5,5.0)` | `20000000` |

`easy/smooth/test` 是 `easytest` 的别名，`hard/rough` 是 `hardtest` 的别名。输出文件名包含规范化后的数据类型，例如 `poisson_hardtest_10000-128-128.mat`。文件内同时保存 `dataset_type`、GRF 参数和 `generation_seed`；NS 使用 HDF5 attrs 保存这些元数据。可先添加 `--dry-run` 检查完整配置而不生成数据。

## 只生成 test 数据

统一 endpoint-pair HDF5 入口支持只生成 test split：

```bash
python data/DataGen/python/generate_pair_h5s.py \
  --pde heat \
  --out-root /large_storage/zhangxf/PDEdata \
  --split test \
  --n-test 10000 \
  --overwrite
```

批量生成 `heat`、`wave`、`advection_diffusion`、`steady_heat_conduction` 的 test 数据：

```bash
SPLIT=test N_TEST=10000 bash data/DataGen/run_generate_pair_h5s_50k_10k_fulltraj.sh
```

Reaction-diffusion 已有 test-only split：

```bash
python data/DataGen/time_dependent/gen_rd.py \
  --save-path /large_storage/zhangxf/PDEdata/reaction_diffusion \
  --total-samples 10000 \
  --samples-per-file 10000 \
  --split test \
  --seed-offset 10000000 \
  --overwrite
```

Non-bounded Navier-Stokes 和 shallow-water 也提供参数化 test-only CLI：

```bash
python data/DataGen/time_dependent/gen_nbns.py \
  --dataset-type easytest \
  --total-samples 10000 \
  --resolution 128 \
  --device cuda:0 \
  --overwrite

python data/DataGen/time_dependent/gen_swe.py \
  --split test \
  --total-samples 10000 \
  --resolution 128 \
  --overwrite
```

## 1. Darcy Flow

- 公式：
  $$-\nabla\cdot(a(x,y)\nabla p(x,y)) = 1,\quad (x,y)\in[0,1]^2.$$
- 初边值条件：系数场 `a` 由 GRF threshold 得到；压力满足齐次 Dirichlet 边界 `p=0`。
- 生成代码：`data/DataGen/static/generate_darcy.m`，依赖 `GRF.m` 和 `solve_gwf.m`。存储的 `a,p` 位于单元中心；`solve_gwf.m` 先用 MATLAB `interp2(...,'spline')` 把 `a` 和 `f` 映射到包含端点的节点网格，在节点网格上用面系数算术平均组装守恒离散算子并施加 `p=0`，求解后再用同一类 spline 映射回单元中心。PDE residual 必须重建这套 spline→节点算子→spline 布局，不能直接在存储网格上套中心差分。
- 磁盘格式：新入口写为 `darcy/darcy_{train|easytest|hardtest}_N-S-S.mat`，HDF5/MATLAB v7.3 key 包括 `thresh_a_data`、`thresh_p_data` 和生成配置元数据，场形状通常为 `[H,W,N]`。
- FM4PDE 读入：`[a,p]`，即 `[N,2,H,W]`；物理量为渗透/扩散系数 `a` 和压力 `p`。

## 2. Poisson

- 公式：
  $$\Delta\phi(x,y)=f(x,y),\quad (x,y)\in[0,1]^2.$$
- 初边值条件：源项 `f` 从 GRF 采样；齐次 Dirichlet 边界 `phi=0`。
- 生成代码：`data/DataGen/static/generate_poisson.m`，MATLAB 五点差分线性系统。
- 磁盘格式：新入口写为 `poisson/poisson_{train|easytest|hardtest}_N-S-S.mat`，key 为 `f_data`、`phi_data` 和生成配置元数据，场形状 `[N,H,W]`。
- FM4PDE 读入：`[f,phi]`，即 `[N,2,H,W]`；物理量为源项 `f` 和势场 `phi`。

## 3. Helmholtz

- 生成器实际方程：
  $$(\Delta+k^2)\psi(x,y)=f(x,y),\quad (x,y)\in[0,1]^2.$$
- 生成代码：`data/DataGen/static/generate_inhom_helmholtz.m`，默认固定波数 `k=1`。代码先修改一维矩阵 `L` 的首末行，再构造 `kron(I,L)+kron(L,I)`；它没有把完整二维边界行替换成单位行，因此生成数据并不严格满足通常意义的二维齐次 Dirichlet `psi=0`。当前 residual 按该 Kronecker 线性系统（并把 RHS 边界置零）精确复现，不再额外叠加与生成数据矛盾的 `psi=0` loss。
- 磁盘格式：新入口写为 `helmholtz/helmholtz_{train|easytest|hardtest}_N-S-S-kK.mat`，key 为 `f_data`、`psi_data`、`k` 和生成配置元数据，场形状 `[N,H,W]`。
- FM4PDE 读入：`[f,psi]`，即 `[N,2,H,W]`；物理量为源项 `f`、波场 `psi` 和生成侧固定参数 `k`。

## 4. Non-bounded Navier-Stokes

- 公式，涡量形式：
  $$\partial_t\omega+u\partial_x\omega+v\partial_y\omega=\nu\Delta\omega+f(x,y),\quad \nu=10^{-3}.$$
  速度由流函数恢复，固定 forcing 为 `0.1*(sin(2*pi*(x+y))+cos(2*pi*(x+y)))`。
- 初边值条件：初始涡量 `w0` 从二维 GRF 采样；空间采用周期谱方法。
- 生成代码：`data/DataGen/time_dependent/gen_nbns.py`，依赖 `no_bound_ns/ns_2d.py` 和 `no_bound_ns/random_fields.py`；时间 `T=1`，内部求解步长 `solver_dt=1e-4`，记录 `10` 个正时间快照。`w` 不包含初值，full trajectory 必须拼成 `[w0,w(t_1),...,w(t_10)]` 共 11 帧；快照差分间隔是 `0.1`，绝不能使用 attrs 中的内部 `dt=1e-4`。三种初值分布见统一配置表，PDE 系数和 forcing 保持相同。
- 磁盘格式：新入口写为 `nsnonbounded/nsnonbounded_TYPE_N-S-S-STEPS[_SHARD].mat`；key 包括 `w0`、`w`、`vx0`、`vy0`、`vx`、`vy`、`t`，attrs 包括 `dataset_type`、`grf_alpha` 和 `grf_tau`。
- FM4PDE 读入：当前训练 loader 取 `[w0,wT]`，即 `[N,2,H,W]`；物理量为初始涡量和终态涡量，速度场只作为磁盘附加量保存。

## 5. Burgers

- 公式：
  $$\partial_tu+\frac12\partial_x(u^2)=\nu\partial_{xx}u,\quad x\in[0,1],\quad \nu=0.01.$$
- 初边值条件：一维周期边界；初值 `u0` 从周期 GRF 采样。
- 生成代码：`data/DataGen/static/gen_burgers1.m` 和 `burgers1.m`，MATLAB/Chebfun `spin` 时间推进。默认 `steps=127` 且 `tspan=linspace(0,1,steps+1)`，因此输出有 128 帧（包含初值和终值），相邻快照 `dt=1/127`。空间为 128 个不重复周期点，`dx=1/128`；输出组织为 `128 x 128` 的 time×space 时空图。三种初值分布见统一配置表，`sigma=tau^(gamma-0.5)` 以延续训练分布原有的幅值归一化规则。
- 磁盘格式：新入口写为 `burgers/burger_{train|easytest|hardtest}_N-X-T.mat`；key 为 `output`、`input`、`tspan` 和生成配置元数据，`output` 形状 `[N,T,X]`。
- FM4PDE 读入：`[u]`，即 `[N,1,128,128]`；采样端当前把同一时空场作为 coef/sol 单通道状态处理。

## 6. Reaction-Diffusion

- 公式：
  $$\partial_tu=D_u\Delta u+u-u^3-k-v,$$
  $$\partial_tv=D_v\Delta v+u-v.$$
- 初边值条件：定义域 `[-1,1]^2`；齐次 Neumann 边界；初值支持 `init_mode=grf` 和 `init_mode=iid`，正式训练建议显式过滤 `grf` 或 `iid`，避免混合。
- 生成代码：`data/DataGen/time_dependent/gen_rd.py`，依赖 `pdebench/data_gen/src/sim_diff_react.py`；train 与 test 默认均为 `D_u=2e-3`、`D_v=4e-3`、`k=3e-3`、`T=1`、`n_save_steps=10`，因此保存 `11` 帧。
- 磁盘格式：`reaction_diffusion_{grf|iid}_{total}-128-128-T1-steps10[_shardNNN].h5`；test 为 `reaction_diffusion_test_{grf|iid}_...h5`。每个样本 group 含 `data`，形状 `[T,H,W,2]`，并含 `grid/x`、`grid/y`、`grid/t`、attrs 和 `sample_seed`。
- FM4PDE 读入：`[u0,v0,uT,vT]`，即 `[N,4,H,W]`；`T,D_u,D_v,k,n_save_steps,tdim,domain,sample_seed,init_mode` 等进入 metadata/pde_params。

## 7. Shallow Water

- 公式，守恒变量为 `q=(h,hu,hv)`：
  $$\partial_th+\partial_x(hu)+\partial_y(hv)=0,$$
  $$\partial_t(hu)+\partial_x(hu^2+\frac12gh^2)+\partial_y(huv)=0,$$
  $$\partial_t(hv)+\partial_x(huv)+\partial_y(hv^2+\frac12gh^2)=0.$$
- 初边值条件：径向溃坝初值，`h=h_inner` inside dam radius、外部 `h=1`，`hu=hv=0`；边界为 extrapolation/零阶 Neumann。
- 生成代码：`data/DataGen/time_dependent/gen_swe.py`，依赖 `pdebench/data_gen/src/sim_radial_dam_break.py` 和 Clawpack/PyClaw；默认 `g=1`、`T=1`、10 个推进区间并保存 11 帧。数组空间轴为 `[x,y]`，定义域 `[-2.5,2.5]^2`，单元中心间距 `dx=dy=5/128`。extrapolation 通过 ghost cells 实现，不等价于要求第一、第二个物理单元值相等。
- 磁盘格式：test 为 `shallow_water/shallow_water_test_1000-128-128-10.h5`；HDF5 group 每个样本含 `data/h`、`data/hu`、`data/hv`、`grid/x`、`grid/y`、`grid/t`，并记录 `dam_radius`、`inner_height` 等 attrs。
- FM4PDE 读入：`[h0,hu0,hv0,hT,huT,hvT]`，即 `[N,6,H,W]`；物理量为水深和两个方向动量。

## 8. Heat

- 公式：
  $$\partial_tu=\alpha\Delta u,\quad (x,y)\in[0,1]^2.$$
- 初边值条件：默认周期边界；可选 Neumann 生成分支；初值由平滑 GRF 采样。
- 生成代码：`data/DataGen/python/generate_heat.py`，统一入口为 `generate_pair_h5s.py`；默认 `alpha~U(5e-4,5e-3)`、`T=1`、`n_time=11`，谱方法精确推进。
- 磁盘格式：`heat/heat_10000-128-128_i.h5` 和 `heat/heat_test_1000-128-128.h5`；key 为 `input_data=[N,1,H,W]`、`output_data=[N,1,H,W]`、可选 `full_trajectory=[N,1,T,H,W]`，随机 alpha 存为 `alpha=[N]`，固定 alpha 存 attrs `fixed_alpha`。
- FM4PDE 读入：当前 loader 返回 `[u0,uT]`，即 `[N,2,H,W]`；`alpha,T,dt` 作为 metadata/pde_params。

## 9. Wave

- 公式：
  $$\partial_{tt}u=c^2\Delta u,\quad u(0)=u_0,\quad \partial_tu(0)=v_0.$$
- 初边值条件：周期边界；`u0` 从平滑 GRF 采样，默认 `v0=0`，可选随机初速度；默认固定 `c=1`。
- 生成代码：`data/DataGen/python/generate_wave.py`；固定 scalar `c` 时使用 Fourier 精确公式，`variable_c=True` 当前禁用。
- 磁盘格式：`wave/wave_10000-128-128_i.h5` 和 test 文件；`input_data=[N,2,H,W]` 存 `[u0,v0]`，`output_data=[N,2,H,W]` 存 `[uT,vT]`，`full_trajectory=[N,2,T,H,W]` 保存完整 `[u(t),v(t)]` 状态。随机 `c` 存 dataset，固定 `c` 存 attrs。
- FM4PDE 读入：`[u0,v0,uT,vT]`，即 `[N,4,H,W]`；`c,T,dt` 作为 metadata/pde_params。

## 10. Advection-Diffusion

- 公式：
  $$\partial_tu+b_x\partial_xu+b_y\partial_yu=\kappa\Delta u,\quad (x,y)\in[0,1]^2.$$
- 初边值条件：周期边界；`u0` 从平滑 GRF 采样。
- 生成代码：`data/DataGen/python/generate_advection_diffusion.py`；默认 `b_x,b_y~U(-1,1)`、`\kappa~U(5e-4,5e-3)`，Fourier 精确推进。
- 磁盘格式：`advection_diffusion/advection_diffusion_10000-128-128_i.h5` 和 test 文件；`input_data=[N,1,H,W]`、`output_data=[N,1,H,W]`，scalar dataset 为 `b_x,b_y,kappa`，可选 `full_trajectory`。
- FM4PDE 读入：当前 loader 返回 `[u0,uT]`，即 `[N,2,H,W]`；`b_x,b_y,kappa,T,dt` 作为 metadata/pde_params。

## 11. Steady Heat Conduction

- 公式：
  $$-\nabla\cdot(\lambda(u)\nabla u)=f(x,y),\quad \lambda(u)=1+0.05(u-298).$$
- 初边值条件：底部 Dirichlet `u=u_D`；顶部、左侧、右侧零 Neumann；源项由 1 到 4 个 Gaussian heat sources 叠加。
- 生成代码：`data/DataGen/python/generate_steady_heat_conduction.py`；Picard 迭代求解非线性椭圆方程，`lambda` 下限 clamp 为 `0.1`，默认 `picard_max_iter=30`、`picard_tol=1e-5`。
- 磁盘格式：`steady_heat_conduction/steady_heat_conduction_10000-128-128_i.h5` 和 test 文件；`input_data=[N,1,H,W]` 存 `f`，`output_data=[N,1,H,W]` 存 `u`，并保存 `u_D,residual_norm,picard_iters,converged,n_sources,source_x,source_y,source_amp,source_sigma`。
- FM4PDE 读入：当前 loader 返回 `[f,u]`，即 `[N,2,H,W]`；边界温度和求解诊断量作为 metadata/pde_params。

## FM4PDE 数据变换

当前训练不再使用 Min-Max 归一化，而是使用 `data/transform.py` 中的 `PDEStandardizer` 做逐通道标准化。对训练张量 `data=[N,C,H,W]`：

```python
mean = data.mean(dim=(0, 2, 3), keepdim=True)
std = data.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
z = (data - mean) / std
```

- `mean/std` 形状为 `[1,C,1,1]`，按每个 FM4PDE 通道独立统计。
- 若某通道标准差小于 `eps`，会用 `1` 保护，避免除零。
- 训练时保存 `normalizer.pt`、`normalization.json`，并把 normalizer 写入 checkpoint。
- 采样必须使用 checkpoint 中保存的 normalizer；不能用测试样本重新估计统计量。
- PDE residual 和 observation loss 在物理空间计算：采样端先对模型状态做 inverse transform，再拆分为物理量。

## PDE residual 时间模式

- `full_trajectory_fd`：采样引导与生成结果评估中仅适用于模型直接输出完整 `[T,X]` 时空场的 Burgers。数据文件保存的真实轨迹不得代替模型输出进入 PDE loss；显式轨迹接口仅保留给离线数据诊断。
- `endpoint_secant`：只使用初值和终值，在中点状态上计算割线近似；这是明确标记为 approximate 的两层近似。
- FM4PDE 采样对端点模型默认使用仅依赖预测 `q0/qT` 的 `hermite_bridge` 或 `endpoint_secant`。Heat、Wave、Advection-Diffusion、Reaction-Diffusion、Shallow-Water、NS 还可显式选择 `near_endpoint_temporal`：`q0/qT` 仍是模型输出，只额外读取 `q(dt)`、`q(T-dt)` 的稀疏真实观测；未观测值被清零且完整近端帧不进入 PDE loss。这是唯一的真实场辅助输入例外。
- 周期数据的首末网格点是不同的物理点，周期性由 FFT/roll 算子编码，不能额外强迫二者相等。RD 的 Neumann 与 SWE 的 extrapolation 由 ghost-cell 算子编码，也不能强迫相邻物理单元相等。

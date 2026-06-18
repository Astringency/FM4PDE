# FM4PDE 数据生成与读入格式

本文只记录当前 FM4PDE 正式使用的 11 个 PDE 数据集：公式、初边值条件、生成代码设置、磁盘格式、FM4PDE 读入格式、物理量，以及训练/采样时的数据变换方式。

## 通用约定

- 默认空间分辨率为 `128 x 128`，除 Burgers 外均表示二维空间网格。
- 训练 loader 输出 BCHW 张量 `[N,C,H,W]`，标签由 `data/load.py` 中的 PDE 顺序给出。
- 对于 `pair_h5` 数据，磁盘中的 scalar 物理参数默认不展开成模型通道，而是进入 `PDEloader.pde_params`、训练 `data_metadata` 和采样 `pde_params`。
- 采样端会把模型状态反标准化回物理量后再计算 observation loss 和 PDE residual。

## 1. Darcy Flow

- 公式：
  $$-\nabla\cdot(a(x,y)\nabla p(x,y)) = 1,\quad (x,y)\in[0,1]^2.$$
- 初边值条件：系数场 `a` 由 GRF threshold 得到；压力满足齐次 Dirichlet 边界 `p=0`。
- 生成代码：`data/DataGen/static/generate_darcy.m`，依赖 `GRF.m` 和 `solve_gwf.m`；MATLAB 有限体积/有限差分椭圆求解，默认每 shard `10000` 个样本、5 个 train shard。
- 磁盘格式：`darcy/darcy_10000-128-128_i.mat`，HDF5/MATLAB v7.3 key 为 `thresh_a_data`、`thresh_p_data`，形状通常为 `[H,W,N]`。
- FM4PDE 读入：`[a,p]`，即 `[N,2,H,W]`；物理量为渗透/扩散系数 `a` 和压力 `p`。

## 2. Poisson

- 公式：
  $$-\Delta\phi(x,y)=f(x,y),\quad (x,y)\in[0,1]^2.$$
- 初边值条件：源项 `f` 从 GRF 采样；齐次 Dirichlet 边界 `phi=0`。
- 生成代码：`data/DataGen/static/generate_poisson.m`，MATLAB 五点差分线性系统。
- 磁盘格式：`poisson/poisson_10000-128-128_i.mat`，key 为 `f_data`、`phi_data`，形状 `[N,H,W]`。
- FM4PDE 读入：`[f,phi]`，即 `[N,2,H,W]`；物理量为源项 `f` 和势场 `phi`。

## 3. Helmholtz

- 公式：
  $$(-\Delta-k^2)\psi(x,y)=f(x,y),\quad (x,y)\in[0,1]^2.$$
- 初边值条件：源项 `f` 从 GRF 采样；默认固定波数 `k=1`；齐次 Dirichlet 边界 `psi=0`。
- 生成代码：`data/DataGen/static/generate_inhom_helmholtz.m`，MATLAB Kronecker 二维 Laplacian 线性系统。
- 磁盘格式：`helmholtz/helmholtz_10000-128-128_i.mat`，key 为 `f_data`、`psi_data`，形状 `[N,H,W]`。
- FM4PDE 读入：`[f,psi]`，即 `[N,2,H,W]`；物理量为源项 `f`、波场 `psi` 和生成侧固定参数 `k`。

## 4. Non-bounded Navier-Stokes

- 公式，涡量形式：
  $$\partial_t\omega+u\partial_x\omega+v\partial_y\omega=\nu\Delta\omega+f(x,y),\quad \nu=10^{-3}.$$
  速度由流函数恢复，固定 forcing 为 `0.1*(sin(2*pi*(x+y))+cos(2*pi*(x+y)))`。
- 初边值条件：初始涡量 `w0` 从二维 GRF 采样；空间采用周期谱方法。
- 生成代码：`data/DataGen/time_dependent/gen_nbns.py`，依赖 `no_bound_ns/ns_2d.py` 和 `no_bound_ns/random_fields.py`；时间 `T=1`，内部步长 `1e-4`，记录 `10` 个快照。
- 磁盘格式：`nsnonbounded/nsnonbounded_10000-128-128-10_i_new.mat`，key 包括 `w0`、`w`、`vx0`、`vy0`、`vx`、`vy`、`t`。
- FM4PDE 读入：当前训练 loader 取 `[w0,wT]`，即 `[N,2,H,W]`；物理量为初始涡量和终态涡量，速度场只作为磁盘附加量保存。

## 5. Burgers

- 公式：
  $$\partial_tu+\frac12\partial_x(u^2)=\nu\partial_{xx}u,\quad x\in[0,1],\quad \nu=0.01.$$
- 初边值条件：一维周期边界；初值 `u0` 从周期 GRF 采样。
- 生成代码：`data/DataGen/static/gen_burgers1.m` 和 `burgers1.m`，MATLAB/Chebfun `spin` 时间推进；输出被组织为 `[space,time]` 的 `128 x 128` 时空图。
- 磁盘格式：`burgers/burger_10000-128-128_i.mat`，key 为 `output`，形状 `[N,128,128]`。
- FM4PDE 读入：`[u]`，即 `[N,1,128,128]`；采样端当前把同一时空场作为 coef/sol 单通道状态处理。

## 6. Reaction-Diffusion

- 公式：
  $$\partial_tu=D_u\Delta u+u-u^3-k-v,$$
  $$\partial_tv=D_v\Delta v+u-v.$$
- 初边值条件：定义域 `[-1,1]^2`；齐次 Neumann 边界；初值支持 `init_mode=grf` 和 `init_mode=iid`，正式训练建议显式过滤 `grf` 或 `iid`，避免混合。
- 生成代码：`data/DataGen/time_dependent/gen_rd.py`，依赖 `pdebench/data_gen/src/sim_diff_react.py`；默认 `D_u=2e-3`、`D_v=4e-3`、`k=3e-3`、`T=1`、`n_save_steps=10`，因此保存 `11` 帧。
- 磁盘格式：新格式为 `reaction_diffusion_{grf|iid}_{total}-128-128-T1-steps10[_shardNNN].h5`；test 为 `reaction_diffusion_test_{grf|iid}_...h5`。每个样本 group 含 `data`，形状 `[T,H,W,2]`，并含 `grid/x`、`grid/y`、`grid/t`、attrs 和 `sample_seed`。
- FM4PDE 读入：`[u0,v0,uT,vT]`，即 `[N,4,H,W]`；`T,D_u,D_v,k,n_save_steps,tdim,domain,sample_seed,init_mode` 等进入 metadata/pde_params。

## 7. Shallow Water

- 公式，守恒变量为 `q=(h,hu,hv)`：
  $$\partial_th+\partial_x(hu)+\partial_y(hv)=0,$$
  $$\partial_t(hu)+\partial_x(hu^2+\frac12gh^2)+\partial_y(huv)=0,$$
  $$\partial_t(hv)+\partial_x(huv)+\partial_y(hv^2+\frac12gh^2)=0.$$
- 初边值条件：径向溃坝初值，`h=h_inner` inside dam radius、外部 `h=1`，`hu=hv=0`；边界为 extrapolation/零阶 Neumann。
- 生成代码：`data/DataGen/time_dependent/gen_swe.py`，依赖 `pdebench/data_gen/src/sim_radial_dam_break.py` 和 Clawpack/PyClaw；默认 `g=1`、`T=1`、`10` 个时间步。
- 磁盘格式：HDF5 group 每个样本含 `data/h`、`data/hu`、`data/hv`、`grid/x`、`grid/y`、`grid/t`，并记录 `dam_radius`、`inner_height` 等 attrs。
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
- 磁盘格式：`wave/wave_10000-128-128_i.h5` 和 test 文件；`input_data=[N,2,H,W]` 存 `[u0,v0]`，`output_data=[N,2,H,W]` 存 `[uT,vT]`，随机 `c` 存 dataset，固定 `c` 存 attrs。
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

# `~/share/PDEdata` PDE residual 验证

验证日期：2026-08-19。验证对象为本工作区中的 `FM4PDE` 与 `FM4PDEbaseline` 修正后实现。测试直接读取 `/home/tat512/share/PDEdata` 的真实文件，每个方程取第一个样本；计算使用 `float64`，并比较两项目的 interior residual tensor。

## 实际参数与布局检查

- Heat、wave、advection-diffusion、steady heat 的 train/test 文件均为 `T=1`；随机系数从各自样本 dataset 读取，不能从 split 推断。
- 当前 reaction-diffusion train/test 文件均为 `Du=0.002,Dv=0.004,k=0.003,T=1`，网格为 `[-1,1]^2` 单元中心。legacy profile 是另一套数据，不得按 train/test 名称自动套用。
- NS test attrs 中 `dt=0.0001` 是内部求解步长；`t=[0.1,...,1.0]`，与 `w0` 拼接后的 11 帧差分间隔为 `0.1`。旧 train 文件没有 attrs，但保存的 `t` 同样给出快照时间；PDE 固定 `nu=0.001,T=1`。
- SWE test group attrs 给出 `g=1,T=1,x_range=y_range=[-2.5,2.5]`；旧 train group 没有 `T` root attr，但 `grid/t=[0,0.1,...,1]` 且 group 中保存 `g` 和空间范围。
- Burgers 真实 MAT header 为 `input=(10000,128)`、`output=(10000,128,128)`；128 帧包含初值，故 `dt=1/127`，周期空间 `dx=1/128`。
- 已验证的旧 Wave 文件 endpoint 为 `[u,v]` 两通道，但 `full_trajectory=(N,1,11,128,128)` 只保存位移；当前生成器已改为保存 `[N,2,T,H,W]` 的完整 `[u,v]` 轨迹。旧文件用于 `near_endpoint_temporal` 时，loader 按同一常系数谱解重建近端速度，再仅保留稀疏观测值。

## 静态方程

| PDE | 实际文件 | FM4PDE RMS | baseline RMS | tensor RMS 差 |
|---|---|---:|---:|---:|
| Darcy | `darcy_test_10000-128-128.mat` | `1.325e-12` | `1.325e-12` | `0` |
| Poisson | `poisson_1000-128-128_test.mat` | `4.340e-14` | `4.340e-14` | `0` |
| Helmholtz | `helmholtz_1000-128-128_test.mat` | `3.562e-14` | `3.562e-14` | `0` |
| Steady heat | `steady_heat_conduction_test_10000-128-128.h5` | `4.221e-1` | `4.221e-1` | `0` |

Darcy、Poisson、Helmholtz 已达到生成器离散系统的浮点精度。Steady heat 文件把解保存为 `float32`；生成时、写盘前记录的 `residual_norm` 为 `9.442e-4`，但对写盘后的 `float32` 温度重新应用含 `1/h²` 的算子得到 `4.221e-1`。两实现完全一致，这一差值来自数据量化而不是 residual 公式错误。另按生成器 `build_linear_system()` 的 bottom → sides → top-interior 行优先级检查了原始（不除以网格间距）边界方程，实际样本 boundary RMS 为 `0`。

## 时变方程

`full` 使用全部保存时间点的内部中心差分；`endpoint` 只使用初值和终值的 midpoint/secant 近似。

| PDE | full RMS | endpoint RMS | 两实现 full tensor RMS 差 | 两实现 endpoint tensor RMS 差 |
|---|---:|---:|---:|---:|
| Burgers | `2.196e-5` | 不适用 | `0` | 不适用 |
| Heat | `2.906e-1` | `1.138e1` | `0` | `0` |
| Wave（位移二阶式） | `1.928e2` | `1.299e2` | `0` | `0` |
| Advection-diffusion | `1.918` | `1.915e1` | `2.153e-14` | `1.792e-14` |
| Reaction-diffusion | `1.182e-1` | `1.728` | `5.670e-17` | `9.850e-17` |
| Shallow water | `1.093` | `1.153` | `2.449e-16` | `9.560e-17` |
| NS vorticity | `8.856e-4` | `1.534e-2` | `1.001e-17` | `1.014e-17` |

较大的绝对 RMS（尤其 wave）主要反映只保存 11 帧时中心时间差分的截断误差；它不是把内部求解步长误当快照步长造成的。修正后的两个项目在相同变量、参数、网格与时间模式下给出相同 residual tensor。

## 结论

- 静态方程已按实际生成器离散算子实现；Darcy 不再是存储网格代理，Helmholtz 不再额外施加与生成器矛盾的标准零 Dirichlet loss。
- Burgers 的 `128` 帧、`dt=1/127`、`dx=1/128`、`nu=0.01` 已得到真实数据验证。
- 所有其余时变方程的底层诊断均能运行 full-trajectory 与 endpoint-only 两条路径；full 模式从保存的时间坐标计算快照间隔。FM4PDE 端点生成模型不会把真实 full trajectory 用于采样 guidance 或生成结果 PDE 指标。唯一例外是显式选择的 `near_endpoint_temporal`：它允许六类端点时变 PDE 额外读取 `q(dt)`、`q(T-dt)` 的稀疏真实观测，但 `q0/qT` 仍必须来自模型输出。
- 训练/test 参数从真实文件逐样本/逐 group 读取；只有生成器不保存的固定常数才使用明确的 generator-profile fallback。

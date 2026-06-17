# 偏微分方程 (PDE) 数据生成技术总结报告

## 1. 总体概述：高斯随机场 (GRF) 作为核心随机源

在科学机器学习（SciML）与算子学习（如 Fourier Neural Operator, FNO）的数据集构建中，**高斯随机场 (Gaussian Random Field, GRF)** 是生成多样化、连续且可控的系统初始条件、边界条件或系数场的核心工具。本数据集的所有 PDE 源项或系数项均从定义在 $\Omega = [0,1]^2$ 上的零均值高斯随机场中采样。

### 协方差算子与平滑度控制
GRF 的协方差算子 $C$ 定义为：
$$C = (-\Delta + \tau^2 I)^{-\alpha}$$

其中：
* $\Delta$ 是拉普拉斯算子，根据具体的 PDE 问题施加**齐次 Neumann 边界条件**（如 Darcy Flow、Poisson、Helmholtz）或**周期边界条件**（如 Navier-Stokes）。
* 参数 $\alpha$ 控制场的**平滑度（Smoothness）**，$\alpha$ 越大，生成的场越光滑。
* 参数 $\tau$ 控制系统的**相关长度（Correlation Length）**，$\tau$ 越大，波长越短，空间相关性衰减越快。

### 离散化与采样方法
在数值实现中，利用 **Karhunen-Loève (KL) 展开** 在傅里叶基（周期边界）或余弦基（Neumann 边界）上进行谱分解。其对应的特征值为：
$$\lambda_{k_1,k_2} = \tau^{\alpha-1} \left( \pi^2(k_1^2 + k_2^2) + \tau^2 \right)^{-\alpha/2}$$

**采样步骤**：
1. 在频域/波谱空间生成独立同分布的标准正态分布随机变量 $\xi_{k_1,k_2} \sim \mathcal{N}(0, 1)$。
2. 将特征值的平方根与随机变量相乘：$\hat{f}_{k_1,k_2} = \sqrt{\lambda_{k_1,k_2}} \cdot \xi_{k_1,k_2}$。
3. 执行**逆离散余弦变换 (IDCT)** 或 **逆快速傅里叶变换 (IFFT)** 将其投影回物理空间。
4. *注：对于 Navier-Stokes 方程，引入了 `GRF_zero` 变体，通过在物理空间额外乘以窗函数，使边界平滑地衰减至零。*

---

## 2. 各偏微分方程 (PDE) 数据生成详解

### 2.1 Darcy Flow (达西流 - 稳态)
* **数学形式**：
    $$-\nabla \cdot \big( a(\mathbf{x}) \nabla p(\mathbf{x}) \big) = f(\mathbf{x}), \quad \mathbf{x} \in \Omega = [0,1]^2$$
    其中源项为常数流：$f(\mathbf{x}) \equiv 1$。
* **系数场 $a(\mathbf{x})$ 类型**：
    * **Log-normal 连续型**：$a(\mathbf{x}) = \exp(\text{GRF})$，保证渗透率始终为正。
    * **Thresholded 分段常数型**：
        $$a(\mathbf{x}) = \begin{cases} 12, & \text{当 } \text{GRF}(\mathbf{x}) \geq 0 \\ 4, & \text{当 } \text{GRF}(\mathbf{x}) < 0 \end{cases}$$
* **边界条件**：齐次 Dirichlet 边界条件，$p|_{\partial\Omega} = 0$（在数值解法中对压力矩阵 $P$ 边缘补零）。
* **数值解法**：采用**有限体积法 (FVM)** 与标准五点网格（5-point stencil）。导热/扩散系数在相邻网格点之间采用**调和平均**或**算术平均**进行离散。
* **数据结构**：包含 2 个通道：`[a, p]`（输入系数场 $a$，输出压力场 $p$）。

### 2.2 Poisson Equations (泊松方程 - 稳态)
* **数学形式**：
    $$-\Delta \phi(\mathbf{x}) = f(\mathbf{x}), \quad \mathbf{x} \in \Omega = [0,1]^2$$
* **源项 $f(\mathbf{x})$**：直接从 GRF 采样，参数设定为 $\alpha=3, \tau=4$。
* **边界条件**：齐次 Dirichlet 边界条件，$\phi|_{\partial\Omega} = 0$。
* **数值解法**：采用二阶精度的**五点有限差分法 (FDM)**。设网格间距 $h = \frac{1}{S-1}$，内部点离散格式为：
    $$\frac{-4\phi_{i,j} + \phi_{i-1,j} + \phi_{i+1,j} + \phi_{i,j-1} + \phi_{i,j+1}}{h^2} = f_{i,j}$$
    边界点强制令 $\phi = 0$。构造完线性方程组 $A\phi = f$ 后，在 MATLAB 中直接使用反斜杠 `\` 算子高效求解。
* **数据结构**：包含 2 个通道：`[f, phi]`（输入源项 $f$，输出势场 $\phi$）。

### 2.3 Helmholtz Equations (亥姆霍兹方程 - 稳态非齐次)
* **数学形式**：
    $$(-\Delta - k^2 I) \psi(\mathbf{x}) = f(\mathbf{x}), \quad \mathbf{x} \in \Omega = [0,1]^2$$
* **参数设定**：源项 $f(\mathbf{x}) \sim \text{GRF}(\alpha=3, \tau=4)$；波数 $k$ 默认取固定值 $1$。
* **边界条件**：齐次 Dirichlet 边界条件，$\psi|_{\partial\Omega} = 0$。
* **数值解法**：利用 **Kronecker 积** 构造二维拉普拉斯矩阵。令 $L$ 为一维带 Dirichlet 边界的二阶微分矩阵，则二维全矩阵为：
    $$L_{\text{full}} = (I \otimes L) + (L \otimes I)$$
    最终的系统矩阵为 $A = L_{\text{full}} + k^2 I$。在构造矩阵 $L$ 时，通过将边界行（第一行和最后一行）设置为单位向量，直接将 Dirichlet 条件内置。
* **数据结构**：包含 2 个通道：`[f, psi]`（输入源项 $f$，输出波场 $\psi$）。

### 2.4 Navier-Stokes Equations (纳维-斯托克斯方程 - 时变 2D 周期)
* **数学形式 (涡量-流函数形式)**：
    $$\frac{\partial \omega}{\partial t} + \mathbf{u} \cdot \nabla \omega = \nu \Delta \omega + f(\mathbf{x})$$
    * $\omega = \nabla \times \mathbf{u}$ 为标量涡量场。
    * 速度场由流函数 $\psi$ 恢复：$\mathbf{u} = (\partial_y \psi, -\partial_x \psi)$。
    * 流函数与涡量满足泊松关系：$\omega = -\Delta \psi$。
    * 运动粘性系数 $\nu = 1 \times 10^{-3}$。
    * 固定强迫项（背景驱动）：$f(\mathbf{x}) = 0.1 \left( \sin(2\pi(x+y)) + \cos(2\pi(x+y)) \right)$。
* **边界条件**：双向完全周期边界条件（隐含于傅里叶谱方法中），物理域为环面 (Torus) $[0,1]^2$。
* **初值条件**：初始涡量 $\omega_0(\mathbf{x}) \sim \text{GRF}(\alpha=2.5, \tau=7)$，结合周期基底。
* **数值解法**：采用**傅里叶伪谱方法 (Fourier Pseudospectral Method)** 进行空间离散，时间推进采用 **Crank-Nicolson (CN) 隐式积分与显式 Euler 混合格式**，并引入 **2/3 去混叠法则 (Dealiasing Mask)** 消除非线性对流项的高频误差。
    * *线性粘性项*：$\frac{\hat{\omega}^{n+1} - \hat{\omega}^n}{\Delta t} = -\nu \frac{|\mathbf{k}|^2}{2}(\hat{\omega}^{n+1} + \hat{\omega}^n)$
    * *流函数求解*：在谱空间中为简单的代数除法 $\hat{\psi} = \frac{\hat{\omega}}{4\pi^2 |\mathbf{k}|^2}$。
    * 时间步长 $\Delta t = 1 \times 10^{-4}$，总模拟时间 $T = 1.0$，全生命周期均匀记录 10 个快照 (Snapshots)。
* **数据结构**：共 11 个通道：`[w0, w_t1, ..., w_t10]`（1帧初始状态 + 10帧时间序列快照），每个通道分辨率为 $128 \times 128$。

### 2.5 Burgers' Equation (伯格斯方程 - 时变 1D)
* **数学形式**：
    $$\frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial}{\partial x}(u^2) = \nu \frac{\partial^2 u}{\partial x^2}, \quad x \in [0,1]$$
    其中粘性系数 $\nu = 0.01$。
* **边界条件**：周期边界条件（GRF 采样基于 "periodic" 模式，利用 Chebfun 的 `'trig'` 三角函数展开）。
* **初值条件**：$u_0(x)$ 从一维高斯随机场采样，参数为 $\gamma=2.5, \tau=7, \sigma=7^2$。
* **数值解法**：利用基于 MATLAB 的 **Chebfun 库中的 `spin` 求解器**（用于刚性 PDE 的指数时间差分法积分器），空间采用傅里叶谱方法离散。
* **数据结构**：空间分辨率 $S = 128$，时间离散为 128 个步长（$t \in [0,1]$），输出表现为 $128 \times 128$ 的时空演化矩阵。

### 2.6 Reaction-Diffusion Equations (反应扩散方程 - 时变 2D)
* **数学形式 (FitzHugh-Nagumo 类型双组分系统)**：
    $$\begin{aligned}
    \frac{\partial u}{\partial t} &= D_u \Delta u + (u - u^3 - k - v) \\
    \frac{\partial v}{\partial t} &= D_v \Delta v + (u - v)
    \end{aligned}$$
    物理域为 $\Omega = [-1, 1]^2$。
* **控制参数 (分布差异)**：
    * **训练集**：$D_u = 1\times 10^{-3}, D_v = 5\times 10^{-3}, k = 5\times 10^{-3}$
    * **测试集**：$D_u = 2\times 10^{-3}, D_v = 4\times 10^{-3}, k = 3\times 10^{-3}$ （通过轻微扰动系统参数来评估神经网络算子的泛化性能）。
* **边界条件**：齐次 Neumann 边界条件（$\partial_n u = 0, \partial_n v = 0$）。在数值实现中，边界网格点上的扩散通量（Diffusive Flux）做减半处理。
* **初值条件**：$u_0(\mathbf{x}), v_0(\mathbf{x})$ 独立从标准正态分布 $\mathcal{N}(0, I)$ 中直接采样。
* **数值解法**：空间离散采用**有限体积法 (FVM, Cell-centered 网格)**，Laplacian 算子由稀疏五点 stencil 表达；时间推进采用 `scipy.integrate.solve_ivp` 内置的 **RK45 自适应变步长积分器**。
* **数据时间与结构**：总演化时间 $T=5$，全过程记录 100 帧（`tdim=100`）。最终数据截取中间态与终态进行配对：输入为第 50 帧（或100帧）作为初值状态，输出为最后一帧。数据格式为 `[u(t_mid), v(t_mid), u(t_final), v(t_final)]` 共 4 个通道，各通道分辨率为 $128 \times 128$。

### 2.7 Shallow Water Equations (浅水方程 - 时变 2D 非线性)
* **数学形式 (双曲守恒律形式)**：
    $$\begin{aligned}
    \frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} + \frac{\partial (hv)}{\partial y} &= 0 \\
    \frac{\partial (hu)}{\partial t} + \frac{\partial}{\partial x}\left(hu^2 + \frac{1}{2}gh^2\right) + \frac{\partial}{\partial y}(huv) &= 0 \\
    \frac{\partial (hv)}{\partial t} + \frac{\partial}{\partial x}(huv) + \frac{\partial}{\partial y}\left(hv^2 + \frac{1}{2}gh^2\right) &= 0
    \end{aligned}$$
    其中 $h$ 为水流深度，$(hu, hv)$ 为 x 和 y 方向的动量，$g = 1.0$ 为重力加速度。
* **初值条件 (径向溃坝模拟 - Radial Dam Break)**：
    $$h_0(x, y) = \begin{cases} h_{\text{inner}}, & \text{当 } r \leq R_{\text{dam}} \\ 1.0, & \text{当 } r > R_{\text{dam}} \end{cases}, \quad hu_0 = 0, \quad hv_0 = 0$$
    其中径向距离 $r = \sqrt{x^2 + y^2}$。大坝半径 $R_{\text{dam}} \sim U(0.4, 0.8)$，内部初始水深 $h_{\text{inner}} \sim U(2, 3)$（测试集参数随机采样）。
* **边界条件**：零阶外推边界条件（Extrapolation / Zero-order Neumann），即允许波无反射地穿过边界。
* **数值解法**：使用高级非线性守恒律求解库 **Clawpack/PyClaw**。采用**有限体积法 (FVM)** 结合 **Roe 近似黎曼求解器 (Roe Riemann Solver)**（内置熵修正机制），并使用 **MC (Monotonized Central) TVD 全变差递减限制器** 抑制激波处的数值振荡。计算域 $\Omega = [-2.5, 2.5]^2$，网格规模 $128 \times 128$。
* **数据时间与结构**：总演化时间 $T = 1.0$，记录 10 个时间步。最终提取初态与终态构成 6 通道结构：`[h0, hu0, hv0, h_final, hu_final, hv_final]`，分辨率 $128 \times 128$。

### 2.8 Heat Equation (热方程 - 时变 2D)
* **数学形式**：
    $$u_t = \alpha \Delta u, \quad (x,y)\in[0,1]^2$$
* **边界条件**：默认周期边界条件，使用 Fourier 谱方法精确推进；`--bc neumann` 可切换到 DCT 余弦基的齐次 Neumann 谱解。
* **初值与参数**：$u_0$ 从平滑周期 GRF 采样；$\alpha \sim U(5\times10^{-4}, 5\times10^{-3})$。训练与测试使用独立 seed 区间。
* **数值解法**：
    $$\hat{u}(t,k)=\exp(-\alpha |k|^2t)\hat{u}_0(k)$$
* **数据结构**：`input_data=[u0]`，`output_data=[uT]`，`data=[u0,uT]`，总通道数 2；可选 `full_trajectory=[u(t_0),...,u(t_K)]`。

### 2.9 Wave Equation (波方程 - 时变 2D)
* **数学形式**：
    $$u_{tt}=c(x,y)^2\Delta u,\quad u(x,y,0)=u_0,\quad u_t(x,y,0)=v_0$$
* **边界条件**：默认周期边界条件。
* **初值与参数**：$u_0$ 从平滑 GRF 采样；默认 $v_0=0$，可用 `--random-v0` 开启随机初速度；默认常数波速 $c=1$，可用 `--variable-c` 生成平滑随机波速场并用 velocity-Verlet 有限差分推进。
* **默认数值解法**：
    $$\hat{u}(t,k)=\hat{u}_0(k)\cos(c|k|t)+\hat{v}_0(k)\frac{\sin(c|k|t)}{c|k|}$$
    零频模式使用 $\hat{u}_0+t\hat{v}_0$ 单独处理。
* **数据结构**：默认 `input_data=[u0,v0]`，`output_data=[uT,vT]`，`data=[u0,v0,uT,vT]`，总通道数 4，保证 FM4PDE 按半通道拆分时输入/输出通道数相等。

### 2.10 Advection-Diffusion Equation (对流扩散方程 - 时变 2D)
* **数学形式**：
    $$u_t+b_xu_x+b_yu_y=\kappa\Delta u,\quad (x,y)\in[0,1]^2$$
* **边界条件**：周期边界条件。
* **初值与参数**：$u_0$ 从平滑周期 GRF 采样；$b_x,b_y\sim U(-1,1)$；$\kappa\sim U(5\times10^{-4},5\times10^{-3})$。参数场以常数场通道保存。
* **数值解法**：
    $$\hat{u}(t,k)=\exp(-(\kappa |k|^2+i(b_xk_x+b_yk_y))t)\hat{u}_0(k)$$
* **数据结构**：`input_data=[u0,b_x,b_y,kappa]`，`output_data=[uT,b_x,b_y,kappa]`，`data=[u0,b_x,b_y,kappa,uT,b_x,b_y,kappa]`，总通道数 8。

### 2.11 Python Future PDE 生成器接口
新增生成代码位于 `data/DataGen/python/`：
* `generate_future_pdes.py`：统一入口，支持 `--pde {heat,wave,advection_diffusion,all}`。
* `generate_heat.py`、`generate_wave.py`、`generate_advection_diffusion.py`：各 PDE 数值生成。
* `common.py`：HDF5 预分配、chunked 写入、seed 管理、metadata、no-leakage check。

默认完整命令：
```bash
python data/DataGen/python/generate_future_pdes.py \
  --pde all \
  --out-root <DATA_ROOT> \
  --resolution 128 \
  --n-train 50000 \
  --n-test 1000 \
  --train-shards 5 \
  --n-time 11 \
  --overwrite
```

CPU quick-test 命令：
```bash
python data/DataGen/python/generate_future_pdes.py \
  --pde all \
  --out-root /tmp/fm4pde_future_pdes \
  --resolution 16 \
  --n-train 8 \
  --n-test 4 \
  --train-shards 2 \
  --n-time 5 \
  --quick-test \
  --overwrite
python -m compileall -q data
```

HDF5 扁平 key：
* `input_data: [N,C_in,H,W]`
* `output_data: [N,C_out,H,W]`
* `data: [N,C_in+C_out,H,W]`
* `full_trajectory: [N,1,T,H,W]`（默认保存，可用 `--no-trajectory` 关闭）
* `x,y,t,sample_id,sample_seed` 以及 PDE 参数数组

文件命名：
* `heat/heat_10000-128-128_1.h5` ... `heat/heat_10000-128-128_5.h5`
* `heat/heat_test_1000-128-128.h5`
* `wave/wave_10000-128-128_1.h5` ... `wave/wave_test_1000-128-128.h5`
* `advection_diffusion/advection_diffusion_10000-128-128_1.h5` ... `advection_diffusion/advection_diffusion_test_1000-128-128.h5`

无泄漏设计：
* train 默认 `base_seed=0`，test 默认 `base_seed=10000000`。
* 每个样本使用 `sample_seed=base_seed+global_sample_id`，参数随机化也由该样本 RNG 产生。
* train/test 分开生成，不从一个 51000 样本大数组切分。
* shard 只改变文件位置，不改变 `global_sample_id` 或随机性；重新生成第 3 个 shard 不影响其他 shard。
* 每个 PDE 目录写入 `no_leakage_check.json`，检查 seed 不重合、抽样 input hash 不重复、train/test 文件名不覆盖。

FM4PDE 读取：
* `data/load.py` 新增 `heat=7`、`wave=8`、`advection_diffusion=9` 标签，训练读取 `<DATA_ROOT>/<pde>/` 下 train shards。
* `configs/heat.yaml`、`configs/wave.yaml`、`configs/advection_diffusion.yaml` 使用 `loadby: future_h5`，采样侧从 test HDF5 的 `input_data/output_data` 读取单样本。
* 当前仓库尚未提供这三个 PDE 的预训练模型和 PDE residual guidance，因此新增 YAML 默认 `guide: False`、`zeta_pde: 0`，仅保证数据接口与偶数通道拆分兼容。

与 PDEBench、iFNO、RecFNO 的关系：仅参考其 HDF5/轨迹组织、operator-learning 输入输出配对和热/波场数据形态，不依赖其外部数据文件或大型求解依赖。

---

## 3. 数据归一化策略 (`transform.py`)

为了消除不同物理量量纲和数值范围的差异，提高神经网络训练的收敛速度，对所有 PDE 的特征数据进行了 **Min-Max 归一化**，将其缩放到 $[0, 1]$ 区间内：

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min} + \varepsilon}$$

**核心实现机制**：
1. **独立归一化**：在训练模式下，输入通道与输出通道分别独立计算统计量（例如前 $C/2$ 个通道作为 Input 整体，后 $C/2$ 个通道作为 Output 整体）。
2. **空间与样本聚合**：最小值 $x_{\min}$ 和最大值 $x_{\max}$ 是在**所有训练样本**的**整个空间维度（像素点）**上共同聚合计算得出的标量。
3. **数值稳定性**：分母中引入微小的偏置项 $\varepsilon = 1\times 10^{-5}$，防止除零崩溃。

---

## 4. 数据结构汇总矩阵

| PDE 问题名称 | 物理类型 | 总通道数 | 输入通道组成 | 输出通道组成 | 空间分辨率 | 单一任务样本总量 |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **Darcy Flow** | 稳态 | 2 | $a(\mathbf{x})$ (扩散系数场) | $p(\mathbf{x})$ (压力场) | $128 \times 128$ | 50,000 |
| **Poisson** | 稳态 | 2 | $f(\mathbf{x})$ (非齐次源项) | $\phi(\mathbf{x})$ (势场) | $128 \times 128$ | 50,000 |
| **Helmholtz** | 稳态 | 2 | $f(\mathbf{x})$ (非齐次源项) | $\psi(\mathbf{x})$ (波场) | $128 \times 128$ | 50,000 |
| **Navier-Stokes** | 时变 | 11 | $\omega_0(\mathbf{x})$ (初始涡量) | $\omega(t_1) \dots \omega(t_{10})$ (演化序列) | $128 \times 128$ | 50,000 |
| **Burgers' (1D)** | 时变 | 1 | - (时空一体表示) | - (时空一体表示) | $128 \times 128$ | 50,000 |
| **Reaction-Diffusion** | 时变 | 4 | $u_0, v_0$ (双组分初值) | $u_T, v_T$ (终态双组分) | $128 \times 128$ | 50,000 |
| **Shallow Water** | 时变 | 6 | $h_0, hu_0, hv_0$ (初始状态) | $h_T, hu_T, hv_T$ (终态守恒量) | $128 \times 128$ | 50,000 |
| **Heat** | 时变 | 2 | $u_0$ | $u_T$ | $128 \times 128$ | 50,000 |
| **Wave** | 时变 | 4 | $u_0, v_0$ | $u_T, v_T$ | $128 \times 128$ | 50,000 |
| **Advection-Diffusion** | 时变 | 8 | $u_0,b_x,b_y,\kappa$ | $u_T,b_x,b_y,\kappa$ | $128 \times 128$ | 50,000 |

---

## 5. 总结

该数据集的设计展现了面向算子学习（Operator Learning）的工业级标准：
1. **随机源的数理一致性**：利用无限维 Hilbert 空间中的高斯随机场（GRF）并结合 KL 展开，提供了具备严格平滑度控制的无限随机输入，能极好地模拟现实中的复杂不确定性。
2. **数值求解的严谨性**：针对椭圆型（Poisson/Helmholtz）、抛物型（Reaction-Diffusion/Burgers）和双曲型激波（Shallow Water）的不同数理特性，分别选用了谱方法、有限差分法、有限体积法以及成熟的 Riemann 求解器，保证了基准标签（Ground Truth）的高精度。
3. **标准化接口**：统一将各种维度的输入输出对整合成固定网格分辨率 ($128 \times 128$) 和多通道张量，并辅以规范的全局 Min-Max 数据变换，极其利于诸如 FNO, U-Net, DeepONet 等先进神经算子架构的无缝调用。

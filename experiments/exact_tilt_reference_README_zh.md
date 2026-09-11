# FM4PDE：可实现的精确目标引导参考算法

## 文件与适用范围

- `fm4pde_exact_guidance_algorithms.py`：PyTorch 参考实现。
- `fm4pde_guidance_test_report.json`：小规模测试的实际运行输出。

本实现没有加载 FM4PDE 仓库的真实 checkpoint，也没有完成 128×128 PDE 基准实验。它实现了可接入现有模型的通用函数，并用线性高斯及一维潜变量非高斯例子核对计算。

两种实现的精确性不同：

1. `linear_gaussian_endpoint`：在完整已知的线性高斯先验、线性观测和有限二次能量下，计算精确模型的条件端点均值和完整倾斜速度。剩余误差是线性求解容差、浮点误差，以及使用它时的外层数值积分误差。
2. `pcn_endpoint_mean`：使用 pCN–Metropolis 内层链计算条件端点积分。接受率保证正确的不变目标，而不是有限步已到平稳分布。适当遍历性、可积性条件下的长链均值一致；有限运行有混合误差和 Monte Carlo 误差。

## 1. 通用接口：确定性生成器 G 和能量 energy

设潜变量 `z ~ N(0,I)`，生成标准化端点 `r=G(z)`。G 可以是冻结 FM 模型的完整、无引导、确定性 ODE rollout，也可以是已知数据生成器。不同 G 对应不同先验。

```python
import torch
from fm4pde_exact_guidance_algorithms import (
    make_ode_generator, pcn_endpoint_mean, moment_guided_sample,
)

# model 必须由你的 checkpoint 加载代码提供。
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# 按仓库实际 forward 签名修改这个适配器；已知物理参数在闭包中传入。
def velocity(x, t_vector):
    return model(x, t_vector)

# 注意：这是完整的无引导生成，不是 x+(1-t)*v(x,t)。
G = make_ode_generator(velocity, steps=100, method="heun")
```

`G` 输入 `[chains, *latent_shape]`，返回 `[chains, *endpoint_shape]`。一次 pCN 调用中的所有链针对同一个物理观测实例。

`energy(r)` 返回 `[chains]`，即每条链一个能量。必须使用标准化端点反变换后的物理量计算原稿观测和物理残差；不要对链维度平均。原稿各残差分量自己的 MSE 归一化应保留。

下面给出双标量场的能量适配示例。`train_mean/train_std`、观测填充数组、mask、固定权重来自现有实验配置，`physical_components` 来自现有离散残差实现。

```python
# train_mean/train_std: [2]，仅来自训练集。
# ya_full/yu_full: [H,W]，只有观测位置有观测值，其余位置填0。
# mask_a/mask_u: [H,W]，0/1。
# physical_components(z) 返回 [(weight, residual), ...]；
# residual: [chains, ...]，weight 已包含 zeta_pde 与 BC/IC 等分量系数。
# 所有张量应放在 G 输出所在设备。
def energy(r):
    z = (r.double() * train_std.double().view(1,2,1,1)
         + train_mean.double().view(1,2,1,1))
    a, u = z[:,0], z[:,1]
    ans = torch.zeros(z.shape[0], dtype=torch.float64, device=z.device)
    if mask_a.sum() > 0:
        ea = (a-ya_full.double()) * mask_a
        ans += zeta_a * ea.flatten(1).square().sum(1) / mask_a.sum()
    if mask_u.sum() > 0:
        eu = (u-yu_full.double()) * mask_u
        ans += zeta_u * eu.flatten(1).square().sum(1) / mask_u.sum()
    for weight, residual in physical_components(z):
        ans += weight * residual.double().flatten(1).square().mean(1)
    return ans
```

这里的 `energy` 是固定的终端能量；不是随 flow time 突然开关的损失。不同任务只改变观测、mask 及预先固定的终端权重。

## 2. 内层条件积分怎样计算

给定一个当前标准化流状态 x 和 t<1，潜变量目标相对 N(0,I) 的密度为

    exp(-energy(G(z)) - ||x-t*G(z)||² / (2*(1-t)²)).

算法逐链提议

    z_new = sqrt(1-beta²)*z + beta*normal_noise

用 `min(1, exp(V_old-V_new))` 接受。高斯先验项已经由 pCN 提议可逆性抵消，不要再次加入先验范数差。拒绝后必须保留旧状态并把它计入样本均值。

```python
# 示例数值仅为调用格式，不是已验证的 PDE 收敛预算。
z0 = torch.randn(4, 2, 128, 128, device=device)
x_t = torch.randn(2, 128, 128, device=device)
result = pcn_endpoint_mean(
    G, energy, z0, x=x_t, t=0.4,
    beta=0.1, warmup=200, keep=400,
)
mean_tilted_endpoint = result.mean
full_tilted_velocity = (result.mean-x_t)/(1-0.4)
print(result.acceptance)
```

桥接高斯核的范数必须对全部标准化端点坐标求 **sum**。它不是可以任意更改成 MSE 的目标项。

输出是该生成先验的独立仿射路径对应的完整倾斜速度。不要再加一次 `v_theta`。若要单独输出能量引导，可用同一 G、`energy=0` 再计算未倾斜条件均值，然后将两个均值之差除以 `(1-t)`。有限训练网络的原始中间边缘未必等于其自身生成端点分布所诱导的独立仿射边缘，因此不能无条件把该均值差加回原网络。

## 3. 外层引导采样

```python
# x0 与 latent z0 必须为独立高斯抽样。
z0 = torch.randn(4, 2, 128, 128, device=device)
x0 = torch.randn(2, 128, 128, device=device)
res = moment_guided_sample(
    G, energy, z0, x0,
    outer_steps=32, stop_t=0.95,
    beta=0.1, first_warmup=500, warmup=100, keep=200,
)
prediction_standardized = res.sample
```

这个外层循环使用 `(conditional_endpoint_mean-x)/(1-t)` 做 Euler 积分，到 `stop_t<1` 后，条件抽取一个端点而不是直接返回条件均值。最后一步在正确中间边缘及精确条件抽样下保持目标；有限内层 MCMC 和外层积分误差仍存在。

`last_local_mean` 是最后流状态下的条件均值，不是整体 `E[r | observations]`。

一般而言 t=0 的内层目标就是完整后验，已经可能很难抽样。因此此算法是高成本参考，并不是一次反传引导的等价成本替代。

## 4. 更经济的条件偏差审计：直接对终端后验做 pCN

只为了比较当前 FM4PDE 条件样本与相同生成先验的目标后验时，不必再套外层流：

```python
res = pcn_endpoint_mean(
    G, energy, z0,
    beta=0.1, warmup=1000, keep=2000,
    # x 和 t 均省略 => 直接针对终端能量倾斜后验
)
posterior_mean_estimate = res.mean
posterior_sample = res.last_endpoints[0]
```

这不是沿当前 flow time 的引导，而是对相同终端目标的 MCMC 参考。仍需用多链、ESS、R-hat、能量和可识别低频等统计量检查混合；预算示例不保证收敛。

## 5. Poisson／固定波数 Helmholtz：矩阵自由的精确高斯引导

使用完整的已知 GRF 和线性离散解算子，构造标准化端点

    r = mu + A*z,  z ~ N(0,I).

`A` 可组合：全分辨率 GRF 生成、线性 PDE 求解、固定仿射标准化的线性部分。`AT` 必须是这一完整复合映射的离散伴随。注意 DCT 源项协方差与 PDE 解的边界条件不同；必须遵循实际生成代码的振幅、边界行、网格和标准化。

将能量写成

    0.5*(B*r-y)^T Rinv (B*r-y).

线性物理残差可堆叠进入 B。仿射偏移吸收进 y。原稿一个 `w/m*||residual||²` 分量对应方差 `m/(2w)`。

代码在每个 t,x 解潜变量上的 SPD 系统：

    Q_t(z) = s²*z + t²*AT(A(z)) + s²*AT(BT(Rinv(B(A(z)))))
    b_t    = t*AT(x-t*mu) + s²*AT(BT(Rinv(y-B(mu))))
    s = 1-t

    zbar = CG(Q_t, b_t)
    mean = mu + A(zbar)
    full_velocity = (mean-x)/s

```python
from fm4pde_exact_guidance_algorithms import linear_gaussian_endpoint
res = linear_gaussian_endpoint(
    x, t, mu, A, AT, B, BT, Rinv, y,
    rtol=1e-10, atol=1e-12, maxiter=2000,
)
v = res.velocity
```

无需形成联合场的巨大稠密协方差，也无需估计条件协方差。CG 不收敛时会抛出异常而不是静默返回“精确”结果；病态系统需要正确预条件和容差研究。真残差小不自动保证解误差小，后者也受条件数影响。

这个高斯先验是数据生成先验，而不是将现有 U-Net 自动变成了精确先验；它主要是解析基准。用训练样本估计协方差、截断 GRF 模态、换边界算子或用单个高斯拟合非高斯先验，都会改变精确性所针对的目标。

当前实现要求 R 为正定的有限软约束。精确无噪声硬条件应另用约束高斯条件化／秩感知线性代数处理，不应直接设置无限权重。

## 6. 成本和不可省略的检查

通用内层 pCN 的每次新提议，都要计算完整 G，而不是只调用一次速度网络。若外层 N 次、P 条链、每次 L 次内层提议、生成器一次花 F 次速度网络评估，则总量约为 `N*P*L*F`，另加初始及终端计算。批量可降延迟，但不消除总计算量。

必须检查：

- G 在相同潜变量下可复现，处于 eval 模式，禁用重新加噪及 dropout；不能把 t 时刻预测均值当作完整 G。
- energy 每链一个数；桥接核对全部标准化坐标求和；物理能量保持原先的分量归一化。
- pCN 拒绝状态不能丢弃；接受率高不等于混合好。
- 不要给完整倾斜速度再叠加 v_theta；不要随意加额外裁剪或开启时刻后继续宣称相同目标精确。
- 多链、延长预算、观测拟合、物理残差、场的低频模态均应检查。内层均值误差在速度中除以 `1-t`，临近端点需要更严格的内层精度。
- 先验学习误差不会由精确条件化自动消失。

## 7. 运行测试

```bash
python fm4pde_exact_guidance_algorithms.py --self-test --report local_report.json
```

测试覆盖：标准高斯先验的后验均值／方差；给定中间状态的条件均值／方差；非高斯端点 `(z,z²)` 与一维积分对照；矩阵自由 CG 与独立稠密高斯求解对照；ODE 适配器和外层循环的基本运行检查。

## 理论来源与实现归属

- 用户稿件 Section 2.4、Proposition 4：终端能量倾斜与精确速度恒等式。
- Feng et al., On the Guidance of Flow Matching, ICML 2025, PMLR 267:16993–17029, arXiv:2502.02150：Flow Matching 引导与 Monte Carlo／训练型近似框架。
- Cotter, Roberts, Stuart and White, MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster, Statistical Science 28(3):424–446 (2013), DOI:10.1214/13-STS421：pCN 提议和接受率。

本文件中的潜变量桥接势能、pCN 内层与 FM 端点条件均值的组合，以及线性高斯潜变量 CG 写法，是针对当前问题给出的参考实现；不是声称上述文献原封不动提供了同一 PDE 代码。

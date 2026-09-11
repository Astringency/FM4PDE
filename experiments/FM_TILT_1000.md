# FM4PDE 学习先验的 tilt 消融

本实现直接加载 FM4PDE 检查点、推理权重和训练归一化器。Poisson 的第一组正式评估使用主实验 `both / ID / random500` 的全部 1000 个输入（0–999）；两场的实际共享观测掩码从历史预测读取并核验。开发试跑仅使用 1100–1103，不用正式评估误差选择提议参数。

## 数学对象

记 `G_100` 为原检查点的 100 步无引导 Euler 生成器，`D` 为物理量反归一化，`z ~ N(0,I)`。每个输入有独立的目标：

```
pi(z | observations) ∝ exp[-||z||²/2 - L(D(G_100(z)); observations)]
L = zeta_a MSE_observed_a + zeta_u MSE_observed_u
    + zeta_pde (MSE_interior + bc_weight MSE_boundary + ...)
```

`L` 调用原 PDE/边界残差与分项归约代码，逐输入计算，不因批量或链数而除以额外因子。原采样器的时间门控和梯度裁剪不属于固定终端能量。此实现目前绑定无噪声主实验记录；带噪声观测需要另行冻结实际噪声张量，不能默认为干净真值。

本消融的主实现是直接终端 tilt MCMC，可检验同一模型先验在该 tilt 目标下的重建表现。它**不是**逐时刻条件矩嵌套流，也不能单独归因于替换了某一项速度。如果评估嵌套流，应另外加入桥条件项并计算整个 `(m_t-x)/(1-t)`，不能把这里的终端链叫作已实现的嵌套流。

## 接受校正与求导

提议为 `z' = rho*z - d*g(z) + beta*epsilon`，其中 `rho=sqrt(1-beta²)`、`d=1-rho`。`g` 可取较短 Euler 积分的能量梯度；它只影响提议。接受概率始终包含完整 `G_100` 的目标能量和正、反两方向高斯提议密度。因此梯度近似不替换目标分布，但可能降低接受率、拖慢混合。

逐步反向求导保存 Euler 状态，并逐步重算网络的向量雅可比积。这是离散生成器的求导，不用近似的连续 ODE 伴随；峰值网络激活显存约为一步。独立测试验证接受比、pCN 特例、非线性生成器的数值积分后验矩，以及逐步反向求导与完整计算图的一致性。

MCMC 的“目标正确”不等于有限链已经收敛。预热后冻结步长，拒绝状态保留；保存每个输入的至少 4 条独立链、能量和带符号低频模态轨迹。必要检查为 rank/folded split R-hat < 1.01、监测量 ESS >= 100；通过也不证明整个高维后验已充分混合。

## 评估口径

- 主指标：每个输入固定取第 0 条链的最终状态，分别报告 a/u 相对 L2，与同一输入的历史 FM4PDE 单次重建配对。
- 链均值作为另一种点估计单列，不能把多样本平均带来的改善全部称为引导改善。
- 同时保存带符号低频误差、功率比、对齐、耗时和混合诊断。1000 指独立 PDE 测试输入，链数和保存次数不增加测试样本数。
- 汇总程序要求 1000 个唯一输入全部完成，逐个核对真值/掩码/预测文件和重算误差，拒绝把部分结果写成 1000 样本结论。诊断失败的输入保留在总体中，不挑选收敛或改善的子集。
- 置信区间以输入为重采样单位；两项主要场误差使用各 97.5% 配对区间。谱指标作描述性诊断。

## 存储与运行

唯一长期主目录：server197

```
/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ablations/fm_tilt_1000_20260912/
```

`inputs/` 保存核验后的 1000 输入及基线；`pilot_*` 和 `benchmark/` 保存开发试跑；正式完整运行有自己的目录；`execution/` 保存 tmux 启动记录、日志和退出码；`sources/` 保存 Git 归档。原采样器、原模型和历史实验不改动。

在隔离检出中运行模块。以下是接口，不代表正式 1000 输入运行已经完成：

```
python -m experiments.prepare_fm_tilt_1000 --record ABS_MAIN_CELL_JSON --output ABS_OUTPUT
python -m experiments.run_fm_tilt --output ABS_OUTPUT --name formal --chains 4 --warmup W --keep K --force-steps S --beta B --batch-inputs N
python -m experiments.report_fm_tilt_1000 --output ABS_OUTPUT --name formal
```

不传 `--ids` 即使用冻结记录中的全部 1000 个输入。`--ids` 仅用于显式子集或开发试跑，最终完整汇总会检查 1000 个 ID。每个批次独立保存状态与随机数状态，可在相同协议下继续；改变科学参数须另开运行目录。

方法背景：[Cotter et al., MCMC Methods for Functions (2013)](https://arxiv.org/abs/1202.0709)。本提议的接受比还通过直接高斯密度公式和非线性积分基准独立验证。

# 稿件采样实现与清理说明

本次清理统一了采样公式、残差和配置入口。现有训练权重与历史实验输出没有重写；历史误差不能直接视为清理后实现的结果。新实验应使用独立输出目录。

## 当前采样规则

- 端点预测为 `x + (1 - t) * v(x, t)`，在 `t = 1` 时精确返回 `x`。
- 确定性提议为 Euler 更新 `x + dt * v(x, t)`。
- 随机提议为 `(1 - t_next) * noise + t_next * endpoint`。
- 确定性引导乘数为 `dt * (1 - safe_t) / safe_t`，其中 `safe_t = max(t, 1e-6)`；首步 `t = 0` 时跳过引导修正。其余步骤采用稿件默认的 `s_k = 1`。
- 随机引导乘数为 `c_zeta * (1 - t)`，主配置中的 `c_zeta` 保持 `0.1`。
- 先求加权分量梯度之和，再对每个样本做一次全局范数裁剪。梯度通过所选损失状态对当前采样状态求导。
- PDE 引导按步骤 `k >= ceil(pde_guidance_start_ratio * N)` 启用，默认比例为 `0.8`。
- 时间网格只保留均匀网格和确定性采样使用的几何网格。31 份主配置均为 100 步；稿件的步数消融仍保留。
- 非有限梯度或更新会报告失败，不再把修正量静默置零。

## 残差变化与保留项

- Darcy 残差直接使用反标准化后的预测物理系数，删除 4/12 投影以及 softplus、clamp、floor 替代分支。独立静态求解器的 `binary` 参数一并删除。
- 数据生成阶段的 GRF 二值化和 OOD 的真实 4/12 系数构造保持不变。
- 浅水通量的分母保护 `max(h, eps)` 保持原样；论文说明由作者另行补充。
- 割线残差继续使用此前更新的端点动力学平均 `0.5 * (G(a) + G(u))`。
- Hermite 残差固定使用 `1/4, 1/2, 3/4` 三个配点，删除额外积分惩罚。
- 近端点方法的稀疏辅助观测、观测掩码及其归一化保留。
- 删除端点模型读取完整真实轨迹计算 PDE 残差的备用分支；Burgers 保留完整**预测**轨迹残差，统一名称为 `full_time_space`。
- Navier–Stokes 固定采用稿件的 2/3 去混叠投影。
- PDE 的物理边界条件和稳态热传导本构关系中的下限保留。

## 删除的 35 个采样配置字段

| 类别 | 字段 |
| --- | --- |
| Darcy 系数变换 | `coef_positive_mode`, `coef_positive_floor` |
| 确定性额外更新规则 | `deterministic_bt_mode`, `deterministic_bt_max_scale`, `deterministic_guidance_coeff`, `deterministic_guidance_start_ratio`, `deterministic_guidance_ramp_ratio`, `deterministic_correction_max_rms`, `deterministic_numerical_guard` |
| 额外积分与端点预测 | `step_method`, `deterministic_endpoint_mode`, `deterministic_rollout_checkpoint` |
| 权重调度与时钟 | `guidance_schedule`, `obs_decay`, `obs_decay_start_ratio`, `polynomial_power`, `cosine_mode`, `pde_guidance_clock`, `pde_guidance_ramp_ratio`, `stochastic_guidance_time` |
| 旧版损失与自动重标定 | `guidance_operator`, `legacy_obs_multiplier`, `obs_guidance_reduction`, `pde_guidance_reduction`, `obs_l2_reference_mse_zeta_a`, `obs_l2_reference_mse_zeta_u` |
| 求导、裁剪与残差区域 | `gradient_target`, `clip_mode`, `pde_residual_region` |
| Hermite 扩展 | `hermite_collocation_times`, `hermite_num_collocation`, `hermite_include_integral_residual`, `hermite_integral_weight` |
| 其他推理分支 | `ns_operator_mode`, `cfg_scale` |

同时删除采样模块中非 CondOT 调度和额外训练目标转换、旧残差模块 `legacy_guidance.py`、未使用的训练评估 `--edm_schedule` 与 CFG 推理混合。正常模型条件输入和内存检查点功能保留。

配置校验、实验入口、日志、汇总和相关绘图均同步调整。旧配置中的已删除字段会被拒绝，不会静默解释成新算法。正文使用的采样阶段、引导位置、时间残差、观测布局／密度／噪声和条件样本平均消融继续保留。

相邻基线仓库的适配器同步移除了 Darcy 4/12 投影，并禁用上游的 3/12 输出投影；反标准化保持不变。该调整记录在新运行的 `physical_postprocessing` 元数据中。

## 验证

在具备 PyTorch、NumPy、SciPy、PyYAML 和项目运行依赖的环境中，从仓库根目录运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest sampling.validate_manuscript_sampling sampling.validate_secant_residual -v
python -m sampling.validate_configs
```

本次验证通过 15 项数值／回归测试，以及 31 份主配置和 6,448 个实验配置组合的解析校验。数值测试包括解析公式、有限差分梯度、逐样本裁剪、浅水零水深保护、保留的时间残差，以及使用解析测试模型的五种设置各 100 步完整采样。基线适配器另经 DDIS 和 FunDPS 的实际归一化类验证，预测系数和反标准化梯度均保持不变。

这些检查验证实现与接口，不替代训练模型在完整测试集上的误差评估；本次没有启动调参或完整测试集重评。

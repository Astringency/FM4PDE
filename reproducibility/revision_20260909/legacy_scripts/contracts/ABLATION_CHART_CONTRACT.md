# 主消融的图表设计

用途：JMLR 论文中的静态 PDF 图和内嵌 LaTeX 表格；不制作在线报告或图表组件。

比较单位是主消融的 ID 0 物理样本。744 条重跑替换对应的原配置，Helmholtz、Darcy、Burgers 的原配置和另外 45 条时序/端点控制仍保留。各图只使用对应配置的实际预测，不能用 32 ID 的重复研究数据替代。完整替换要求 1,060 条配置均有来源，744 条指定重跑完成；开发阶段可输出注明范围的独立文件，不能称为完整论文结果。

| 内容 | 问题与可支持的解释 | 图表与数据 | 计划文件 |
|---|---|---|---|
| 引导组成 | 无引导、物理、观测、联合引导对各个场的影响；不由单样本推断总体显著性 | 分面点图；每 PDE 的 a/u 分开，Burgers 只有 u；另外展示主样本的重建及绝对误差 | ablation_guidance_fields、ablation_guidance_reconstruction_* |
| 损失计算位置 | 当前状态、下一状态、端点对随机/确定性采样的影响 | 表格；21 个物理场分别列 6 种设置 | ablation_loss_state_fields.tex |
| 采样阶段 | 不同阶段顺序如何改变重建结构 | 每个 PDE 展示真值、S、D、D→S、S→D；混合阶段预定选 switch=0.2，不按误差挑选；a/u 及多分量场均展示，图下注明整个场的相对误差 | ablation_phase_reconstruction_* |
| 步数 | 固定引导参数下的数值预算与重建变化 | 单独预算图或样本图，保留 10/50/100/200/500/1000/2000 全部预算；完整数值表移至附录，主文不同时放数值表和热力图 | ablation_budget_fields、完整附表 |
| 观测覆盖 | 在预定 5 个传感器数量下重建如何变化 | 分面点/连线图；明确横轴是离散实验条件，不推断连续曲线；所有 a/u 分别显示 | ablation_coverage_fields |
| 采样轨迹 | 主样本更新过程中两个场的误差演化 | 每 PDE 分开 a/u，观测与联合引导两条曲线；无总体置信带 | ablation_trajectories_fields |
| 其他控制 | 网格、积分方法、布局、噪声、时序近似和五种子敏感性 | 附录完整数值表，含各个 a/u；五种子表按场给均值和样本 SD | ablation_*_fields.tex |

标量场用直接热图；RD、SWE、Wave 的每个物理分量单独画图，误差标签始终使用整个 a 或 u 向量场的范数，不用某一分量替代整个场。Burgers 图中横轴为空间，纵轴为物理时间。

视觉约束：Times New Roman，包括数学标签；PDF 嵌入字体。论文栏宽约 6.2 英寸；每页最多 6 个分量行、每行 5 个采样设置，过多行分图。真值与预测在同一行共享完整色域；绝对误差从零起。使用蓝/金两色根的连续色图与中性色；曲线由线型与直接标签补充颜色区别。无装饰性徽标。标题为描述性名称，图注说明场、误差单位、样本、采样设置。

检查：逐图检查导出 PNG 与最终 PDF；逐张核对标签与源预测、共同真值/掩码、场分量、误差和步数。局部样本图不代表 1,000 个样本的主实验均值，也不附上 32 ID 研究的置信区间。

Numeric labels: retain two significant digits for small nonzero errors and changes; do not display them as 0.00. Steady Heat has errors below 0.005%, so use decimal or scientific notation as needed.

# 归档后实际复现入口核查

本次仅只读审查源码、冻结协议、当前文件布局及参数解析；未执行 GPU、远程操作、collector 或生产程序。审查范围为新源码/环境整理工具，以及新 NS、K-scaling、744 个消融重跑、原三次预测平均和 Burgers 的主要入口。归档脚本尚在完善，不据此判断归档已经完成。

后续受授权完成的 CPU 路径迁移回归另见 `RELOCATION_TEST.md` / `RELOCATION_TEST.json`。其中对 744 修订消融和原三次预测研究从 raw 重新导出，使用新的 11-PDE 相对链接视图，并更新 `STUDY_RECIPES.md` 的独立采样/离线复算分支。下文保留首次静态核查依据；其“未执行 collector”的范围仅指首次核查。 随后完整 server197 归档重算已完成：744 修订、316 原始对照和 1,056 条三次预测全部核验，20 表数字与排名保持一致，8,862 个来源文件前后 SHA 不变；见 `RELOCATION_SERVER197.md`。其中已落实 11-PDE 结果视图和全部新读取路径，PNG 的跨版本渲染差异另行明确记录。

最初识别的 Git 身份阻断已修复并实查通过；K exporter 的显式输入重定位已有单池回归证据。其余应在交付复现入口前落实：Burgers 登记跨研究引用的 FM 权重；NS 输入中的绝对符号链接解析到归档内实体；消融的 PDE 统一视图、新路径索引和完整原始对照已由后续服务器重算落实。其余问题主要是明确入口参数和区分旧结果核查与新采样。

## 1. 源码解包目录不能直接通过计时程序的 Git 版本断言

**已解决：根提交 `7a3dd7a`。** 复查新版 `restore_tree` 从哈希校验过的 bundle 克隆、detach 并验证 HEAD/tree；本地 DiffusionPDE、FM4PDEbaseline 恢复目录均有独立 `.git`，实际 HEAD/tree 与各自 manifest 相符。README 已同步。以下保留原问题及其影响，说明为何需要真实 checkout。

本次最初读取的 `source_snapshots.py::restore_tree` 仅解出源文件并写 `.FM4PDE_SOURCE.json`，没有独立 `.git`。README 的恢复示例将 DiffusionPDE 解到当前 FM4PDE 仓库内。此时 `git rev-parse HEAD` 会向父目录查找并返回 FM4PDE 的 HEAD，而非 DiffusionPDE 的版本；若解到任何 Git 仓库外，则直接报错。

直接受影响的代码是 `plot/run_diffusion_fm_timing.py` 中两条断言：

```python
assert git_head_of_fm == protocol['fm_commit']
assert git_head_of_diffusion == protocol['diffusion_commit']
```

新 NS 计时的冻结 `timing_inputs/protocol.json` 要求：

- FM4PDE：`0d239b3d1ed1dae25e66bc4017cc972ae3d11a13`。
- DiffusionPDE：`151e721b9991404154ad4b430ab85cdaedbaa399`。

应从经过 SHA256 核验的 Git bundle 克隆到新目录，再 detach 到指定 commit，并核对 HEAD 与 tree。运行 FM 程序时也要进入对应生产 checkout，不能从最新主仓库执行历史计时再传一个旧 Diffusion 目录。其他生产器通常只记录 Git HEAD，未全部断言；错误父仓库身份在这些程序中会造成错误的来源记录而不是立即失败。

本次已只读核查清单中 23 个 FM、一个 DiffusionPDE 和一个 FM4PDEbaseline tree，均可在本地 Git 解析，且没有 tracked symlink；因此当前 catalog 没有因 `restore_tree` 拒绝 symlink 而额外失效的已知条目。`collect_environments.py` 的解释器路径由 `--python` 明确提供，读取包元数据而不导入 torch/初始化 CUDA；未发现需要修正的路径解析问题。它产生的是事后环境观测，不是可安装的完整环境锁，README 已准确限定这一点。

## 2. Burgers 的 FM 权重不在 `burgers_weights/`

本地 `paper_revision_20260908/burgers_weights/` 只有 `pretrained-burgers.pkl`。Burgers 的正式 FM 权重实际复用了旧消融的：

```
/home/tat512/C01Python/audit/paper_revision_20260908/inputs/burger/weights.pth
```

本次读取并计算该文件 SHA256，与 `burgers_inputs/sampling_protocol_v3.json` 完全一致：

| 参数 | 应使用的来源 | 字节数 / SHA256 |
|---|---|---|
| `--fm-weights` | 消融归档中的 `inputs/burger/weights.pth`，或相同字节的 Burgers 专用副本 | 385948627；`b76ea10874c37b04061d41e909a6e05bfb954f3e6797fd70895bf27f49abf956` |
| `--dm-weights` | `burgers_weights/pretrained-burgers.pkl` | 协议 SHA：`a24eddebaff43e477e015e8a0f4869e27bc0aeaae224f24f319c16e3dad50edf` |
| `--sampling-protocol` | `burgers_inputs/sampling_protocol_v3.json` | 保留完整原协议；其中 `shards=10`，不是当前 CLI 的默认 5 |
| `--diffusion-root` | 对应 DiffusionPDE Git checkout | `scripts/generate_burgers.py` 必须匹配协议 SHA `ddcd165ef0de0dc848474d587b0efb599b299749aaca7b299b42a572b3695cc2` |

`run_burgers_revision.py::predict` 会用 `--fm-weights` 重写 `checkpoint_path`，输入真值和观测来自六个冻结的 `*_random.pt` / `*_structured.pt`。因此，协议中旧 `fm_config.checkpoint_path` 和 `/large_storage/.../*.mat` 字符串**不会在正式 worker 中触发再次读取原数据**，不应修改这些冻结字符串。只归档 Burgers 自己的 `burgers_weights/` 则缺少重新采样的 FM 权重。

源码还必须满足两个哈希；本次确认 FM commit `23d14d3066704170dcf471ce2f2556179111f9ba` 包含所需字节：

- `plot/run_burgers_revision.py`：`bbbb6649e75397f030ea7923a91c9fa89e1098590ed2abae9fef8c6a90f8567c`。
- `plot/diffusion_timing_adapter.py`：`fa62dbe5f09d0a1d7268d1884c5f4639550e3df5fa7199a6720f458887138aa3`。

原 `run_burgers_parallel.py work` 还要求 assignment 中的 hostname、GPU 编号和 executor SHA 全部匹配。不能把历史 `burgers_parallel_assignment_2242.json` 直接交给集中后的单台 server197 并期望所有 shard 通过。重新采样可从上述固定源码的普通 `worker` 入口执行冻结协议的 0–9 原始 shard，输出到新目录；或为新的资源分配生成一份新的 assignment，明确这是新的执行布局。后者不应修改原任务分配或原 receipts。

已有 Burgers 数值复核不需要重新采样：`export_burgers_revision.py export` 的 `--results` 可接收多个原来源目录，并在每个目录内按相对路径找 tensor/receipt。选择每个 job 的一个权威来源即可；同一结果同时传入原始层与其重复副本会触发 duplicate-result 断言。

## 3. NS 主实验与计时输入需要两个不同的可用布局

新 NS 主采样的 `--inputs` 指向直接包含 `protocol.json`、`weights.pth`、`id.npz`、`smooth.npz`、`rough.npz` 的目录。它从这三个 NPZ 读取真值；协议中的原 HDF5 `sources[dist]['path']` 仅用于配置/元数据，正式 worker 不再读取原 HDF5。冻结数据可用时无需复制完整训练数据，也无需重跑 `prepare`。

独立 `audit_ns_main_residual_0909.py` 的 `--audit` 则指向研究根目录，要求：

```
<NS_STUDY>/weights.pth
<NS_STUDY>/selection.json
<NS_STUDY>/inputs/protocol.json
<NS_STUDY>/inputs/{id,smooth,rough}.npz
```

本地 `inputs/weights.pth` 是指向研究根 `weights.pth` 的绝对符号链接。只有 `inputs/` 或只有 raw `main_results/` 都不足以满足完整审计；归档索引需同时登记根权重、选择文件和输入。该审计用保存的 checkpoint 字符串重建配置并比较，但实际验证的是 `<NS_STUDY>/weights.pth` 的内容，不需要恢复原远程 checkpoint 绝对路径。

新 NS `timing_inputs/` 更依赖链接。本次实查以下路径均为绝对符号链接：

| timing_inputs 内路径 | 原目标 |
|---|---|
| `protocol.json` | `ns_main_revision_0909/timing_protocol.json` |
| `masks.npz` | `diffusion_fm_revision_20260907/timing_inputs_v2/masks.npz` |
| `source` | `diffusion_fm_revision_20260907/timing_inputs_v2/source/` |
| `weights/fm_nsnonbounded.pth` | 新 NS 研究根 `weights.pth` |
| `weights/pretrained-ns-nonbounded.pkl` | 原计时 `timing_inputs_v2/weights/pretrained-ns-nonbounded.pkl` |

迁移时应保存这些目标实体，或在新的读取视图中建立归档内可解析的链接，并逐项通过 protocol `artifacts` 的原哈希。仅复制旧符号链接不能在删除旧位置后复核计时。`export_diffusion_fm_timing.py --pde-overrides` 的 `nsnonbounded.inputs` 应指向这个完整的**新 NS timing_inputs**；不能用主 NS 的 endpoint-pair `inputs` 代替。`nsnonbounded.results` 应指向下层有 `nsnonbounded/` 的结果目录。

## 4. 744 消融与原三次平均需要显式的按 PDE 结果汇合视图

**已提供实际路径回归：** `check_ablation_relocation.py` 建立相对链接视图并从 raw 新建 snapshot。相同两线程环境下，原路径与新路径的 1,060 条记录（除读取路径）、未舍入 CSV、20 张表和 84 个三次预测频谱数组严格一致。历史 ablation snapshot 另有至多约 $4\times10^{-15}$ 的浮点归约差，未改变显示数值或排名；该差异独立记录，未改旧数据或计算公式。本地缺失的 271 个 unchanged raw 仍属于归档完整性事项，不能由本回归代替。

`collect_paper_ablation_fields.py` 和 `export_paper_seed_ensemble.py` 都只接受一个 `--results` 根目录，随后读取 `<results>/<pde>/...`。当前本地 `paper_revision_20260908/output/` 并不是一个完全独立的实体树：

- `poisson`、`reaction_diffusion` 是本地 `output/` 下的实体目录。
- `helmholtz`、`shallow_water`、`wave`、`burger`、`darcy`、`nsnonbounded`、`steady_heat_conduction`、`heat`、`advection_diffusion` 是指向本地 `output_197/<pde>` 的绝对链接。

按来源分别归档是正确的，但 exporter 还需要一个明确的 11-PDE 读取视图：每个 PDE 连接到归档清单指定的权威实体目录，包含 `selection.json`、completion marker 和相应 raw 子树。把 `$ABL` 定义为研究父目录不会自动得到这个视图。原三次平均本身已经按相对位置搜索 `result.pt` / `masks.pt`，不会依赖 receipt 内的旧 `result_path`；正确汇合视图足以解决其路径问题。

作图还需区别“旧出版索引”与“迁移后新索引”。`plot_paper_ablation_fields.py` 会直接 `Path(record['result_path'])` 并读取旁边的 `curves.csv`。原 `ablation_publication_snapshot/records.json` 保存的是 `/home/tat512/C01Python/audit/...` 绝对路径；搬动该 JSON 后仅指定新的 `--source` 不会重定位。应使用现有 collector 对归档 raw 重建一份**新输出**的 snapshot，再用它画图；保留原 snapshot 及其哈希作为历史记录。

对未重跑的原实验，collector 的 `--original-root` 会拼接 `item['source_result']` 在 `/FM4PDE/` 之后的路径，例如 `outputs/ablations/.../result.pt`。server197 原 `FM4PDE` 根目录可作为这一参数，只要原结果继续原地保留。本地 `original_predictions/` 只是为现有图挑选的子集：当前出版索引中 316 个 unchanged 记录有 **271 个 `result_path=null`**。这不影响它们已保留的原 summary 数值，但不能把这一子集描述为全部原始控制预测已集中齐全；应从 server197 现存原 `outputs/ablations/` 登记这些文件。

我提供的 `STUDY_RECIPES.md` 有一处需要在最终路径落实时明确：重新采样写入 `$REPLAY/results`，而紧随其后的 exporter 示例读取 `$ABL/output`。后者是“核查既有结果”的命令，不会核查刚刚的新采样。最终命令应分别命名 `$ARCHIVED_RESULTS_VIEW` 与 `$NEW_RESULTS`，在相应分支中保持 producer 输出和 exporter 输入一致。原三次平均在新输出中执行前也需要每个 PDE 的冻结 `selection.json`；只创建空 `$REPLAY/results` 会在读取该文件时失败。

`prepare` 是原始输入缓存建立阶段，不是归档复核的前置条件。它仍需要原数据文件、旧 `resolved_config.yaml`、原权重和 `outputs/tuning/six_pde_sampling_refine/recommended_params_standard.json`，且读取所用 checkout 的默认 YAML。日常复现应先复用哈希校验后的 `inputs/<pde>/{truths.pt,weights.pth,protocol.json}` 和冻结选择；不能在最新默认配置上重建缓存并默认等同于原实验。

## 5. K-scaling：两个 `--inputs` 的层级不同，collector 仍属于原部署

根代理新增的 exporter `--inputs` / `--selection` 解决了核心旧路径依赖：它从迁移后文件读字节，但用 environment 中记录的旧 source 字符串重建原 config，随后核验 protocol/truth/weight/selection 哈希。这一职责划分正确。已读取根新增 `k_relocation_validation.json`：server197 forward/offset1500 的完整 1000-draw 池及五份 K 原始结果通过哈希核验，新旧 exporter 的数值 CSV 逐字相同、7 个数组逐元素相同。该证据仅涵盖这一输入/任务，不是整个研究的迁移回归。

必须在文档中明确：

- `run_conditional_sample_scaling.py` 和 `validate_conditional_sample_scaling.py`：`--inputs` 指向**包含 poisson/ 的父目录**。
- `export_conditional_sample_scaling.py`：新增 `--inputs` 指向**poisson/ 本身**，直接含 `protocol.json`、`truths.pt`、`weights.pth`。
- `--selection` 使用原 Poisson 冻结选择 JSON；与 environment 中 `selection_sha256` 一致，不能重新写一份值相同但字节不同的 JSON。

集中到 server197 后不应使用原 `collect_conditional_sample_scaling.py` 来离线复核：其 `HOSTS`、远端 checkout、`production` 路径、tmux 启动命令仍是原部署，并会 SSH 回原服务器。它是生产期监测器，不是本地归档分析入口。

离线复核应对每个原 host 的 result 根分别执行：

```bash
python plot/export_conditional_sample_scaling.py \
  --results "$HOST_PRODUCTION" --inputs "$FROZEN_INPUTS/poisson" \
  --selection "$POISSON_SELECTION" --output "$REVIEW_EXPORT_FOR_HOST"
```

然后用已有 `plot/plot_conditional_sample_scaling.py --exports <各host导出目录> --output <新统计目录> --figures <新图目录>` 汇合。不要把不同 host 的 environment 文件扁平合并后只跑一次 exporter：当前代码要求它们的 `inputs` 原始字符串一致，并明确按原 host 分开导出。对正式完整复核不得使用 `--allow-partial`。

## 结论边界

新 NS 主采样、Burgers worker、原三次平均在提供上述完整冻结输入和正确结果视图后，旧数据路径字符串并不必然阻止复现。优先补齐具体缺失实体/视图和真实 Git checkout，保留已冻结的协议内容。环境采集、复制源码、文件 SHA 检查与重新计算全部原始结果属于不同层次；本次仅完成静态入口审查，不声称完整数值迁移回归已通过。

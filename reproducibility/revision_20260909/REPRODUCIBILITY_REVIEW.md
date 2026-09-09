# 归档后实际复现入口核查

本文保留 2026-09-09 首次只读入口审查及其发现；后续实施状态按本节和各节开头的更新阅读。首次审查没有执行生产程序，并不表示后续数值复算仍未完成。

截至本次说明更新，真实 Git 恢复、Burgers FM 专用权重副本、NS 主/计时链接实体化、十一类 PDE 结果视图及全部 316 个原始对照均已落实。完整 server197 消融重算见 `RELOCATION_SERVER197.md`；新 NS 的 15,000 例实际归档 CPU 重算见 `NS_ARCHIVED_RECOMPUTATION.md`；18 个 baseline 原生缓存、186 个数据/掩码协议和四个模型的小批次真实推理见 `BASELINE_FROZEN_REPLAY.md`。这些事实分别由对应报告及哈希支持。

K 的四个生产分片和 collector 已结束，全部 480 个结果回执、96,000 条 canonical 轨迹和 106,944 条实际计时轨迹通过；完整 compact 数值、统计和三图审阅也通过。其三项结果归档也已于 19:38:49 完成独立校验，至此全部 67 项完成，见 `ARCHIVE_COMPLETED_67.md`。随后从实际 server197 目标读取全部 96 池及 480 条 compact 数据的 CPU 数值重算也通过，见 `K_ARCHIVED_RECOMPUTATION.md`。最终命令和来源边界见 `STUDY_RECIPES.md`；先前 64 项记录保留为历史证据。

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

首次已只读核查清单中 23 个 FM、一个 DiffusionPDE 和一个 FM4PDEbaseline tree，均可在本地 Git 解析，且没有 tracked symlink；因此当前 catalog 没有因 `restore_tree` 拒绝 symlink 而额外失效的已知条目。后续完整源码核验最终扩大到 60 版本（49 FM、3 Diffusion、8 baseline），70 归档、27,029 文件内容/权限及三个独立 Git bundle 恢复均通过，记录见 `source_restore_coverage.json`（SHA `dd6228c136845aeb2836293fc92ff008aeba3ad8659e3cb0361a0e07fa4079a1`）。`collect_environments.py` 的解释器路径由 `--python` 明确提供，读取包元数据而不导入 torch/初始化 CUDA；未发现需要修正的路径解析问题。它产生的是事后环境观测，不是可安装的完整环境锁，README 已准确限定这一点。

## 2. Burgers 的 FM 权重不在 `burgers_weights/`

**2026-09-09 后续已落实。** 专用 FM 权重已复制并校验到 `outputs/main/revision_20260909/burgers_revision_20260908/selected_models/fm/weights.pth`，保留本节记录的 `b76ea108…` 原 SHA 和 385,948,627 字节。原有 64 项完成记录包含该依赖；不再依赖仓库外的旧绝对链接。以下说明保留原依赖识别过程及重新采样参数。

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

**2026-09-09 后续已落实。** 完整主研究归档的链接均保存为目标实体，实际读取根为 `outputs/main/revision_20260909/ns_main_revision_0909/local/complete_local_ns_study`。该归档的 1,146 个文件及原哈希在 15,000 例 CPU 重算前后保持一致，见 `NS_ARCHIVED_RECOMPUTATION.md`。主 `inputs` 与 `timing_inputs` 仍须按各自用途区分；以下绝对链接表描述首次检查时的原来源布局。

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

**已提供实际路径回归：** `check_ablation_relocation.py` 建立相对链接视图并从 raw 新建 snapshot。相同两线程环境下，原路径与新路径的 1,060 条记录（除读取路径）、未舍入 CSV、20 张表和 84 个三次预测频谱数组严格一致。历史 ablation snapshot 另有至多约 $4\times10^{-15}$ 的浮点归约差，未改变显示数值或排名；该差异独立记录，未改旧数据或计算公式。首次本地回归缺少 271 个 unchanged raw；随后已在 server197 对全部 316 个原始对照逐文件登记并实际重算，见 `in_place_reference_checks.json` 与 `RELOCATION_SERVER197.md`。这一本地子集限制不再是最终 server197 归档的缺项。

`collect_paper_ablation_fields.py` 和 `export_paper_seed_ensemble.py` 都只接受一个 `--results` 根目录，随后读取 `<results>/<pde>/...`。首次检查时的本地 `paper_revision_20260908/output/` 并不是一个完全独立的实体树：

- `poisson`、`reaction_diffusion` 是本地 `output/` 下的实体目录。
- `helmholtz`、`shallow_water`、`wave`、`burger`、`darcy`、`nsnonbounded`、`steady_heat_conduction`、`heat`、`advection_diffusion` 是指向本地 `output_197/<pde>` 的绝对链接。

按来源分别归档是正确的，但 exporter 还需要一个明确的 11-PDE 读取视图：每个 PDE 连接到归档清单指定的权威实体目录，包含 `selection.json`、completion marker 和相应 raw 子树。把 `$ABL` 定义为研究父目录不会自动得到这个视图。原三次平均本身已经按相对位置搜索 `result.pt` / `masks.pt`，不会依赖 receipt 内的旧 `result_path`；正确汇合视图足以解决其路径问题。

作图还需区别“旧出版索引”与“迁移后新索引”。`plot_paper_ablation_fields.py` 会直接 `Path(record['result_path'])` 并读取旁边的 `curves.csv`。原 `ablation_publication_snapshot/records.json` 保存的是 `/home/tat512/C01Python/audit/...` 绝对路径；搬动该 JSON 后仅指定新的 `--source` 不会重定位。应使用现有 collector 对归档 raw 重建一份**新输出**的 snapshot，再用它画图；保留原 snapshot 及其哈希作为历史记录。

对未重跑的原实验，collector 的 `--original-root` 会拼接 `item['source_result']` 在 `/FM4PDE/` 之后的路径，例如 `outputs/ablations/.../result.pt`。server197 原 `FM4PDE` 根目录可作为这一参数，只要原结果继续原地保留。本地 `original_predictions/` 只是为现有图挑选的子集：首次本地出版索引中 316 个 unchanged 记录有 **271 个 `result_path=null`**。这不影响它们已保留的原 summary 数值，但不能把这一子集描述为全部原始控制预测已集中齐全；应从 server197 现存原 `outputs/ablations/` 登记这些文件。

**2026-09-09 已修正 `STUDY_RECIPES.md` 的对应命令。** 原先有一处需要区分两种工作：重新采样写入 `$REPLAY/results`，而紧随其后的 exporter 示例读取 `$ABL/output`。后者是“核查既有结果”的命令，不会核查刚刚的新采样。最终命令应分别命名 `$ARCHIVED_RESULTS_VIEW` 与 `$NEW_RESULTS`，在相应分支中保持 producer 输出和 exporter 输入一致。原三次平均在新输出中执行前也需要每个 PDE 的冻结 `selection.json`；只创建空 `$REPLAY/results` 会在读取该文件时失败。

`prepare` 是原始输入缓存建立阶段，不是归档复核的前置条件。它仍需要原数据文件、旧 `resolved_config.yaml`、原权重和 `outputs/tuning/six_pde_sampling_refine/recommended_params_standard.json`，且读取所用 checkout 的默认 YAML。日常复现应先复用哈希校验后的 `inputs/<pde>/{truths.pt,weights.pth,protocol.json}` 和冻结选择；不能在最新默认配置上重建缓存并默认等同于原实验。

## 5. K-scaling：两个 `--inputs` 的层级不同，collector 仍属于原部署

**2026-09-09 后续完整审阅已通过。** 本节单池回归之后，四个分片、两份完整 host export、全部 480 compact 行及 672 数组已经过完整复核。数值审阅完成 2,361 项比较；科学审阅 SHA 为 `8d3a539b5118a72981c4b62a340968d4880e1631693fdf4357a114aa5d4d0801`。三项 K 归档现已全部复制并独立校验，完整目标路径复算也已完成。`K_ARCHIVED_RECOMPUTATION.md` 记录 96 池重导出的 480 行及 672 数组精确一致、原源文件前后 SHA 不变，以及独立 compact checker 的 2,361 项比较；初次环境记录排列顺序断言和修正后的完整重跑均保留。实际入口是 `verify_conditional_scaling_archived_host.py` 和 `verify_conditional_scaling_summary.py --paper ... --audit ... --output ...`，不是原 collector。

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

原始路径字符串保留在冻结协议中作为来源信息；实际读取使用已校验的目标实体和明确入口参数。首次识别的具体实体、视图和 Git 身份问题已由本页列出的后续操作落实。完整消融、新 NS 和完整 K 研究均已从实际 server197 归档执行 CPU 数值复算；全部 67 项结果归档也已独立校验。环境记录、文件校验、保存结果复算、原权重推理和重新训练仍是不同层次。未保存的六个 VIVID 权重及原 Diffusion HEAD 缺失等历史限制见 `REPRODUCTION_COVERAGE.md`，不因归档工作完成而消失。

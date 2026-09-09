from pathlib import Path
import csv,datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent
read=lambda p:list(csv.DictReader(p.open()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
main=ROOT/"report_v2"
probe=ROOT/"step_probe_report"
audit=json.loads((main/"audit_manifest.json").read_text())
pa=json.loads((probe/"audit_manifest.json").read_text())
assert audit["status"]==pa["status"]=="complete"
assert audit["new_calls"]==audit["new_finite_calls"]==1152
assert pa["new_calls"]==pa["new_finite_calls"]==72
rows=read(main/"summary.csv")
models=[("current_common100","FM4PDE 当前模型（260712），100 步"),("v260904_common100","FM4PDE 260904，100 步"),("bak_legacy100","FM4PDE_bak，旧版引导，100 步"),("DiffusionPDE_100","DiffusionPDE，100 步"),("DiffusionPDE_1000","DiffusionPDE，1000 步")]
lines=["# NS 模型版本与采样对照", "", "已完成 1152 次主对照和 72 次探索性步数诊断。原始预测场、真值、掩码、配置、逐次误差与文件校验值均已保存；另有 41 次加速迁移前的结果保留在历史目录中。", "", "主对照使用同一批 32 个 Smooth 输入，每个输入 3 个种子、500 个观测点。表中先对每个输入的三个种子误差取均值，再给出 32 个输入的均值 ± 样本标准差，单位 %。联合任务在表中分别展示两个字段；预设主要指标 max(a,u) 及其统计仍完整保留在详细报告。这是补充评估，不是论文的 1000 输入主表。", "", "| 模型与采样设置 | 正向 u | 逆向 a | 联合 a | 联合 u |", "|---|---:|---:|---:|---:|"]
for key,label in models:
    cells=[]
    for task,metric in [("forward","rel_l2_u"),("inverse","rel_l2_a"),("both","rel_l2_a"),("both","rel_l2_u")]:
        r=next(r for r in rows if r["variant"]==key and r["task"]==task)
        assert int(r["complete_inputs"])==32
        cells.append(f'{float(r[metric+"_mean_pct"]):.2f} ± {float(r[metric+"_sd_pct"]):.2f}')
    lines.append("| "+label+" | "+" | ".join(cells)+" |")
lines += ["", "260904 在三项任务的主要指标上均优于当前权重，但仍未超过 DiffusionPDE 1000 步。备份模型配合旧版引导在逆问题上略优：32 个输入中有 30 个误差更低，配对均值差为 −0.4243 个百分点，逐输入 bootstrap 的点态 95% 区间为 [−0.4951, −0.3489] 个百分点。正向任务及联合任务的两个字段仍是 DiffusionPDE 1000 步更低；这不能表述为备份模型在全部 NS 任务上胜出。", "", "`bak` 直接使用当前模型引导参数的额外结果也完整保留在 [全部比较表](report_v2/RESULTS.md) 中。归一化尺度和损失缩减不同会改变有效引导强度；部分正向、联合预测出现极大误差，不能将这类参数移植失稳解释为旧模型训练质量差。", "", "[配对差异及逐输入 bootstrap 区间](report_v2/paired_effects.csv) 使用 32 个输入作为重采样单位，不把三个随机种子当作三个独立物理输入。FM 100 步对应 100 次网络计算；DiffusionPDE 100/1000 步对应 199/1999 次。DiffusionPDE 参考复用既有同输入结果，时间列不用于跨环境速度结论。", "", "关于训练不足，目前的证据不能将差距简单归因于 epoch 不够。当前模型和 260904 最后 20 个 epoch 的平均验证损失分别为 0.07225456 与 0.07232274，后期均处于平台；历史验证划分未证实完全一致，这两个损失值不构成严格配对比较，而且相近的流匹配损失并不保证条件重建误差相近。260904 改用了约 4412 万参数的轻量网络、关闭数值 Fourier 特征；当前模型约 3.87 亿参数，两者不是固定架构下延长训练的对照。当前日志还有恢复训练初期的学习率和损失突增；它值得进一步检查，但仅凭日志不能认定是最终差距的原因。旧版 bak 约 9881 万参数，保存到第 500 个 epoch，缺少验证曲线与历史训练归一化统计，不能把其训练损失绝对值与两个新模型直接比较。优化器记录的更新次数分别为当前模型 222600、260904 模型 210900、bak 模型 9500；名义有效 batch 分别为 64、64、2560，因此 epoch 数本身也不是等价训练预算。详见 [优化器记录](results_v2/evidence/optimizer_inventory.json)。", "", "此前同一当前权重的独立引导校准已将 NS 逆问题误差从约 20.73% 降至 12.47%，无需重新训练。因此，采样配置是已经得到实验证据支持的影响因素。不同 PDE 的架构、数据、物理残差和观测引导也不同，现有跨 PDE 排序不能单独证明训练充分性或方法原理上的普遍优势。", "", "[训练曲线](report_v2/ns_training_histories.pdf)、[频谱诊断](report_v2/ns_checkpoint_spectra.pdf)、[逆问题预测场](report_v2/ns_fields_inverse_a.pdf)、[正向预测场](report_v2/ns_fields_forward_u.pdf)、[联合初值](report_v2/ns_fields_both_a.pdf)、[联合终态](report_v2/ns_fields_both_u.pdf)。图中文字使用 Times New Roman。图中频谱是涡量 Fourier 系数的平方谱，不是速度场的动能谱；按每个输入的总参考平方谱归一化，避免在几乎无参考信号的高频尾部误读比例。", "", "频带统计进一步显示，当前 FM 模型正向 u 与逆向 a 的平均归一化平方误差分别有 99.93% 和 88.64% 位于 0 < |k| ≤ 8 的低频带。这表明目前误差包括大尺度结构重建问题，不能仅解释为高频细节学习不足。Smooth 终态在 |k| > 32 的参考平方谱占比只有约 4.89×10⁻¹⁵，不能用这一尾部极小的分母夸大相对误差。这里诊断的是当前模型的频带误差位置，并未隔离训练与条件采样各自的因果贡献。详见 [频带数值](report_v2/frequency_band_summary.csv)。", "", "另外完成了固定前 4 个输入 [130,196,651,484]、每个输入 3 个种子的逆问题步数诊断。下表仅用于观察增加采样步数的影响，不能据此宣布 32 或 1000 输入上的排名。", "", "| 模型 / 引导 | 100 步逆问题误差，% | 1000 步逆问题误差，% |", "|---|---:|---:|"]
pr=read(probe/"summary.csv")
for base,label in [("current_common","当前模型 / 当前引导"),("v260904_common","260904 / 当前引导"),("bak_legacy","bak / 旧版引导"),("DiffusionPDE_","DiffusionPDE / 原始引导")]:
    cells=[]
    for steps in [100,1000]:
        r=next(r for r in pr if r["variant"]==base+str(steps))
        cells.append(f'{float(r["mean_pct"]):.2f} ± {float(r["sd_pct"]):.2f}')
    lines.append("| "+label+" | "+" | ".join(cells)+" |")
lines += ["", "[步数诊断完整记录](step_probe_report/RESULTS.md)。所有模型与两个步数均保留，没有按评估误差选择权重或调参。当前随机采样的每步引导采用 c(1-t)（旧版采用 c(1-t_next)），没有乘积分步长；步数增加还会增加引导次数和噪声重采样次数，因此这里同时改变了累计引导作用，不能把差异单独归因于积分精度。另存有各方法对三个预测场取平均后的 [结果](report_v2/mean3_summary.csv) 和 [误差分解](report_v2/seed_variance_identity.csv)，每种方法均计入三次采样成本。", "", "两个新实验重复的 36 次逆问题 100 步采样在不同 GPU 上逐元素完全一致；见 [跨实验复现核对](overlap_100_step_qa.json)。[主对照校验清单](report_v2/audit_manifest.json) 与 [步数诊断校验清单](step_probe_report/audit_manifest.json) 可追溯到每个预测文件及输入。"]
(ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")
pdfs=list(main.glob("*.pdf"))
assert len(pdfs)==6
font_reports={}
for pdf in pdfs:
    fonts=subprocess.check_output(["pdffonts",str(pdf)],text=True)
    assert "TimesNewRoman" in fonts and "DejaVu" not in fonts, pdf
    font_reports[pdf.name]=fonts
state=dict(status="numerically_complete_pending_visual_review",checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),main_calls=1152,step_probe_calls=72,preserved_additional_history_calls=41,main_audit_sha256=sha(main/"audit_manifest.json"),step_audit_sha256=sha(probe/"audit_manifest.json"),pdf_sha256={p.name:sha(p) for p in pdfs},font_reports=font_reports,summary_sha256=sha(ROOT/"RESULTS.md"))
(ROOT/"final_status.json").write_text(json.dumps(state,indent=2)+"\n")
print("Written",ROOT/"RESULTS.md")

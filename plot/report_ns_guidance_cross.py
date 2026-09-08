"""Audit and report all six cells of the fixed NS checkpoint x guidance design."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import json
import subprocess

import numpy as np
import torch

import report_ns_checkpoint_comparison as parent_report
from report_ns_checkpoint_comparison import csv_write, font_setup, save
from run_ns_loss_study import sha, write

METRICS = ['primary_error', 'rel_l2_a', 'rel_l2_u']
LABELS = {
    'current_common100': '260712 / current guidance',
    'current_legacy100': '260712 / legacy guidance',
    'v260904_common100': '260904 / current guidance',
    'v260904_legacy100': '260904 / legacy guidance',
    'bak_common100': 'Backup / current guidance',
    'bak_legacy100': 'Backup / legacy guidance',
    'DiffusionPDE_100': 'DiffusionPDE / 100 steps',
    'DiffusionPDE_1000': 'DiffusionPDE / 1000 steps',
}


def audit_new(args, protocol):
    ph = sha(args.results / 'protocol.json')
    parent = json.loads((args.base_results / 'protocol.json').read_text())
    assert sha(args.base_results / 'protocol.json') == protocol['parent_protocol_sha256']
    for key in ['source_sha256', 'fields_sha256', 'evaluation_ids', 'pilot_ids', 'seeds',
                'tasks', 'common_configs', 'legacy_configs', 'workers']:
        assert parent[key] == protocol[key], key
    for key, info in protocol['checkpoints'].items():
        assert parent['checkpoints'][key] == info
    assert [v['name'] for v in protocol['variants']] == ['current_legacy100', 'v260904_legacy100']
    assert sha(args.inputs / 'source.json') == protocol['source_sha256']
    assert sha(args.inputs / 'fields_masks.npz') == protocol['fields_sha256']
    fields = np.load(args.inputs / 'fields_masks.npz')
    hashes = {str(args.results / 'protocol.json'): ph}
    for shard in range(4):
        env_path = args.results / f'environment_{shard}.json'
        env = json.loads(env_path.read_text())
        reference = protocol['worker_reference_environments'][str(shard)]
        reference_path = args.base_results / f'environment_{shard}.json'
        assert sha(reference_path) == reference['source_sha256']
        assert env['protocol_sha256'] == ph and env['tf32'] is False
        for key in ['host', 'uuid', 'torch', 'cuda', 'gpu']:
            assert env[key] == reference[key], (shard, key)
        hashes[str(env_path)] = sha(env_path)
        pilot_path = args.results / f'implementation_check_{shard}.json'
        check = json.loads(pilot_path.read_text())
        assert check['protocol_sha256'] == ph and check['status'] == 'pass'
        assert len(check['checks']) == 2
        for item in check['checks']:
            if 'error' not in item:
                assert item['repeat'] == item['hidden'] == item['native_runner'] == 0
                assert item['observation_sensitivity'] > 0 and item['nfe'] == 100
        hashes[str(pilot_path)] = sha(pilot_path)
        terminal = args.results / f'complete_{shard}.json'
        receipt = json.loads(terminal.read_text())
        assert receipt['protocol_sha256'] == ph and receipt['expected_calls'] == 144
        hashes[str(terminal)] = sha(terminal)
        for label, info in protocol['checkpoints'].items():
            path = args.results / f'loaded_weight_{shard}_{label}.json'
            loaded = json.loads(path.read_text())
            assert loaded['protocol_sha256'] == ph
            assert loaded['inference_signature'] == info['inference_signature']
            hashes[str(path)] = sha(path)
    rows, paths = [], {}
    overridden = {'device', 'sample_seed', 'batch_size', 'offset', 'num_steps', 'model_profile',
                  'save_plots', 'save_intermediate', 'save_per_sample_curves', 'checkpoint_path',
                  'output_dir', 'initial_noise_source_indices', 'initial_noise_source_batch_size',
                  'model_gradient_checkpointing'}
    for task in protocol['tasks']:
        for variant in protocol['variants']:
            for i in protocol['evaluation_ids']:
                for seed in protocol['seeds']:
                    key = (task, variant['name'], i, seed)
                    path = args.results / 'evaluation' / task / variant['name'] / f'sample{i}_seed{seed}.json'
                    r = json.loads(path.read_text())
                    assert (r['task'], r['variant'], r['sample_id'], r['seed']) == key
                    assert r['protocol_sha256'] == ph and r['checkpoint'] == variant['checkpoint']
                    assert r['worker'] == protocol['evaluation_ids'].index(i) % 4
                    assert r['steps'] == 100
                    hashes[str(path)] = sha(path)
                    if 'prediction_sha256' in r:
                        pt = path.with_suffix('.pt')
                        assert sha(pt) == r['prediction_sha256']
                        hashes[str(pt)] = r['prediction_sha256']
                        data = torch.load(pt, map_location='cpu', weights_only=False)
                        cfg = data['config']
                        for setting, value in protocol['legacy_configs'][task].items():
                            if setting not in overridden:
                                assert cfg[setting] == value, (key, setting)
                        assert cfg['num_steps'] == 100 and cfg['dtype'] == 'float32'
                        assert cfg['model_profile'] == protocol['checkpoints'][variant['checkpoint']]['model_profile']
                        assert cfg['sample_seed'] == seed and cfg['offset'] == i and cfg['batch_size'] == 1
                        assert cfg['model_gradient_checkpointing'] is False
                        errors = []
                        for j, field in enumerate(['a', 'u']):
                            truth = torch.from_numpy(fields[f'{field}_{i}'])
                            mask = torch.from_numpy(fields[f'mask_{"a" if task == "both" else field}_{i}'])
                            if (task, field) in [('forward', 'u'), ('inverse', 'a')]:
                                mask = torch.zeros_like(mask)
                            assert torch.equal(data['truth'][j], truth)
                            assert torch.equal(data['masks'][j], mask)
                            pred = data['prediction'][j].double()
                            assert pred.shape == truth.shape
                            if r['status'] == 'complete':
                                assert bool(torch.isfinite(pred).all())
                                error = float((pred - truth.double()).norm() / truth.double().norm())
                                assert np.isclose(error, r[f'rel_l2_{field}'], rtol=1e-12, atol=1e-12)
                                errors.append(error)
                        assert r['nfe'] == 100
                        if r['status'] == 'complete':
                            primary = errors[1] if task == 'forward' else errors[0] if task == 'inverse' else max(errors)
                            assert np.isclose(primary, r['primary_error'], rtol=1e-12, atol=1e-12)
                            paths[key] = pt
                    else:
                        assert r['status'] == 'unstable'
                    rows.append(r)
    assert len(rows) == 576
    assert len(list((args.results / 'evaluation').rglob('sample*.json'))) == 576
    return rows, paths, hashes


def aggregate(rows, protocol, output):
    per_input, summary = [], []
    for task in protocol['tasks']:
        for variant in LABELS:
            for i in protocol['evaluation_ids']:
                rr = [r for r in rows if (r['task'], r['variant'], r['sample_id']) == (task, variant, i)]
                assert len(rr) == 3 and sorted(r['seed'] for r in rr) == protocol['seeds']
                finite = sum(r['status'] == 'complete' for r in rr)
                item = dict(task=task, variant=variant, sample_id=i, calls=3, finite_calls=finite)
                for metric in METRICS:
                    item[metric] = float(np.mean([r[metric] for r in rr])) if finite == 3 else None
                per_input.append(item)
            rr = [r for r in per_input if (r['task'], r['variant']) == (task, variant)]
            complete = all(r['finite_calls'] == 3 for r in rr)
            item = dict(task=task, variant=variant, expected_inputs=32,
                        complete_inputs=sum(r['finite_calls'] == 3 for r in rr),
                        finite_calls=sum(r['finite_calls'] for r in rr), expected_calls=96)
            for metric in METRICS:
                values = np.array([r[metric] for r in rr], dtype=float)
                item[metric + '_mean_pct'] = float(values.mean() * 100) if complete else None
                item[metric + '_sd_pct'] = float(values.std(ddof=1) * 100) if complete else None
            summary.append(item)
    csv_write(output / 'per_call.csv', rows)
    csv_write(output / 'per_input.csv', per_input)
    csv_write(output / 'summary.csv', summary)
    effects = []
    contrasts = []
    for label in ['current', 'v260904', 'bak']:
        contrasts.append(('guidance_change', f'{label}_legacy100', f'{label}_common100'))
    for guidance in ['common', 'legacy']:
        for a, b in [('v260904', 'current'), ('bak', 'current'), ('bak', 'v260904')]:
            contrasts.append(('checkpoint_change', f'{a}_{guidance}100', f'{b}_{guidance}100'))
    for label in ['current', 'v260904', 'bak']:
        for guidance in ['common', 'legacy']:
            for budget in [100, 1000]:
                contrasts.append(('diffusion_reference', f'{label}_{guidance}100', f'DiffusionPDE_{budget}'))
    for task in protocol['tasks']:
        for kind, variant, control in contrasts:
            aa = sorted([r for r in per_input if (r['task'], r['variant']) == (task, variant)], key=lambda r:r['sample_id'])
            bb = sorted([r for r in per_input if (r['task'], r['variant']) == (task, control)], key=lambda r:r['sample_id'])
            assert [r['sample_id'] for r in aa] == [r['sample_id'] for r in bb]
            for metric in METRICS:
                row = dict(kind=kind, task=task, variant=variant, control=control, metric=metric, n_inputs=32)
                if all(r['finite_calls'] == 3 for r in aa + bb):
                    delta = np.array([a[metric] - b[metric] for a,b in zip(aa,bb)]) * 100
                    rng = np.random.default_rng(20260913)
                    boot = delta[rng.integers(0, 32, (4000, 32))].mean(axis=1)
                    lo, hi = np.quantile(boot, [.025, .975])
                    row.update(status='complete', mean_delta_pp=float(delta.mean()), ci95_low_pp=float(lo),
                               ci95_high_pp=float(hi), input_wins=int((delta < 0).sum()),
                               resamples=4000, bootstrap_seed=20260913,
                               interval='pointwise paired-input percentile bootstrap')
                else:
                    row.update(status='not_estimated_due_to_nonfinite_outcomes')
                effects.append(row)
    csv_write(output / 'paired_effects.csv', effects)
    return summary, effects


def figures(args, protocol, paths, *, only_tasks=None):
    import matplotlib.pyplot as plt
    fields = np.load(args.inputs / 'fields_masks.npz')
    i, seed = protocol['evaluation_ids'][0], 0
    variants = ['current_common100', 'current_legacy100', 'v260904_common100', 'v260904_legacy100', 'DiffusionPDE_1000']
    names = ['260712 / current\n100 steps', '260712 / legacy\n100 steps',
             '260904 / current\n100 steps', '260904 / legacy\n100 steps', 'DiffusionPDE\n1000 steps']
    for task, field, j in [('forward', 'u', 1), ('inverse', 'a', 0), ('both', 'a', 0), ('both', 'u', 1)]:
        if only_tasks is not None and task not in only_tasks:
            continue
        truth = fields[f'{field}_{i}'].squeeze().astype(float)
        preds = []
        for variant in variants:
            path = paths.get((task, variant, i, seed))
            preds.append(torch.load(path, map_location='cpu', weights_only=False)['prediction'][j].numpy().squeeze().astype(float) if path else None)
        vmax = max(float(np.abs(x).max()) for x in [truth] + preds if x is not None)
        emax = max(float(np.abs(x - truth).max()) for x in preds if x is not None)
        fig, axes = plt.subplots(2, 6, figsize=(15.5, 5.2), layout='constrained')
        im = axes[0,0].imshow(truth, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        axes[0,0].set_title('Reference')
        observed = 'u' if task == 'inverse' else 'a'
        mask = fields[f'mask_{observed}_{i}'].squeeze()
        axes[1,0].imshow(mask, cmap='Greys', vmin=0, vmax=1, origin='lower')
        axes[1,0].set_title(f'{int(mask.sum())} {"paired" if task == "both" else observed} sensors')
        for k, pred in enumerate(preds, 1):
            axes[0,k].set_title(names[k-1])
            if pred is None:
                axes[0,k].text(.5,.5,'Nonfinite outcome',ha='center',transform=axes[0,k].transAxes)
                continue
            axes[0,k].imshow(pred, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
            error = np.abs(pred-truth)
            errim = axes[1,k].imshow(error, cmap='magma', vmin=0, vmax=emax, origin='lower')
            rel = np.linalg.norm(error.ravel()) / np.linalg.norm(truth.ravel()) * 100
            axes[1,k].set_title(f'Relative L2: {rel:.2f}%')
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[0,:], shrink=.7, pad=.01, label='Field value (shared scale)')
        fig.colorbar(errim, ax=axes[1,1:], shrink=.7, pad=.01, label='Absolute error (shared scale)')
        fig.suptitle(f'NS {task}: {field}, input {i}, seed {seed}\nPrespecified example; checkpoint normalization retained', fontsize=14)
        save(fig, args.output / f'ns_guidance_cross_{task}_{field}')


def display(summary, task, variant, metric):
    r = next(r for r in summary if (r['task'], r['variant']) == (task, variant))
    if r['complete_inputs'] != 32:
        return f"{r['finite_calls']}/96 finite"
    mean, sd = r[metric + '_mean_pct'], r[metric + '_sd_pct']
    fmt = '.2e' if abs(mean) >= 1000 else '.2f'
    return f'{mean:{fmt}} ± {sd:{fmt}}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['results', 'base-results', 'inputs', 'diffusion-results', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    font_setup()
    protocol = json.loads((args.results / 'protocol.json').read_text())
    new_rows, new_paths, new_hashes = audit_new(args, protocol)
    base_out = args.output / 'baseline_reaudit'
    base_out.mkdir(exist_ok=True)
    base_args = SimpleNamespace(results=args.base_results, inputs=args.inputs,
                                diffusion_results=args.diffusion_results, output=base_out)
    base_protocol = json.loads((args.base_results / 'protocol.json').read_text())
    _, _, base_paths = parent_report.audit(base_args, base_protocol)
    # Reconstruct receipt rows from their independently reaudited raw sources.
    import csv
    base_rows = list(csv.DictReader((base_out / 'per_call.csv').open()))
    for row in base_rows:
        for key in ['sample_id', 'seed', 'steps', 'nfe']:
            if row.get(key):
                row[key] = int(row[key])
        for metric in METRICS:
            row[metric] = float(row[metric]) if row.get(metric) else None
    rows = base_rows + new_rows
    assert len(rows) == 2304
    summary, effects = aggregate(rows, protocol, args.output)
    figures(args, protocol, {**base_paths, **new_paths})
    lines = ['NS 权重 × 引导交叉对照', '',
             '新增 576 次采样，补齐三个权重 × 两套引导的六个组合。复核既有 1152 次 FM 结果和 576 次 DiffusionPDE 参考结果。所有组合保留各自权重的网络与归一化。', '',
             '同一批 32 个 Smooth 输入，每个输入 3 个种子，500 个观测位置；同一输入沿用上轮同一台服务器和同一张物理 GPU。所有 FM 均为 100 步、100 次网络计算。先对每个输入的三个种子误差取均值，再给出 32 个输入的均值 ± 样本标准差，单位 %。', '',
             '| 权重 / 引导 | 正向 u | 逆向 a | 联合 a | 联合 u |', '|---|---:|---:|---:|---:|']
    for variant, label in LABELS.items():
        cells = [display(summary, t, variant, m) for t,m in [('forward','rel_l2_u'), ('inverse','rel_l2_a'), ('both','rel_l2_a'), ('both','rel_l2_u')]]
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '旧版引导相对当前引导的配对变化（百分点；负值表示误差降低）。区间以物理输入为单位进行 4000 次 bootstrap，是点态 95% 区间，未作多重比较校正。', '',
              '| 权重 | 任务 / 字段 | 均值差 | 95% 区间 | 改善输入数 |', '|---|---|---:|---|---:|']
    for label in ['current', 'v260904', 'bak']:
        for task, metric in [('forward','rel_l2_u'), ('inverse','rel_l2_a'), ('both','rel_l2_a'), ('both','rel_l2_u')]:
            r = next(r for r in effects if r['kind']=='guidance_change' and r['variant']==f'{label}_legacy100' and r['task']==task and r['metric']==metric)
            if r['status']=='complete':
                lines.append(f"| {label} | {task} / {metric[-1]} | {r['mean_delta_pp']:.4g} | [{r['ci95_low_pp']:.4g}, {r['ci95_high_pp']:.4g}] | {r['input_wins']}/32 |")
            else:
                lines.append(f'| {label} | {task} / {metric[-1]} | 存在非有限结果，未估计 | — | — |')
    lines += ['', '该实验比较整套引导配置：观测损失缩减、残差算子、各项权重、物理引导开启时间、随机修正系数和裁剪设置同时改变。它能衡量固定权重下切换整套引导的效果，不能分离其中某个组件的作用。旧版 NS 引导使用空间差分代理项，不是完整 NS 演化残差。没有重新训练或依据评估误差调参。', '',
              '所有巨大有限误差、非有限结果和运行失败均保留；若某组合存在非有限结果，不报告仅基于成功样本的总体均值。联合任务预设主要指标为逐次 max(a,u)，另保留两字段误差；完整数据见 [summary.csv](summary.csv)、[per_input.csv](per_input.csv) 和 [paired_effects.csv](paired_effects.csv)。', '',
              '图固定展示原有输入顺序的首个输入、seed 0，没有按效果选图。Times New Roman 字体，预测和绝对误差分别共享色标：[正向](ns_guidance_cross_forward_u.pdf)、[逆向](ns_guidance_cross_inverse_a.pdf)、[联合初值](ns_guidance_cross_both_a.pdf)、[联合终态](ns_guidance_cross_both_u.pdf)。', '',
              '这是 32 输入的补充对照；论文的 1000 输入主表保持原有评估口径。DiffusionPDE 参考来自既有配对评估，100/1000 步分别为 199/1999 次网络计算，时间值不用于跨环境速度结论。']
    (args.output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    artifacts = {str(path):sha(path) for path in args.output.iterdir() if path.is_file() and path.name != 'audit_manifest.json'}
    write(args.output / 'audit_manifest.json', dict(
        status='complete', new_calls=576, new_finite_calls=sum(r['status']=='complete' for r in new_rows),
        base_fm_calls=1152, diffusion_reference_calls=576, total_calls=2304,
        summary_rows=len(summary), per_input_rows=32*3*8, paired_effect_rows=len(effects),
        protocol_sha256=sha(args.results/'protocol.json'), new_source_hashes=new_hashes,
        baseline_reaudit_sha256=sha(base_out/'audit_manifest.json'), artifact_sha256=artifacts,
        report_source_sha256=sha(Path(__file__)),
        report_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=parent_report.ROOT,text=True).strip(),
        checks='Complete planned identities, frozen settings, checkpoint signatures, same host/physical GPU pairing, new pilots, exact truth/masks, tensor hashes, independently recomputed float64 errors, measured NFE, input-level aggregation.'))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()

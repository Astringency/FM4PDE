"""Audit all frozen Rough NS predictions and report input-paired errors."""
import argparse
import csv
import json
import math
from pathlib import Path

from run_ns_rough_diffusion import sha, write

OUTCOMES = [('forward', 'u', 1), ('inverse', 'a', 0), ('both', 'a', 0), ('both', 'u', 1)]
NAMES = {('forward', 'u'): '正向 u', ('inverse', 'a'): '逆向 a', ('both', 'a'): '联合 a', ('both', 'u'): '联合 u'}


def csv_write(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(study, protocol):
    import numpy as np
    import torch
    from report_ns_checkpoint_comparison import font_setup, save
    font_setup()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    # Field comparisons have different signed values and strictly positive errors.
    signed = LinearSegmentedColormap.from_list('blue_orange', ['#245681', '#FAFAF7', '#BE681A'])
    absolute = LinearSegmentedColormap.from_list('gold', ['#FFFDF5', '#DDAF45', '#62491B'])
    data = np.load(study / 'inputs/fields_masks_bak.npz')
    i = protocol['evaluation_ids'][0]
    paths = {task: study / 'output/results' / task / 'DiffusionPDE_1000' / f'sample{i}_seed0.pt' for task in protocol['tasks']}
    if not all(p.exists() for p in paths.values()):
        print('Prespecified example is not complete yet; no alternative sample selected.')
        return
    payloads = {task: torch.load(path, map_location='cpu', weights_only=False) for task, path in paths.items()}
    fig, axes = plt.subplots(4, 5, figsize=(12.3, 9.7), layout='constrained')
    english = ['Forward u', 'Inverse a', 'Joint a', 'Joint u']
    for row, (task, field, k) in enumerate(OUTCOMES):
        truth = data[f'{task}_{i}_{field}'].squeeze().astype(float)
        bak = data[f'{task}_{i}_bak_{field}'].squeeze().astype(float)
        dm = payloads[task]['prediction'][k].squeeze().numpy()
        assert np.isfinite(dm).all()
        fields = [truth, bak, dm]
        errors = [np.abs(v - truth) for v in [bak, dm]]
        limit = max(np.abs(v).max() for v in fields)
        error_limit = max(v.max() for v in errors)
        labels = ['Reference', 'bak, 100 steps', 'DiffusionPDE, 1000 steps', 'bak absolute error', 'DiffusionPDE absolute error']
        for col, values in enumerate(fields + errors):
            ax = axes[row, col]
            im = ax.imshow(values, origin='lower', extent=[0, 1, 0, 1],
                           cmap=signed if col < 3 else absolute,
                           vmin=-limit if col < 3 else 0, vmax=limit if col < 3 else error_limit)
            title = labels[col]
            if col in [1, 2]:
                rel = 100 * np.linalg.norm(values - truth) / np.linalg.norm(truth)
                title += f'\nRel. L2 = {rel:.2f}%'
            ax.set_title(title, fontsize=9)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            if row == 3: ax.set_xlabel('x')
            if col == 0: ax.set_ylabel(english[row] + '\ny')
            if col == 2: fig.colorbar(im, ax=list(axes[row, :3]), shrink=.68, pad=.015)
            if col == 4: fig.colorbar(im, ax=list(axes[row, 3:]), shrink=.68, pad=.015)
    fig.suptitle(f'Rough NS reconstruction: fixed input {i}\nIdentical inputs and masks; 500 observations per observed field; DiffusionPDE seed 0', fontsize=13)
    save(fig, study / 'report/ns_rough_reconstruction')


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--figure-only', action='store_true')
    args = parser.parse_args()
    study, out = args.study, args.study / 'report'
    out.mkdir(exist_ok=True)
    p = json.loads((study / 'inputs/protocol.json').read_text())
    ph = sha(study / 'inputs/protocol.json')
    assert p['steps'] == [1000] and p['formal_calls'] == 96
    assert sha(study / 'inputs/source.json') == p['source_sha256']
    assert sha(study / 'inputs/fields_masks_bak.npz') == p['fields_sha256']
    if args.figure_only:
        figures(study, p)
        return
    arrays = np.load(study / 'inputs/fields_masks_bak.npz')
    receipts = list((study / 'output/results').glob('*/*/*.json'))
    expected = {(t, i, 1000, 0) for t in p['tasks'] for i in p['evaluation_ids']}
    actual, failures, verified, per_input = set(), [], [], []
    for path in receipts:
        r = json.loads(path.read_text())
        key = (r['task'], r['sample_id'], r['steps'], r['seed'])
        assert key not in actual and key in expected
        actual.add(key)
        assert r['protocol_sha256'] == ph and r['prediction_sha256'] == sha(path.with_suffix('.pt'))
        assert r['nfe'] == 1999
        payload = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
        task, i = r['task'], r['sample_id']
        for field, k in [('a', 0), ('u', 1)]:
            truth = torch.from_numpy(arrays[f'{task}_{i}_{field}']).double()
            mask = torch.from_numpy(arrays[f'{task}_{i}_mask_{field}']).double()
            assert torch.equal(payload['truth'][k].double(), truth)
            assert torch.equal(payload['masks'][k].double(), mask)
            assert payload['effective_config']['test']['iterations'] == 1000
        if r['status'] != 'complete':
            failures.append(r)
            continue
        assert all(bool(torch.isfinite(x).all()) for x in payload['prediction'])
        for field, k in [('a', 0), ('u', 1)]:
            truth = payload['truth'][k].double()
            dm = payload['prediction'][k].double()
            bak = torch.from_numpy(arrays[f'{task}_{i}_bak_{field}']).double()
            dm_error = float((dm - truth).norm() / truth.norm())
            bak_error = float((bak - truth).norm() / truth.norm())
            assert math.isclose(dm_error, r['relative_l2'][k], rel_tol=1e-10, abs_tol=1e-12)
            per_input.append(dict(task=task, field=field, sample_id=i, dm_steps=1000, bak_steps=100,
                 dm_seed=0, dm_error_pct=100 * dm_error, bak_error_pct=100 * bak_error,
                 dm_minus_bak_pp=100 * (dm_error - bak_error), source=str(path.relative_to(study))))
        verified.append(dict(source=str(path.relative_to(study)), prediction_sha256=r['prediction_sha256']))
    assert actual == expected, f'Missing {len(expected - actual)} calls; final report requires complete frozen cohort.'
    assert not failures, 'Numerical failures retained; report failure rate before summarizing successful samples.'
    for shard in range(p['workers']):
        pilot = json.loads((study / f'output/pilot_{shard}.json').read_text())
        assert pilot['status'] == 'pass' and pilot['protocol_sha256'] == ph
    csv_write(out / 'per_input.csv', per_input)
    summary = []
    rng = np.random.default_rng(20260911)
    for task, field, _ in OUTCOMES:
        selected = {r['sample_id']: r for r in per_input if (r['task'], r['field']) == (task, field)}
        assert set(selected) == set(p['evaluation_ids'])
        values = [selected[i] for i in p['evaluation_ids']]
        dm = np.array([r['dm_error_pct'] for r in values])
        bak = np.array([r['bak_error_pct'] for r in values])
        delta = dm - bak
        draws = rng.integers(0, 32, (10000, 32))
        lo, hi = np.quantile(delta[draws].mean(axis=1), [.025, .975])
        summary.append(dict(task=task, field=field, n_inputs=32, inference_seeds=1,
            dm_mean_pct=float(dm.mean()), dm_sd_pct=float(dm.std(ddof=1)),
            bak_mean_pct=float(bak.mean()), bak_sd_pct=float(bak.std(ddof=1)),
            dm_minus_bak_pp=float(delta.mean()), paired_ci95_low_pp=float(lo), paired_ci95_high_pp=float(hi),
            dm_lower_error_inputs=int((delta < 0).sum())))
    csv_write(out / 'summary.csv', summary)
    smooth_path = Path('/home/tat512/C04Papers/fm4pde_jmlr/source_data/diffusion_comparison_summary.csv')
    with smooth_path.open() as stream:
        smooth = {(r['task'], r['field']): r for r in csv.DictReader(stream)
                  if r['pde'] == 'nsnonbounded' and r['method'] == 'DiffusionPDE' and r['steps'] == '1000'}
    lines = ['# DiffusionPDE 在 Rough NS 上的固定输入测试', '',
        '1000 步评测已完成：32 个预先固定的 Rough 输入，正向、逆向、联合三个任务，共 96 次采样，全部得到有限预测。每个输入使用一个 DiffusionPDE 种子（0）；对照为同一真值、同一任务观测掩码下已保存的 bak＋旧版引导 100 步预测。', '',
        '相对 L2 误差（%，均值 ± 样本标准差，越低越好）。Smooth 列为此前 1000 输入主实验，仅作为背景，不属于本次配对样本。', '',
        '| 任务/场 | Rough：bak 100 步，32 输入 | Rough：DiffusionPDE 1000 步，32 输入 | Smooth：DiffusionPDE 1000 步，1000 输入 |',
        '|---|---:|---:|---:|']
    for r in summary:
        s = smooth[r['task'], r['field']]
        lines.append(f"| {NAMES[r['task'],r['field']]} | {r['bak_mean_pct']:.2f} ± {r['bak_sd_pct']:.2f} | {r['dm_mean_pct']:.2f} ± {r['dm_sd_pct']:.2f} | {100*float(s['mean']):.2f} ± {100*float(s['sd']):.2f} |")
    lines += ['', '配对差值为 DiffusionPDE − bak，负数代表 DiffusionPDE 误差较低。区间为以输入为单位重采样 10000 次的点态 95% bootstrap 区间，未作多重比较校正。', '',
        '| 任务/场 | 均值差（百分点） | 配对 95% 区间 | DiffusionPDE 误差较低的输入数 |', '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {NAMES[r['task'],r['field']]} | {r['dm_minus_bak_pp']:+.2f} | [{r['paired_ci95_low_pp']:+.2f}, {r['paired_ci95_high_pp']:+.2f}] | {r['dm_lower_error_inputs']}/32 |")
    lines += ['', '## 范围与来源', '',
        '- 评测仅运行用户指定的 1000 步；独立 pilot 的短步数运行只验证实现，未计入结果。',
        '- 每个被观测场使用 500 个观测值，联合任务对两个场各观测 500 个值；掩码逐条复用 bak 归档。',
        '- 沿用当前 DiffusionPDE 存档权重、物理尺度变换和引导参数，无 Rough 调参或重训练。原生 NS 引导为历史空间导数代理项，不是完整 NS 演化残差。',
        '- Rough 与 Smooth 的存档样本群体、观测掩码不同，跨分布误差差异仅作描述；本次可逐样本配对的是 Rough DiffusionPDE 与 Rough bak。',
        '- 每个输入仅一个采样种子，标准差反映输入间差异；不能据此判断跨种子稳定性，也不替代 1000 输入的全量评测。',
        '- 逐一校验所有 96 个预测文件哈希、真值、掩码、1999 次网络前向调用和从预测张量复算的相对 L2 误差。',
        '- inputs/protocol.json 固定样本、采样配置、代码与权重哈希；inputs/source.json 保存 bak 来源；output/results 保存全部新预测。',
        '- report/per_input.csv 与 report/summary.csv 为可复用数值表。', '',
        '[固定输入 903 的采样图](ns_rough_reconstruction.pdf)：按预先固定输入顺序选首个输入，未按效果选图。Times New Roman 字体，场值和误差分别共享每行色标。', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    write(out / 'audit.json', dict(status='complete', protocol_sha256=ph, calls=96, inputs=32,
          seeds=[0], failures=[], all_truths_masks_exactly_match_bak=True,
          all_prediction_metrics_recomputed=True, all_nfe_1999=True, verified_predictions=verified,
          source_code_sha256=sha(Path(__file__)), smooth_context_sha256=sha(smooth_path),
          pairing='Within Rough only; no matched-noise or paired Smooth/Rough claim.'))
    figures(study, p)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

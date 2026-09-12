"""Verify saved endpoint-term pairs and write a standalone diagnostic report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

PDES = ['nsnonbounded', 'reaction_diffusion', 'shallow_water', 'heat', 'wave', 'advection_diffusion']
NAMES = ['Navier–Stokes', 'Reaction–diffusion', 'Shallow water', 'Heat', 'Wave', 'Advection–diffusion']


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'complete.json').read_text())
    rows = manifest['rows']
    assert len(rows) == 12
    assert {(r['pde'], r['include_endpoint']) for r in rows} == {(p, b) for p in PDES for b in [True, False]}
    pairs = []
    for pde in PDES:
        pair = {r['include_endpoint']: r for r in rows if r['pde'] == pde}
        on, off = pair[True], pair[False]
        different = {k for k in on['config'] if on['config'][k] != off['config'][k]}
        assert different == {'hermite_include_integral_residual', 'output_dir'}
        for k in ['environment', 'source_weights_sha256', 'source_truth_sha256']:
            assert on[k] == off[k]
        payloads, masks = {}, {}
        for include, row in pair.items():
            remote_path = Path(row['result_path'])
            suffix = remote_path.parts[remote_path.parts.index(root.name) + 1:]
            local_path = root.joinpath(*suffix)
            assert digest(local_path) == row['result_sha256']
            assert digest(local_path.parent / 'masks.pt') == row['mask_sha256']
            payloads[include] = torch.load(local_path, map_location='cpu', weights_only=False)
            masks[include] = torch.load(local_path.parent / 'masks.pt', map_location='cpu', weights_only=False)
            assert payloads[include]['config']['hermite_include_integral_residual'] is include
        summary = {'pde': pde}
        for field, key in [('a', 'coef'), ('u', 'sol')]:
            truth = payloads[True][key + '_ground_truth'].double()
            assert torch.equal(truth, payloads[False][key + '_ground_truth'].double())
            assert torch.equal(masks[True][key], masks[False][key])
            for include, label in [(True, 'on'), (False, 'off')]:
                pred = payloads[include][key + '_final'].double()
                mask = masks[include][key].double()
                err = float((pred - truth).norm() / truth.norm())
                obs = float(((pred - truth)*mask).norm() / (truth*mask).norm())
                assert abs(err - pair[include]['errors'][field]['full']) < 1e-12
                assert abs(obs - pair[include]['errors'][field]['observed']) < 1e-12
                summary[field + '_' + label + '_pct'] = 100 * err
            summary[field + '_delta_pp'] = summary[field + '_off_pct'] - summary[field + '_on_pct']
            summary[field + '_paired_prediction_change_pct'] = 100 * float(
                (payloads[True][key + '_final'].double() - payloads[False][key + '_final'].double()).norm() / truth.norm())
            summary[field + '_on_replay_change_pct'] = 100 * on['prediction_difference_from_archived'][field]
        for include, label in [(True, 'on'), (False, 'off')]:
            d = pair[include]['common_residual_diagnostics']
            summary[label + '_interior_mse'] = d['interior_mse']
            summary[label + '_endpoint_mse'] = d['endpoint_mse']
        pairs.append(summary)
    repeats = []
    repeat_pair_deltas = []
    repeat_file = root / 'repeat_control/complete.json'
    if repeat_file.exists():
        for row in json.loads(repeat_file.read_text())['rows']:
            previous = next(r for r in rows if (r['pde'], r['include_endpoint']) ==
                            (row['pde'], row['include_endpoint']))
            assert row['environment'] == previous['environment']
            assert {k for k in row['config'] if row['config'][k] != previous['config'][k]} == {'output_dir'}
            predictions = []
            for r in [previous, row]:
                remote = Path(r['result_path'])
                local = root.joinpath(*remote.parts[remote.parts.index(root.name) + 1:])
                assert digest(local) == r['result_sha256']
                predictions.append(torch.load(local, map_location='cpu', weights_only=False))
            entry = {'pde': row['pde'], 'include_endpoint': row['include_endpoint']}
            for f, key in [('a', 'coef'), ('u', 'sol')]:
                truth = predictions[0][key + '_ground_truth'].double()
                assert torch.equal(truth, predictions[1][key + '_ground_truth'].double())
                assert torch.equal(predictions[0]['masks'][key], predictions[1]['masks'][key])
                entry[f + '_error_change_pp'] = 100*(row['errors'][f]['full'] - previous['errors'][f]['full'])
                entry[f + '_prediction_change_pct'] = 100*float(
                    (predictions[0][key + '_final'].double() - predictions[1][key + '_final'].double()).norm()/truth.norm())
            repeats.append(entry)
        repeat_rows = json.loads(repeat_file.read_text())['rows']
        for pde in sorted({r['pde'] for r in repeat_rows}):
            pair = {r['include_endpoint']: r for r in repeat_rows if r['pde'] == pde}
            assert set(pair) == {True, False}
            repeat_pair_deltas.append({'pde': pde, **{f + '_delta_pp':
                100*(pair[False]['errors'][f]['full'] - pair[True]['errors'][f]['full']) for f in ['a', 'u']}})
    (root / 'comparison.json').write_text(json.dumps({'pairs': pairs, 'repeat_controls': repeats,
        'repeat_pair_deltas': repeat_pair_deltas,
        'validation': 'passed', 'num_pairs': 6, 'samples_per_pde': 1, 'sample_seed': 0},
        indent=2, allow_nan=False) + '\n')
    with (root / 'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    lines = ['# Hermite 梯形端点项开关测试', '',
        '仅用于诊断；未修改论文。六个 PDE 各使用原 Temporal Residuals 实验的第 0 个样本，'
        '随机种子为 0，100 步采样。同一 GPU 上重新运行有端点项与无端点项两组，'
        '模型权重、真值、观测掩码和其余采样参数完全相同。复用已训练模型，没有重新训练。'
        '沿用原配置，在最后 20 步加入 PDE 引导，没有重新调节引导强度。', '',
        '唯一计算配置差异为 `hermite_include_integral_residual=True/False`。'
        '开启组保留原权重 `hermite_integral_weight=endpoint_bc_weight=1`。'
        '关闭组的实际端点残差通道数、端点损失和端点残差范数均为 0。', '',
        '误差为全部物理通道上的 $100\\|\\hat x-x\\|_2/\\|x\\|_2$。'
        'Δ = 不加 − 加，单位为百分点；负值表示去掉该项后误差下降。', '',
        '| PDE | a 加 (%) | a 不加 (%) | Δa | u 加 (%) | u 不加 (%) | Δu |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r, name in zip(pairs, NAMES):
        vals = [r[k] for k in ['a_on_pct', 'a_off_pct', 'a_delta_pp', 'u_on_pct', 'u_off_pct', 'u_delta_pp']]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.6f}' for v in vals) + ' |')
    max_replay = max(r[f + '_on_replay_change_pct'] for r in pairs for f in ['a', 'u'])
    lines += ['', f'开启组相对原归档预测的最大差异（以真值范数归一化）为 {max_replay:.9g}%。', '',
        '这是每个方程单个样本、单个种子的配对诊断，没有估计均值、方差或统计显著性。'
        '不能据此断言该项对所有样本都有效或都无效。', '',
        '另在两组最终预测上，以相同公式、CPU float64 重算配点残差 MSE 和梯形端点残差 MSE，'
        '存于 comparison.csv。无端点项组的端点残差仅作事后诊断，未加入该组采样。'
        '残差值的比例不等于梯度贡献的比例。', '',
        '原始预测、逐步日志、配置和 SHA-256 校验值保存在各 PDE 子目录的 receipt.json 及 result.pt；'
        '运行命令及退出码位于 execution/；独立代码归档位于 sources/code.bundle。', '',
        '运行环境：' + json.dumps(manifest['environment'], ensure_ascii=False), '']
    if repeats:
        lines += ['重复运行检查（相同配置、样本和种子）：', '',
                  '| PDE | 端点项 | a 误差变化（百分点） | u 误差变化（百分点） |',
                  '|---|---|---:|---:|']
        for r in repeats:
            lines.append(f"| {r['pde']} | {'加' if r['include_endpoint'] else '不加'} | "
                         f"{r['a_error_change_pp']:.9g} | {r['u_error_change_pp']:.9g} |")
        lines.append('')
        lines += ['重复配对得到的不加 − 加误差差值：', '',
                  '| PDE | 第一次 Δa | 第二次 Δa | 第一次 Δu | 第二次 Δu |',
                  '|---|---:|---:|---:|---:|']
        for r in repeat_pair_deltas:
            first = next(x for x in pairs if x['pde'] == r['pde'])
            lines.append(f"| {r['pde']} | {first['a_delta_pp']:.6f} | {r['a_delta_pp']:.6f} | "
                         f"{first['u_delta_pp']:.6f} | {r['u_delta_pp']:.6f} |")
        lines += ['', '相同种子下的重复运行也存在数值波动。若差值大小或符号随重复运行改变，'
                  '不能将第一次的差值直接解释为该项带来的稳定改善或退化。', '']
    (root / 'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:18]))


if __name__ == '__main__':
    main()

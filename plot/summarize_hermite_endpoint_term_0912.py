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
    (root / 'comparison.json').write_text(json.dumps({'pairs': pairs, 'validation': 'passed',
        'num_pairs': 6, 'samples_per_pde': 1, 'sample_seed': 0}, indent=2, allow_nan=False) + '\n')
    with (root / 'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    lines = ['# Hermite 梯形端点项开关测试', '',
        '仅用于诊断；未修改论文。六个 PDE 各使用原 Temporal Residuals 实验的第 0 个样本，'
        '随机种子为 0，100 步采样。同一 GPU 上重新运行有端点项与无端点项两组，'
        '模型权重、真值、观测掩码和其余采样参数完全相同。', '',
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
    (root / 'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:18]))


if __name__ == '__main__':
    main()

"""Build a portable report input from audited main-model continuation results.

The default permits an explicitly partial progress snapshot. --require-complete
requires all five 50-epoch runs and all 66 paired thousand-input evaluations.
This collector never starts training, selects a checkpoint, or changes results.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


PDES = ('nsnonbounded', 'poisson', 'darcy', 'helmholtz', 'burger')
LABELS = dict(nsnonbounded='NS', poisson='Poisson', darcy='Darcy',
              helmholtz='Helmholtz', burger='Burgers')
TITLE = 'FM4PDE main-model continuation and evaluation'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_fields(pde, task):
    return ('u',) if pde == 'burger' or task == 'forward' else (
        ('a',) if task == 'inverse' else ('a', 'u'))


def interval_label(low, high):
    return '改善' if high < 0 else '退化' if low > 0 else '区间含 0'


def verify_comparison(before, after, expected_ids, reported):
    """Recompute every summary from exactly 1000 correctly paired score rows."""
    require(len(expected_ids) == len(set(expected_ids)) == 1000, 'Expected 1000 unique inputs')
    for rows in (before, after):
        require([r['sample_id'] for r in rows] == expected_ids, 'Score input membership/order mismatch')
    fields = set(reported)
    require(fields and all(set(r['fields']) == fields for r in before + after), 'Field mismatch')
    for field in fields:
        metrics = set(reported[field])
        require(all(set(r['fields'][field]) == metrics for r in before + after), 'Metric mismatch')
        for metric, saved in reported[field].items():
            a = [r['fields'][field][metric] for r in before]
            b = [r['fields'][field][metric] for r in after]
            require(all(math.isfinite(v) for v in a + b), 'Nonfinite score')
            delta = [y - x for x, y in zip(a, b)]
            mean = statistics.fmean(delta)
            half = 1.96 * statistics.stdev(delta) / math.sqrt(1000)
            actual = dict(n=1000, baseline_mean=statistics.fmean(a),
                          resumed_mean=statistics.fmean(b), mean_change=mean)
            if metric.endswith('relative_l2'):
                require(actual['baseline_mean'] > 0, 'Undefined relative change')
                actual['relative_change'] = actual['resumed_mean'] / actual['baseline_mean'] - 1
            for key, value in actual.items():
                require(math.isclose(saved[key], value, rel_tol=1e-10, abs_tol=1e-12),
                        f'Summary mismatch: {field}/{metric}/{key}')
            require(len(saved['paired_95ci']) == 2, 'Invalid interval')
            for saved_value, value in zip(saved['paired_95ci'], (mean - half, mean + half)):
                require(math.isclose(saved_value, value, rel_tol=1e-10, abs_tol=1e-12),
                        f'Interval mismatch: {field}/{metric}')


def collect(study, require_complete=False):
    study = Path(study).resolve()
    evidence = {}

    def read(relative):
        p = study / relative
        data = p.read_bytes()
        evidence[str(relative)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    plan = read('study_plan.json')
    jobs = {j['pde']: j for j in plan['jobs']}
    require(set(jobs) == set(PDES) and len(plan['jobs']) == 5, 'Wrong main-model scope')
    binding = {r['pde']: r for r in read('main_checkpoint_binding.json')['models']}
    require(set(binding) == set(PDES), 'Incomplete model bindings')
    catalog = read('evaluation_inputs/catalog.json')
    cells = catalog['cells']
    require(catalog['status'] == 'complete' and len(cells) == 66, 'Incomplete main input catalog')
    require(len({r['cell'] for r in cells}) == 66, 'Duplicate main cells')
    canonical = Path(jobs['nsnonbounded']['output']).parent
    training, metrics, selected, pending = [], [], [], []
    for pde in PDES:
        job, bound = jobs[pde], binding[pde]
        require(job['source'] == bound['resume_checkpoint'] and job['epochs'] == 50, 'Source/epoch mismatch')
        require(bound['epoch'] == 299 and all(bound[k] for k in
                ('inference_weights_equal', 'normalizer_equal', 'architecture_equal')), 'Unverified main model')
        if pde == 'nsnonbounded':
            require('/260904-' in job['source'], 'NS must use the 260904 main model')
        expected = [c for c in cells if c['pde'] == pde]
        require(len(expected) == (6 if pde == 'burger' else 15), 'Incorrect PDE cell count')
        require(all(c['n'] == 1000 for c in expected), 'Incorrect catalog input count')
        progress_path = Path(pde) / 'progress.json'
        progress = read(progress_path) if (study / progress_path).exists() else {}
        epochs = progress.get('additional_epoch', 0)
        updates = progress.get('updates', 0)
        require(0 <= epochs <= 50 and updates == epochs * 704, 'Training progress mismatch')
        for name in ('exit.json', 'evaluation.exit.json', 'final_audit.exit.json'):
            path = Path(pde) / name
            if (study / path).exists():
                require(read(path)['exit_code'] == 0, f'{pde}/{name} reports failure')
        ready = all((study / pde / name).exists() for name in
                    ('training_complete.json', 'exit.json', 'evaluation/complete.json',
                     'evaluation.exit.json', 'final_artifact_audit.json', 'final_audit.exit.json'))
        training.append(dict(pde=LABELS[pde], key=pde, completed_epochs=epochs,
                             planned_epochs=50, updates=updates, audited_cells=len(expected) if ready else 0,
                             planned_cells=len(expected), status='评估已核验' if ready else '评估待完成'))
        if not ready:
            pending.append(pde)
            continue
        done = read(Path(pde) / 'training_complete.json')
        complete = read(Path(pde) / 'evaluation/complete.json')
        audit = read(Path(pde) / 'final_artifact_audit.json')
        selection = read(Path(pde) / 'evaluation/selection.json')
        require(done['completed_epochs'] == 50 and done['updates'] == 35200, 'Incomplete continuation')
        require(epochs == 50 and updates == 35200, 'Final progress does not match completion')
        require(audit['training_verified'] and audit['evaluation_verified'] and
                audit['epochs'] == 50 and audit['updates'] == 35200, 'Incomplete artifact audit')
        require(audit['evaluation_complete_sha256'] == evidence[f'{pde}/evaluation/complete.json'] and
                audit['selection_sha256'] == evidence[f'{pde}/evaluation/selection.json'], 'Stale final audit')
        require(complete['status'] == 'complete' and complete['samples_per_cell'] == 1000 and
                complete['steps'] == 100 and complete['seeds'] == 1 and
                complete['paired_masks_noise_config_verified'], 'Incorrect sampling protocol')
        require(done['source_sha256'] == complete['source_sha256'] == bound['resume_sha256'], 'Source hash mismatch')
        require(complete['selected_sha256'] == selection['checkpoint_sha256'] and
                complete['selection_sha256'] == evidence[f'{pde}/evaluation/selection.json'] and
                selection['test_results_used'] is False, 'Invalid checkpoint selection')
        reports = {r['cell']: r for r in complete['reports']}
        require(set(reports) == {c['cell'] for c in expected} and
                len(complete['reports']) == complete['cells'] == len(expected), 'Missing/duplicate comparison')
        selected.append(dict(pde=LABELS[pde], completed_epochs=selection['completed_epochs'],
                             checkpoint=Path(selection['checkpoint']).name,
                             sha256=selection['checkpoint_sha256'], validation_score=selection['score']))
        for cell in expected:
            record_relative = Path(cell['record_path']).relative_to(canonical)
            record = read(record_relative)
            require(evidence[str(record_relative)] == cell['record_sha256'], 'Changed input record')
            folder = Path(pde) / 'evaluation/main' / cell['dist'] / cell['setting']
            report = read(folder / 'comparison.json')
            require(report == reports[cell['cell']], 'Changed cell comparison')
            scores = [read(folder / label / 'scores.json') for label in ('baseline', 'resumed')]
            for label in ('baseline', 'resumed'):
                require(evidence[str(folder / label / 'scores.json')] == report[f'{label}_scores_sha256'],
                        'Changed paired score file')
            verify_comparison(*scores, record['sample_ids'], report['full'])
            for field, values in report['full'].items():
                for metric, values_row in values.items():
                    low, high = values_row['paired_95ci']
                    metrics.append(dict(pde=LABELS[pde], key=pde, cell=cell['cell'], dist=cell['dist'],
                        setting=cell['setting'], task=cell['task'], field=field, metric=metric,
                        target=field in target_fields(pde, cell['task']), n=values_row['n'],
                        baseline=values_row['baseline_mean'], resumed=values_row['resumed_mean'],
                        mean_change=values_row['mean_change'], relative_change=values_row.get('relative_change'),
                        ci_low=low, ci_high=high,
                        interval=interval_label(low, high) if metric.endswith('relative_l2') else '诊断量'))
    if require_complete:
        require(not pending, 'Full study is pending: ' + ', '.join(pending))
    return dict(status='partial' if pending else 'complete', pending_pdes=pending,
                training=training, selected=selected, metrics=metrics,
                audited_cells=sum(r['audited_cells'] for r in training), planned_cells=66,
                samples_per_cell=1000, evidence=evidence,
                generated_at=datetime.now(timezone.utc).isoformat())


def artifact(summary):
    date = summary['generated_at']
    full = summary['status'] == 'complete'
    sources = [dict(id='results', label='summary.json: verified experiment snapshot and source file hashes',
                    path='summary.json')]
    blocks, charts, tables = [], [], []
    datasets = dict(training=summary['training'])

    def paragraph(key, body, sourced=False):
        row = dict(id=key, type='markdown', body=body)
        if sourced:
            row['sourceId'] = 'results'
        blocks.append(row)

    def bar(key, title, dataset, x, y, percent=False):
        charts.append(dict(id=key, title=title, type='bar', dataset=dataset, sourceId='results', layout='full',
            encodings=dict(x=dict(field=x, type='nominal'), y=dict(field=y, type='quantitative',
                format='percent' if percent else 'number')),
            settings=dict(orientation='horizontal', groupMode='single', sort='none', showValues=True),
            palette=dict(kind='diverging' if percent else 'categorical', midpoint=0) if percent
                    else dict(kind='categorical'), maxRows=50))
        blocks.append(dict(id=key + '-block', type='chart', chartId=key, layout='full'))

    def table(key, title, dataset, columns, sort):
        tables.append(dict(id=key, title=title, dataset=dataset, sourceId='results', density='dense',
                           columns=columns, defaultSort=dict(field=sort, direction='asc')))
        blocks.append(dict(id=key + '-block', type='table', tableId=key, layout='full'))

    paragraph('title', '# ' + TITLE)
    targets = [r for r in summary['metrics'] if r['metric'] == 'relative_l2' and r['target']]
    if full:
        counts = {key: sum(r['interval'] == key for r in targets) for key in ('改善', '退化', '区间含 0')}
        paragraph('summary', '## 技术摘要\n\n五个主实验模型均完成连续 50 轮续训，全部 66 组各完成 '
            '1000 个唯一输入的原模型/续训模型配对采样并通过审计。'
            f'任务目标场指标中，{counts["改善"]} 项的配对区间低于 0，{counts["退化"]} 项高于 0，'
            f'{counts["区间含 0"]} 项包含 0。不同任务可能有相反变化，不能据此宣称所有场景统一提升。', True)
    else:
        paragraph('summary', '## 技术摘要\n\n**实验尚未完成，当前不能给出新模型整体精度结论。** '
            f'已审计的完整评估为 {summary["audited_cells"]}/66 组。下图只表示各模型最后已记录的完整训练轮数；'
            '0 表示尚无完成轮次记录。等待中的 PDE 不会被补成零误差或算作持平。', True)
    paragraph('definitions', '## 每组固定 1000 个配对输入\n\n'
        '误差为物理场相对 L2，越小越好；变化率为续训均值除以原模型均值再减 1。'
        '正向任务以解场 u 为目标，反向任务以输入场 a 为目标，联合任务分别报告 a、u；Burgers 报告 u。'
        '每个单元使用一个噪声种子和 100 个采样步，原模型与续训模型使用相同输入、观测掩码、噪声和 native 采样器。'
        '1000 是每组唯一输入数；同一输入跨任务重复使用，不能将所有组拼成独立样本。', True)
    paragraph('training-progress', '## 连续续训的完成轮次\n\n'
        '每个主实验模型的目标是连续新增 50 轮。下图的单位是完整 epoch，详细表同时给出更新次数和已审计评估组数。'
        '完成训练不等于已完成采样评估。', True)
    bar('epochs', '各模型已完成的新增轮次', 'training', 'pde', 'completed_epochs')
    table('training-table', '训练与评估覆盖', 'training', [dict(field=k, label=v) for k, v in
        [('pde', '模型'), ('completed_epochs', '新增轮次'), ('updates', 'Adam 更新'),
         ('audited_cells', '已审计组数'), ('planned_cells', '计划组数'), ('status', '评估状态')]], 'pde')
    for pde in PDES:
        rows = [dict(r, label=f'{r["dist"]} / {r["setting"]} / {r["field"]}')
                for r in targets if r['key'] == pde]
        if not rows:
            continue
        datasets[pde] = rows
        counts = {key: sum(r['interval'] == key for r in rows) for key in ('改善', '退化', '区间含 0')}
        paragraph(pde + '-finding', f'## {LABELS[pde]} 的任务目标场变化\n\n'
            f'{counts["改善"]} 项区间支持误差下降，{counts["退化"]} 项支持上升，'
            f'{counts["区间含 0"]} 项尚未分辨变化。图中负值表示误差均值下降；'
            '是否能分辨变化需同时看表中的配对区间。所有分布和任务均保留，未只挑选改善项。', True)
        bar(pde + '-change', LABELS[pde] + ' 相对 L2 均值变化率', pde, 'label', 'relative_change', True)
        table(pde + '-table', LABELS[pde] + ' 任务目标场误差', pde,
            [dict(field=k, label=v) for k, v in [('label', '分布 / 设置 / 场'), ('n', '输入数'),
             ('baseline', '原模型'), ('resumed', '续训模型'), ('mean_change', '配对均值差'),
             ('ci_low', '区间下界'), ('ci_high', '区间上界'), ('interval', '区间判断')]], 'label')
    paragraph('design', '## 主模型绑定与训练验证集选模\n\n'
        'NS 固定使用 260904 主模型，其余 PDE 使用论文主实验对应的完整 checkpoint。'
        '恢复 Adam 的一、二阶矩和 step；每轮使用 45000 个训练输入和 704 次更新，'
        'FP32 张量并启用 TF32。NS 学习率从 1e-5、其余从 3e-6 开始，预热后余弦下降。'
        '从每 5 轮 checkpoint 和 FM 验证最优 checkpoint 中，用 32 个训练验证输入的实际条件采样选择模型；'
        '在查看正式测试结果前冻结选择。旧主实验误差最大的 25 个输入仅用于诊断，随后补齐剩余 975 个。', True)
    if summary['selected']:
        datasets['selected'] = summary['selected']
        table('selected-models', '训练验证集选中的 checkpoint', 'selected',
            [dict(field=k, label=v) for k, v in [('pde', '模型'), ('completed_epochs', '新增轮次'),
             ('checkpoint', '文件名'), ('validation_score', '验证采样分数'), ('sha256', 'SHA-256')]], 'pde')
    paragraph('uncertainty', '## 区间含 0 不证明等价\n\n'
        '95% 区间来自 1000 个输入的配对差，未作多重比较校正，且不同任务共享输入。'
        '当前一个噪声种子不衡量跨种子稳定性；历史难例上的改善不代表总体改善。'
        '旧六月模型可能在更早预训练中见过当前验证池，本轮续训严格排除该池；'
        'NS 260904 使用原先相同的数据拆分。历史 NS 采样器和硬件差异通过本轮重新运行配对基线控制。'
        '低频误差、功率比和对齐度保留在完整指标 CSV 中；功率相近不能替代系数方向一致。')
    paragraph('next', '## 后续行动\n\n' + ('逐 PDE 和目标任务判断是否采用续训模型，保留退化场景；'
        '整体均值变化不能替代场景级比较。' if full else '继续完成五个模型的 50 轮训练、每组 1000 输入的配对评估和原始产物审计；'
        '在完成前保持旧主实验模型与结果不变。'))
    paragraph('questions', '## 仍需回答的问题\n\n'
        '改善是否跨噪声种子稳定？哪些任务或分布出现相反变化？低频幅度与模态对齐的变化是否和场误差一致？'
        '这些问题需要结合完整指标及必要的后续复验，不能仅由 FM 训练损失回答。')
    result = dict(surface='report', manifest=dict(version=1, surface='report', title=TITLE, generatedAt=date,
        blocks=blocks, charts=charts, tables=tables, sources=sources),
        snapshot=dict(version=1, generatedAt=date, status='ready' if full else 'partial', datasets=datasets), sources=sources)
    if not full:
        result['snapshot']['accessIssues'] = [dict(id='pending-evaluation', scope='final paired evaluation',
            message='等待完整训练和评估：' + ', '.join(LABELS[p] for p in summary['pending_pdes']))]
    return result


def write_outputs(output, summary):
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary.json', summary), ('artifact.json', artifact(summary))]:
        temp = output / (name + '.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temp.replace(output / name)
    if summary['metrics']:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(summary['metrics'][0]))
        writer.writeheader()
        writer.writerows(summary['metrics'])
        (output / 'metrics.csv').write_text(buffer.getvalue())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    summary = collect(args.study, args.require_complete)
    write_outputs(args.output.resolve(), summary)
    print(json.dumps(dict(status=summary['status'], audited_cells=summary['audited_cells'],
                         pending=summary['pending_pdes']), ensure_ascii=False))

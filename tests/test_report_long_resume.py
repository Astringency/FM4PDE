import json
from pathlib import Path

import pytest

from scripts.train.report_long_resume import (
    PDES, artifact, collect, interval_label, sha, target_fields, verify_comparison,
)


def put(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture_study(root):
    jobs, bindings, cells = [], [], []
    for pde in PDES:
        source = f'/canonical/formal/{pde}/260904-model/checkpoint.pth'
        jobs.append(dict(pde=pde, epochs=50, source=source, output=f'/canonical/study/{pde}'))
        bindings.append(dict(pde=pde, resume_checkpoint=source, epoch=299, resume_sha256='a' * 64,
                             inference_weights_equal=True, normalizer_equal=True, architecture_equal=True))
        for i in range(6 if pde == 'burger' else 15):
            cells.append(dict(pde=pde, cell=f'{pde}/id/setting{i}', dist='id', setting=f'setting{i}',
                              task='both', n=1000, record_path=f'/canonical/study/inputs/{pde}/{i}.json'))
    put(root, 'study_plan.json', dict(jobs=jobs))
    put(root, 'main_checkpoint_binding.json', dict(models=bindings))
    put(root, 'evaluation_inputs/catalog.json', dict(status='complete', cells=cells))
    return cells


def test_partial_snapshot_cannot_be_presented_as_full_study(tmp_path):
    fixture_study(tmp_path)
    put(tmp_path, 'nsnonbounded/progress.json', dict(additional_epoch=17, updates=11968))
    summary = collect(tmp_path)
    assert summary['status'] == 'partial' and summary['audited_cells'] == 0
    assert summary['metrics'] == [] and summary['training'][0]['completed_epochs'] == 17
    assert artifact(summary)['snapshot']['status'] == 'partial'
    with pytest.raises(ValueError, match='Full study is pending'):
        collect(tmp_path, require_complete=True)


def test_wrong_ns_checkpoint_and_failed_process_are_rejected(tmp_path):
    fixture_study(tmp_path)
    for name in ['study_plan.json', 'main_checkpoint_binding.json']:
        p = tmp_path / name
        p.write_text(p.read_text().replace('nsnonbounded/260904-', 'nsnonbounded/260712-'))
    with pytest.raises(ValueError, match='260904'):
        collect(tmp_path)
    fixture_study(tmp_path)
    put(tmp_path, 'burger/exit.json', dict(exit_code=1))
    with pytest.raises(ValueError, match='reports failure'):
        collect(tmp_path)


def test_thousand_input_pairing_and_summary_must_agree():
    ids = list(range(1000))
    a = [dict(sample_id=i, fields=dict(u=dict(relative_l2=1.0))) for i in ids]
    b = [dict(sample_id=i, fields=dict(u=dict(relative_l2=0.5))) for i in ids]
    report = dict(u=dict(relative_l2=dict(n=1000, baseline_mean=1.0, resumed_mean=0.5,
        mean_change=-0.5, relative_change=-0.5, paired_95ci=[-0.5, -0.5])))
    verify_comparison(a, b, ids, report)
    for wrong in (b[:-1], b[::-1], b[:-1] + [b[0]]):
        with pytest.raises(ValueError, match='membership/order'):
            verify_comparison(a, wrong, ids, report)
    report['u']['relative_l2']['n'] = 999
    with pytest.raises(ValueError, match='Summary mismatch'):
        verify_comparison(a, b, ids, report)


def test_zero_crossing_and_observed_fields_are_not_counted_as_wins():
    assert interval_label(-0.2, 0.1) == interval_label(0.0, 0.0) == '区间含 0'
    assert interval_label(-0.2, -0.1) == '改善'
    assert interval_label(0.1, 0.2) == '退化'
    assert target_fields('nsnonbounded', 'inverse') == ('a',)
    assert target_fields('poisson', 'forward') == ('u',)
    assert target_fields('darcy', 'both') == ('a', 'u')
    assert target_fields('burger', 'both') == ('u',)


def test_only_current_audited_cells_enter_accuracy_report(tmp_path):
    cells = fixture_study(tmp_path)
    ids = list(range(1000))
    reports = []
    for i, cell in enumerate(c for c in cells if c['pde'] == 'burger'):
        relative = Path(cell['record_path']).relative_to('/canonical/study')
        cell['record_sha256'] = sha(put(tmp_path, relative, dict(sample_ids=ids)))
        folder = Path('burger/evaluation/main/id') / f'setting{i}'
        hashes = {}
        for label, value in [('baseline', 1.0), ('resumed', 0.5)]:
            rows = [dict(sample_id=k, fields=dict(u=dict(relative_l2=value))) for k in ids]
            hashes[label] = sha(put(tmp_path, folder / label / 'scores.json', rows))
        report = dict(cell=cell['cell'], baseline_scores_sha256=hashes['baseline'],
            resumed_scores_sha256=hashes['resumed'], full=dict(u=dict(relative_l2=dict(n=1000,
                baseline_mean=1.0, resumed_mean=0.5, mean_change=-0.5, relative_change=-0.5,
                paired_95ci=[-0.5, -0.5]))))
        put(tmp_path, folder / 'comparison.json', report)
        reports.append(report)
    put(tmp_path, 'evaluation_inputs/catalog.json', dict(status='complete', cells=cells))
    put(tmp_path, 'burger/progress.json', dict(additional_epoch=50, updates=35200))
    put(tmp_path, 'burger/training_complete.json', dict(completed_epochs=50, updates=35200, source_sha256='a' * 64))
    for name in ('exit.json', 'evaluation.exit.json', 'final_audit.exit.json'):
        put(tmp_path, 'burger/' + name, dict(exit_code=0))
    selection_hash = sha(put(tmp_path, 'burger/evaluation/selection.json', dict(checkpoint_sha256='b' * 64,
        test_results_used=False, checkpoint='/canonical/study/burger/005.pth', completed_epochs=5, score=0.5)))
    complete = dict(status='complete', cells=6, reports=reports, samples_per_cell=1000, steps=100, seeds=1,
        paired_masks_noise_config_verified=True, source_sha256='a' * 64, selected_sha256='b' * 64,
        selection_sha256=selection_hash)
    complete_hash = sha(put(tmp_path, 'burger/evaluation/complete.json', complete))
    put(tmp_path, 'burger/final_artifact_audit.json', dict(training_verified=True, evaluation_verified=True,
        epochs=50, updates=35200, evaluation_complete_sha256=complete_hash, selection_sha256=selection_hash))
    summary = collect(tmp_path)
    assert summary['status'] == 'partial' and summary['audited_cells'] == 6
    assert len(summary['metrics']) == 6 and all(r['n'] == 1000 for r in summary['metrics'])
    assert len(artifact(summary)['snapshot']['datasets']['burger']) == 6
    with pytest.raises(ValueError, match='Full study is pending'):
        collect(tmp_path, require_complete=True)
    complete['steps'] = 99
    put(tmp_path, 'burger/evaluation/complete.json', complete)
    with pytest.raises(ValueError, match='Stale final audit'):
        collect(tmp_path)

"""Verify checkpoint tensors and paired evaluation artifacts from a resume study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def checkpoint_audit(path, source, pde):
    from train import _build_lr_scheduler

    value = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert value['checkpoint_schema_version'] == 3
    assert value['args'].dataset == pde and value['epoch'] > source['epoch']
    same(value['model_config'], source['model_config'])
    same(value['normalizer'], source['normalizer'])
    weights = value['model_for_resume']
    original = source['model_for_resume']
    assert weights.keys() == original.keys()
    same(weights, value['model'])
    change2, original2, changed_tensors = 0.0, 0.0, 0
    for key, tensor in weights.items():
        reference = original[key]
        assert tensor.shape == reference.shape and tensor.dtype == reference.dtype
        assert torch.isfinite(tensor).all(), key
        if tensor.is_floating_point():
            delta = tensor.double() - reference.double()
            change2 += float(delta.square().sum())
            original2 += float(reference.double().square().sum())
            changed_tensors += int(bool(torch.any(delta != 0)))
    assert changed_tensors > 0 and change2 > 0
    old_states, states = source['optimizer']['state'], value['optimizer']['state']
    assert states.keys() == old_states.keys()
    updates = int(value['resume_study']['updates'])
    assert updates > 0
    steps = set()
    for key, state in states.items():
        step = int(state['step'])
        assert step == int(old_states[key]['step']) + updates
        steps.add(step)
        for name in ('exp_avg', 'exp_avg_sq'):
            assert state[name].shape == old_states[key][name].shape
            assert state[name].dtype == torch.float32
            assert torch.isfinite(state[name]).all(), (key, name)
        assert (state['exp_avg_sq'] >= 0).all()
    assert len(steps) == 1
    # Exercise the native scheduler constructor against the saved metadata.
    prototype = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=value['args'].lr)
    scheduler = _build_lr_scheduler(prototype, value['args'], value['resolved_lr_scheduler'])
    scheduler.load_state_dict(value['lr_schedule'])
    same(scheduler.state_dict(), value['lr_schedule'])
    return dict(path=str(path), sha256=sha(path), epoch=int(value['epoch']),
                updates=updates, optimizer_step=steps.pop(), optimizer_states=len(states),
                changed_weight_tensors=changed_tensors,
                relative_model_state_change=(change2/max(original2, 1e-30))**0.5,
                normalizer_unchanged=True, architecture_unchanged=True,
                native_scheduler_roundtrip_verified=True,
                weights_and_optimizer_moments_finite=True)


def audit_job(job, require_evaluation):
    out = Path(job['output'])
    assert '/outputs/pretrained/' in str(out.resolve())
    complete = json.loads((out/'complete.json').read_text())
    assert json.loads((out/'exit.json').read_text())['exit_code'] == 0
    protocol = json.loads((out/'protocol.json').read_text())
    assert all('test' not in x['path'].lower() for x in protocol['training_files'])
    source_path = Path(job['checkpoint'])
    assert sha(source_path) == protocol['checkpoint_sha256']
    source = torch.load(source_path, map_location='cpu', weights_only=False, mmap=True)
    checkpoints = {name: checkpoint_audit(out/f'{name}_resume.pth', source, job['pde'])
                   for name in ('best', 'last')}
    assert checkpoints['best']['sha256'] == complete['best_sha256']
    assert checkpoints['best']['epoch'] == complete['best_epoch']
    assert checkpoints['best']['optimizer_step'] == complete['best_optimizer_step']
    assert checkpoints['last']['epoch'] == complete['last_epoch']
    split = torch.load(out/'split_indices.pt', map_location='cpu', weights_only=True)
    train_ids, val_ids = set(split['train_indices'].tolist()), set(split['val_indices'].tolist())
    assert len(train_ids) == 45000 and len(val_ids) == 5000 and not train_ids & val_ids
    assert train_ids | val_ids == set(range(50000))
    dev, confirm = set(split['development_positions'].tolist()), set(split['confirmation_positions'].tolist())
    assert len(dev) == 512 and len(confirm) == 4488 and not dev & confirm
    before = json.loads((out/'baseline_confirmation.json').read_text())
    after = json.loads((out/'selected_confirmation.json').read_text())
    assert before['positions'] == after['positions'] == split['confirmation_positions'].tolist()
    b, a = np.asarray(before['per_sample']), np.asarray(after['per_sample'])
    assert b.shape == a.shape == (4488,) and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.isclose(b.mean(), complete['baseline_confirmation_mse'], rtol=1e-12, atol=1e-14)
    assert np.isclose(a.mean(), complete['selected_confirmation_mse'], rtol=1e-12, atol=1e-14)
    result = dict(pde=job['pde'], training_artifacts_verified=True,
                  source_sha256=protocol['checkpoint_sha256'], checkpoints=checkpoints,
                  current_continuation_train_validation_disjoint=True,
                  confirmation_positions_disjoint_from_development=True,
                  historical_validation_caveat='See validation_provenance.json; historical unseen status is not established.',
                  evaluation_verified=False)
    evaluation = out/'evaluation'
    if (evaluation/'complete.json').exists():
        final = json.loads((evaluation/'complete.json').read_text())
        assert final['status'] == 'complete' and final['samples'] == 32 and final['seeds'] == 2
        assert final['paired_masks_and_rng_verified']
        assert final['checkpoint_sha256'] == {'baseline':protocol['checkpoint_sha256'], 'resumed':complete['best_sha256']}
        tasks = ['both'] if job['pde'] == 'burger' else ['both', 'inverse']
        fields = ['u'] if job['pde'] == 'burger' else ['a', 'u']
        expected = {(task, field, metric) for task in tasks for field in fields
                    for metric in ('relative_l2', 'low_relative_l2', 'low_power_ratio', 'low_alignment')}
        assert {(r['task'], r['field'], r['metric']) for r in final['comparisons']} == expected
        for row in final['comparisons']:
            b, a = np.asarray(row['per_input_baseline']), np.asarray(row['per_input_resumed'])
            assert b.shape == a.shape == (32,) and np.isfinite(a).all() and np.isfinite(b).all()
            assert np.isclose(b.mean(), row['baseline'], rtol=1e-12, atol=1e-14)
            assert np.isclose(a.mean(), row['resumed'], rtol=1e-12, atol=1e-14)
        receipts = {}
        for task in tasks:
            for label in ('baseline', 'resumed'):
                rows = [json.loads(path.read_text()) for path in (evaluation/task/label).glob('*/receipt.json')]
                for seed in (0, 1):
                    ids = [i for row in rows if row['seed'] == seed for i in row['sample_ids']]
                    assert sorted(ids) == list(range(1500, 1532))
                for row in rows:
                    assert sha(row['result_path']) == row['result_sha256']
                    assert row['checkpoint_sha256'] == final['checkpoint_sha256'][label]
                    receipts[(task, label, row['seed'], tuple(row['sample_ids']))] = row
        for (task, label, seed, ids), row in receipts.items():
            if label != 'baseline':
                continue
            other = receipts[(task, 'resumed', seed, ids)]
            for key in ('mask_sha256', 'initial_noise_sha256', 'rng_after_initial_sha256'):
                assert row[key] == other[key]
        result.update(evaluation_verified=True, verified_prediction_batches=len(receipts),
                      evaluation_summary_sha256=sha(evaluation/'complete.json'))
    if require_evaluation:
        assert result['evaluation_verified'], 'Paired sampling is still pending'
    temporary = out/'artifact_audit.json.writing'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(out/'artifact_audit.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--pde')
    parser.add_argument('--require-evaluation', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = json.loads((args.study/'study_plan.json').read_text())
    jobs = [job for job in plan['jobs'] if args.pde is None or job['pde'] == args.pde]
    assert jobs, 'No matching PDE job'
    for job in jobs:
        print(json.dumps(audit_job(job, args.require_evaluation)), flush=True)

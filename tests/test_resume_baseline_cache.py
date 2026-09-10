"""A cached baseline must preserve the scientific comparison and provenance."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.train import evaluate_resume_study as evaluation


@pytest.fixture
def baseline_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'IDS', [1500, 1501, 1502, 1503])
    source, destination = tmp_path/'source', tmp_path/'destination'
    checkpoint = str(tmp_path/'original.pth')
    protocol = dict(pde='burger', sample_ids=evaluation.IDS, seeds=[0, 1], tasks=['both'],
        num_steps=100, checkpoint_sha256={'baseline': 'a'*64, 'resumed': 'b'*64},
        truths_sha256='c'*64, precision={'parameters': 'float32', 'matmul_tf32': True},
        configs={'both': {'num_obs': 500}})
    evaluation.write(source/'protocol.json', protocol)
    evaluation.write(source/'complete.json', dict(status='complete', paired_masks_and_rng_verified=True,
        checkpoint_sha256=protocol['checkpoint_sha256'], samples=4, seeds=2, tasks=['both'], batch_size=2))
    evaluation.write(source/'batch_selection.json', dict(batch_size=2, selected_batch_peak_bytes=24*2**30))
    cases = [(0, [0], True), (0, [0, 0], True)]
    cases += [(seed, evaluation.IDS[start:start+2], False) for seed in (0, 1) for start in (0, 2)]
    for seed, ids, probe in cases:
        folder = source/(f'probe_batch{len(ids)}' if probe else 'both')/'baseline'/f'seed{seed}_id{ids[0]}'
        folder.mkdir(parents=True)
        prediction = folder/'result.pt'
        prediction.write_bytes(b'immutable baseline prediction')
        masks = {'coef': torch.ones(len(ids), 1, 2, 2), 'sol': torch.zeros(len(ids), 1, 2, 2)}
        torch.save(masks, folder/'masks.pt')
        config = evaluation.sampling_config('burger', 'both', checkpoint, folder, seed, ids).asdict()
        evaluation.write(folder/'receipt.json', dict(pde='burger', task='both', label='baseline',
            seed=seed, sample_ids=ids, checkpoint_sha256='a'*64, config=config,
            result_path=str(prediction), result_sha256=evaluation.sha(prediction),
            mask_sha256=hashlib.sha256(masks['coef'].numpy().tobytes()+masks['sol'].numpy().tobytes()).hexdigest(),
            initial_noise_sha256='d'*64, rng_after_initial_sha256='e'*64,
            fields={'u': {'relative_l2': [0.1]*len(ids)}},
            peak_bytes=(20 if len(ids) == 1 else 24)*2**30, seconds=10.0))
    current = copy.deepcopy(protocol)
    current['checkpoint_sha256']['resumed'] = 'f'*64
    return source, destination, current, checkpoint


def test_reuse_keeps_original_prediction_and_records_its_source(baseline_cache):
    source, destination, current, checkpoint = baseline_cache
    result = evaluation.reuse_baseline_receipts(destination, source, current, checkpoint)
    assert result['status'] == 'reused' and result['receipts'] == 6
    assert result['avoided_sampling_seconds'] == 60
    receipts = list(destination.rglob('receipt.json'))
    assert len(receipts) == 6
    for path in receipts:
        row = json.loads(path.read_text())
        assert Path(row['result_path']).is_relative_to(source)
        assert evaluation.sha(row['reused_from_receipt']) == row['reused_from_receipt_sha256']
        assert evaluation.sha(row['result_path']) == row['result_sha256']


@pytest.mark.parametrize('change', ['precision', 'truths', 'checkpoint', 'steps', 'config'])
def test_different_scientific_protocol_cannot_share_baseline(baseline_cache, change):
    source, destination, current, checkpoint = baseline_cache
    if change == 'precision': current['precision']['matmul_tf32'] = False
    elif change == 'truths': current['truths_sha256'] = '0'*64
    elif change == 'checkpoint': current['checkpoint_sha256']['baseline'] = '0'*64
    elif change == 'steps': current['num_steps'] = 99
    else: current['configs']['both']['num_obs'] = 499
    with pytest.raises(AssertionError, match='Baseline protocols differ'):
        evaluation.reuse_baseline_receipts(destination, source, current, checkpoint)
    assert not destination.exists()


@pytest.mark.parametrize('damage', ['prediction', 'mask', 'seed', 'missing_receipt'])
def test_invalid_last_batch_cannot_leave_partially_imported_cache(baseline_cache, damage):
    source, destination, current, checkpoint = baseline_cache
    folder = source/'both/baseline/seed1_id1502'
    if damage == 'prediction': (folder/'result.pt').write_bytes(b'changed')
    elif damage == 'mask':
        masks = torch.load(folder/'masks.pt', weights_only=False)
        masks['sol'][0, 0, 0, 0] = 1
        torch.save(masks, folder/'masks.pt')
    elif damage == 'seed':
        row = json.loads((folder/'receipt.json').read_text())
        row['config']['sample_seed'] += 1
        evaluation.write(folder/'receipt.json', row)
    else: (folder/'receipt.json').unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        evaluation.reuse_baseline_receipts(destination, source, current, checkpoint)
    assert not destination.exists()

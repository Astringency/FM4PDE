from copy import deepcopy
from argparse import Namespace

import torch

from scripts.train.resume_study import restore, save_checkpoint


def test_resume_lr_restart_preserves_adam_history_and_next_update():
    model = torch.nn.Linear(2, 1)
    old = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.tensor([[0.4, -0.7]])
    for _ in range(4):
        old.zero_grad()
        model(x).square().sum().backward()
        old.step()
    source = deepcopy({'model_for_resume': model.state_dict(), 'optimizer': old.state_dict()})
    resumed = torch.nn.Linear(2, 1)
    optim = torch.optim.AdamW(resumed.parameters(), lr=0.5)
    restore(resumed, optim, deepcopy(source), 0.00003)
    reference = torch.nn.Linear(2, 1)
    reference.load_state_dict(source['model_for_resume'])
    ref_optim = torch.optim.AdamW(reference.parameters())
    ref_optim.load_state_dict(deepcopy(source['optimizer']))
    for group in ref_optim.param_groups:
        group['lr'] = 0.00003
    for s, old_s in zip(optim.state.values(), source['optimizer']['state'].values()):
        assert torch.equal(s['exp_avg'], old_s['exp_avg'])
        assert torch.equal(s['exp_avg_sq'], old_s['exp_avg_sq'])
        assert int(s['step']) == 4
    for m, opt in [(resumed, optim), (reference, ref_optim)]:
        opt.zero_grad()
        m(x).square().sum().backward()
        opt.step()
    for actual, expected in zip(resumed.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(int(s['step']) == 5 for s in optim.state.values())


def test_continuation_checkpoint_reloads_model_optimizer_and_epoch(tmp_path):
    model = torch.nn.Linear(2, 1)
    optim = torch.optim.AdamW(model.parameters(), lr=0.00003)
    model(torch.ones(1, 2)).square().sum().backward()
    optim.step()
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=4)
    source = {'epoch': 299, 'args': Namespace(dataset='poisson'),
              'normalizer': {'mean': torch.zeros(1), 'std': torch.ones(1)},
              'checkpoint_schema_version': 3}
    metadata = {'learning_rate': 0.00003, 'batch_size': 16,
                'source_checkpoint': 'original.pth', 'updates': 1}
    path = tmp_path/'continued.pth'
    save_checkpoint(path, source, model, optim, schedule, 300, metadata)
    loaded = torch.load(path, weights_only=False)
    fresh = torch.nn.Linear(2, 1)
    fresh_optim = torch.optim.AdamW(fresh.parameters())
    restore(fresh, fresh_optim, loaded, 0.00001)
    assert loaded['epoch'] == 300
    assert loaded['normalizer']['std'].item() == 1
    assert loaded['resume_study']['updates'] == 1
    assert loaded['args'].accum_iter == 4
    for actual, expected in zip(fresh.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(int(s['step']) == 1 for s in fresh_optim.state.values())

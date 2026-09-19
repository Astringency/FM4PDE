import copy
import argparse
import torch

from experiments.optimizer_diagnostics.study import layer_stats, restore
from training.load_and_save import _apply_resume_optimizer_betas
from train import _reset_resume_lr_schedule


def test_resume_overrides_keep_moments_and_do_not_mutate_source():
    model=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),foreach=False)
    model(torch.ones(4,3)).square().mean().backward()
    opt.step()
    source=copy.deepcopy(dict(model_for_resume=model.state_dict(),optimizer=opt.state_dict()))
    original=copy.deepcopy(source)
    restore(model,opt,source,lr=3e-5,betas=(.8,.99))
    assert opt.param_groups[0]['lr']==3e-5
    assert opt.param_groups[0]['betas']==(.8,.99)
    for state,old in zip(opt.state.values(),original['optimizer']['state'].values()):
        for k in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(state[k],old[k],rtol=0,atol=0)
    model(torch.ones(4,3)).square().mean().backward()
    opt.step()
    for state,old in zip(source['optimizer']['state'].values(),original['optimizer']['state'].values()):
        for k in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(state[k],old[k],rtol=0,atol=0)
    restore(model,opt,source)
    assert opt.param_groups[0]['lr']==1e-4
    assert opt.param_groups[0]['betas']==(.9,.999)


def test_measured_update_detects_fp32_roundoff_despite_nonzero_gradient():
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-12,weight_decay=0)
    model(torch.ones(1,1)).square().sum().backward()
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    opt.step()
    stats=layer_stats(model,opt,before)['summary']
    assert stats['grad_norm']>0
    assert stats['update_norm']==0
    assert stats['update_zero_fraction']==1


def test_native_beta_override_preserves_states_and_validates_range():
    model=torch.nn.Linear(2,1)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    model(torch.ones(3,2)).sum().backward()
    opt.step()
    states=copy.deepcopy(opt.state_dict()['state'])
    _apply_resume_optimizer_betas(opt,(.8,.99))
    assert opt.param_groups[0]['betas']==(.8,.99)
    for state,old in zip(opt.state_dict()['state'].values(),states.values()):
        for k in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(state[k],old[k],rtol=0,atol=0)
    for invalid in ((.9,1),(.9,-.1),(float('nan'),.99),(.9,)):
        try:
            _apply_resume_optimizer_betas(opt,invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid betas {invalid}')


def test_explicit_schedule_reset_uses_remaining_epochs_and_preserves_adam():
    model=torch.nn.Linear(2,1)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-8)
    model(torch.ones(2,2)).square().mean().backward()
    opt.step()
    states=copy.deepcopy(opt.state_dict()['state'])
    args=argparse.Namespace(resume='existing.pth',epochs=320,start_epoch=300,
        lr=3e-6,min_lr=1e-6,warmup_epochs=0,warmup_start_factor=.1)
    scheduler=_reset_resume_lr_schedule(opt,args,'warmup_cosine')
    assert scheduler.T_max==20
    assert opt.param_groups[0]['lr']==3e-6
    for current,old in zip(opt.state_dict()['state'].values(),states.values()):
        for k in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(current[k],old[k],rtol=0,atol=0)
    opt.zero_grad(set_to_none=True)
    for _ in range(20):
        opt.step()
        scheduler.step()
    assert abs(opt.param_groups[0]['lr']-1e-6)<1e-15

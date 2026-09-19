import copy
import torch

from experiments.optimizer_diagnostics.study import layer_stats, restore


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

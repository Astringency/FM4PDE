import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('ns_revision',Path(__file__).parents[1]/'plot/run_ns_main_revision_0909.py')
ns=importlib.util.module_from_spec(spec);spec.loader.exec_module(ns)


def config(task):
    from sampling.config import load_config
    return load_config(Path(__file__).parents[1]/f'configs/main/{task}/nsnonbounded.yaml')


def test_observations_remove_hidden_fields_for_all_tasks():
    fields=torch.randn(2,2,128,128)
    for task in ['forward','inverse','both']:
        c=config(task)
        gt,m=ns.observations(fields,[2000,2001],c,{'nu':.001,'T':1.},'cpu')
        assert torch.equal(gt.coef,fields[:,0:1]*m.coef)
        assert torch.equal(gt.sol,fields[:,1:2]*m.sol)
        assert set(gt.pde_params)=={'nu','T'}
        assert not torch.any(gt.coef[m.coef==0]) and not torch.any(gt.sol[m.sol==0])
        if task=='forward':assert not torch.any(m.sol)
        if task=='inverse':assert not torch.any(m.coef)


def test_per_sample_metric_and_batch_gradient_scaling():
    from sampling.losses import compute_guidance_losses
    from sampling.state import SplitState
    c=config('both');fields=torch.randn(2,2,128,128)
    gt,m=ns.observations(fields,[2000,2001],c,{'nu':.001,'T':1.},'cpu')
    pred=(fields*.8).requires_grad_()
    loss=compute_guidance_losses(SplitState(pred[:,0:1],pred[:,1:2]),gt,m,c)
    g=torch.autograd.grad(2*(loss.guidance_L_obs_a+loss.guidance_L_obs_u),pred)[0]
    for j in range(2):
        one=pred[j:j+1].detach().requires_grad_()
        onegt,onem=ns.observations(fields[j:j+1],[2000+j],c,{'nu':.001,'T':1.},'cpu')
        l=compute_guidance_losses(SplitState(one[:,0:1],one[:,1:2]),onegt,onem,c)
        ref=torch.autograd.grad(l.guidance_L_obs_a+l.guidance_L_obs_u,one)[0]
        torch.testing.assert_close(g[j:j+1],ref,atol=0,rtol=0)
    metrics=ns.score(pred.detach(),fields,gt,m,c)
    for row in metrics:
        assert abs(row['error_a']-.2)<1e-6 and abs(row['error_u']-.2)<1e-6


def test_formal_and_development_examples_are_disjoint():
    assert len(ns.EVAL)==len(set(ns.EVAL))==1000
    assert set(ns.EVAL).isdisjoint(ns.DEV)
    assert len(ns.SETTINGS)*3*len(ns.EVAL)==15000

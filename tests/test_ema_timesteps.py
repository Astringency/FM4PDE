"""Protect Adam continuity, EMA/raw separation, and fixed validation semantics."""
import argparse
import copy

import pytest
import torch

from models.ema import EMA
from training.load_and_save import load_model, CHECKPOINT_SCHEMA_VERSION
from training.timesteps import sample_timesteps
from training.train_loop import validate_one_epoch
from experiments.ema_timesteps.evaluate import evaluate_pair


class Velocity(torch.nn.Module):
    num_classes = None

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.1))

    def forward(self, x, t, extra=None):
        return x * self.scale


def test_add_ema_retains_adam_trajectory_and_roundtrips(tmp_path):
    base = Velocity()
    original = torch.optim.AdamW(base.parameters(), lr=.001)
    for _ in range(3):
        original.zero_grad(); base.scale.square().backward(); original.step()
    state = copy.deepcopy(original.state_dict())
    checkpoint = dict(checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        model_for_resume=base.state_dict(), optimizer=state, epoch=2,
        normalizer={'mean':torch.zeros(1),'std':torch.ones(1)}, has_ema=False)
    path = tmp_path/'raw.pth'; torch.save(checkpoint,path)
    model = EMA(Velocity())
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.9)
    args = argparse.Namespace(resume=str(path), dataset='nsnonbounded', resume_add_ema=True)
    load_model(args, model, opt, None, None)
    assert opt.param_groups[0]['lr'] == original.param_groups[0]['lr']
    assert int(model.num_updates) == 0
    assert torch.equal(model.shadow_params[0],base.scale)
    for name in ['step','exp_avg','exp_avg_sq']:
        assert torch.equal(opt.state[model.model.scale][name], original.state[base.scale][name])
    for parameter, optimizer in [(base.scale,original),(model.model.scale,opt)]:
        optimizer.zero_grad(); parameter.square().backward(); optimizer.step()
    model.update_ema()
    assert torch.equal(base.scale,model.model.scale)
    assert not torch.equal(model.shadow_params[0],model.model.scale)
    # A second resume loads the EMA history instead of reinitializing it.
    checkpoint.update(model_for_resume=model.state_dict(), optimizer=opt.state_dict(), has_ema=True,
                      use_ema=True, ema_decay=model.decay)
    path=tmp_path/'ema.pth';torch.save(checkpoint,path)
    restored=EMA(Velocity()); opt2=torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad))
    args.resume=str(path);args.resume_add_ema=False
    load_model(args,restored,opt2,None,None)
    for k,v in model.state_dict().items(): assert torch.equal(v,restored.state_dict()[k])
    assert torch.equal(opt2.state[restored.model.scale]['exp_avg'],opt.state[model.model.scale]['exp_avg'])


def test_old_ema_optimizer_frozen_shadows_can_resume(tmp_path):
    model=EMA(Velocity());old=torch.optim.AdamW(model.parameters())
    old.zero_grad();model.model.scale.square().backward();old.step();model.update_ema()
    path=tmp_path/'legacy-ema.pth'
    torch.save(dict(checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
                    model_for_resume=model.state_dict(),optimizer=old.state_dict(),epoch=0,
                    normalizer={'mean':torch.zeros(1),'std':torch.ones(1)},has_ema=True),path)
    new=EMA(Velocity());opt=torch.optim.AdamW((p for p in new.parameters() if p.requires_grad))
    load_model(argparse.Namespace(resume=str(path),dataset='nsnonbounded'),new,opt,None,None)
    assert len(opt.param_groups[0]['params'])==1
    assert torch.equal(opt.state[new.model.scale]['exp_avg'],old.state[model.model.scale]['exp_avg'])


def test_time_distributions_and_assignment():
    count=10000
    t=sample_timesteps(count,'cpu','stratified_uniform',generator=torch.Generator().manual_seed(9))
    ordered=torch.sort(t).values; edges=torch.arange(count)/count
    assert bool(((ordered >= edges-2e-7)&(ordered <= edges+1/count+2e-7)).all())
    assert not torch.equal((t*count).long(),torch.arange(count))
    center=sample_timesteps(count,'cpu','logit_normal',generator=torch.Generator().manual_seed(9))
    skew=sample_timesteps(count,'cpu','legacy_skewed',generator=torch.Generator().manual_seed(9))
    assert .48 < float(center.median()) < .52
    assert .74 < float(skew.median()) < .79
    assert bool(((center>0)&(center<1)).all())


def test_raw_ema_evaluation_restores_weights_and_random_stream():
    model=EMA(Velocity());data=torch.randn(8,1,4,4)
    with torch.no_grad():model.model.scale.fill_(.4)
    state={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    result=evaluate_pair(model,data,4,bins=10)
    assert torch.equal(rng,torch.get_rng_state())
    assert model.training and model.model.training
    assert result['raw']['mean'] != result['ema']['mean']
    for k,v in state.items():assert torch.equal(v,model.state_dict()[k])
    assert result==evaluate_pair(model,data,4,bins=10)


def test_native_validation_is_fixed_uniform_and_keeps_training_rng():
    model=Velocity();data=torch.randn(20,1,4,4)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data,torch.zeros(20,dtype=torch.long)),batch_size=5)
    args=argparse.Namespace(skewed_timesteps=True,timestep_sampling='logit_normal',sampling_dtype='float32')
    rng=torch.get_rng_state().clone()
    first=validate_one_epoch(model,loader,torch.device('cpu'),0,args)
    assert torch.equal(rng,torch.get_rng_state())
    second=validate_one_epoch(model,loader,torch.device('cpu'),1,args)
    assert first==second and sum(first['time_bin_counts'])==20
    assert first['time_distribution']=='uniform'

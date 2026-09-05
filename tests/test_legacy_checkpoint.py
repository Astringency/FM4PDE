from __future__ import annotations

import argparse
from collections import OrderedDict

import pytest
import torch

from data.transform import LegacyAffineNormalizer, PDEStandardizer
from models import legacy_checkpoint as compat
from models.model_configs import instantiate_model
from sampling.model_io import load_fm4pde_checkpoint_bundle
from training.load_and_save import load_model, save_model


@pytest.mark.parametrize("pde,low,high", [
    ("poisson", [0,0], [2.5,1/36.5]),
    ("helmholtz", [-2.1313,-.0269], [2.1616,.0277]),
    ("darcy", [7.5,.9/115], [12.5,1.9/115]),
    ("nsnonbounded", [0,0], [1.6,1.6]),
    ("burger", [0], [1.415]),
])
def test_legacy_inference_scale_matches_old_inverse(pde,low,high):
    norm=compat.legacy_sampling_normalizer(pde)
    z=torch.tensor([-1.,0.,1.]).view(1,1,1,3).expand(1,len(low),1,3)
    expected=torch.tensor(low).view(1,-1,1,1)+(z+1)/2*(torch.tensor(high)-torch.tensor(low)).view(1,-1,1,1)
    torch.testing.assert_close(norm.inverse_transform(z),expected,atol=2e-7,rtol=1e-6)
    restored=PDEStandardizer.from_state_dict(norm.state_dict())
    assert isinstance(restored,LegacyAffineNormalizer)
    torch.testing.assert_close(restored.inverse_transform(z),expected,atol=2e-7,rtol=1e-6)


def test_legacy_training_scale_matches_minmax_then_minus_one():
    x=torch.randn(4,2,8,8)
    norm=LegacyAffineNormalizer.fit(x)
    low=x.amin((0,2,3),keepdim=True);high=x.amax((0,2,3),keepdim=True)
    torch.testing.assert_close(norm.transform(x),2*(x-low)/(high-low+1e-8)-1)
    torch.testing.assert_close(norm.inverse_transform(norm.transform(x)),x)
    union=LegacyAffineNormalizer.fit_pools([x[:2],x[2:]])
    torch.testing.assert_close(union.mean,norm.mean)
    torch.testing.assert_close(union.std,norm.std)


def tiny_config(pde):
    return dict(in_channels=2,out_channels=2,model_channels=32,num_res_blocks=1,
                attention_resolutions=(),channel_mult=(1,),dropout=0.,dims=2,
                architecture_profile='legacy',architecture_family='legacy_test')


def old_payload(model,optimizer=None,scheduler=None):
    return dict(model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer else {},
                lr_schedule=scheduler.state_dict() if scheduler else {},epoch=2,scaler={},
                args=argparse.Namespace(dataset='poisson',use_ema=False,decay_lr=True))


def test_legacy_loader_and_dataset_shape_checks(tmp_path,monkeypatch):
    monkeypatch.setattr(compat,'legacy_model_config',tiny_config)
    model=instantiate_model('poisson',False,model_config=tiny_config('poisson'))
    path=tmp_path/'old.pth';torch.save(old_payload(model),path)
    loaded,norm,payload=load_fm4pde_checkpoint_bundle(str(path),'poisson','cpu',model_profile='legacy')
    assert payload['selected_model_profile']=='legacy'
    assert isinstance(norm,LegacyAffineNormalizer)
    assert not loaded.model.training
    with pytest.raises(ValueError,match='does not match requested PDE'):
        load_fm4pde_checkpoint_bundle(str(path),'darcy','cpu')
    broken=old_payload(model);broken['model']=dict(broken['model']);broken['model']['out.2.bias']=torch.zeros(3)
    torch.save(broken,path)
    with pytest.raises(RuntimeError,match='does not match its saved state_dict'):
        load_fm4pde_checkpoint_bundle(str(path),'poisson','cpu')


class Scaler:
    def state_dict(self):return {}
    def load_state_dict(self,state):pass


def test_resume_restores_adam_next_update_and_saves_modern_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(compat,'legacy_model_config',tiny_config)
    cfg=tiny_config('poisson')
    model=instantiate_model('poisson',False,model_config=cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False)
    scheduler=torch.optim.lr_scheduler.LinearLR(optimizer,total_iters=10,start_factor=1.,end_factor=.0001)
    for p in model.parameters():p.grad=torch.full_like(p,.01)
    optimizer.step();scheduler.step();optimizer.zero_grad()
    path=tmp_path/'old.pth';torch.save(old_payload(model,optimizer,scheduler),path)
    restored=instantiate_model('poisson',False,model_config=cfg)
    opt2=torch.optim.AdamW(restored.parameters(),lr=1e-4,foreach=False)
    sched2=torch.optim.lr_scheduler.LinearLR(opt2,total_iters=10,start_factor=1.,end_factor=.0001)
    args=argparse.Namespace(resume=str(path),dataset='poisson',lr_scheduler='linear',
                            start_epoch=0,output_dir=str(tmp_path),use_ema=False,model_profile='legacy')
    checkpoint=load_model(args,restored,opt2,Scaler(),sched2)
    assert args.start_epoch==3
    assert opt2.param_groups[0]['lr']==optimizer.param_groups[0]['lr']
    for network,opt in ((model,optimizer),(restored,opt2)):
        for p in network.parameters():p.grad=torch.full_like(p,-.02)
        opt.step();opt.zero_grad()
    for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    bad=dict(checkpoint);bad['model']=OrderedDict(reversed(list(bad['model'].items())))
    with pytest.raises(ValueError,match='parameter order'):
        compat.validate_legacy_optimizer(restored,bad)
    norm=LegacyAffineNormalizer.fit(torch.randn(4,2,8,8))
    save_model(args,3,restored,restored,opt2,sched2,Scaler(),final=True,normalizer=norm,
               model_config=cfg,model_config_metadata=cfg,num_channels=2)
    loaded,loaded_norm,roundtrip=load_fm4pde_checkpoint_bundle(str(tmp_path/'fm4poisson.pth'),'poisson','cpu')
    assert roundtrip['checkpoint_schema_version']==3
    assert 'legacy_compatibility' not in roundtrip  # Refitted training statistics are now authoritative.
    torch.testing.assert_close(loaded_norm.mean,norm.mean)
    for a,b in zip(restored.parameters(),loaded.model.parameters()):torch.testing.assert_close(a,b)


def test_legacy_loss_scale_and_darcy_multiplier():
    from sampling.config import AblationConfig
    from sampling.guidance import make_zeta_schedule
    from sampling.losses import _legacy_masked_l2_mean
    from sampling.legacy_guidance import legacy_pde_loss
    x=torch.ones(2,1,4,4,requires_grad=True);mask=torch.ones_like(x)
    loss=_legacy_masked_l2_mean(x,torch.zeros_like(x),mask)
    torch.testing.assert_close(loss,torch.tensor(.25))
    cfg=AblationConfig(task='inverse',guidance_operator='legacy',legacy_obs_multiplier=.1,
                       zeta_obs_u=50000000,zeta_pde=1,pde_guidance_start_ratio=0)
    schedule=make_zeta_schedule(cfg,torch.tensor(.2),torch.tensor(.3),torch.tensor(1.))
    assert schedule.zeta_obs_u_t.item()==5000000
    assert schedule.zeta_pde_t.item()==1
    assert legacy_pde_loss('nsnonbounded',x,x).isfinite()


def test_bak_configs_are_separate_and_valid():
    from pathlib import Path
    from sampling.config import load_config
    files=list(Path('configs/bak').glob('*/*.yaml'))
    assert len(files)==12
    for path in files:
        cfg=load_config(str(path));cfg.validate()
        assert cfg.model_profile=='legacy' and cfg.guidance_operator=='legacy'
        assert cfg.num_steps==100 and cfg.num_obs==500
        assert cfg.output_dir=='outputs/bak'


def test_legacy_gradient_and_noise_replay_are_batch_independent(tmp_path):
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    from sampling.runner import run_single_ablation
    torch.manual_seed(17)
    fields=torch.randn(2,2,8,8)*.1
    mask=torch.zeros(2,1,8,8);mask[...,::2,::2]=1
    norm=LegacyAffineNormalizer(torch.zeros(2),torch.ones(2),eps=0)
    class Velocity:
        def __call__(self,x,t,**kw):return x*.1
    bundle=(Velocity(),norm,{})
    def run(indices,source_indices):
        x=fields[indices]
        gt=PDEGroundTruth('poisson',x[:,:1],x[:,1:],x,{},['a'],['u'],
                          {'sample_ids':[str(i) for i in indices],'synthetic':False})
        masks=PairMasks(mask[indices],mask[indices],{})
        cfg=AblationConfig(pde='poisson',task='both',output_dir=str(tmp_path/str(indices)),
            num_steps=3,batch_size=len(indices),img_resolution=8,guidance_operator='legacy',
            obs_guidance_reduction='legacy_l2_mean',pde_guidance_reduction='legacy_l2_mean',
            zeta_obs_a=.01,zeta_obs_u=.01,zeta_pde=.0001,pde_guidance_start_ratio=0,
            clip_mode='none',stochastic_guidance_coeff=.2,stochastic_guidance_time='t_next',
            initial_noise_source_batch_size=4,initial_noise_source_indices=source_indices)
        out=run_single_ablation(cfg,bundle,ground_truth=gt,observation_masks=masks)
        from pathlib import Path
        return torch.load(Path(out['run_dir'])/'result.pt',weights_only=False)
    batched=run([0,1],[0,2])
    for i,source in enumerate([0,2]):
        single=run([i],[source])
        for field in ['coef_final','sol_final']:
            torch.testing.assert_close(single[field],batched[field][i:i+1],rtol=1e-5,atol=1e-6)


def test_comparison_retries_prefer_success_and_keep_failures(tmp_path):
    import json
    from scripts.compare_bak import read_results,summarize
    base={'pde':'poisson','task':'inverse','test_type':'rough','sample_id':0,
          'current_a':.5,'current_u':.1}
    success=dict(base,status='ok',bak_a=.4,bak_u=.05)
    failure=dict(base,status='failed',error='OOM')
    (tmp_path/'paired_results.worker0.jsonl').write_text(json.dumps(failure)+'\n')
    (tmp_path/'paired_results.jsonl').write_text(json.dumps(success)+'\n')
    assert read_results(tmp_path)==[success]
    rows=summarize(tmp_path)
    assert len(rows)==2 and all(row['n']==1 and not row['significant'] for row in rows)

from copy import deepcopy
from argparse import Namespace
import json

import pytest
import torch

from scripts.train.resume_long_study import assert_same_inference, protocol_arguments
from scripts.train.resume_study import restore, save_checkpoint


def test_recovery_memory_guard_requires_a_saved_job_and_verified_batch_probe(tmp_path):
    from scripts.train.resume_long_study import startup_memory_requirement
    spool=tmp_path/'spool';spool.mkdir()
    assert startup_memory_requirement(tmp_path,spool)==70*2**30
    (tmp_path/'resume_verification.json').write_text(json.dumps(dict(
        microbatch=32,model_restored_after_probe=True,adam_restored_after_probe=True)))
    (tmp_path/'batch_probe.json').write_text(json.dumps([
        dict(batch_size=32,peak_bytes=41*2**30,peak_reserved_bytes=47.5*2**30)]))
    assert startup_memory_requirement(tmp_path,spool)==70*2**30
    (spool/'00000000').mkdir();(spool/'00000000/job.json').write_text('{}')
    required=startup_memory_requirement(tmp_path,spool)
    assert 51*2**30<required<53*2**30


def test_optional_spool_preserves_existing_resume_protocol_arguments():
    old=Namespace(pde='nsnonbounded',epochs=50,lr=1e-5)
    new=Namespace(**vars(old),checkpoint_spool=None)
    assert protocol_arguments(new)==vars(old)
    new.checkpoint_spool='/task/cache/checkpoints'
    assert protocol_arguments(new)['checkpoint_spool']=='/task/cache/checkpoints'


def test_main_identity_rejects_same_architecture_different_weights_or_normalizer():
    source=dict(model_config={'width':2},model_for_resume={'w':torch.ones(2)},
                normalizer={'mean':torch.zeros(1),'std':torch.ones(1),'pde':'ns'})
    inference=dict(model_config={'width':2},model={'w':torch.ones(2)},normalizer=deepcopy(source['normalizer']))
    assert_same_inference(source,inference)
    inference['model']['w'][0]=2
    with pytest.raises(AssertionError):assert_same_inference(source,inference)
    inference['model']['w'][0]=1
    inference['normalizer']['std'][0]=2
    with pytest.raises(AssertionError):assert_same_inference(source,inference)


def test_epoch_boundary_resume_matches_uninterrupted_adam_and_cosine(tmp_path):
    torch.manual_seed(0)
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=1e-6)
    def epoch(m,o,s):
        o.zero_grad();m(torch.tensor([[.4,-.7]])).square().sum().backward();o.step();s.step()
    for _ in range(17):epoch(model,optimizer,scheduler)
    source={'epoch':299,'args':Namespace(dataset='poisson'),'checkpoint_schema_version':3}
    metadata={'learning_rate':3e-6,'batch_size':16,'source_checkpoint':'main.pth','training_dtype':'float32',
              'history':list(range(17)),'planned_epochs':50}
    path=tmp_path/'last.pth'
    save_checkpoint(path,source,model,optimizer,scheduler,316,metadata)
    saved=torch.load(path,weights_only=False)
    resumed=torch.nn.Linear(2,1)
    opt=torch.optim.AdamW(resumed.parameters(),lr=3e-6)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-6)
    restore(resumed,opt,saved,saved['optimizer']['param_groups'][0]['lr'])
    sched.load_state_dict(saved['lr_schedule'])
    for _ in range(17,50):
        epoch(model,optimizer,scheduler);epoch(resumed,opt,sched)
        assert opt.param_groups[0]['lr']==optimizer.param_groups[0]['lr']
        for a,b in zip(model.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert sched.last_epoch==50 and opt.param_groups[0]['lr']==1e-6
    for a,b in zip(opt.state.values(),optimizer.state.values()):
        for key in ['step','exp_avg','exp_avg_sq']:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)

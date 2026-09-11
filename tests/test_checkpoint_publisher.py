from argparse import Namespace
from copy import deepcopy
import json
import threading

import pytest
import torch

from scripts.train.checkpoint_publisher import CheckpointPublisher
from scripts.train.resume_study import file_sha, restore


def setup_model():
    torch.manual_seed(7)
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=1e-6)
    source={'epoch':299,'args':Namespace(dataset='poisson'),'checkpoint_schema_version':3,
            'normalizer':{'mean':torch.zeros(1),'std':torch.ones(1)}}
    return model,optimizer,scheduler,source


def step(model,optimizer,scheduler):
    optimizer.zero_grad();model(torch.tensor([[.4,-.7]])).square().sum().backward()
    optimizer.step();scheduler.step()


def metadata(updates):
    return dict(learning_rate=3e-6,batch_size=16,source_checkpoint='main.pth',
                training_dtype='float32',updates=updates,planned_epochs=50)


def test_training_can_advance_while_all_checkpoint_versions_publish_in_order(tmp_path):
    model,opt,sched,source=setup_model()
    output=tmp_path/'primary';publisher=CheckpointPublisher(output,tmp_path/'spool','a'*64)
    entered,release=threading.Event(),threading.Event()
    original=publisher._copy
    def delayed(job):
        entered.set();assert release.wait(timeout=10)
        return original(job)
    publisher._copy=delayed
    try:
        step(model,opt,sched)
        first=deepcopy(model.state_dict())
        rng=torch.random.get_rng_state().clone()
        publisher.save_checkpoint(output/'last_resume.pth',source,model,opt,sched,300,metadata(1))
        assert entered.wait(timeout=10)
        publisher.save_checkpoint(output/'checkpoints/resume_epoch_001.pth',source,model,opt,sched,300,metadata(1))
        step(model,opt,sched)
        publisher.save_checkpoint(output/'last_resume.pth',source,model,opt,sched,301,metadata(2))
        assert torch.equal(torch.random.get_rng_state(),rng)
        assert not (output/'last_resume.pth').exists()
        recovered=torch.load(publisher.latest_local_checkpoint(),weights_only=False)
        assert recovered['epoch']==301 and recovered['args'].output_dir==str(output)
        fresh=torch.nn.Linear(2,1);fresh_opt=torch.optim.AdamW(fresh.parameters())
        restore(fresh,fresh_opt,recovered,recovered['optimizer']['param_groups'][0]['lr'])
        for a,b in zip(model.parameters(),fresh.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        for a,b in zip(opt.state.values(),fresh_opt.state.values()):
            for key in ['step','exp_avg','exp_avg_sq']:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    finally:
        release.set()
    done=publisher.finish()
    assert done['published_versions']==3
    for relative,row in done['targets'].items():assert file_sha(output/relative)==row['sha256']
    saved=torch.load(output/'last_resume.pth',weights_only=False)
    assert saved['epoch']==301
    early=torch.load(output/'checkpoints/resume_epoch_001.pth',weights_only=False)
    for key,value in first.items():torch.testing.assert_close(early['model_for_resume'][key],value,atol=0,rtol=0)


def test_failed_publication_keeps_recoverable_state_and_can_be_retried(tmp_path):
    model,opt,sched,source=setup_model();step(model,opt,sched)
    output,spool=tmp_path/'primary',tmp_path/'spool'
    publisher=CheckpointPublisher(output,spool,'b'*64)
    def failed(job):raise OSError('simulated transport failure')
    publisher._copy=failed
    publisher.save_checkpoint(output/'last_resume.pth',source,model,opt,sched,300,metadata(1))
    with pytest.raises(RuntimeError,match='publication failed'):publisher.finish()
    assert not (output/'checkpoint_publication_complete.json').exists()
    checkpoint=publisher.latest_local_checkpoint()
    assert torch.load(checkpoint,weights_only=False)['epoch']==300
    resumed=CheckpointPublisher(output,spool,'b'*64)
    done=resumed.finish()
    assert done['published_versions']==1 and file_sha(output/'last_resume.pth')==file_sha(checkpoint)


def test_recovery_uses_newer_complete_fixed_checkpoint_when_last_is_stale(tmp_path):
    model,opt,sched,source=setup_model()
    output=tmp_path/'primary'
    publisher=CheckpointPublisher(output,tmp_path/'spool','e'*64)
    step(model,opt,sched)
    publisher.save_checkpoint(output/'last_resume.pth',source,model,opt,sched,300,metadata(1))
    step(model,opt,sched)
    publisher.save_checkpoint(output/'checkpoints/resume_epoch_002.pth',source,model,opt,sched,301,metadata(2))
    publisher.finish()
    recovered=torch.load(publisher.latest_local_checkpoint(),weights_only=False)
    assert recovered['epoch']==301 and recovered['resume_study']['updates']==2
    fresh,fresh_opt,fresh_sched,_=setup_model()
    restore(fresh,fresh_opt,recovered,recovered['optimizer']['param_groups'][0]['lr'])
    fresh_sched.load_state_dict(recovered['lr_schedule'])
    step(model,opt,sched);step(fresh,fresh_opt,fresh_sched)
    for a,b in zip(model.parameters(),fresh.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(opt.state.values(),fresh_opt.state.values()):
        for key in ['step','exp_avg','exp_avg_sq']:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)


@pytest.mark.parametrize('fail_active_upload', [False, True])
def test_coalescing_keeps_all_local_states_and_every_fixed_checkpoint(tmp_path, fail_active_upload):
    model,opt,sched,source=setup_model()
    output,spool=tmp_path/'primary',tmp_path/'spool'
    publisher=CheckpointPublisher(output,spool,'c'*64,coalesce_mutable=True)
    entered,release=threading.Event(),threading.Event()
    original=publisher._copy
    copied=[]
    def delayed(job):
        entered.set();assert release.wait(timeout=10)
        if fail_active_upload:raise OSError('simulated failure with a coalesced backlog')
        result=original(job);copied.append(job['id']);return result
    publisher._copy=delayed
    try:
        step(model,opt,sched)
        publisher.save_checkpoint(output/'last_resume.pth',source,model,opt,sched,300,metadata(1))
        assert entered.wait(timeout=10)
        fixed_states={}
        for update in [2,3]:
            step(model,opt,sched)
            for name in ['last_resume.pth','best_fm_resume.pth',f'checkpoints/resume_epoch_{update:03d}.pth']:
                publisher.save_checkpoint(output/name,source,model,opt,sched,299+update,metadata(update))
            fixed_states[update]=deepcopy(model.state_dict())
        state=publisher.state()
        assert state['active']==0 and state['saved_versions']==7 and state['superseded_versions']==2
        assert state['queued']==4
        assert torch.load(publisher.latest_local_checkpoint(),weights_only=False)['resume_study']['updates']==3
        # Superseded versions remain complete local recovery artifacts, with honest receipts.
        for number in [1,2]:
            directory=spool/f'{number:08d}'
            assert (directory/'checkpoint.pth').is_file() and not (directory/'published.json').exists()
            receipt=json.loads((directory/'superseded.json').read_text())
            assert receipt['superseded_by']>number and receipt['local_checkpoint_retained']
    finally:
        release.set()
    if fail_active_upload:
        with pytest.raises(RuntimeError,match='publication failed'):publisher.finish()
        assert not (output/'checkpoint_publication_complete.json').exists()
        publisher=CheckpointPublisher(output,spool,'c'*64,coalesce_mutable=True)
    done=publisher.finish()
    assert done['saved_versions']==7
    assert done['published_versions']==(4 if fail_active_upload else 5)
    assert done['superseded_versions']==(3 if fail_active_upload else 2)
    if not fail_active_upload:assert copied==[0,3,4,5,6]
    assert len(list(spool.glob('[0-9]*/checkpoint.pth')))==7
    for name in ['last_resume.pth','best_fm_resume.pth']:
        saved=torch.load(output/name,weights_only=False)
        assert saved['resume_study']['updates']==3
        for a,b in zip(opt.state.values(),saved['optimizer']['state'].values()):
            for key in ['step','exp_avg','exp_avg_sq']:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    for update,expected in fixed_states.items():
        saved=torch.load(output/f'checkpoints/resume_epoch_{update:03d}.pth',weights_only=False)
        for key,value in expected.items():torch.testing.assert_close(saved['model_for_resume'][key],value,atol=0,rtol=0)
    for relative,row in done['targets'].items():assert file_sha(output/relative)==row['sha256']


def test_coalescing_never_skips_a_fixed_checkpoint_upload(tmp_path):
    model,opt,sched,source=setup_model();step(model,opt,sched)
    output=tmp_path/'primary'
    publisher=CheckpointPublisher(output,tmp_path/'spool','d'*64,coalesce_mutable=True)
    entered,release=threading.Event(),threading.Event()
    original=publisher._copy
    def delayed(job):
        entered.set();assert release.wait(timeout=10)
        return original(job)
    publisher._copy=delayed
    try:
        publisher.save_checkpoint(output/'last_resume.pth',source,model,opt,sched,300,metadata(1))
        assert entered.wait(timeout=10)
        for _ in range(2):
            publisher.save_checkpoint(output/'checkpoints/fixed.pth',source,model,opt,sched,300,metadata(1))
        assert publisher.state()['superseded_versions']==0
    finally:
        release.set()
    done=publisher.finish()
    assert done['published_versions']==3 and done['superseded_versions']==0

from argparse import Namespace
from copy import deepcopy
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

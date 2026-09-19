"""Verify the full-training-pool continuation, snapshots and Adam counters."""
import argparse
import json
from pathlib import Path
import torch
from experiments.optimizer_diagnostics.study import sha, tensor_sha, write


def audit(root):
    base=root/'continuation/nsnonbounded'
    assert json.loads((base/'queue.json').read_text())['state']=='complete'
    pool=json.loads((base/'training_pool.json').read_text())
    assert sha(base/'full_train.pt')==pool['sha256']
    data=torch.load(base/'full_train.pt',map_location='cpu',weights_only=False,mmap=True)
    assert len(data['train'])==len(data['train_ids'])==pool['count']==45000
    assert tensor_sha(data['train'])==pool['tensor_sha256']
    assert tensor_sha(data['train_ids'])==pool['train_ids_sha256']
    screen=torch.load(root/'inputs/nsnonbounded/data.pt',map_location='cpu',weights_only=False,mmap=True)
    assert not set(data['train_ids'].tolist()) & set(screen['development_ids'].tolist()+screen['confirmation_ids'].tolist())
    original=torch.load(root/'inputs/nsnonbounded/source.pth',map_location='cpu',weights_only=False,mmap=True)
    steps={int(s['step']) for s in original['optimizer']['state'].values()};assert len(steps)==1
    initial_step=next(iter(steps));records=[];protocols=[]
    for arm in ['selected_resume','lr_control']:
        assert json.loads((base/f'{arm}.exit.json').read_text())['exit_code']==0
        out=base/arm
        protocol=json.loads((out/'protocol.json').read_text());protocols.append(protocol)
        assert protocol['parent']['sha256']==sha(root/'runs/nsnonbounded'/f'{arm}.pth')
        complete=json.loads((out/'complete.json').read_text())
        training=complete['training']
        assert [r['total_updates'] for r in training]==list(range(129,513))
        assert all(r['lr']==1e-5 and r['betas']==protocol['config']['betas'] for r in training)
        assert all(r['grad_norm']>0 for r in training)
        for n in [256,512]:
            path=out/f'step_{n:04d}.pth';receipt=json.loads(path.with_suffix('.json').read_text())
            assert sha(path)==receipt['sha256']
            payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert payload['optimizer_diagnostics']['additional_updates']==n
            assert payload['continuation_parent']==protocol['parent']
            assert {int(s['step']) for s in payload['optimizer']['state'].values()}=={initial_step+n}
            assert payload['model_config']==original['model_config']
            for k,v in original['normalizer'].items():
                assert torch.equal(v,payload['normalizer'][k]) if torch.is_tensor(v) else v==payload['normalizer'][k]
            changed=0
            for k,v in payload['model_for_resume'].items():
                assert torch.isfinite(v).all()
                changed+=int(not torch.equal(v,original['model_for_resume'][k]))
                assert torch.equal(v,payload['model'][k])
            assert changed>0
            for s in payload['optimizer']['state'].values():
                assert torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all()
                assert (s['exp_avg_sq']>=0).all()
            for group in payload['optimizer']['param_groups']:
                assert group['lr']==1e-5 and list(group['betas'])==protocol['config']['betas']
            records.append(dict(arm=arm,total_updates=n,optimizer_step=initial_step+n,sha256=sha(path),changed_tensors=changed))
    for key in ['training_pool_sha256','common_rng_seed','effective_batch','microbatch']:
        assert protocols[0][key]==protocols[1][key]
    write(base/'audit.json',dict(status='verified',training_pool_count=45000,
        validation_membership_preserved=True,paired_training_protocol=True,checkpoints=records))
    print('CONTINUATION_AUDIT_VERIFIED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    torch.set_num_threads(4);audit(p.parse_args().root)

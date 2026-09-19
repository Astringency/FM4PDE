"""Recompute physical metrics and verify full-state continuation invariants."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.optimizer_diagnostics.study import sha, write
from experiments.aligned_sampling.input_sources import load_cell
from experiments.aligned_sampling.run_inference import score
from models.model_configs import instantiate_model


def checkpoint(root,pde,arm,epoch):
    folder=root/'runs'/pde/arm
    path=folder/f'epoch_{epoch:04d}.pth'
    ready=json.loads(path.with_suffix('.ready.json').read_text())
    assert sha(path)==ready['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    source=torch.load(root/'inputs'/pde/'source.pth',map_location='cpu',weights_only=False,mmap=True)
    profile=json.loads((root/'profiles'/pde/'complete.json').read_text())
    meta=saved['ema_timesteps']; steps=703*epoch
    assert meta['additional_updates']==steps and meta['completed_epochs']==epoch
    assert saved['epoch']==source['epoch']+epoch
    assert int(saved['model_for_resume']['num_updates'])==steps
    assert saved['ema_decay']==.999 and not saved['ema_warmup']
    assert saved['has_ema'] and saved['use_ema']
    for name,value in source['normalizer'].items():
        other=saved['normalizer'][name]
        assert torch.equal(value,other) if torch.is_tensor(value) else value==other
    assert saved['model_config']==source['model_config']
    baseline_steps={int(state['step']) for state in source['optimizer']['state'].values()}
    final_steps={int(state['step']) for state in saved['optimizer']['state'].values()}
    assert final_steps=={n+steps for n in baseline_steps}
    for group in saved['optimizer']['param_groups']:
        assert tuple(group['betas'])==tuple(meta['config']['betas'])
        assert group['lr']==meta['config']['lr']
    model=instantiate_model(pde,use_ema=False,model_config=saved['model_config'])
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert sum(len(g['params']) for g in saved['optimizer']['param_groups'])==len(named)
    for i,(name,_) in enumerate(named):
        assert torch.equal(saved['model'][name],saved['model_for_resume']['model.'+name])
        assert torch.equal(saved['model_ema'][name],saved['model_for_resume'][f'shadow_params.{i}'])
    assert any(not torch.equal(saved['model'][n],saved['model_ema'][n]) for n,_ in named)
    for key in ['model','model_ema']:
        for tensor in saved[key].values():
            if torch.is_tensor(tensor):assert torch.isfinite(tensor).all()
    for state in saved['optimizer']['state'].values():
        for value in state.values():
            if torch.is_tensor(value):assert torch.isfinite(value).all()
    rows=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['step']<=steps]
    assert len(rows)==steps and [r['step'] for r in rows]==list(range(1,steps+1))
    assert all(math.isfinite(r['loss']) for r in rows)
    assert all(r['grad_norm']>0 for r in rows if r['grad_norm'] is not None)
    for e in range(1,epoch+1):
        metrics=json.loads((folder/f'epoch_{e:04d}.json').read_text())
        assert sum(metrics['training_time_histogram'])==703*64
        assert metrics['ema_updates']==e*703
        for weight in ['raw','ema']:
            m=metrics[weight]
            assert m['input_count']==512 and m['bins']==10
            assert len(m['per_time_input'])==10
            assert abs(sum(m['by_time_bin'])/10-m['mean'])<1e-6
    output=dict(status='verified',pde=pde,arm=arm,epoch=epoch,additional_updates=steps,
                optimizer_steps=sorted(final_steps),ema_updates=int(saved['model_for_resume']['num_updates']),
                raw_ema_separate_and_consistent=True,normalizer_and_architecture_retained=True,
                finite_model_and_optimizer=True,checkpoint_sha256=ready['sha256'],
                initial_pool=profile['input_metadata']['data_sha256'])
    write(folder/f'audit_{epoch:04d}.json',output)
    return output


def sampling(root,pde):
    folder=root/'evaluation'/pde
    selection=json.loads((root/'evaluation_inputs'/pde/'selection.json').read_text())
    data=load_cell(root,selection['cell'])
    ids=selection['indices'];batch=selection['batch_size'];variants=[]
    assert len(ids)==len(set(ids))==32 and len(selection['hard'])==len(selection['ordinary'])==16
    assert not set(selection['hard']) & set(selection['ordinary'])
    original=folder/'variants/original'
    assert json.loads((original/'execution_gate.json').read_text())['passed']
    for complete in sorted((folder/'variants').glob('*/complete.json')):
        result=json.loads(complete.read_text());identity=json.loads((complete.parent/'identity.json').read_text())
        assert identity['weight']==result['weight']
        count=0
        for seed in selection['seeds']:
            reported={r['index']:r for r in result['results'][str(seed)]}
            assert list(reported)==ids
            for start in range(0,len(ids),batch):
                name=f'seed{seed}_batch{start:02d}.pt';path=complete.parent/name
                receipt=json.loads(path.with_suffix('.json').read_text())
                assert sha(path)==receipt['sha256']
                payload=torch.load(path,map_location='cpu',weights_only=False)
                baseline=torch.load(original/name,map_location='cpu',weights_only=False)
                assert payload['indices']==ids[start:start+batch]==baseline['indices']
                assert payload['input_hashes']==baseline['input_hashes']
                assert payload['runtime']['initial_noise_sha256']==baseline['runtime']['initial_noise_sha256']
                assert payload['runtime']['bridge_noise_sha256']==baseline['runtime']['bridge_noise_sha256']
                assert len(payload['runtime']['bridge_noise_sha256'])==100
                configs=[dict(x['effective_config']) for x in [payload,baseline]]
                for cfg in configs:
                    for key in ['checkpoint_path','output_dir']:cfg.pop(key,None)
                assert configs[0]==configs[1]
                recomputed=score(payload['prediction'],data,payload['indices'],SimpleNamespace(pde=pde))
                for row in recomputed:
                    stored=reported[row['index']]
                    assert stored['sample_id']==row['sample_id']
                    for field in ['u'] if pde=='burger' else ['u','a']:
                        assert math.isclose(row[f'rel_l2_{field}'],stored[f'rel_l2_{field}'],rel_tol=1e-10,abs_tol=1e-12)
                count+=len(recomputed)
        assert count==len(ids)*len(selection['seeds'])
        variants.append(dict(variant=result['variant'],weight=result['weight'],prediction_count=count))
    output=dict(status='verified',pde=pde,variants=variants,
                cpu_float64_metrics_recomputed=True,all_101_noise_draws_paired=True,
                inputs_masks_configs_verified=True)
    write(folder/'audit.json',output)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['checkpoint','sampling'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--pde',required=True)
    p.add_argument('--arm');p.add_argument('--epoch',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(checkpoint(a.root,a.pde,a.arm,a.epoch) if a.mode=='checkpoint' else sampling(a.root,a.pde)))

"""Recompute physical metrics and verify full-state continuation invariants."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from experiments.optimizer_diagnostics.study import sha, tensor_sha, write
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


def confirmation(root, pde, selection_path):
    folder = root/'confirmation'/pde
    selection = json.loads(selection_path.read_text())
    digest = sha(selection_path)
    identity = json.loads((folder/'selection_identity.json').read_text())
    assert selection['choices_frozen_before_confirmation'] is True
    assert identity == dict(sha256=digest, selection=selection)
    assert json.loads((folder/'launch.json').read_text())['selection_sha256'] == digest
    profile = json.loads((root/'profiles'/pde/'complete.json').read_text())
    data = torch.load(root/'inputs'/pde/'full_data.pt', map_location='cpu', mmap=True, weights_only=False)
    expected = profile['input_metadata']['tensor_sha256']
    for key in ['confirmation', 'confirmation_ids', 'development_ids', 'train_ids']:
        assert tensor_sha(data[key]) == expected[key]
    ids = data['confirmation_ids'].tolist()
    assert len(ids) == len(set(ids)) == 512
    assert not set(ids) & set(data['development_ids'].tolist())
    assert not set(ids) & set(data['train_ids'].tolist())
    entries = [dict(name='original', path='inputs/'+pde+'/source.pth', weight='raw',
                    sha256=profile['input_metadata']['source_checkpoint_sha256'])]
    entries += selection['pdes'][pde]['checkpoints']
    assert len({e['name'] for e in entries}) == len(entries)
    hashes, records = {}, {}
    for entry in entries:
        path = root/entry['path']
        if path not in hashes:
            hashes[path] = sha(path)
        assert hashes[path] == entry['sha256']
        record = json.loads((folder/(entry['name']+'.json')).read_text())
        assert record['identity'] == dict(checkpoint_sha256=entry['sha256'], weight=entry['weight'],
                                         selection_sha256=digest)
        assert record['confirmation_ids'] == ids and record['microbatch'] == profile['microbatch']
        metric = record['metrics']
        values = np.asarray(metric['per_time_input'], dtype=np.float64)
        assert values.shape == (10, 512) and np.isfinite(values).all() and (values >= 0).all()
        assert metric['seed'] == 20260925 and metric['bins'] == 10 and metric['input_count'] == 512
        assert metric['distribution'] == 'uniform; equal mass in every time bin'
        np.testing.assert_allclose(values.mean(axis=0), metric['per_input'], rtol=2e-6, atol=1e-9)
        np.testing.assert_allclose(values.mean(axis=1), metric['by_time_bin'], rtol=2e-6, atol=1e-9)
        assert math.isclose(values.mean(), metric['mean'], rel_tol=2e-6)
        records[entry['name']] = metric

    def differences(candidate, reference):
        delta = np.asarray(candidate['per_input']) - np.asarray(reference['per_input'])
        se = delta.std(ddof=1)/math.sqrt(len(delta))
        return dict(change=float(delta.mean()), ci95=[float(delta.mean()-1.96*se), float(delta.mean()+1.96*se)],
                    relative_change_pct=float(100*delta.mean()/np.mean(reference['per_input'])), count=len(delta))

    summary = json.loads((folder/'summary.json').read_text())
    assert summary['selection_sha256'] == digest
    assert set(summary['comparisons']) == set(records)-{'original'}
    for name, reported in summary['comparisons'].items():
        actual = differences(records[name], records['original'])
        assert reported['count'] == actual['count']
        assert math.isclose(reported['change'], actual['change'], abs_tol=1e-12)
        np.testing.assert_allclose(reported['ci95'], actual['ci95'], rtol=1e-10, atol=1e-12)
        assert abs(reported['relative_change_pct']-actual['relative_change_pct']) < 5e-5
        assert len(reported['time_bins']) == 10
        for index, reported_bin in enumerate(reported['time_bins']):
            actual_bin = differences(dict(per_input=records[name]['per_time_input'][index]),
                                     dict(per_input=records['original']['per_time_input'][index]))
            np.testing.assert_allclose(reported_bin['ci95'], actual_bin['ci95'], rtol=1e-10, atol=1e-12)
            assert math.isclose(reported_bin['change'], actual_bin['change'], abs_tol=1e-12)
            assert abs(reported_bin['relative_change_pct']-actual_bin['relative_change_pct']) < 5e-5
    later = {}
    for name, candidate in records.items():
        earlier = name.replace('_e10_', '_e05_')
        if earlier != name and earlier in records:
            later[name+'_vs_'+earlier] = differences(candidate, records[earlier])
    output = dict(status='verified', pde=pde, selection_sha256=digest,
                  variants=list(records), input_count=512,
                  reserved_tensors_and_disjoint_ids_verified=True,
                  snapshot_and_weight_identities_verified=True,
                  fixed_uniform_metrics_and_paired_intervals_recomputed=True,
                  epoch10_vs_epoch5=later,
                  scope='Stored fixed-FM metric audit, not an independent model rerun or physical sampling confirmation; historical pretraining exposure is not ruled out for older PDE models.')
    write(folder/'audit.json', output)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['checkpoint','sampling','confirmation'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--pde',required=True)
    p.add_argument('--selection', type=Path)
    p.add_argument('--arm');p.add_argument('--epoch',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(4)
    if a.mode == 'checkpoint':
        result = checkpoint(a.root,a.pde,a.arm,a.epoch)
    elif a.mode == 'sampling':
        result = sampling(a.root,a.pde)
    else:
        result = confirmation(a.root,a.pde,a.selection or a.root/'confirmation_selections'/(a.pde+'.json'))
    print(json.dumps(result))

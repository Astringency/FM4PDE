"""Native sampling with explicit sample, mask, noise and checkpoint identities."""
from __future__ import annotations
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from data.specs import get_pde_spec
from sampling.config import AblationConfig
from sampling.data import PDEGroundTruth
from sampling.masks import PairMasks, make_pair_masks
import sampling.runner as runner
from scripts.train.resume_study import file_sha, write
from scripts.train.evaluate_resume_study import field_scores


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def params_signature(value):
    if torch.is_tensor(value):return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_sha(value))
    if isinstance(value,dict):return {k:params_signature(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [params_signature(v) for v in value]
    return value


def slice_params(params,indices,total,device):
    result={}
    for key,value in params.items():
        if isinstance(value,dict):result[key]=slice_params(value,indices,total,device)
        elif torch.is_tensor(value):
            result[key]=(value[indices] if value.ndim and len(value)==total else value).to(device)
        elif isinstance(value,list) and len(value)==total:result[key]=[value[i] for i in indices]
        else:result[key]=value
    return result


def ground_truth(pde,fields,ids,params,device,*,source):
    spec=get_pde_spec(pde)
    fields=fields.to(device)
    coef,sol=(fields,fields) if pde=='burger' else (fields[:,:spec.coef_channels],fields[:,spec.coef_channels:])
    assert torch.isfinite(fields).all() and len(fields)==len(ids)
    return PDEGroundTruth(pde,coef,sol,fields,params,list(spec.coef_channel_names),list(spec.sol_channel_names),
        dict(synthetic=False,sample_offsets=ids,sample_ids=[str(i) for i in ids],
             offset=ids[0],batch_size=len(ids),data_path=source,source=source,
             endpoint_pair=pde!='burger',channel_names=list(spec.channel_names)))


def validation_inputs(cache,config,indices,device):
    fields=cache['fields'][indices]
    ids=cache['original_training_file_pool_ids'][indices].tolist()
    params=slice_params(cache['loader_metadata'][cache['pde']].get('pde_params',{}),indices,32,device)
    gt=ground_truth(cache['pde'],fields,ids,params,device,source='training_validation_cache')
    masks=make_pair_masks(gt.coef.shape,gt.sol.shape,config['num_obs'],config['sensor_mode'],
        config['shared_mask'],config['mask_seed'],device=device,num_sensor_columns=config.get('num_sensor_columns'))
    if config['task']=='forward':masks.sol.zero_()
    if config['task']=='inverse':masks.coef.zero_()
    return gt,masks,ids


def historical_inputs(record,rows,device):
    assert len({r['result_path'] for r in rows})==1
    path=rows[0]['result_path']
    batch=next(b for b in record['batches'] if b['result_path']==path)
    assert file_sha(path)==batch['result_sha256']
    payload=torch.load(path,map_location='cpu',weights_only=False)
    indices=[r['result_row'] for r in rows];ids=[r['sample_id'] for r in rows]
    if record['pde']=='nsnonbounded':
        fields=payload['truths'][indices]
        mask=payload['masks'][indices].to(device)
        masks=PairMasks(mask[:,:1],mask[:,1:],dict(source='archived main NS masks'))
        params={k:torch.full((len(rows),),v,device=device) for k,v in {'nu':.001,'T':1.,'solver_dt':.0001}.items()}
    else:
        assert not payload['ground_truth_metadata'].get('synthetic',False)
        fields=(payload['coef_ground_truth'][indices] if record['pde']=='burger' else
                torch.cat([payload['coef_ground_truth'][indices],payload['sol_ground_truth'][indices]],1))
        masks=PairMasks(payload['masks']['coef'][indices].to(device),payload['masks']['sol'][indices].to(device),
                        dict(payload['masks']['metadata']))
        params=slice_params(payload.get('pde_params',{}),indices,len(payload['coef_ground_truth']),device)
    gt=ground_truth(record['pde'],fields,ids,params,device,source=record['config']['data_path'])
    return gt,masks,ids


def execution_config(base,checkpoint,out,ids,pool,noise_indices,device):
    cfg=deepcopy(base)
    cfg.update(checkpoint_path=str(checkpoint),model_profile='auto',output_dir=str(out),device=str(device),
        dtype='float32',batch_size=len(ids),offset=ids[0],initial_noise_source_batch_size=pool,
        initial_noise_source_indices=noise_indices,save_plots=False,save_intermediate=False,
        save_per_sample_curves=False,allow_synthetic_data=False,empty_cache_each_step=False)
    result=AblationConfig(**cfg);result.validate()
    assert result.noise_level==0 and result.num_steps==100
    return result


def sample_once(base,bundle,checkpoint,checkpoint_sha,out,gt,masks,ids,pool,noise_indices,*,device='cuda:0'):
    """Restart-safe native sampler invocation; existing receipts are fully bound."""
    out=Path(out)
    cfg=execution_config(base,checkpoint,out,ids,pool,noise_indices,device)
    request=dict(config=cfg.asdict(),checkpoint_sha256=checkpoint_sha,sample_ids=ids,
        truth_sha256=tensor_sha(gt.pair),mask_sha256=tensor_sha(torch.cat([masks.coef,masks.sol],1)),
        params_sha256=json_sha(params_signature(gt.pde_params)))
    receipt=out/'receipt.json'
    if receipt.exists():
        row=json.loads(receipt.read_text())
        assert row['request']==request, f'Cached sampling request differs: {receipt}'
        assert file_sha(row['result_path'])==row['result_sha256']
        return row
    out.mkdir(parents=True,exist_ok=True)
    captured={}
    original=runner._sample_initial_noise
    def initial(*args,**kwargs):
        x=original(*args,**kwargs)
        captured['initial_noise_sha256']=tensor_sha(x)
        state=torch.cuda.get_rng_state(device) if torch.device(device).type=='cuda' else torch.get_rng_state()
        captured['rng_after_initial_sha256']=tensor_sha(state)
        return x
    runner._sample_initial_noise=initial
    start=time.monotonic()
    cuda=torch.device(device).type=='cuda'
    if cuda:torch.cuda.reset_peak_memory_stats(device)
    try:
        with (out/'run.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
            result=runner.run_single_ablation(cfg,checkpoint_bundle=bundle,ground_truth=gt,observation_masks=masks)
    finally:runner._sample_initial_noise=original
    assert result['status']=='ok' and not result.get('synthetic_data')
    path=Path(result['run_dir'])/'result.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    fields={'u':field_scores(saved['sol_final'],saved['sol_ground_truth'],cfg.pde)}
    if cfg.pde!='burger':fields['a']=field_scores(saved['coef_final'],saved['coef_ground_truth'],cfg.pde)
    row=dict(request=request,fields=fields,result_path=str(path),result_sha256=file_sha(path),
        seconds=time.monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,
        **captured)
    write(receipt,row)
    return row


def paired_identity(a,b):
    for k in ['sample_ids','truth_sha256','mask_sha256','params_sha256']:assert a['request'][k]==b['request'][k],k
    for k in ['initial_noise_sha256','rng_after_initial_sha256']:assert a[k]==b[k],k
    def settings(row):
        return {k:v for k,v in row['request']['config'].items() if k not in {'checkpoint_path','output_dir'}}
    assert settings(a)==settings(b)


def flatten_scores(receipts,expected_ids):
    rows={}
    for receipt in receipts:
        for i,sample_id in enumerate(receipt['request']['sample_ids']):
            assert sample_id not in rows, f'Duplicate sample {sample_id}'
            rows[sample_id]={field:{k:v[i] for k,v in metrics.items() if k!='basis'}
                             for field,metrics in receipt['fields'].items()}
    assert set(rows)==set(expected_ids) and len(expected_ids)==len(set(expected_ids))
    return [dict(sample_id=i,fields=rows[i]) for i in expected_ids]


def selection_score(baseline,candidate,tasks,pde):
    """Equal weight per setting; within joint, equal normalized weight per field."""
    scores=[]
    assert set(baseline)==set(candidate)==set(tasks)
    for setting,task in tasks.items():
        fields=['u'] if pde=='burger' or task=='forward' else ['a'] if task=='inverse' else ['a','u']
        assert [r['sample_id'] for r in baseline[setting]]==[r['sample_id'] for r in candidate[setting]]
        ratios=[]
        for field in fields:
            before=np.mean([r['fields'][field]['relative_l2'] for r in baseline[setting]])
            after=np.mean([r['fields'][field]['relative_l2'] for r in candidate[setting]])
            assert np.isfinite(before) and before>0 and np.isfinite(after)
            ratios.append(float(after/before))
        scores.append(float(np.mean(ratios)))
    return float(np.mean(scores))


def paired_summary(baseline,resumed):
    assert [r['sample_id'] for r in baseline]==[r['sample_id'] for r in resumed]
    result={}
    for field in baseline[0]['fields']:
        result[field]={}
        for metric in baseline[0]['fields'][field]:
            a=np.asarray([r['fields'][field][metric] for r in baseline])
            b=np.asarray([r['fields'][field][metric] for r in resumed])
            delta=b-a;half=1.96*delta.std(ddof=1)/np.sqrt(len(delta))
            row=dict(n=len(delta),baseline_mean=float(a.mean()),resumed_mean=float(b.mean()),
                mean_change=float(delta.mean()),paired_95ci=[float(delta.mean()-half),float(delta.mean()+half)])
            if metric.endswith('relative_l2'):row['relative_change']=float(b.mean()/a.mean()-1)
            result[field][metric]=row
    return result


def main_batches(record,batch_size):
    """Run exactly the hard 25 first, then the other 975, preserving old noise pools."""
    hard={r['sample_id'] for r in record['hardest_25']}
    batches=[]
    for stage in ['hard25','remaining975']:
        for original in record['batches']:
            rows=[r for r in record['rows'] if r['result_path']==original['result_path']
                  and ((r['sample_id'] in hard)==(stage=='hard25'))]
            for start in range(0,len(rows),batch_size):
                group=rows[start:start+batch_size]
                assert len({r['noise_source_size'] for r in group})==1
                batches.append(dict(stage=stage,rows=group))
    ids=[r['sample_id'] for b in batches for r in b['rows']]
    assert sorted(ids)==sorted(record['sample_ids']) and len(ids)==1000
    assert sum(len(b['rows']) for b in batches if b['stage']=='hard25')==25
    return batches

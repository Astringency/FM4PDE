"""Paired sampling on cases selected only from archived original-model errors."""
import argparse
import copy
import csv
import gc
import json
from pathlib import Path
import time
import subprocess
import sys
import os

import numpy as np
import torch

from experiments.aligned_sampling.input_sources import load_cell
from experiments.aligned_sampling.run_inference import (
    configure_runtime, effective_config, observation_batch, infer, score,
    relative_differences, tensor_digest, runtime_environment,
)
from experiments.optimizer_diagnostics.study import sha, write
from sampling.model_io import load_fm4pde_checkpoint_bundle

CELL='supervised/nsnonbounded/id/sparse_joint'
REFERENCE=Path(os.environ.get('FM_OPT_REFERENCE_ROOT','/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/baseline_pairing_20260915_57k'))


def location(root,pde):
    return root/'hard_sampling' if pde=='nsnonbounded' else root/'hard_sampling'/pde


def paired_infer(cfg,bundle,gt,masks,indices):
    import sampling.sampler_wrappers as sampler
    original=sampler._stochastic_bridge_noise_like
    hashes=[]
    def tracked(*args,**kwargs):
        value=original(*args,**kwargs)
        hashes.append(tensor_digest(value))
        return value
    sampler._stochastic_bridge_noise_like=tracked
    try:
        pred,meta=infer(cfg,bundle,gt,masks,indices)
    finally:
        sampler._stochastic_bridge_noise_like=original
    assert len(hashes)==100
    meta['bridge_noise_sha256']=hashes
    return pred,meta


def setup(root,pde='nsnonbounded'):
    out=location(root,pde)
    if (out/'selection.json').exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((REFERENCE/'protocol.json').read_text())
    cell_id=f'supervised/{pde}/id/sparse_joint'
    cell=next(c for c in protocol['cells'] if c['cell_id']==cell_id)
    with (REFERENCE/'audits/final/per_sample.csv').open() as stream:
        pool=[r for r in csv.DictReader(stream) if r['cell_id']==cell_id]
    assert len(pool)==1000 and len({r['index'] for r in pool})==1000
    rows=sorted(pool,key=lambda r:(-float(r['error_u_percent']),int(r['index'])))[:16]
    ids=[int(r['index']) for r in rows]
    data=load_cell(REFERENCE,cell)
    source=torch.load(root/'inputs'/pde/'source.pth',map_location='cpu',weights_only=False,mmap=True)
    historical_path=REFERENCE/cell['checkpoint']['path']
    assert sha(historical_path)==cell['checkpoint']['sha256']
    historical=torch.load(historical_path,map_location='cpu',weights_only=False,mmap=True)
    state=historical['model']
    assert state.keys()==source['model_for_resume'].keys()
    assert all(torch.equal(v,source['model_for_resume'][k]) for k,v in state.items())
    for key,value in historical['normalizer'].items():
        other=source['normalizer'][key]
        assert torch.equal(value,other) if torch.is_tensor(value) else value==other
    cached={}
    predictions=[]
    for row in rows:
        path=Path(row['prediction_file'])
        if path not in cached:
            assert sha(path)==row['prediction_sha256']
            cached[path]=torch.load(path,map_location='cpu',weights_only=False)
        pack=cached[path]
        i=int(row['index'])
        pred=pack['prediction'][pack['indices'].index(i)]
        truth=data['sol_ground_truth'][i].double()
        solution=pred[:1] if pde=='burger' else pred[1:2]
        error=float((solution.double()-truth).norm()/truth.norm())*100
        assert abs(error-float(row['error_u_percent']))<1e-8
        predictions.append(pred)
    torch.save(dict(indices=ids,prediction=torch.stack(predictions),
        coef_truth=data['coef_ground_truth'][ids],sol_truth=data['sol_ground_truth'][ids],
        coef_mask=data['masks']['coef'][ids],sol_mask=data['masks']['sol'][ids]),out/'selected_reference.pt')
    write(out/'selection.json',dict(cell=cell,indices=ids,rows=rows,
        source_metrics=str(REFERENCE/'audits/final/per_sample.csv'),
        source_metrics_sha256=sha(REFERENCE/'audits/final/per_sample.csv'),
        reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        selected_reference_sha256=sha(out/'selected_reference.pt'),
        original_training_and_historical_sampling_weights_bitwise_equal=True,
        selection='Worst 16 of 1000 original seed-0 solution-field relative L2 errors; fixed before new sampling',
        seeds=[0,1],primary_metric='solution relative L2; coefficient error also reported',
        improvement_thresholds_pct=[10,20],
        scope='Selected difficult ID cases; seed 1 repeats the same cases independently; not a population performance estimate'))
    print('HARD_CASES_FIXED',ids,flush=True)


def run(root,variant,checkpoint,batch_size,pde='nsnonbounded'):
    out=location(root,pde)
    selection=json.loads((out/'selection.json').read_text())
    ids=selection['indices']
    data=load_cell(REFERENCE,selection['cell'])
    checkpoint=Path(checkpoint)
    while not checkpoint.exists():
        print('WAIT_CHECKPOINT',checkpoint,flush=True)
        time.sleep(30)
    configure_runtime('cuda:0',False,4)
    bundle=load_fm4pde_checkpoint_bundle(str(checkpoint),pde,'cuda:0',prefer_ema=False)
    target=out/'runs'/variant
    target.mkdir(parents=True,exist_ok=True)
    identity=dict(variant=variant,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        selection_sha256=sha(out/'selection.json'),batch_size=batch_size,
        environment=runtime_environment('cuda:0',False))
    if (target/'complete.json').exists():
        assert json.loads((target/'identity.json').read_text())['checkpoint_sha256']==identity['checkpoint_sha256']
        return
    write(target/'identity.json',identity)
    archived=torch.load(out/'selected_reference.pt',map_location='cpu',weights_only=False)
    # Historical batch 64 is numerically sensitive on the selected hard cases.
    # Gate exact current-batch reproducibility, and measure every candidate against
    # a fresh original-model baseline with this SAME batch and all noise draws.
    if variant=='original' and not (out/'paired_execution_gate.json').exists():
        diagnosis=json.loads((out/'replay_diagnosis.json').read_text()) if (out/'replay_diagnosis.json').exists() else None
        if diagnosis is not None:
            assert diagnosis['batch1_bank1']['first_case_vs_native']['max_relative']==0
            assert batch_size==4
        group=ids[:batch_size]
        cfg=effective_config(selection['cell'],checkpoint,'cuda:0',group,1000,target)
        gt,masks,_=observation_batch(data,cfg,group,'cuda:0')
        pred,meta=paired_infer(cfg,bundle,gt,masks,group)
        if diagnosis is not None:
            previous=torch.load(out/'replay_batch4_bank1.pt',map_location='cpu',weights_only=False)
        else:
            previous,_=paired_infer(cfg,bundle,gt,masks,group)
        diff=relative_differences(pred,previous)
        write(out/'paired_execution_gate.json',dict(difference=diff,runtime=meta,tolerance=1e-6,
            passed=diff['max_relative']<1e-6,historical_difference=relative_differences(pred,archived['prediction'][:batch_size]),
            comparison_policy=f'Fresh original-model baseline at identical batch {batch_size}, inputs, masks and all 101 noise draws; historical errors select cases only',
            historical_batch_sensitivity=diagnosis))
        assert diff['max_relative']<1e-6,diff
        free,_=torch.cuda.mem_get_info()
        assert meta['peak_allocated_bytes']*1.5<free-10*2**30
        del pred,gt,masks
    assert json.loads((out/'paired_execution_gate.json').read_text())['passed']
    results={}
    for seed in selection['seeds']:
        rows=[];predictions=[];batches=[]
        for start in range(0,len(ids),batch_size):
            group=ids[start:start+batch_size]
            saved=target/f'seed{seed}_batch{start:02d}.pt'
            receipt=saved.with_suffix('.json')
            if saved.exists() and receipt.exists():
                record=json.loads(receipt.read_text())
                assert sha(saved)==record['sha256']
                payload=torch.load(saved,map_location='cpu',weights_only=False)
                assert payload['indices']==group
            else:
                cell=copy.deepcopy(selection['cell'])
                cell['config']['sample_seed']=seed
                cfg=effective_config(cell,checkpoint,'cuda:0',group,1000,target)
                gt,masks,hashes=observation_batch(data,cfg,group,'cuda:0')
                pred,meta=paired_infer(cfg,bundle,gt,masks,group)
                payload=dict(indices=group,prediction=pred,metrics=score(pred,data,group,cfg),
                    input_hashes=hashes,runtime=meta,effective_config=cfg.asdict())
                torch.save(payload,saved)
                write(receipt,dict(sha256=sha(saved),indices=group,seed=seed))
                del gt,masks,pred
            rows.extend(payload['metrics']);predictions.append(payload['prediction']);batches.append(payload['runtime'])
            print('SAMPLING_PROGRESS',variant,seed,start+len(group),len(ids),flush=True)
        pred=torch.cat(predictions)
        replay=relative_differences(pred,archived['prediction']) if variant=='original' and seed==0 else None
        results[str(seed)]=dict(rows=rows,batches=batches,historical_replay=replay)
    write(target/'complete.json',dict(variant=variant,results=results,checkpoint_sha256=identity['checkpoint_sha256']))
    print('SAMPLING_COMPLETE',variant,flush=True)


def report(root,pde='nsnonbounded'):
    out=location(root,pde)
    original=json.loads((out/'runs/original/complete.json').read_text())
    selection=json.loads((out/'selection.json').read_text())
    variants=[];details=[]
    for path in sorted((out/'runs').glob('*/complete.json')):
        candidate=json.loads(path.read_text())
        if candidate['variant']=='original': continue
        result=dict(variant=candidate['variant'],fields={})
        for field in (['u'] if pde=='burger' else ['u','a']):
            b=np.array([[r[f'rel_l2_{field}'] for r in original['results'][str(seed)]['rows']] for seed in selection['seeds']])
            c=np.array([[r[f'rel_l2_{field}'] for r in candidate['results'][str(seed)]['rows']] for seed in selection['seeds']])
            assert b.shape==c.shape==(len(selection['seeds']),16)
            for seed_row,seed in enumerate(selection['seeds']):
                for i,idx in enumerate(selection['indices']):
                    details.append(dict(variant=candidate['variant'],seed=seed,index=idx,field=field,
                        original=float(b[seed_row,i]),candidate=float(c[seed_row,i]),improvement_pct=float(100*(1-c[seed_row,i]/b[seed_row,i]))))
            base=b.mean(0);new=c.mean(0);diff=new-base
            rng=np.random.default_rng(20260919)
            bootstrap=diff[rng.integers(0,16,size=(10000,16))].mean(1)
            improvement=100*(1-new/base)
            result['fields'][field]=dict(original_mean=float(base.mean()),candidate_mean=float(new.mean()),
                aggregate_improvement_pct=float(100*(1-new.mean()/base.mean())),
                median_case_improvement_pct=float(np.median(improvement)),improved=int((improvement>0).sum()),
                improved_ge10=int((improvement>=10).sum()),improved_ge20=int((improvement>=20).sum()),
                paired_difference_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                seed_improvement_pct=(100*(1-c.mean(1)/b.mean(1))).tolist())
        # Verify paired observation and random stream identities from saved batch payloads.
        for p in path.parent.glob('seed*_batch*.pt'):
            candidate_batch=torch.load(p,map_location='cpu',weights_only=False)
            baseline_batch=torch.load(out/'runs/original'/p.name,map_location='cpu',weights_only=False)
            assert candidate_batch['input_hashes']==baseline_batch['input_hashes']
            assert candidate_batch['runtime']['initial_noise_sha256']==baseline_batch['runtime']['initial_noise_sha256']
            assert candidate_batch['runtime']['bridge_noise_sha256']==baseline_batch['runtime']['bridge_noise_sha256']
            assert candidate_batch['indices']==baseline_batch['indices']
            config0=dict(baseline_batch['effective_config']);config1=dict(candidate_batch['effective_config'])
            for cfg in [config0,config1]:
                for key in ['checkpoint_path','output_dir']: cfg.pop(key,None)
            assert config0==config1
        variants.append(result)
    write(out/'summary.json',dict(selection=selection['selection'],variants=variants,
        paired_inputs_masks_noise_and_sampler_verified=True,scope=selection['scope']))
    if details:
        with (out/'per_sample.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    lines=['# 困难样本采样对照','',
        f"{pde} / ID / 稀疏观测恢复；按原模型已有 1000 个结果的解场误差，预先固定最差 16 例。100 步采样、原观测、原指导参数；使用 seeds {selection['seeds']}，原模型与候选严格配对。",
        '历史误差仅用于选样。所有训练收益均相对于相同批量重新运行的原模型基线计算；固定 batch 的重复执行门限为 1e-6，所有 101 次噪声抽样逐次核对。NS 已发现明显的批量敏感性，相关诊断保留。',
        '先按样本平均两个种子，再计算逐例改善与配对 bootstrap 区间。正改善率表示误差降低；这组困难样本不能代表总体泛化表现。','',
        f"| 候选 | 场 | 原误差 | 新误差 | 相对改善 | 改善例数 | ≥10% / ≥20% | seeds {selection['seeds']} 改善 |",
        '|---|---|---:|---:|---:|---:|---|---|']
    for v in variants:
        for field,d in v['fields'].items():
            lines.append(f"| {v['variant']} | {field} | {100*d['original_mean']:.3f}% | {100*d['candidate_mean']:.3f}% | {d['aggregate_improvement_pct']:+.3f}% | {d['improved']}/16 | {d['improved_ge10']} / {d['improved_ge20']} | {d['seed_improvement_pct'][0]:+.3f}% / {d['seed_improvement_pct'][1]:+.3f}% |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(variants),flush=True)


def queue(root,pde='nsnonbounded',screen_root=None):
    out=location(root,pde)
    out.mkdir(parents=True,exist_ok=True)
    screen_root=Path(screen_root) if screen_root else root/'runs'
    for name,path in [('original',root/'inputs'/pde/'source.pth'),
                      ('lr_control_128',screen_root/pde/'lr_control.pth'),
                      ('beta2_128' if pde=='nsnonbounded' else 'selected_128',screen_root/pde/'selected_resume.pth')]:
        if name!='original':
            status=screen_root/pde/'run.exit.json'
            while not status.exists():
                print('WAIT_FINISHED_SCREEN',pde,flush=True);time.sleep(30)
            assert json.loads(status.read_text())['exit_code']==0
        cmd=[sys.executable,'-u','-m','experiments.optimizer_diagnostics.hard_sampling',
            'run','--root',str(root),'--pde',pde,'--variant',name,'--checkpoint',str(path),
            '--batch-size','1' if pde=='helmholtz' else '4']
        with (out/f'{name}.log').open('a') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            write(out/'queue.json',dict(state='running',variant=name,child_pid=child.pid,time=time.time()))
            code=child.wait()
        write(out/f'{name}.exit.json',dict(exit_code=code,child_pid=child.pid,time=time.time()))
        if code: raise RuntimeError(f'Sampling failed: {name}, exit {code}')
        if name!='original': report(root,pde)
    write(out/'queue.json',dict(state='complete',time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['setup','run','report','queue'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pde',choices=['poisson','helmholtz','darcy','nsnonbounded','burger'],default='nsnonbounded')
    p.add_argument('--screen-root')
    p.add_argument('--variant')
    p.add_argument('--checkpoint')
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args()
    torch.set_num_threads(4)
    if a.mode=='setup': setup(a.root,a.pde)
    elif a.mode=='report': report(a.root,a.pde)
    elif a.mode=='queue': queue(a.root,a.pde,a.screen_root)
    else: run(a.root,a.variant,a.checkpoint,a.batch_size,a.pde)

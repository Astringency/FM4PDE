"""Freeze the paper's 66 complete main cells and their 25 hardest inputs.

Reads actual saved predictions/configurations, verifies all 1000 input identities
and errors per cell, and retains source hashes. No training or sampling occurs.
"""
from __future__ import annotations
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from scripts.train.resume_study import ROOT, file_sha, write
from scripts.train.launch_long_resume import NS_MAIN


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def historical_run(path):
    relative=str(path).split('outputs/',1)[1]
    if relative.startswith('main/'):return ROOT/'outputs'/relative
    return ROOT/'outputs/main'/relative


def metric_errors(pred,truth):
    return (torch.linalg.vector_norm((pred-truth).double().flatten(2),dim=2)/
            torch.linalg.vector_norm(truth.double().flatten(2),dim=2)).numpy()


def hardest(rows,task,pde,n=25):
    def value(row):
        if pde=='burger' or task=='forward':return row['error_u']
        if task=='inverse':return row['error_a']
        return max(row['error_a'],row['error_u'])
    return [dict(sample_id=x['sample_id'],ranking_error=value(x),error_a=x['error_a'],error_u=x['error_u'])
            for x in sorted(rows,key=lambda x:(-value(x),x['sample_id']))[:n]]


def main(study):
    assert study.is_absolute() and '/outputs/pretrained/' in str(study)
    out=study/'evaluation_inputs';out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    papers=read_csv(study/'main_hyperparameters_verified.csv')
    papers=[r for r in papers if r['pde'] in ['poisson','nsnonbounded','darcy','helmholtz','burger']]
    assert len(papers)==66
    plan=json.loads((study/'study_plan.json').read_text())
    source_by_pde={j['pde']:j for j in plan['jobs']}
    @lru_cache(None)
    def table(root):return read_csv(ROOT/'outputs/main'/root/'metrics_per_sample_all.csv')
    ns_protocol=json.loads((NS_MAIN/'inputs/protocol.json').read_text())
    ns_selection=json.loads((NS_MAIN/'selection.json').read_text())
    catalog=[]
    for paper in papers:
        pde,task,dist,obs=(paper[k] for k in ['pde','task','distribution','observations'])
        dist=dist.lower()
        setting=('full_' if obs=='full' else 'sparse_')+('joint' if task=='both' else task)
        if pde=='burger':setting=obs
        cell=f'{pde}/{dist}/{setting}'
        rows=[];batches=[];configs=[]
        if pde=='nsnonbounded':
            directory=NS_MAIN/'main_results'/dist/setting
            for rp in sorted(directory.glob('offset*.json')):
                receipt=json.loads(rp.read_text());path=rp.with_suffix('.pt')
                assert file_sha(path)==receipt['result_sha256']
                payload=torch.load(path,map_location='cpu',weights_only=False)
                cfg=payload['config'];configs.append(cfg)
                errors=metric_errors(payload['predictions'],payload['truths'])
                assert len(receipt['ids'])==len(errors)
                assert receipt['protocol_sha256']==file_sha(NS_MAIN/'inputs/protocol.json')
                assert receipt['selection_sha256']==file_sha(NS_MAIN/'selection.json')
                for i,sample_id in enumerate(receipt['ids']):
                    old=receipt['rows'][i]
                    assert np.allclose(errors[i],[old['error_a'],old['error_u']],rtol=1e-7,atol=1e-10)
                    rows.append(dict(sample_id=sample_id,error_a=float(errors[i,0]),error_u=float(errors[i,1]),
                        result_path=str(path),result_row=i,noise_source_size=1000,noise_source_index=sample_id-2000))
                batches.append(dict(result_path=str(path),result_sha256=receipt['result_sha256'],
                    receipt_path=str(rp),receipt_sha256=file_sha(rp),sample_ids=receipt['ids'],config=cfg))
        else:
            matches=[r for r in table(paper['source_root']) if r['pde']==pde and r['task']==task
                     and r['sensor_mode']==paper['sensor_mode'] and int(r['num_obs'])==int(paper['num_obs'])
                     and historical_run(r['run_dir']).is_relative_to(ROOT/'outputs/main'/paper['source_root']/pde/task)]
            by_run={}
            for row in matches:by_run.setdefault(row['run_dir'],[]).append(row)
            for run,group in sorted(by_run.items()):
                directory=historical_run(run)
                path=directory/'result.pt';config_path=directory/'resolved_config.yaml'
                cfg=yaml.safe_load(config_path.read_text());configs.append(cfg)
                assert cfg['checkpoint_path']==paper['checkpoint_path']
                payload=torch.load(path,map_location='cpu',weights_only=False)
                pred=torch.cat([payload['coef_final'],payload['sol_final']],1)
                truth=torch.cat([payload['coef_ground_truth'],payload['sol_ground_truth']],1)
                errors=metric_errors(pred,truth)
                sample_ids=[]
                for row in sorted(group,key=lambda x:int(x['sample_index'])):
                    sample_id,index=int(row['sample_id']),int(row['sample_index'])
                    assert sample_id==int(cfg['offset'])+index
                    assert int(cfg['batch_size'])==len(errors)==20
                    assert np.allclose(errors[index],[float(row['rel_l2_a']),float(row['rel_l2_u'])],rtol=2e-6,atol=2e-8)
                    rows.append(dict(sample_id=sample_id,error_a=float(errors[index,0]),error_u=float(errors[index,1]),
                        result_path=str(path),result_row=index,noise_source_size=20,noise_source_index=index))
                    sample_ids.append(sample_id)
                batches.append(dict(result_path=str(path),result_sha256=file_sha(path),
                    config_path=str(config_path),config_sha256=file_sha(config_path),
                    masks_path=str(directory/'masks.pt'),masks_sha256=file_sha(directory/'masks.pt'),
                    sample_ids=sample_ids,config=cfg))
        rows.sort(key=lambda x:x['sample_id'])
        ids=[r['sample_id'] for r in rows]
        expected=list(range(2000,3000)) if pde=='nsnonbounded' else list(range(1000))
        assert ids==expected, (cell,len(ids),len(set(ids)))
        runtime_keys={'batch_size','offset','output_dir','device','checkpoint_path','save_plots','save_intermediate',
            'save_per_sample_curves','ablation_name','initial_noise_source_indices'}
        scientific=[{k:v for k,v in c.items() if k not in runtime_keys} for c in configs]
        assert all(c==scientific[0] for c in scientific), f'Configurations vary within {cell}'
        for k in ['num_steps','num_obs','mask_seed','sample_seed','zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold']:
            assert float(configs[0][k])==float(paper[k]),(cell,k,configs[0][k],paper[k])
        for k in ['sampler_phase','time_grid','step_method','sensor_mode','loss_state']:
            assert configs[0][k]==paper[k],(cell,k)
        hard=hardest(rows,task,pde)
        target=out/(cell+'.json')
        record=dict(cell=cell,pde=pde,dist=dist,task=task,setting=setting,observations=obs,
            n=1000,source_main_row=paper,source_checkpoint=source_by_pde[pde]['source'],
            inference_checkpoint=source_by_pde[pde]['inference'],config=configs[0],sample_ids=ids,
            rows=rows,batches=batches,hardest_25=hard,
            hard_selection_rule='descending u error for forward/Burgers, a error for inverse, max(a,u) for joint; stable input-ID tie break',
            hard_selection_role='diagnostic only; never checkpoint/hyperparameter selection',
            historical_mean_a=float(np.mean([r['error_a'] for r in rows])),
            historical_mean_u=float(np.mean([r['error_u'] for r in rows])))
        write(target,record)
        catalog.append(dict(cell=cell,pde=pde,dist=dist,task=task,setting=setting,n=1000,
            record_path=str(target),record_sha256=file_sha(target),hardest_ids=[x['sample_id'] for x in hard],
            historical_mean_a=record['historical_mean_a'],historical_mean_u=record['historical_mean_u']))
        write(out/'progress.json',dict(completed_cells=len(catalog),total_cells=66,last=cell,updated_unix=time.time()))
        print('CELL_VERIFIED',cell,len(rows),'hardest',catalog[-1]['hardest_ids'],flush=True)
    assert len(catalog)==66
    write(out/'catalog.json',dict(status='complete',cells=catalog,total_cells=66,samples_per_cell=1000,
        total_input_setting_pairs=66000,steps=100,main_source_csv_sha256=file_sha(study/'main_hyperparameters_verified.csv'),
        primary_comparison='full main1000 errors; report all cells, even if continued model is worse',
        historical_error_recomputation='all saved predictions, all 66000 input-setting pairs; float64 norm',
        final_sampling_status='pending'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    main(p.parse_args().study.resolve())

#!/usr/bin/env python3
"""Audit completed conditional-scaling draws and export compact mean fields.

Run on each result host. Raw predictions remain beside the hash receipts;
compact files are sufficient for merging and plotting on the paper host.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'plot'))
from run_paper_ablation_revision import digest,write
from run_conditional_sample_scaling import KS,configuration


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path,
                   help='Relocated Poisson input directory containing protocol.json, truths.pt, and weights.pth')
    p.add_argument('--selection',type=Path,
                   help='Relocated frozen selection JSON; its original receipt hash is required')
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    gpu_inventory='unavailable (CPU-only audit; nvidia-smi is absent)'
    if shutil.which('nvidia-smi'):
        inventory=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True)
        gpu_inventory=inventory.stdout if inventory.returncode==0 else 'unavailable (nvidia-smi query failed)'
    resources=dict(memory_available_bytes=mem['MemAvailable'],load_average=list(os.getloadavg()),
        gpu_inventory=gpu_inventory,
        export_device='cpu',export_threads=2)
    assert resources['memory_available_bytes']>=3*(1<<30),'Insufficient CPU memory for independent tensor audit'
    write(args.output/'resources.json',resources)
    envs=[json.loads(x.read_text()) for x in args.results.glob('environment_run_*.json')]
    assert envs
    recorded_source=Path(envs[0]['inputs'])
    source=recorded_source if recorded_source.is_absolute() else args.results.parent.parent/'conditional_scaling_20260909'/recorded_source
    if args.inputs is not None:
        source=args.inputs
    source=source.resolve()
    assert digest(source/'truths.pt')==envs[0]['truth_sha256']
    assert digest(source/'weights.pth')==envs[0]['weights_sha256']
    assert digest(source/'protocol.json')==envs[0]['protocol_sha256']
    original_truths=torch.load(source/'truths.pt',weights_only=False,map_location='cpu')
    protocol=json.loads((source/'protocol.json').read_text())
    selection_path=args.selection or ROOT/'plot/conditional_sample_scaling_selection.json'
    assert digest(selection_path)==envs[0]['selection_sha256']
    selection=json.loads(selection_path.read_text())
    expected=set()
    for e in envs:
        # Relocation changes where bytes are read, never the frozen identity.
        assert e['truth_sha256']==envs[0]['truth_sha256']
        assert e['weights_sha256']==envs[0]['weights_sha256']
        assert e['protocol_sha256']==envs[0]['protocol_sha256']
        assert e['selection_sha256']==envs[0]['selection_sha256']
        assert e['inputs']==envs[0]['inputs'],'Export each original host separately before merging compact results'
        alljobs=[(t,i) for t in e['args']['tasks'] for i in e['args']['offsets']]
        expected.update(v for j,v in enumerate(alljobs) if j%e['args']['num_shards']==e['args']['shard_index'])
        assert e['tf32'] and e['args']['fused_guidance'] and e['K']==KS
    rows=[];proofs=[];arrays={};completed=set()
    for task,offset in sorted(expected):
        folder=args.results/task/f'offset{offset}'
        if not all((folder/f'K{k}.json').exists() for k in KS):
            if args.allow_partial:continue
            raise RuntimeError(f'Incomplete task/offset: {task} {offset}')
        receipts={k:json.loads((folder/f'K{k}.json').read_text()) for k in KS}
        canonical=torch.load(folder/'K1000.pt',weights_only=False,map_location='cpu')
        assert canonical['predictions'].shape==(1000,2,128,128)
        pred=canonical['predictions'].double()
        truth=canonical['truth'].double()
        assert torch.isfinite(pred).all() and torch.isfinite(truth).all()
        assert len({hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in canonical['predictions']})==1000
        assert canonical['offset']==offset and canonical['task']==task
        cfg=canonical['config']
        expected_cfg=configuration(protocol,selection,task,recorded_source)
        expected_cfg.offset=offset;expected_cfg.mask_seed=20260912+offset;expected_cfg.sample_seed=20260912+offset
        assert cfg==expected_cfg.asdict()
        original_pair=torch.cat([original_truths[offset].coef,original_truths[offset].sol],dim=1)[0].double()
        assert torch.equal(truth,original_pair)
        assert cfg['num_steps']==100 and cfg['num_obs']==500 and cfg['noise_level']==0
        from sampling.masks import make_pair_masks
        masks=make_pair_masks((1,1,128,128),(1,1,128,128),cfg['num_obs'],cfg['sensor_mode'],cfg['shared_mask'],cfg['mask_seed'])
        key=f'{task}_{offset}'
        arrays[key+'_truth']=truth.numpy()
        arrays[key+'_mask']=torch.cat([masks.coef,masks.sol],dim=1)[0].numpy()
        for k in KS:
            receipt=receipts[k]
            assert digest(folder/f'K{k}.pt')==receipt['result_sha256']
            assert receipt['offset']==offset and receipt['task']==task and receipt['K']==k
            assert receipt['num_steps']==100 and receipt['nfe_per_draw']==100
            indices=[i for b in receipt['batches'] for i in b['seed_indices']]
            assert indices==list(range(k)) and all(b['nfe']==100 for b in receipt['batches'])
            standalone=torch.load(folder/f'K{k}.pt',weights_only=False,map_location='cpu')
            assert standalone['config']==cfg
            assert torch.equal(standalone['truth'],canonical['truth'])
            assert torch.equal(standalone['mean'],standalone['predictions'].double().mean(0).float())
            mean=pred[:k].mean(0)
            prefix_error=[]
            for j in range(2):
                denominator=torch.linalg.vector_norm(pred[:k,j])
                discrepancy=float(torch.linalg.vector_norm(standalone['predictions'][:,j].double()-pred[:k,j])/denominator)
                # Independent batch partitions preserve stochastic paths; TF32
                # kernel rounding is checked separately and is not called exact.
                assert discrepancy < 0.005,(task,offset,k,j,discrepancy)
                prefix_error.append(discrepancy)
            arrays[key+f'_K{k}']=mean.numpy()
            from sampling.config import AblationConfig
            from sampling.data import PDEGroundTruth
            from sampling.state import SplitState
            from sampling.losses import compute_guidance_losses
            gt=PDEGroundTruth('poisson',truth[None,0:1],truth[None,1:2],truth[None],{},['a'],['u'],{'synthetic':False})
            loss=compute_guidance_losses(SplitState(mean[None,0:1],mean[None,1:2]),gt,masks,AblationConfig(**cfg))
            assert loss.pde_residual_status!='error'
            row=dict(task=task,offset=offset,K=k,seconds=receipt['seconds'],compute_seconds=receipt['compute_seconds'],
                     peak_bytes=receipt['peak_bytes'],L_obs_a=float(loss.L_obs_a),L_obs_u=float(loss.L_obs_u),
                     L_pde=float(loss.L_pde),standalone_prefix_difference_a=prefix_error[0],standalone_prefix_difference_u=prefix_error[1])
            for j,f in enumerate(['a','u']):
                row[f'rel_l2_{f}']=float(torch.linalg.vector_norm(mean[j]-truth[j])/torch.linalg.vector_norm(truth[j]))
                single_squared=((pred[:k,j]-truth[j])**2).flatten(1).sum(1)/(truth[j]**2).sum()
                spread=((pred[:k,j]-mean[j])**2).flatten(1).sum(1).mean()/(truth[j]**2).sum()
                assert np.isclose(float(single_squared.mean()),row[f'rel_l2_{f}']**2+float(spread),rtol=1e-10,atol=1e-12)
                row[f'mean_individual_error_{f}']=float(single_squared.sqrt().mean())
                row[f'normalized_spread_{f}']=float(spread)
            rows.append(row)
            proofs.append(dict(task=task,offset=offset,K=k,sha256=receipt['result_sha256'],path=str(folder/f'K{k}.pt')))
        completed.add((task,offset));print('AUDITED',task,offset,flush=True)
    assert rows,'No complete task/offset available'
    write_csv(args.output/'conditional_scaling_per_input.csv',rows)
    np.savez_compressed(args.output/'conditional_scaling_fields.npz',**arrays)
    write(args.output/'conditional_scaling_manifest.json',dict(expected_jobs=sorted(expected),completed_jobs=sorted(completed),
        complete=completed==expected,conditional_trajectories=len(completed)*1000,
        independent_timing_trajectories=len(completed)*sum(KS),hash_verified_results=len(proofs),
        all_fields_finite=True,variance_identity_verified=True,batch_noise_prefixes_verified=True,
        original_input_truths_verified=True,frozen_guidance_configurations_verified=True,distinct_predictions_per_pool_verified=1000,
        accuracy_source='Mean of first K physical draws from the stored 1000-draw pool',
        timing_source='Separate synchronized execution for each K on the same GPU for each task/offset',
        environments=envs,export_resources=resources,results=proofs,exporter_sha256=digest(__file__),
        resolved_inputs=str(source),resolved_selection=str(selection_path.resolve())))


if __name__=='__main__':main()

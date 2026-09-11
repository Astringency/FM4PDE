"""Validation selection followed by paired, complete thousand-input main sampling."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import torch

from models.legacy_checkpoint import read_checkpoint
from sampling.model_io import load_fm4pde_checkpoint_bundle
from scripts.train.resume_study import ROOT, file_sha, write
from scripts.train.main_resume_sampling import (
    flatten_scores, historical_inputs, main_batches, paired_identity, paired_summary,
    sample_once, selection_score, validation_inputs,
)


def release(bundle):
    # Caller drops its reference immediately after this helper.
    bundle[0].model.cpu()


def main(study,pde,*,memory_limit_gib=56.,minimum_free_gib=70.,maximum_batch=16,buffer_step_metrics=False):
    assert study.is_absolute() and '/outputs/pretrained/' in str(study)
    assert 2 < memory_limit_gib < minimum_free_gib
    assert maximum_batch in [1,2,4,8,16]
    memory_limit=memory_limit_gib*2**30
    training=study/pde
    done=json.loads((training/'training_complete.json').read_text())
    assert done['completed_epochs']==50 and done['updates']==35200
    assert json.loads((training/'exit.json').read_text())['exit_code']==0
    plan=json.loads((study/'study_plan.json').read_text())
    job=next(j for j in plan['jobs'] if j['pde']==pde)
    source=job['source'];source_sha=file_sha(source)
    assert source_sha==done['source_sha256']
    catalog=json.loads((study/'evaluation_inputs/catalog.json').read_text())
    assert catalog['total_cells']==66 and catalog['samples_per_cell']==1000
    cells=[]
    for item in catalog['cells']:
        if item['pde']!=pde:continue
        assert file_sha(item['record_path'])==item['record_sha256']
        cells.append(json.loads(Path(item['record_path']).read_text()))
    assert len(cells)==(6 if pde=='burger' else 15)
    id_cells={r['setting']:r for r in cells if r['dist']=='id'}
    tasks={k:v['task'] for k,v in id_cells.items()}
    cache_path=training/'sampling_validation_inputs.pt'
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    assert cache['pde']==pde and len(cache['fields'])==32
    validation_ids=cache['original_training_file_pool_ids'].tolist()
    split=torch.load(training/'split_indices.pt',map_location='cpu',weights_only=False)
    assert set(validation_ids).isdisjoint(split['train_indices'].tolist())
    assert torch.equal(cache['original_training_file_pool_ids'],split['val_indices'][cache['validation_positions']])
    out=training/'evaluation';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    assert torch.cuda.mem_get_info()[0]>minimum_free_gib*2**30
    torch.manual_seed(0)
    protocol=dict(version=1,pde=pde,source_checkpoint=source,source_sha256=source_sha,
        training_complete_sha256=file_sha(training/'training_complete.json'),
        validation_cache_sha256=file_sha(cache_path),validation_original_pool_ids=validation_ids,
        validation_settings=tasks,validation_seed=0,validation_noise_pool_size=32,
        selection_rule='minimize setting-balanced relative-L2 ratio against baseline on 32 training-validation inputs; forward:u, inverse:a, joint:equal a/u; ties by checkpoint filename; baseline is a reference, choose among resumed candidates',
        final_cells=[r['cell'] for r in cells],samples_per_final_cell=1000,steps=100,
        primary_comparison='new baseline and selected resumed sampling with identical native sampler/configuration/masks/noise/precision/batch partition',
        historical_comparison='secondary comparison to archived original main predictions; NS archived fused gradients and hardware can differ',
        hard_cases='25 preselected archived worst inputs per cell; sampled first after validation selection; diagnostic only',
        inference='float32 parameters and tensors, TF32 enabled; no autocast',
        resource_limits=dict(memory_limit_bytes=memory_limit,
            minimum_free_bytes=minimum_free_gib*2**30,maximum_batch=maximum_batch,
            buffer_step_metrics=buffer_step_metrics),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:write(out/'protocol.json',protocol)
    write(out/'process.json',dict(pid=os.getpid(),host=socket.gethostname(),started_unix=time.time(),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid)))
    bundle=load_fm4pde_checkpoint_bundle(source,pde,'cuda:0',model_profile='auto')

    def validation_batch(setting,indices,label,model,path,digest,folder):
        record=id_cells[setting]
        gt,masks,ids=validation_inputs(cache,record['config'],indices,'cuda:0')
        row=sample_once(record['config'],model,path,digest,folder,gt,masks,ids,32,indices,
            buffer_step_metrics=buffer_step_metrics)
        assert row['peak_bytes']<memory_limit
        del gt,masks
        return row

    if (out/'batch_selection.json').exists():
        batch=int(json.loads((out/'batch_selection.json').read_text())['batch_size'])
        assert batch<=maximum_batch
    else:
        measurements=[]
        for setting in tasks:
            row=validation_batch(setting,[0],'baseline',bundle,source,source_sha,out/'probes'/setting/'batch1')
            measurements.append(dict(setting=setting,batch=1,peak_bytes=row['peak_bytes'],seconds=row['seconds']))
        worst=max(measurements,key=lambda x:x['peak_bytes'])
        previous=worst;batch=1
        for candidate in [2,4,8,16]:
            if candidate>maximum_batch:break
            projected=2.6*previous['peak_bytes']+2*2**30
            if projected>memory_limit-2*2**30:break
            torch.cuda.empty_cache()
            available,_=torch.cuda.mem_get_info()
            if projected-torch.cuda.memory_allocated()+2*2**30>=available:break
            row=validation_batch(worst['setting'],list(range(candidate)),'baseline',bundle,source,source_sha,
                out/'probes'/worst['setting']/f'batch{candidate}')
            assert row['peak_bytes']<memory_limit
            previous=dict(setting=worst['setting'],batch=candidate,peak_bytes=row['peak_bytes'],seconds=row['seconds'])
            measurements.append(previous);batch=candidate
        write(out/'batch_selection.json',dict(batch_size=batch,measurements=measurements,
            single_gpu_limit_bytes=memory_limit,probe_inputs='training-validation cache, not main test inputs'))

    def validate_candidate(label,model,path,digest):
        scores={};receipts={}
        for setting in tasks:
            rows=[]
            for start in range(0,32,batch):
                indices=list(range(start,min(start+batch,32)))
                row=validation_batch(setting,indices,label,model,path,digest,
                    out/'validation'/label/setting/f'chunk{start:03d}')
                rows.append(row)
                print('VALIDATION_BATCH',pde,label,setting,start,flush=True)
            scores[setting]=flatten_scores(rows,validation_ids);receipts[setting]=rows
        write(out/'validation'/label/'scores.json',scores)
        return scores,receipts

    baseline,baseline_receipts=validate_candidate('baseline',bundle,source,source_sha)
    release(bundle);del bundle;gc.collect();torch.cuda.empty_cache()
    candidates=sorted(Path(p) for p in done['checkpoint_candidates'])
    assert len(candidates)==10
    assert [p.name for p in candidates]==[f'resume_epoch_{i:03d}.pth' for i in range(5,51,5)]
    candidates.append(Path(done['best_fm_checkpoint']))
    if (out/'selection.json').exists():
        selection=json.loads((out/'selection.json').read_text())
        assert selection['protocol_sha256']==file_sha(out/'protocol.json')
        assert file_sha(selection['checkpoint'])==selection['checkpoint_sha256']
    else:
        choices=[]
        for path in candidates:
            digest=file_sha(path);label=path.stem
            bundle=load_fm4pde_checkpoint_bundle(str(path),pde,'cuda:0',model_profile='auto')
            payload=bundle[2]
            assert payload['resume_study']['source_sha256']==source_sha
            assert payload['resume_study']['completed_epochs']>=1
            scores,receipts=validate_candidate(label,bundle,str(path),digest)
            for setting in tasks:
                for a,b in zip(baseline_receipts[setting],receipts[setting]):paired_identity(a,b)
            score=selection_score(baseline,scores,tasks,pde)
            choices.append(dict(checkpoint=str(path),checkpoint_sha256=digest,label=label,score=score,
                completed_epochs=payload['resume_study']['completed_epochs'],epoch=payload['epoch'],
                validation_scores_sha256=file_sha(out/'validation'/label/'scores.json')))
            write(out/'selection_progress.json',dict(candidates=choices,expected=len(candidates)))
            release(bundle);del bundle,payload;gc.collect();torch.cuda.empty_cache()
        best=min(choices,key=lambda x:(x['score'],x['label']))
        selection=dict(**best,candidates=choices,protocol_sha256=file_sha(out/'protocol.json'),
            baseline_scores_sha256=file_sha(out/'validation/baseline/scores.json'),
            frozen_unix=time.time(),test_results_used=False,rule=protocol['selection_rule'])
        write(out/'selection.json',selection)
    selected=selection['checkpoint'];selected_sha=selection['checkpoint_sha256']
    # Each label is sampled with one resident network. Both labels use the same
    # frozen batch schedule, including the exact historical Gaussian pool rows.
    schedules={r['cell']:main_batches(r,batch) for r in cells}
    write(out/'main_batch_schedule.json',{k:[dict(stage=b['stage'],ids=[r['sample_id'] for r in b['rows']],
        pool=b['rows'][0]['noise_source_size'],indices=[r['noise_source_index'] for r in b['rows']]) for b in v]
        for k,v in schedules.items()})
    for stage in ['hard25','remaining975']:
        for label,path,digest in [('baseline',source,source_sha),('resumed',selected,selected_sha)]:
            bundle=load_fm4pde_checkpoint_bundle(path,pde,'cuda:0',model_profile='auto')
            for record in cells:
                receipts=[]
                hard_ids={r['sample_id'] for r in record['hardest_25']}
                for number,group in enumerate(schedules[record['cell']]):
                    if stage=='hard25' and group['stage']!='hard25':continue
                    rows=group['rows'];gt,masks,ids=historical_inputs(record,rows,'cuda:0')
                    folder=out/'main'/record['dist']/record['setting']/label/f'chunk{number:04d}'
                    row=sample_once(record['config'],bundle,path,digest,folder,gt,masks,ids,
                        rows[0]['noise_source_size'],[r['noise_source_index'] for r in rows],
                        buffer_step_metrics=buffer_step_metrics)
                    assert row['peak_bytes']<memory_limit
                    receipts.append(row)
                    write(out/'progress.json',dict(stage=stage,label=label,cell=record['cell'],
                        completed_batches=number+1,total_batches=len(schedules[record['cell']]),
                        completed_samples=sum(len(x['request']['sample_ids']) for x in receipts),pid=os.getpid(),updated_unix=time.time()))
                    print('MAIN_BATCH',pde,label,record['cell'],stage,number,len(ids),flush=True)
                    del gt,masks
                expected=sorted(hard_ids) if stage=='hard25' else record['sample_ids']
                scores=flatten_scores(receipts,expected)
                name='hard25_scores.json' if stage=='hard25' else 'scores.json'
                write(out/'main'/record['dist']/record['setting']/label/name,scores)
            release(bundle);del bundle;gc.collect();torch.cuda.empty_cache()
        if stage=='hard25':
            diagnostic=[]
            for record in cells:
                folder=out/'main'/record['dist']/record['setting']
                before=json.loads((folder/'baseline/hard25_scores.json').read_text())
                after=json.loads((folder/'resumed/hard25_scores.json').read_text())
                diagnostic.append(dict(cell=record['cell'],comparison=paired_summary(before,after)))
            write(out/'hard25_complete.json',dict(status='complete',selection_already_frozen=True,
                selection_sha256=file_sha(out/'selection.json'),cells=diagnostic,
                interpretation='preselected historical worst cases; not an estimate of whole-test-set improvement'))
    reports=[]
    for record in cells:
        cell=out/'main'/record['dist']/record['setting']
        before=json.loads((cell/'baseline/scores.json').read_text())
        after=json.loads((cell/'resumed/scores.json').read_text())
        for number,_ in enumerate(schedules[record['cell']]):
            paired_identity(json.loads((cell/'baseline'/f'chunk{number:04d}/receipt.json').read_text()),
                            json.loads((cell/'resumed'/f'chunk{number:04d}/receipt.json').read_text()))
        hard={r['sample_id'] for r in record['hardest_25']}
        report=dict(cell=record['cell'],full=paired_summary(before,after),
            hard25=paired_summary([r for r in before if r['sample_id'] in hard],[r for r in after if r['sample_id'] in hard]),
            historical_baseline_mean_a=record['historical_mean_a'],historical_baseline_mean_u=record['historical_mean_u'],
            hard25_ids=sorted(hard),baseline_scores_sha256=file_sha(cell/'baseline/scores.json'),
            resumed_scores_sha256=file_sha(cell/'resumed/scores.json'))
        reports.append(report);write(cell/'comparison.json',report)
    assert file_sha(source)==source_sha and file_sha(selected)==selected_sha
    write(out/'complete.json',dict(status='complete',pde=pde,cells=len(cells),samples_per_cell=1000,
        distinct_inputs_per_distribution=1000,seeds=1,steps=100,source_sha256=source_sha,selected_sha256=selected_sha,
        selection_sha256=file_sha(out/'selection.json'),paired_masks_noise_config_verified=True,reports=reports))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--pde',required=True)
    p.add_argument('--memory-limit-gib',type=float,default=56.)
    p.add_argument('--minimum-free-gib',type=float,default=70.)
    p.add_argument('--maximum-batch',type=int,choices=[1,2,4,8,16],default=16)
    p.add_argument('--buffer-step-metrics',action='store_true')
    args=p.parse_args();main(args.study.resolve(),args.pde,
        memory_limit_gib=args.memory_limit_gib,minimum_free_gib=args.minimum_free_gib,
        maximum_batch=args.maximum_batch,buffer_step_metrics=args.buffer_step_metrics)

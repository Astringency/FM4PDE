"""Audit full training recovery and every actual validation/main prediction."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.train.audit_resume_study import checkpoint_audit,same
from scripts.train.resume_study import file_sha,write
from scripts.train.main_resume_sampling import (
    execution_config,flatten_scores,historical_inputs,json_sha,main_batches,
    paired_identity,paired_summary,params_signature,selection_score,tensor_sha,validation_inputs,
)
from scripts.train.evaluate_resume_study import field_scores


def read(path):return json.loads(Path(path).read_text())


def audit_receipt(path,checkpoint_sha,gt,masks,ids,pool,indices,config,*,device='cuda:0'):
    row=read(path);request=row['request']
    assert request['checkpoint_sha256']==checkpoint_sha and request['sample_ids']==ids
    assert request['truth_sha256']==tensor_sha(gt.pair)
    assert request['mask_sha256']==tensor_sha(torch.cat([masks.coef,masks.sol],1))
    assert request['params_sha256']==json_sha(params_signature(gt.pde_params))
    expected=execution_config(config,request['config']['checkpoint_path'],Path(path).parent,ids,pool,indices,device)
    assert request['config']==expected.asdict()
    assert file_sha(row['result_path'])==row['result_sha256']
    value=torch.load(row['result_path'],map_location='cpu',weights_only=False)
    assert torch.equal(value['coef_ground_truth'],gt.coef.cpu())
    assert torch.equal(value['sol_ground_truth'],gt.sol.cpu())
    assert torch.equal(value['masks']['coef'],masks.coef.cpu())
    assert torch.equal(value['masks']['sol'],masks.sol.cpu())
    assert not value['ground_truth_metadata']['synthetic']
    assert value['ground_truth_metadata']['sample_offsets']==ids
    for field in row['fields']:
        prefix='sol' if field=='u' else 'coef'
        prediction,truth=value[prefix+'_final'],value[prefix+'_ground_truth']
        # Independently recompute physical relative L2 with vector norms.
        relative=torch.linalg.vector_norm((prediction.double()-truth.double()).flatten(1),dim=1)/torch.linalg.vector_norm(truth.double().flatten(1),dim=1)
        np.testing.assert_allclose(relative,row['fields'][field]['relative_l2'],rtol=1e-10,atol=1e-12)
        actual=field_scores(prediction,truth,config['pde'])
        assert actual['basis']==row['fields'][field]['basis']
        for key in ['low_relative_l2','low_power_ratio','low_alignment']:
            np.testing.assert_allclose(actual[key],row['fields'][field][key],rtol=1e-10,atol=1e-12)
    for key in ['initial_noise_sha256','rng_after_initial_sha256']:
        assert len(row[key])==64 and all(c in '0123456789abcdef' for c in row[key])
    return row


def audit(study,pde,require_evaluation):
    job=next(j for j in read(study/'study_plan.json')['jobs'] if j['pde']==pde)
    training=study/pde;done=read(training/'training_complete.json')
    assert read(training/'exit.json')['exit_code']==0
    assert done['completed_epochs']==50 and done['updates']==35200
    assert file_sha(job['source'])==done['source_sha256']
    source=torch.load(job['source'],map_location='cpu',weights_only=False,mmap=True)
    paths=done['checkpoint_candidates']+[done['best_fm_checkpoint'],str(training/'last_resume.pth')]
    checks=[checkpoint_audit(Path(p),source,pde) for p in paths]
    assert checks[-1]['epoch']==349 and checks[-1]['updates']==35200
    assert checks[-1]['sha256']==done['last_sha256']
    assert [x['updates'] for x in checks[:10]]==list(range(3520,35201,3520))
    history=read(training/'history.json')
    assert [x['additional_epoch'] for x in history]==list(range(1,51))
    assert [x['updates'] for x in history]==list(range(704,35201,704))
    assert all(x['samples']==45000 for x in history)
    split=torch.load(training/'split_indices.pt',map_location='cpu',weights_only=True)
    train,val=set(split['train_indices'].tolist()),set(split['val_indices'].tolist())
    assert len(train)==45000 and len(val)==5000 and not train&val and train|val==set(range(50000))
    dev,confirm=set(split['development_positions'].tolist()),set(split['confirmation_positions'].tolist())
    assert len(dev)==512 and len(confirm)==4488 and not dev&confirm and dev|confirm==set(range(5000))
    for name in ['baseline_confirmation.json','last_confirmation.json']:
        row=read(training/name)
        assert row['positions']==split['confirmation_positions'].tolist()
        assert len(row['per_sample'])==4488 and np.isfinite(row['per_sample']).all()
        assert np.isclose(np.mean(row['per_sample']),row['mean'],rtol=1e-12,atol=1e-14)
    cache=torch.load(training/'sampling_validation_inputs.pt',map_location='cpu',weights_only=False)
    val_ids=cache['original_training_file_pool_ids'].tolist()
    assert len(val_ids)==len(set(val_ids))==32 and set(val_ids).isdisjoint(train)
    result=dict(pde=pde,training_verified=True,epochs=50,updates=35200,checkpoints=checks,
        validation_selection_inputs_disjoint_from_continuation=True,evaluation_verified=False,checked_unix=time.time())
    out=training/'evaluation'
    if (out/'complete.json').exists():
        assert read(training/'evaluation.exit.json')['exit_code']==0
        protocol=read(out/'protocol.json');selection=read(out/'selection.json');complete=read(out/'complete.json')
        assert protocol['source_sha256']==done['source_sha256']
        assert selection['protocol_sha256']==file_sha(out/'protocol.json')
        assert not selection['test_results_used'] and selection['checkpoint_sha256']==file_sha(selection['checkpoint'])
        assert len(selection['candidates'])==11
        batch=read(out/'batch_selection.json')['batch_size']
        records=[]
        for c in read(study/'evaluation_inputs/catalog.json')['cells']:
            if c['pde']!=pde:continue
            assert file_sha(c['record_path'])==c['record_sha256'];records.append(read(c['record_path']))
        id_records={r['setting']:r for r in records if r['dist']=='id'}
        tasks={k:r['task'] for k,r in id_records.items()}
        all_scores={};all_receipts={};receipt_count=0
        choices=[dict(label='baseline',checkpoint=job['source'],checkpoint_sha256=done['source_sha256'])]+selection['candidates']
        for choice in choices:
            assert file_sha(choice['checkpoint'])==choice['checkpoint_sha256']
            label=choice['label'];all_scores[label]={};all_receipts[label]={}
            for setting,record in id_records.items():
                receipts=[]
                for start in range(0,32,batch):
                    indices=list(range(start,min(start+batch,32)))
                    gt,masks,ids=validation_inputs(cache,record['config'],indices,'cpu')
                    path=out/'validation'/label/setting/f'chunk{start:03d}/receipt.json'
                    receipts.append(audit_receipt(path,choice['checkpoint_sha256'],gt,masks,ids,32,indices,record['config']))
                scores=flatten_scores(receipts,val_ids)
                same(scores,read(out/'validation'/label/'scores.json')[setting])
                all_scores[label][setting]=scores;all_receipts[label][setting]=receipts;receipt_count+=len(receipts)
            if label!='baseline':
                score=selection_score(all_scores['baseline'],all_scores[label],tasks,pde)
                assert score==choice['score']
                for setting in tasks:
                    for a,b in zip(all_receipts['baseline'][setting],all_receipts[label][setting]):paired_identity(a,b)
        winner=min(selection['candidates'],key=lambda x:(x['score'],x['label']))
        for key in ['checkpoint','checkpoint_sha256','score','completed_epochs']:assert selection[key]==winner[key]
        reports=[];main_receipts=0
        for record in records:
            schedule=main_batches(record,batch);scores={};receipts={}
            for label,digest in [('baseline',done['source_sha256']),('resumed',selection['checkpoint_sha256'])]:
                rows=[]
                for number,group in enumerate(schedule):
                    original=group['rows'];gt,masks,ids=historical_inputs(record,original,'cpu')
                    path=out/'main'/record['dist']/record['setting']/label/f'chunk{number:04d}/receipt.json'
                    rows.append(audit_receipt(path,digest,gt,masks,ids,original[0]['noise_source_size'],
                        [r['noise_source_index'] for r in original],record['config']))
                scores[label]=flatten_scores(rows,record['sample_ids']);receipts[label]=rows;main_receipts+=len(rows)
                same(scores[label],read(out/'main'/record['dist']/record['setting']/label/'scores.json'))
            for a,b in zip(receipts['baseline'],receipts['resumed']):paired_identity(a,b)
            report=read(out/'main'/record['dist']/record['setting']/'comparison.json')
            same(report['full'],paired_summary(scores['baseline'],scores['resumed']))
            hard={r['sample_id'] for r in record['hardest_25']}
            same(report['hard25'],paired_summary([r for r in scores['baseline'] if r['sample_id'] in hard],
                [r for r in scores['resumed'] if r['sample_id'] in hard]))
            reports.append(report)
            print('AUDITED_CELL',record['cell'],flush=True)
        same(complete['reports'],reports)
        assert complete['cells']==len(records)==(6 if pde=='burger' else 15)
        assert complete['samples_per_cell']==1000 and complete['steps']==100
        result.update(evaluation_verified=True,validation_prediction_batches=receipt_count,
            main_prediction_batches=main_receipts,main_input_model_pairs=len(records)*1000*2,
            selection_sha256=file_sha(out/'selection.json'),evaluation_complete_sha256=file_sha(out/'complete.json'))
    if require_evaluation:assert result['evaluation_verified'],'Full evaluation is pending'
    write(training/('final_artifact_audit.json' if result['evaluation_verified'] else 'training_artifact_audit.json'),result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--pde',required=True);p.add_argument('--require-evaluation',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(2)
    print(json.dumps(audit(args.study.resolve(),args.pde,args.require_evaluation)),flush=True)

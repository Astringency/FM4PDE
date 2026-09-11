from copy import deepcopy
from pathlib import Path

import pytest
import torch

from data.transform import PDEStandardizer
from sampling.config import load_config
from sampling.model_io import WrappedModel
from scripts.train.main_resume_sampling import (
    flatten_scores, main_batches, paired_identity, sample_once, selection_score,
    validation_inputs,
)


class TinyVelocity(torch.nn.Module):
    def forward(self,x,t,extra=None):
        return .2*x+.01*t.reshape(-1,1,1,1)


def test_native_sampling_pool_rows_survive_batch_partition_and_receipt_binding(tmp_path):
    torch.manual_seed(2)
    fields=torch.randn(32,2,8,8)
    cache=dict(pde='poisson',fields=fields,original_training_file_pool_ids=torch.arange(100,132),
               loader_metadata={'poisson':{'pde_params':{}}})
    cfg=load_config(Path(__file__).parents[1]/'configs/main/both/poisson.yaml',overrides=dict(
        num_steps=100,num_obs=8,img_resolution=8,zeta_obs_a=.1,zeta_obs_u=.1,zeta_pde=.001,
        clip_threshold=5.,device='cpu',save_plots=False)).asdict()
    bundle=(WrappedModel(TinyVelocity()),PDEStandardizer.identity(2),{'num_channels':2})
    gt,masks,ids=validation_inputs(cache,cfg,[0,3],'cpu')
    both=sample_once(cfg,bundle,'source.pth','a'*64,tmp_path/'both',gt,masks,ids,32,[0,3],device='cpu')
    again=sample_once(cfg,bundle,'source.pth','a'*64,tmp_path/'both',gt,masks,ids,32,[0,3],device='cpu')
    assert again==both
    from scripts.train.audit_long_resume import audit_receipt
    audited=audit_receipt(tmp_path/'both/receipt.json','a'*64,gt,masks,ids,32,[0,3],cfg,device='cpu')
    assert audited==both
    with pytest.raises(AssertionError):
        sample_once(cfg,bundle,'source.pth','b'*64,tmp_path/'both',gt,masks,ids,32,[0,3],device='cpu')
    full=torch.load(both['result_path'],weights_only=False)
    for index,row in enumerate([0,3]):
        g,m,single_ids=validation_inputs(cache,cfg,[row],'cpu')
        one=sample_once(cfg,bundle,'source.pth','a'*64,tmp_path/f'one{row}',g,m,single_ids,32,[row],device='cpu')
        payload=torch.load(one['result_path'],weights_only=False)
        for field in ['coef_final','sol_final']:
            torch.testing.assert_close(full[field][index:index+1],payload[field],rtol=2e-5,atol=2e-6)
    changed=deepcopy(both);changed['request']['params_sha256']='different'
    with pytest.raises(AssertionError):paired_identity(both,changed)


def test_full_schedule_covers_thousand_ids_once_with_twenty_five_hard_first():
    rows=[];batches=[]
    start=0
    for number,size in enumerate([20]*10+[50]*16):
        path=f'batch{number}.pt';ids=list(range(start,start+size));start+=size
        batches.append(dict(result_path=path,sample_ids=ids))
        rows.extend(dict(sample_id=i,result_path=path,result_row=j,noise_source_size=size,noise_source_index=j)
                    for j,i in enumerate(ids))
    hard=list(range(0,1000,40))
    record=dict(rows=rows,batches=batches,sample_ids=list(range(1000)),hardest_25=[dict(sample_id=i) for i in hard])
    planned=main_batches(record,16)
    flat=[r for batch in planned for r in batch['rows']]
    assert len(flat)==len({r['sample_id'] for r in flat})==1000
    assert {r['sample_id'] for r in flat[:25]}==set(hard)
    assert all(len(batch['rows'])<=16 for batch in planned)
    assert all(r['noise_source_index']==r['result_row'] for r in flat)


def test_selection_balances_settings_and_joint_fields_and_rejects_wrong_ids():
    def row(i,a,u):return dict(sample_id=i,fields={'a':{'relative_l2':a},'u':{'relative_l2':u}})
    tasks={'forward':'forward','inverse':'inverse','joint':'both'}
    base={k:[row(1,10.,.01),row(2,10.,.01)] for k in tasks}
    candidate={k:[row(1,5.,.005),row(2,5.,.005)] for k in tasks}
    assert selection_score(base,candidate,tasks,'poisson')==.5
    candidate['joint']=[row(1,10.,.01),row(2,10.,.01)]
    assert selection_score(base,candidate,tasks,'poisson')==pytest.approx(2/3)
    candidate['joint'][1]['sample_id']=3
    with pytest.raises(AssertionError):selection_score(base,candidate,tasks,'poisson')


def test_score_collection_rejects_missing_and_duplicate_inputs():
    r={'request':{'sample_ids':[2,4]},'fields':{'u':{'relative_l2':[.2,.4],'basis':'test'}}}
    assert [x['sample_id'] for x in flatten_scores([r],[4,2])]==[4,2]
    with pytest.raises(AssertionError):flatten_scores([r],[2,4,6])
    with pytest.raises(AssertionError):flatten_scores([r,r],[2,4])

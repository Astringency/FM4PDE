"""Run the native sampler to audit the archived baseline and development case."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from experiments.run_fm_tilt import repeated_case
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.masks import make_pair_masks
from scripts.train.main_resume_sampling import sample_once
from scripts.train.resume_study import file_sha,write


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot-inputs',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    out=args.output/'native_checks';out.mkdir(exist_ok=True)
    ip=json.loads((args.output/'inputs/protocol.json').read_text())
    assert file_sha(ip['checkpoint_path'])==ip['checkpoint_sha256']
    assert file_sha(args.output/'inputs/cases.pt')==ip['cases_sha256']
    cases=torch.load(args.output/'inputs/cases.pt',map_location='cpu',weights_only=False)
    bundle=load_fm4pde_checkpoint_bundle(ip['checkpoint_path'],ip['pde'],'cuda:0',model_profile=ip['config']['model_profile'])
    results={}
    gt,masks=repeated_case(ip,cases,[0],1,'cuda:0')
    source=ip['rows'][0]['source'];assert source['sample_id']==0
    row=sample_once(ip['config'],bundle,ip['checkpoint_path'],ip['checkpoint_sha256'],out/'historical_id0',
        gt,masks,[0],source['noise_source_size'],[source['noise_source_index']])
    raw=torch.load(row['result_path'],map_location='cpu',weights_only=False)
    pred=torch.cat([raw['coef_final'],raw['sol_final']],1)
    ref=cases[0]['baseline']
    diff=(torch.linalg.vector_norm((pred-ref).double().flatten(2),dim=2)/torch.linalg.vector_norm(ref.double().flatten(2),dim=2)).tolist()
    results['historical_id0']=dict(relative_prediction_difference=diff,receipt=row,
        close_to_saved_baseline=bool(max(diff[0])<.005),comparison='Current native sampler vs archived original prediction, same input/mask/noise pool/checkpoint; FP32 with TF32 off.')
    write(out/'progress.json',results)
    assert row['peak_bytes']<10*2**30
    cache=torch.load(args.pilot_inputs/'truths.pt',map_location='cpu',weights_only=False)
    one=cache[1103]
    mask=make_pair_masks(one.coef.shape,one.sol.shape,500,'random',True,0)
    cases[1103]=dict(truth=one.pair,params=one.pde_params,masks=dict(coef=mask.coef,sol=mask.sol))
    gt,masks=repeated_case(ip,cases,[1103],4,'cuda:0')
    row=sample_once(ip['config'],bundle,ip['checkpoint_path'],ip['checkpoint_sha256'],out/'development_id1103',
        gt,masks,[1103]*4,20,[0,1,2,3])
    results['development_id1103']=dict(receipt=row,
        comparison='Four native stochastic draws on the separate development input. This is not a 1000-input result.')
    write(out/'complete.json',results)
    print(json.dumps({k:{'close_to_saved_baseline':v.get('close_to_saved_baseline'),'fields':v['receipt']['fields']} for k,v in results.items()},indent=2))


if __name__=='__main__':main()

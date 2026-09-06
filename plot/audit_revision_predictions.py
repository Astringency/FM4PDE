"""Recompute complete-stage field errors and verify frozen truth/mask tensors.

This is independent of the sampler's metric reducer: use CPU float64 vector
norms on saved float32 tensors. Every result must match the frozen ground-truth
cache by actual sample ID. Initial-noise/mask pairing is separately validated
by check_revision_sampling.py; this additionally checks actual stored masks.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_revision_sampling import check


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--dest',type=Path,required=True)
    parser.add_argument('--pdes',nargs='+',default=['poisson','darcy','nsnonbounded','burger'])
    args=parser.parse_args();args.dest.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(2)
    reports=[];values=[];sources=[]
    for pde in args.pdes:
        completeness=check(args.inputs,args.results,pde,'evaluation')
        cache=torch.load(args.inputs/'cache'/f'{pde}_ground_truth.pt',map_location='cpu',weights_only=False)
        truths={int(k):v for k,v in cache['truths'].items()};differences=[];obs_differences=[];paired_masks={};count=0
        for path in sorted((args.results/pde/'evaluation').glob('*/seed*/*/receipt.json')):
            receipt=json.loads(path.read_text());run=args.results/receipt['run_dir'].split('/revision_results/',1)[1]
            payload=torch.load(run/'result.pt',map_location='cpu',weights_only=False)
            ids=list(map(int,payload['ground_truth_metadata']['sample_ids']))
            assert ids==receipt['sample_ids'];assert not payload['ground_truth_metadata'].get('synthetic',False)
            assert len(ids)==4
            h=hashlib.sha256()
            for field in ['coef','sol']:
                prediction=payload[field+'_final'];target=payload[field+'_ground_truth']
                assert torch.isfinite(prediction).all() and torch.isfinite(target).all()
                for i,sample_id in enumerate(ids):
                    assert torch.equal(target[i:i+1],getattr(truths[sample_id],field)),(path,sample_id,field)
                h.update(payload['masks'][field].contiguous().numpy().tobytes())
            assert h.hexdigest()==receipt['mask_tensor_sha256'],path
            key=tuple(ids);assert key not in paired_masks or paired_masks[key]==h.hexdigest()
            paired_masks[key]=h.hexdigest()
            recomputed={};obs_recomputed={}
            for field,label in [('coef','a'),('sol','u')]:
                pred=payload[field+'_final'].double();true=payload[field+'_ground_truth'].double()
                errors=(torch.linalg.vector_norm((pred-true).flatten(1),dim=1)/torch.linalg.vector_norm(true.flatten(1),dim=1).clamp_min(1e-12)).tolist()
                recomputed[label]=errors
                for error,row in zip(errors,receipt['rows']):
                    archived=float(row['rel_l2_'+label]);difference=abs(error-archived)
                    differences.append(difference)
                    assert math.isclose(error,archived,rel_tol=1e-6,abs_tol=2e-8),(path,label,error,archived)
                mask=payload['masks'][field].double()
                observed_errors=(torch.linalg.vector_norm(((pred-true)*mask).flatten(1),dim=1)/
                                 torch.linalg.vector_norm((true*mask).flatten(1),dim=1).clamp_min(1e-12)).tolist()
                obs_recomputed[label]=observed_errors
                for error,row in zip(observed_errors,receipt['rows']):
                    archived=float(row['obs_rel_l2_'+label]);obs_differences.append(abs(error-archived))
                    assert math.isclose(error,archived,rel_tol=1e-6,abs_tol=2e-8),(path,'observation',label,error,archived)
            for i,sample_id in enumerate(ids):
                values.append(dict(pde=pde,variant=receipt['variant']['name'],seed=receipt['seed'],sample_id=sample_id,
                                   rel_l2_a_f64=recomputed['a'][i],rel_l2_u_f64=recomputed['u'][i],
                                   obs_rel_l2_a_f64=obs_recomputed['a'][i],obs_rel_l2_u_f64=obs_recomputed['u'][i]))
            sources.append(dict(pde=pde,receipt=str(path),result=str(run/'result.pt'),sha256=digest(run/'result.pt')))
            count+=1
        reports.append(dict(**completeness,prediction_batches_checked=count,field_errors_recomputed=len(differences),
                            ground_truth_source='Exact tensors from frozen per-ID cache',actual_mask_batches=len(paired_masks),
                            maximum_absolute_error_difference=max(differences),reduction='CPU float64',
                            observation_errors_recomputed=len(obs_differences),
                            maximum_absolute_observation_error_difference=max(obs_differences),
                            scope='Checks stored tensors, IDs, masks and field-error arithmetic; does not re-discretize physical residuals or certify historical training-data independence.'))
        print('AUDITED',pde,count,'prediction batches',flush=True)
    with (args.dest/'prediction_errors_recomputed.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    (args.dest/'prediction_tensor_audit.json').write_text(json.dumps(dict(reports=reports,sources=sources),indent=2)+'\n')


if __name__=='__main__':main()

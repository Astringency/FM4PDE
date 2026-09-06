"""Validate completeness, pairing and finite metrics before reporting a stage."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math


def check(inputs,output,pde,stage):
    protocol=json.loads((inputs/'protocol.json').read_text())
    expected_hash=hashlib.sha256((inputs/'protocol.json').read_bytes()).hexdigest()
    ids=protocol['pilot_ids'] if stage=='pilot' else protocol['evaluation_ids']
    seeds=[0] if stage=='pilot' else protocol['inference_seeds']
    names={v['name'] for v in protocol['variants']}
    receipts=[json.loads(p.read_text()) for p in (output/pde/stage).glob('*/seed*/*/receipt.json')]
    expected={(v,s,tuple(ids[i:i+protocol['batch_size']])) for v in names for s in seeds
              for i in range(0,len(ids),protocol['batch_size'])}
    found=set();noise={};masks={};timing=collections.defaultdict(list)
    for r in receipts:
        key=(r['variant']['name'],r['seed'],tuple(r['sample_ids']))
        assert key not in found,key
        found.add(key)
        assert r['protocol_sha256']==expected_hash
        assert r['status']=='ok' and r['error'] is None,r
        assert [int(x['sample_id']) for x in r['rows']]==r['sample_ids']
        for row in r['rows']:
            for field in ['rel_l2_a','rel_l2_u','pde_residual_norm','obs_rel_l2_a','obs_rel_l2_u']:
                assert math.isfinite(float(row[field])) and float(row[field])>=0,(key,row)
        assert r['model_evaluations']>=r['variant']['num_steps']
        assert math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>0
        pair=(r['seed'],tuple(r['sample_ids']))
        assert r['initial_noise_sha256'] and r['mask_tensor_sha256']
        assert pair not in noise or noise[pair]==r['initial_noise_sha256'],('noise',key)
        noise[pair]=r['initial_noise_sha256']
        mask_pair=tuple(r['sample_ids'])
        assert mask_pair not in masks or masks[mask_pair]==r['mask_tensor_sha256'],('mask',key)
        masks[mask_pair]=r['mask_tensor_sha256']
        timing[r['variant']['name']].append(r['elapsed_seconds'])
    assert expected==found,dict(pde=pde,missing=len(expected-found),extra=len(found-expected))
    return dict(pde=pde,stage=stage,verified_batches=len(receipts),verified_example_runs=sum(len(r['rows']) for r in receipts),
                maximum_allocated_gib=max(r['peak_allocated_bytes'] for r in receipts)/(1<<30),
                total_measured_seconds=sum(r['elapsed_seconds'] for r in receipts),
                paired_initialization_groups=len(noise),frozen_mask_batches=len(masks),protocol_sha256=expected_hash)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['pilot','evaluation'],required=True)
    p.add_argument('--pdes',nargs='+',default=['poisson','darcy','nsnonbounded','burger'])
    args=p.parse_args()
    print(json.dumps([check(args.inputs,args.output,pde,args.stage) for pde in args.pdes],indent=2))

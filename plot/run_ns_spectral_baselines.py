"""Task-specific NS baselines on the loss study's exact physical observations."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_ns_loss_study import sha,write


def make_batch(observed,mask,template,method,sample_id,device):
    import torch
    from baselines.common.data_adapter import PDEBatch
    from baselines.common.voronoi import voronoi_fill
    public=['canonical_layout','coordinate_layout','grid_layout','domain_length','T','nu',
            'final_time','joint_reconstruction','joint_input_channels','joint_solution_channels',
            'observation_source_channel_names','task_channel_names','input_channel_names','target_channel_names']
    meta={k:copy.deepcopy(v) for k,v in template['metadata'].items() if k in public}
    meta.update(num_sensors=int(mask[0,0].sum()),noise_level=0.,deferred_dynamic_sensors=False)
    meta['masked_grid']=observed.to(device)
    if method in ['recfno','voronoicnn']: meta['voronoi_grid']=voronoi_fill(observed,mask).to(device)
    assert torch.equal(mask,mask[:,:1].expand_as(mask))
    indices=mask[0,0].flatten().bool()
    coords=template['coords'][None].to(device)
    x=observed.to(device)
    target=torch.zeros((1,len(template['target_channel_names']),*x.shape[-2:]),device=device)
    # No saved template targets, trajectories or unobserved field values are
    # provided to inference; only public grid/channel conventions are reused.
    return PDEBatch(pde_name='nsnonbounded',task=template['task'],
        full_tensor=torch.zeros((1,2,*x.shape[-2:]),device=device),input_fields=x,target_fields=target,
        coords=coords,mask=mask.to(device),obs_values=x.flatten(2).transpose(1,2)[:,indices],
        obs_coords=coords[:,indices],channel_names=template['channel_names'],
        input_channel_names=template['input_channel_names'],target_channel_names=template['target_channel_names'],
        metadata=meta,pde_params={},split='test',sample_indices=torch.tensor([sample_id],device=device),
        global_sample_ids=[str(sample_id)],file_paths=[])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    sys.path.insert(0,str(args.baseline_root))
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    source=json.loads((args.inputs/'source.json').read_text())
    assert source['joint_shared_mask'] and sha(args.inputs/'fields_masks.npz')==source['fields_sha256']
    manifest=json.loads((args.weights/'source_manifest.json').read_text())
    for r in manifest:assert sha(args.weights/r['path'])==r['sha256']
    protocol=dict(source_sha256=sha(args.inputs/'source.json'),weights_manifest_sha256=sha(args.weights/'source_manifest.json'),
                  runner_sha256=sha(Path(__file__)),baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.baseline_root,text=True).strip(),
                  baseline_clean=not subprocess.check_output(['git','status','--porcelain'],cwd=args.baseline_root,text=True),
                  device=args.device,torch=torch.__version__,examples=32,tasks=['forward','inverse','both'],
                  methods=['recfno','senseiver','voronoicnn'],scope='Task-specific trained baseline checkpoints, exact common physical observations; one deterministic prediction per input; no training or weight selection.')
    assert protocol['baseline_clean']
    args.output.mkdir(parents=True,exist_ok=True)
    existing=args.output/'protocol.json'
    if existing.exists():assert json.loads(existing.read_text())==protocol
    else:write(existing,protocol)
    ph=sha(existing)
    data=np.load(args.inputs/'fields_masks.npz')
    classes={'recfno':'RecFNOBaseline','senseiver':'SenseiverBaseline','voronoicnn':'VoronoiCNNBaseline'}
    for task in protocol['tasks']:
        for method in protocol['methods']:
            folder=args.output/task/method;folder.mkdir(parents=True,exist_ok=True)
            payload=torch.load(args.weights/task/(method+'_checkpoint.pt'),map_location='cpu',weights_only=False)
            template=torch.load(args.weights/task/(method+'_template.pt'),map_location='cpu',weights_only=False)
            cls=getattr(importlib.import_module('baselines.methods.'+method),classes[method])
            config=copy.deepcopy(payload['config']);config['device']=args.device
            model=cls().build(config,payload['data_spec'])
            model.load_state_dict({k:v for k,v in payload['state_dict'].items() if torch.is_tensor(v)},strict=True)
            model.load_payload(payload);model.to(args.device).eval()

            def predict(observed,mask,i,hidden=False):
                batch=make_batch(observed,mask,template,method,i,args.device)
                if hidden:
                    batch.target_fields+=7.
                    batch.full_tensor+=9.
                with torch.no_grad():return model.predict_physical(batch).detach().cpu()

            # Cross-device numerical tolerance against the actual archived
            # GPU prediction, using its original observation mask.
            mask=template['mask'][None].float()
            observed=template['input_fields'][None].float()*mask
            ref=predict(observed,mask,0)
            original=template['prediction'][None].float()
            relative=float((ref-original).norm()/original.norm())
            if method in ['recfno','voronoicnn']:
                from baselines.common.voronoi import voronoi_fill
                regenerated=voronoi_fill(observed,mask)
                assert torch.equal(regenerated,template['metadata']['voronoi_grid'][None].float()), 'Archived observation representation differs'
            # The archive was generated on another device/backend. Check
            # strict checkpoint loading and exact observation representations
            # separately; do not call the CPU prediction bitwise equivalent.
            # Record the actual discrepancy and require <0.1% of prediction
            # norm, while every new score is recomputed on the new prediction.
            assert relative<1e-3,(task,method,'saved prediction mismatch',relative)
            checks=[]
            for i in source['pilot_ids'][:2]:
                a=torch.from_numpy(data[f'a_{i}']).float();u=torch.from_numpy(data[f'u_{i}']).float()
                ma=torch.from_numpy(data[f'mask_a_{i}']).float();mu=torch.from_numpy(data[f'mask_u_{i}']).float()
                field,mask=(a,ma) if task=='forward' else (u,mu) if task=='inverse' else (torch.cat([a,u],1),torch.cat([ma,ma],1))
                observed=field*mask
                base=predict(observed,mask,i);repeat=predict(observed,mask,i);hidden=predict(observed,mask,i,hidden=True);changed=predict(observed*.83,mask,i)
                row=dict(sample_id=i,repeat=float((base-repeat).abs().max()),hidden=float((base-hidden).abs().max()),sensitivity=float((base-changed).abs().max()))
                assert row['repeat']==row['hidden']==0 and row['sensitivity']>0 and torch.isfinite(base).all(),row
                checks.append(row)
            write(folder/'pilot.json',dict(status='pass',reference_relative_difference=relative,
                  reference_tolerance=1e-3,reference_scope='Cross-device numerical check; not bitwise equality. Exact weights and observation representation checked separately.',
                  checks=checks,protocol_sha256=ph))
            for i in source['evaluation_ids']:
                dest=folder/f'sample{i}.pt';receipt=dest.with_suffix('.json')
                if receipt.exists():
                    old=json.loads(receipt.read_text());assert old['protocol_sha256']==ph and sha(dest)==old['sha256'];continue
                a=torch.from_numpy(data[f'a_{i}']).float();u=torch.from_numpy(data[f'u_{i}']).float()
                ma=torch.from_numpy(data[f'mask_a_{i}']).float();mu=torch.from_numpy(data[f'mask_u_{i}']).float()
                field,mask,truth=(a,ma,u) if task=='forward' else (u,mu,a) if task=='inverse' else (torch.cat([a,u],1),torch.cat([ma,ma],1),torch.cat([a,u],1))
                prediction=predict(field*mask,mask,i)
                assert torch.isfinite(prediction).all()
                errors=[float((prediction[:,c:c+1].double()-truth[:,c:c+1].double()).norm()/truth[:,c:c+1].double().norm()) for c in range(truth.shape[1])]
                torch.save(dict(prediction=prediction,truth=truth,observations=field*mask,mask=mask),dest)
                write(receipt,dict(task=task,method=method,sample_id=i,relative_l2=errors,sha256=sha(dest),protocol_sha256=ph))
            write(folder/'complete.json',dict(status='complete',calls=32,protocol_sha256=ph))
            print('COMPLETE',task,method,32,flush=True)
    write(args.output/'complete.json',dict(status='complete',calls=288,protocol_sha256=ph))


if __name__=='__main__':main()

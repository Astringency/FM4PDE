"""Freeze and independently check the selected 4/8-example MSE/RMS study.

Selection is recomputed from the original per-example exports. Every selected
holdout field error is recomputed from prediction tensors; physical diagnostics
are checked against the original per-example export. This does not certify a
continuous PDE residual or historical training-data independence.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1]/'C04Papers/fm4pde_jmlr'
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    torch.set_num_threads(2)
    study=ROOT/'outputs/tuning/pde_loss_pair_197_20260906'
    source=PAPER/'source_data'; source.mkdir(exist_ok=True)
    rows=[]; protocols={}; manifests={}
    for form in ['mse','rms']:
        folder=study/form
        protocols[form]=json.loads((folder/'protocol.json').read_text())
        manifests[form]=json.loads((folder/'sample_manifest.json').read_text())
        rows += [dict(r,form=form) for r in csv.DictReader((folder/'per_sample.csv').open())]
    for key in ['seed','tune_sample_ids','holdout_sample_ids','batch_size','steps','time_grid',
                'sampler','clip_mode','clip_threshold','sensor_mode','num_obs','test_type','schedules','configs']:
        assert protocols['mse'][key]==protocols['rms'][key],key
    protocol=protocols['mse']
    assert not set(protocol['tune_sample_ids']) & set(protocol['holdout_sample_ids'])
    selection={}; comparisons=[]; fingerprints={}; raw=[]; output_rows=[]; trajectories=[]
    metric_count=0; tensor_differences=[]
    for pde,task in protocol['cells']:
        paired={}
        for form in ['mse','rms']:
            candidates={}
            for r in rows:
                if (r['form'],r['pde'],r['task'],r['stage'])==(form,pde,task,'tune'):
                    candidates.setdefault((r['schedule'],float(r['zeta_pde'])),[]).append(r)
            candidates={k:v for k,v in candidates.items() if len(v)==4 and all(r['status']=='ok' for r in v)}
            key=min(candidates,key=lambda k:(statistics.mean(float(r['primary_error']) for r in candidates[k]),k[1],k[0]))
            selection[f'{pde}/{form}']=dict(schedule=key[0],zeta=key[1],selection_ids=sorted(int(r['sample_id']) for r in candidates[key]))
            selected=sorted([r for r in rows if (r['form'],r['pde'],r['task'],r['stage'],r['schedule'],float(r['zeta_pde']))
                             ==(form,pde,task,'holdout',*key)],key=lambda r:int(r['sample_id']))
            assert [int(r['sample_id']) for r in selected]==sorted(protocol['holdout_sample_ids'])
            assert all(r['status']=='ok' and not r['error'] for r in selected)
            checked={}
            for row in selected:
                run=ROOT/row['run_dir'].split('/FM4PDE/',1)[1]
                if run not in checked:
                    payload=torch.load(run/'result.pt',map_location='cpu',weights_only=False)
                    cfg=payload['config']; ids=list(map(int,payload['ground_truth_metadata']['sample_ids']))
                    assert cfg.get('pde_guidance_reduction','mse')==form
                    assert not payload['ground_truth_metadata'].get('synthetic',False)
                    per=list(csv.DictReader((run/'metrics_per_sample.csv').open()))
                    assert [int(r['sample_id']) for r in per]==ids
                    errors={}
                    for field,label in [('coef','a'),('sol','u')]:
                        pred=payload[field+'_final'].double(); truth=payload[field+'_ground_truth'].double()
                        values=(torch.linalg.vector_norm((pred-truth).flatten(1),dim=1)/
                                torch.linalg.vector_norm(truth.flatten(1),dim=1).clamp_min(1e-12)).tolist()
                        errors[label]=values
                        for v,r in zip(values,per):
                            archived=float(r['rel_l2_'+label])
                            tensor_differences.append(dict(pde=pde,form=form,field=label,sample_id=int(r['sample_id']),
                                                           absolute=abs(v-archived),relative=abs(v-archived)/max(abs(v),1e-12)))
                            # Archived GPU float32 reductions differ from CPU float64.
                            assert math.isclose(v,archived,rel_tol=5e-5,abs_tol=1e-8),(run,label,v,r)
                            r['recomputed_rel_l2_'+label]=v
                    h=hashlib.sha256()
                    for t in [payload['coef_ground_truth'],payload['sol_ground_truth'],payload['masks']['coef'],payload['masks']['sol']]:
                        h.update(t.contiguous().numpy().tobytes())
                    paired_keys=['sample_seed','mask_seed','batch_size','num_steps','time_grid','sampler_phase',
                                 'zeta_obs_a','zeta_obs_u','clip_threshold','loss_state','gradient_target','checkpoint_path']
                    h.update(json.dumps({k:cfg[k] for k in paired_keys},sort_keys=True).encode())
                    pair_key=(pde,tuple(ids)); fingerprint=h.hexdigest()
                    assert pair_key not in fingerprints or fingerprints[pair_key]==fingerprint,pair_key
                    fingerprints[pair_key]=fingerprint
                    curves=list(csv.DictReader((run/'curves.csv').open()))
                    assert len(curves)==100 and [int(r['step']) for r in curves]==list(range(100))
                    trajectories += [dict(r,pde=pde,form=form,batch_ids=';'.join(map(str,ids))) for r in curves]
                    raw.append(dict(pde=pde,form=form,run_dir=str(run),sample_ids=ids,config=cfg,
                                    result_sha256=sha(run/'result.pt'),metrics_sha256=sha(run/'metrics_per_sample.csv'),
                                    curves_sha256=sha(run/'curves.csv'),paired_fingerprint=fingerprint,per_sample=per))
                    checked[run]={int(r['sample_id']):r for r in per}
                original=checked[run][int(row['sample_id'])]
                for field in ['rel_l2_a','rel_l2_u','obs_rel_l2_a','obs_rel_l2_u','pde_residual_norm']:
                    assert math.isclose(float(row[field]),float(original[field]),rel_tol=1e-12,abs_tol=1e-14)
                    metric_count+=1
                primary=float(original['rel_l2_u']) if pde=='burger' else (float(original['rel_l2_a'])+float(original['rel_l2_u']))/2
                assert math.isclose(primary,float(row['primary_error']),rel_tol=1e-12,abs_tol=1e-14)
                output_rows.append(dict(row,recomputed_rel_l2_a=original['recomputed_rel_l2_a'],
                                        recomputed_rel_l2_u=original['recomputed_rel_l2_u']))
            paired[form]=selected
        a=statistics.mean(float(r['primary_error']) for r in paired['mse'])
        b=statistics.mean(float(r['primary_error']) for r in paired['rms'])
        ar=statistics.mean(float(r['pde_residual_norm']) for r in paired['mse'])
        br=statistics.mean(float(r['pde_residual_norm']) for r in paired['rms'])
        comparisons.append(dict(pde=pde,n=8,mse_error=a,rms_error=b,error_change_pct=100*(b/a-1),
                                mse_residual=ar,rms_residual=br,residual_change_pct=100*(br/ar-1),
                                **{form+'_'+k:v for form in ['mse','rms'] for k,v in selection[f'{pde}/{form}'].items() if k!='selection_ids'}))
    for name,data in [('loss_holdout_verified.csv',output_rows),('loss_comparison_verified.csv',comparisons),
                      ('loss_trajectories_verified.csv',trajectories)]:
        with (source/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    archive=dict(protocols=protocols,sample_manifests=manifests,selection=selection,raw_runs=raw,
                 verification=dict(selected_runs=len(raw),paired_batches=len(fingerprints),
                                   tensor_field_errors_recomputed=len(tensor_differences),export_metrics_checked=metric_count,
                                   recomputation_dtype='CPU float64 from archived float32 prediction/target tensors',
                                   maximum_absolute_error_difference=max(r['absolute'] for r in tensor_differences),
                                   maximum_relative_error_difference=max(r['relative'] for r in tensor_differences),
                                   limitations=['Physical residual values checked against source exports, not independently re-discretized.',
                                                'Initial noise pairing is inferred from matched seeds/settings; tensors were not archived.',
                                                'Four-example selection and eight-example holdout; one inference seed and fixed checkpoints.']))
    with gzip.open(source/'loss_study_audit.json.gz','wt') as f: json.dump(archive,f,allow_nan=False)
    print(json.dumps(archive['verification'],indent=2)); print(json.dumps(comparisons,indent=2))


if __name__=='__main__': main()

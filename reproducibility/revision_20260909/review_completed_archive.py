#!/usr/bin/env python3
"""Bind completed-study gates and recheck frozen metadata before archive copying."""
import concurrent.futures
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

import build_archive_inventory as inventory

HERE = Path(__file__).resolve().parent
AUDIT = inventory.AUDIT
OUT = AUDIT/'revision_archive_execution_20260909'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate(name, relative, predicate, reason, expected=None):
    path = AUDIT/relative
    data = json.loads(path.read_text())
    assert predicate(data), 'Failed scientific completion gate: ' + str(path)
    value = digest(path)
    if expected:
        assert value == expected, 'Owner-frozen gate SHA mismatch: ' + str(path)
    return name, dict(path=str(path),sha256=value,reason=reason,status='pass')


def main():
    original_path = HERE/'archive_inventory.json'
    old = json.loads(original_path.read_text())
    previous = {e['id']:e for e in old['entries']}
    gates = dict([
        gate('ablation','paper_revision_20260908/ablation_publication_snapshot/manifest.json',lambda d:d['final_ready'] and d['required_reruns']==d['completed_reruns']==744 and not d['pending_ids'],'All 744 required revisions complete; 316 original controls separately verified in place.'),
        gate('three_draw','paper_revision_20260908/ensemble_complete/manifest.json',lambda d:d['full_study'] and d['predictions']==d['expected_full_predictions']==1056 and len(d['pdes'])==11,'All 1056 predictions across 11 PDEs complete; original three-draw study retained.'),
        gate('burgers','paper_revision_20260908/burgers_complete/manifest.json',lambda d:d['final_ready'] and d['completed_new_calls']==d['expected_new_calls']==7000 and not d['pending_jobs'],'All 7000 new calls complete; original 150 random-FM tensor hashes verified in place.'),
        gate('timing','diffusion_fm_revision_20260907/timing_ready_for_visual_review.json',lambda d:d['calls']==400 and d['status']=='audited_integrated_built_ready_for_visual_review','Original 400-call timing audited; the final figure uses 320 original non-NS calls and 80 new NS calls.'),
        gate('matched_timing','fm4pde_jmlr_matched_timing_20260907/report/matched_timing_manifest.json',lambda d:d['calls_verified']==1920 and d['prediction_error_max_discrepancy']==0,'All 1920 matched calls verified with zero recomputation discrepancy.'),
        gate('sampling_controls','fm4pde_jmlr_sampling_20260906/completed_three_pde_report/sampling_confirmation_manifest.json',lambda d:d['stage']=='evaluation' and len(d['protocol']['evaluation_ids'])==32 and len(d['validation'])==3 and all(r['verified_batches']==312 and r['verified_example_runs']==1248 for r in d['validation']),'Final three-PDE report verifies 312 batches / 1248 example-runs per PDE over 32 registered inputs.'),
        gate('ns_supplement','ns_loss_spectrum_20260907/ns_complete_qa/complete_reporting_qa.json',lambda d:d['status']=='pass' and d['calls']==1728,'All 1728 loss-exchange calls and 1536 variance identity checks passed.'),
        gate('ns_calibration','ns_loss_spectrum_20260907/calibration_complete_audit/calibration_audit_manifest.json',lambda d:d['status']=='complete' and d['selection_independently_verified'] and d['outcomes']=={'calibration':{'complete':540},'evaluation':{'complete':576}},'540 calibration and 576 evaluation calls, independent selection, paired-setting and ensemble-convexity checks complete.'),
        gate('ns_model_comparison','ns_checkpoint_comparison_20260908/final_status.json',lambda d:d['status']=='complete' and d['main_calls']==1152 and d['step_probe_calls']==72,'1152 model-comparison calls and 72 probes complete, including legitimate additional history.'),
        gate('ns_guidance_cross','ns_guidance_cross_20260908/final_status.json',lambda d:d['status']=='complete' and d['new_calls']==576 and d['outcomes']=={'complete':576},'All 576 guidance-cross calls complete.'),
        gate('ns_rough','ns_rough_diffusion_20260908/final_status.json',lambda d:d['status']=='complete' and d['formal_calls']==96 and d['failures']==0 and d['all_predictions_recomputed'],'All 96 rough Diffusion calls independently recomputed, no failures.'),
        gate('ns_main_owner','ns_main_revision_0909/FINAL_HANDOFF.json',lambda d:d['status']=='complete_and_validated' and d['main_samples']==15000 and d['timing_calls_new_ns']==80 and d['all_production_gpu_tasks_finished'],'Owner explicitly froze all NS sources after final 15000-example and 80-call timing audits.',expected='3787c15af2ad3b97c722f74c51fc62c3bdb0f368a5d8d137074e9e49f447fb20'),
        gate('ns_main_rel_l2','ns_main_revision_0909/tables/ns_main_validation.json',lambda d:d['status']=='pass' and d['examples']==15000 and d['all_result_hashes_match'] and d['all_nfe_100'],'All 15000 tensors, finite predictions, NFE and independent observation/relative-error checks passed.',expected='124540d4415a4aceac55b955b8e655a86617a068b5d5e04d9de20621f984965b'),
        gate('ns_main_residual','ns_main_revision_0909/tables/ns_main_residual_audit.json',lambda d:d['status']=='pass' and d['complete'] and d['examples']==15000 and d['batches']==450 and all(v for k,v in d.items() if k.startswith('all_')),'All 450 raw batches pass the independent float64 PDE, frozen-input/configuration/mask/runtime audit.',expected='236181acd0b97c5ee25ff65d8ee13a3e613d67ee555e5b2fcb216dba60e5197b'),
    ])
    assert digest(AUDIT/'ns_main_revision_0909/ARCHIVE_INVENTORY.json')=='1069710f114f694b61b307a5b334b97b471961b4bc29bdf20d3fb5167acb4639'
    reference_path = HERE/'in_place_reference_checks.json'
    ref = json.loads(reference_path.read_text())
    assert ref['status']=='passed' and ref['counts']=={'burgers_original_random':150,'unchanged_ablations':316}
    gates['original_fm_in_place'] = dict(path=str(reference_path),sha256=digest(reference_path),status='pass',reason='All 316 unchanged control tensors present; all 150 original random-FM hashes match frozen manifest.')
    # Legacy main outputs are preserved historical evidence, not newly claimed runs.
    for name in ['baseline_effective_counts_verified.csv','baseline_counts_archive.json.gz']:
        path=inventory.PAPER/'source_data'/name
        gates[name]=dict(path=str(path),sha256=digest(path),status='recorded_published_provenance',reason='Published legacy comparison metric identities and effective counts; exact selected raw metric files are SHA-bound.')
    entries, ns = inventory.catalog()
    eligible = [e for e in entries if e['status']=='ready' and e['role'] in ['primary','dependency']]
    observed={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(inventory.probe,h,[e for e in eligible if e['source_host']==h]) for h in ['local',*inventory.BASE]]
        for future in concurrent.futures.as_completed(jobs):
            observed.update({r['id']:r for r in future.result()})
    decisions=[]
    for e in entries:
        review={'id':e['id'],'original_status':previous.get(e['id'],{}).get('status'),'decision':'not_selected','reasons':[]}
        if e not in eligible:
            review['reasons']=['Pending producer, alternate duplicate, or existing/reference-only source; not copied in this batch.']
            decisions.append(review);continue
        now=observed[e['id']];before=previous.get(e['id'],{})
        e['observed']=now;e['source_kind']=now['source_kind']
        if now.get('selected_relative_files'):e['include_files']=now.pop('selected_relative_files')
        errors=[]
        if not now['exists'] or now.get('missing'):errors.append('missing or dangling source')
        expected=e.get('evidence_sha256',{})
        for name,value in expected.items():
            if now['metadata_sha256'].get(name)!=value:errors.append('known SHA mismatch: '+name)
        if before.get('status')=='ready':
            old_meta=before['observed'].get('metadata_sha256',{})
            changed=[p for p in sorted(set(old_meta)|set(now['metadata_sha256'])) if old_meta.get(p)!=now['metadata_sha256'].get(p)]
            if changed:errors.append('metadata changed since initial census')
            review['changed_metadata']=changed
        study=Path(e['destination_relative']).parts[2]
        keys={'paper_revision_20260908':['ablation','three_draw','original_fm_in_place'],
              'burgers_revision_20260908':['burgers','original_fm_in_place'],
              'diffusion_fm_timing_20260907':['timing'], 'matched_baseline_timing_20260907':['matched_timing'],
              'sampling_confirmation_20260906':['sampling_controls'],
              'ns_loss_spectrum_20260907':['ns_supplement','ns_calibration'],
              'ns_checkpoint_comparison_20260908':['ns_model_comparison'],
              'ns_guidance_cross_20260908':['ns_guidance_cross'], 'ns_rough_diffusion_20260908':['ns_rough'],
              'ns_main_revision_0909':['ns_main_owner','ns_main_rel_l2','ns_main_residual'],
              'conditional_scaling_20260909':['ablation'],
              'diffusion_original_main':['baseline_counts_archive.json.gz'],
              'legacy_baseline_extracts':['baseline_effective_counts_verified.csv','baseline_counts_archive.json.gz']}[study]
        review['gate_keys']=keys
        if errors:
            e['status']='hold_for_review';review['decision']='hold_for_review';review['reasons']=errors
        else:
            e['evidence_sha256']={**now['metadata_sha256'],**expected}
            e['completion_review']={k:gates[k] for k in keys}
            review['decision']='approved';review['reasons']=[gates[k]['reason'] for k in keys]+['All registered sources present; frozen metadata and known content hashes match. Full-file transfer SHA verification remains mandatory.']
        review['logical_bytes']=now.get('logical_bytes',0);review['metadata_hashes']=len(now['metadata_sha256'])
        decisions.append(review)
    OUT.mkdir(exist_ok=True)
    reviewed=copy.deepcopy(old);reviewed.update(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='reviewed_for_completed_studies_only',original_inventory_sha256=digest(original_path),entries=entries,completion_gates=gates,review_script_sha256=digest(__file__),ns_owner_inventory_sha256=digest(AUDIT/'ns_main_revision_0909/ARCHIVE_INVENTORY.json'))
    (OUT/'archive_inventory.reviewed.json').write_text(json.dumps(reviewed,indent=2)+'\n')
    report=dict(observed_utc=reviewed['observed_utc'],reviewed_inventory_sha256=digest(OUT/'archive_inventory.reviewed.json'),gates=gates,decisions=decisions,
        approved_entries=sum(r['decision']=='approved' for r in decisions),held_entries=sum(r['decision']=='hold_for_review' for r in decisions),
        approved_bytes=sum(r.get('logical_bytes',0) for r in decisions if r['decision']=='approved'),pending_ids=[e['id'] for e in entries if e['status']=='pending'])
    (OUT/'completion_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['gates','decisions']},indent=2))


if __name__ == '__main__':main()

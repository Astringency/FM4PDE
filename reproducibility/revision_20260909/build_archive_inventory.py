#!/usr/bin/env python3
"""Read-only source census; write the local revision archive inventory.

No transfer, directory mutation on servers, training, or sampling is performed.
The census hashes configuration/protocol metadata, not multi-GB result tensors.
Actual archive execution computes SHA-256 for every copied file independently.
"""
from __future__ import annotations
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = Path('/home/tat512/C01Python/audit')
PAPER = Path('/home/tat512/C04Papers/fm4pde_jmlr')
BASE = {'server197': '/research_data/users/zhangxifeng/C01Python',
        'server193': '/home/zhangxf/C01Python', 'server216': '/data1/zjinzxf2025/C01Python'}
PYTHON = {'server197': '/usr/bin/python3', 'server193': '/home/zhangxf/conda_envs/fm4pdebaseline/bin/python',
          'server216': '/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python'}


def catalog():
    entries = []
    def add(identifier, host, path, category, study, leaf, purpose, status='ready', role='primary', **extra):
        entries.append(dict(id=identifier, source_host=host, source_path=str(path),
            destination_relative=f'{category}/revision_20260909/{study}/{leaf}',
            purpose=purpose, status=status, role=role, **extra))

    for name, category in [('main', 'main'), ('ablations', 'ablations')]:
        add('existing_fm_' + name, 'server197', BASE['server197'] + '/FM4PDE/outputs/' + name,
            category, 'existing_sources', name, 'Already inside requested outputs/main or outputs/ablations. Preserve in place; do not recursively copy the parent.',
            status='reference', role='already_at_target')
    add('baseline_evaluation_mother_tree', 'server197', '/large_storage/zhangxf/outputs/FM4PDEbaseline/runs/evaluations',
        'main', 'references', 'baseline_evaluations', '3.06 TB unfiltered parent. Not an archive copy source; only selected published metrics/subsets/models are in scope.', status='reference', role='reference_only')
    add('fm_training_checkpoint_library', 'server197', BASE['server197'] + '/FM4PDE/outputs/pretrained',
        'main', 'references', 'pretrained_library', '248.5 GB historical checkpoint library. Selected inference models are preserved in frozen inputs; do not copy the entire library.', status='reference', role='reference_only')

    for host in BASE:
        add('ablation_raw_' + host, host, BASE[host] + '/paper_revision_20260908/output', 'ablations', 'paper_revision_20260908',
            host + '/output', 'Original rerun and three-draw production outputs, receipts/configurations and legitimate recorded attempts. Together with preserved old ablations: 744 revised + 316 unchanged configurations and 1056 three-draw predictions.')
    add('ablation_frozen_inputs', 'server197', BASE['server197'] + '/paper_revision_20260908/inputs', 'ablations', 'paper_revision_20260908',
        'frozen_inputs', 'All 11 PDE frozen truths, protocols and selected inference weights.', role='dependency', dereference_symlinks=True)
    add('ablation_canonical_results', 'local', AUDIT/'paper_revision_20260908/output', 'ablations', 'paper_revision_20260908',
        'canonical_results', 'Single 11-PDE authoritative reading view for both ablation and three-draw exporters. Dereference the nine local PDE links; original per-host raw trees remain available separately.', dereference_symlinks=True)
    for name in ['ablation_publication_snapshot', 'ablation_publication_tables', 'ablation_publication_figures', 'ensemble_complete']:
        add('ablation_export_' + name, 'local', AUDIT/'paper_revision_20260908'/name, 'ablations', 'paper_revision_20260908',
            'local_exports/' + name, 'Final complete numerical/figure export with source hashes and publication counts.')

    for host in BASE:
        add('burgers_raw_' + host, host, BASE[host] + '/paper_revision_20260908/burgers_output_v3', 'main', 'burgers_revision_20260908',
            host + '/burgers_output_v3', 'Original final 7000-call Burgers supplementary main evaluation shards and pilot/equivalence records; relative output structure retained.')
    add('burgers_random_fm', 'server197', BASE['server197'] + '/paper_revision_20260908/burgers_fm_random', 'main', 'burgers_revision_20260908',
        'server197/burgers_fm_random', 'Per-input FM random-sensor metrics and manifest binding original prediction hashes. Referenced result.pt files already reside under server197/FM4PDE/outputs/main and are preserved in place.')
    for name in ['burgers_inputs', 'burgers_weights', 'burgers_complete']:
        add('burgers_' + name, 'local', AUDIT/'paper_revision_20260908'/name, 'main', 'burgers_revision_20260908',
            'local/' + name, 'Frozen Burgers dependencies or complete publication export.', role='dependency' if name != 'burgers_complete' else 'primary', dereference_symlinks=True,
            **({'evidence_sha256': {'pretrained-burgers.pkl': 'a24eddebaff43e477e015e8a0f4869e27bc0aeaae224f24f319c16e3dad50edf'}} if name == 'burgers_weights' else {}))
    add('burgers_production_inputs', 'server197', BASE['server197'] + '/paper_revision_20260908/burgers_inputs', 'main', 'burgers_revision_20260908',
        'server197/burgers_inputs', 'Producer-side frozen inputs and selected models; captures dependencies absent from compact local exports.', role='dependency', dereference_symlinks=True)
    add('burgers_selected_fm_weights', 'server197', BASE['server197'] + '/paper_revision_20260908/inputs/burger/weights.pth',
        'main', 'burgers_revision_20260908', 'selected_models/fm',
        'Exact FM checkpoint required by sampling_protocol_v3. Same bytes as the ablation Burger checkpoint; separate explicit copy makes the Burgers entry point self-contained.', role='dependency',
        evidence_sha256={'weights.pth':'b76ea10874c37b04061d41e909a6e05bfb954f3e6797fd70895bf27f49abf956'},
        expected_bytes=385948627, cross_study_reference='ablations/revision_20260909/paper_revision_20260908/frozen_inputs/burger/weights.pth')

    for host in ['server197', 'server216']:
        add('conditional_scaling_raw_' + host, host, BASE[host] + '/conditional_scaling_results_20260909', 'ablations', 'conditional_scaling_20260909',
            host + '/results', 'All K tensors/receipts plus pilot and full-step validation artifacts. Pending until all producer and final scientific gates pass.',
            status='pending', producer_commit='1f1573b', completion_requirement='Four complete_<shard>.json receipts and local READY_FOR_PAPER_REVIEW.json, followed by final scientific review.')
    add('conditional_scaling_inputs', 'server197', BASE['server197'] + '/paper_revision_20260908/inputs/poisson', 'ablations', 'conditional_scaling_20260909',
        'frozen_inputs/poisson', 'Poisson truth/protocol/selected weights; protocol be205a... is shared with the ablation input snapshot.', role='dependency', dereference_symlinks=True)
    add('conditional_scaling_local_audit', 'local', AUDIT/'conditional_scaling_20260909', 'ablations', 'conditional_scaling_20260909',
        'local_audit', 'Collector state, compact exports and validation audit. Do not copy while the collector is updating files.', status='pending')

    add('old_diffusion_main', 'server216', BASE['server216'] + '/DiffusionPDE/outputs', 'main', 'diffusion_original_main',
        'server216/outputs', 'Original native DiffusionPDE main results. Only the registered MAIN1000_100 and MAIN1000_1000 trees are copied.',
        selected_directories=['MAIN1000_100', 'MAIN1000_1000'])
    add('old_diffusion_weights', 'server216', '/data0/zhangxf/Models/pretrained-models', 'main', 'diffusion_original_main',
        'selected_models', 'Native pretrained models required by the original five-PDE comparison.', role='dependency')
    add('diffusion_timing_study', 'local', AUDIT/'diffusion_fm_revision_20260907', 'main', 'diffusion_fm_timing_20260907',
        'local_study', 'Original complete 400-call timing, source/weights/inputs and audit; final figure retains 320 non-NS calls.', dereference_symlinks=True)
    add('matched_baseline_timing', 'local', AUDIT/'fm4pde_jmlr_matched_timing_20260907', 'main', 'matched_baseline_timing_20260907',
        'local_study', 'Controlled matched-baseline timing inputs, raw runs and reports.', dereference_symlinks=True)
    add('sampling_confirmation_local', 'local', AUDIT/'fm4pde_jmlr_sampling_20260906', 'ablations', 'sampling_confirmation_20260906',
        'local_study', 'Earlier repeated guided controls and frequency diagnostics, frozen inputs, raw collected results and complete audits.', dereference_symlinks=True)
    add('sampling_confirmation_original', 'server193', BASE['server193'] + '/FM4PDE_jmlr_20260906/revision_results', 'ablations', 'sampling_confirmation_20260906',
        'server193/revision_results', 'Original completed producer result tree; keeps original receipts/attempts beyond the compact local report.')
    for name in ['ns_loss_spectrum_20260907', 'ns_checkpoint_comparison_20260908', 'ns_guidance_cross_20260908']:
        add(name + '_local', 'local', AUDIT/name, 'ablations', name, 'local_study',
            'Complete merged NS supplementary study: raw predictions, all candidates, protocol, source hashes, diagnostics and exports.', dereference_symlinks=True)
    add('ns_supplement_selected_models', 'server197', BASE['server197'] + '/ns_loss_spectrum_20260907/calibration_weights',
        'ablations', 'ns_loss_spectrum_20260907', 'selected_models', 'Exact old NS inference checkpoint used by calibration/loss-exchange; retains frozen checkpoint fingerprint.', role='dependency', dereference_symlinks=True)
    add('ns_supplement_frozen_inputs', 'server197', BASE['server197'] + '/ns_loss_spectrum_20260907/inputs_v2',
        'ablations', 'ns_loss_spectrum_20260907', 'frozen_inputs', 'Original common-input truth and mask subset for 32 evaluation + four calibration examples.', role='dependency', dereference_symlinks=True)
    add('ns_matched_baseline_inputs', 'server197', BASE['server197'] + '/ns_loss_spectrum_20260907/baseline_inputs',
        'ablations', 'ns_loss_spectrum_20260907', 'baseline_inputs', 'Frozen matched baseline inputs and selected baseline checkpoints.', role='dependency', dereference_symlinks=True)
    for host, path in [('server193', '/FM4PDE_ns_loss_spectrum_20260907/ns_results_v3'),
                       ('server193', '/FM4PDE_ns_loss_spectrum_20260907/ns_results_accelerated_v1'),
                       ('server193', '/FM4PDE_ns_loss_spectrum_20260907/ns_results_accelerated_v2'),
                       ('server197', '/ns_loss_spectrum_20260907/ns_results_accelerated_v1'),
                       ('server197', '/ns_loss_spectrum_20260907/ns_results_accelerated_v2'),
                       ('server216', '/ns_loss_extension_20260907/ns_results_accelerated_v2'),
                       ('server197', '/ns_loss_spectrum_20260907/guidance_calibration_v1')]:
        leaf = path.rsplit('/', 1)[-1]
        add('ns_supplement_origin_' + host + '_' + leaf, host, BASE[host] + path, 'ablations', 'ns_loss_spectrum_20260907',
            host + '/' + leaf, 'Original immutable shard. Complete publication outcomes are merged in local_study; retained as a provenance/alternate source.', role='alternate')
    for host in BASE:
        add('ns_rough_diffusion_' + host, host, BASE[host] + '/ns_rough_diffusion_20260908/output', 'main', 'ns_rough_diffusion_20260908',
            host + '/output', 'Original three-host completed Rough DiffusionPDE main-result shards.')
    add('ns_rough_diffusion_local', 'local', AUDIT/'ns_rough_diffusion_20260908', 'main', 'ns_rough_diffusion_20260908',
        'local_study', 'Merged rough-comparison audit and frozen input subset.', dereference_symlinks=True)

    # Incorporate the NS owner inventory without treating old directory labels as
    # authoritative destinations. Every source receives a distinct origin leaf.
    ns_inventory = AUDIT/'ns_main_revision_0909/ARCHIVE_INVENTORY.json'
    ns = json.loads(ns_inventory.read_text())
    for row in ns['entries']:
        if row['name'] in ['old_timing_inputs', 'old_timing_raw', 'model_selection_study', 'diffusion_external_code']:
            continue
        status = 'ready' if row['status'] == 'complete' else 'pending'
        role = row['copy_role']
        if role not in ['primary', 'alternate', 'dependency']:
            role = 'reference_only'
        add('ns_main_' + row['name'], row['host'], row['source_path'], 'main', 'ns_main_revision_0909',
            row['host'] + '/' + row['name'], row['purpose'], status=status, role=role,
            dereference_symlinks=row.get('dereference_symlinks', False), producer_commit=row.get('expected_commit'),
            owner_inventory=str(ns_inventory), completion_requirement='Owner collection_complete, 15000-example validation, and residual audit must pass before pending entries are promoted.')

    # Exact published legacy baseline metric files, not the 3 TB parent tree.
    main_rows = {r['row'] for r in csv.DictReader((PAPER/'source_data/main_figure_values.csv').open()) if r['file'] == 'results.xlsx'}
    source_rows = [r for r in csv.DictReader((PAPER/'source_data/baseline_effective_counts_verified.csv').open()) if r['row'] in main_rows]
    files = {}; identities = []
    for row in source_rows:
        relative = row['raw_path'].split('/FM4PDEbaseline/runs/', 1)[1]
        files[relative] = row['raw_sha256']
        identities.append({k: row[k] for k in ['row', 'PDE', 'Method', 'TASK', 'DIST', 'field', 'raw_path']})
    add('published_baseline_metrics', 'server197', '/large_storage/zhangxf/outputs/FM4PDEbaseline/runs', 'main', 'legacy_baseline_extracts',
        'runs', 'Exact raw metric files underlying published main-table baseline rows; original relative directory structure retained. No unrelated evaluations/prediction trees.',
        include_files=sorted(files), evidence_sha256=files, published_cells=identities)
    return entries, ns


PROBE = r'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
entries=json.load(sys.stdin);out=[]
for entry in entries:
 p=Path(entry['source_path']);row={'id':entry['id'],'exists':p.exists(),'source_kind':'directory' if p.is_dir() else 'file','symlinks':[],'metadata_sha256':{},'recorded_commits':[],'recorded_script_hashes':[]}
 if not p.exists():out.append(row);continue
 row['resolved_source']=str(p.resolve())
 if entry['role'] in ['reference_only','already_at_target']:
  row['disk_bytes']=int(subprocess.check_output(['du','-s','-B1',str(p)],text=True).split()[0]);out.append(row);continue
 def walk(q,ancestors):
  if q.is_symlink():
   rel=str(q.relative_to(p)) if p.is_dir() else q.name
   row['symlinks'].append({'path':rel,'target':os.readlink(q),'target_exists':q.exists()})
   if not entry.get('dereference_symlinks'):return [q]
  if q.is_dir():
   if q.resolve() in ancestors:raise RuntimeError('Symlink cycle: '+str(q))
   return [f for c in q.iterdir() for f in walk(c,ancestors|{q.resolve()})]
  return [q]
 if entry.get('include_files'):
  paths=[p/x for x in entry['include_files']]
 elif entry.get('selected_directories'):
  paths=[f for name in entry['selected_directories'] for f in walk(p/name,set())]
  row['selected_relative_files']=[str(f.relative_to(p)) for f in paths]
 else:paths=walk(p,set())
 logical=0;files=0;missing=[]
 for f in paths:
  rel=str(f.relative_to(p)) if p.is_dir() else f.name
  if f.is_symlink() and not entry.get('dereference_symlinks'):continue
  if not f.exists():missing.append(rel);continue
  if not f.is_file():continue
  st=f.stat();logical+=st.st_size;files+=1
  if (f.suffix.lower() in ['.json','.yaml','.yml','.toml'] and st.st_size <= 8<<20) or rel in entry.get('evidence_sha256',{}):
   raw=f.read_bytes();row['metadata_sha256'][rel]=hashlib.sha256(raw).hexdigest()
   if f.suffix=='.json':
    try:d=json.loads(raw)
    except Exception:continue
    if isinstance(d,dict):
     for key in ['commit','source_commit','producer_commit']:
      if isinstance(d.get(key),str):row['recorded_commits'].append(d[key])
     for key in ['script_sha256','source_script_sha256']:
      if isinstance(d.get(key),str):row['recorded_script_hashes'].append(d[key])
 row.update(logical_bytes=logical,files=files,missing=missing)
 row['recorded_commits']=sorted(set(row['recorded_commits']));row['recorded_script_hashes']=sorted(set(row['recorded_script_hashes']))
 out.append(row)
print(json.dumps(out))
'''


def probe(host, entries):
    if host == 'local':
        argv = ['python', '-c', PROBE]
    else:
        argv = ['ssh', host, shlex.join([PYTHON[host], '-c', PROBE])]
    result = subprocess.run(argv, input=json.dumps(entries), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def main():
    entries, ns = catalog()
    groups = {h: [e for e in entries if e['source_host'] == h] for h in ['local', *BASE]}
    observed = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(probe, h, values): h for h, values in groups.items()}
        for future in concurrent.futures.as_completed(jobs):
            for row in future.result():
                observed[row['id']] = row
    for entry in entries:
        row = observed[entry['id']]
        entry['observed'] = row
        entry['source_kind'] = row['source_kind']
        if row.get('selected_relative_files'):
            entry['include_files'] = row.pop('selected_relative_files')
        if not row['exists'] or row.get('missing'):
            entry['status'] = 'missing_source'
        expected = entry.get('evidence_sha256', {})
        mismatch = [p for p, h in expected.items() if row['metadata_sha256'].get(p) != h]
        if mismatch:
            entry['status'] = 'source_hash_mismatch'; entry['mismatching_evidence'] = mismatch
        if entry['status'] == 'ready' and not expected:
            entry['evidence_sha256'] = row['metadata_sha256']
    budget = sum(e['observed'].get('logical_bytes', 0) for e in entries if e['role'] in ['primary','dependency'])
    report = dict(schema_version=1, observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='inventory_only_no_transfers', target_host='server197', target_root=BASE['server197']+'/FM4PDE/outputs',
        layout='outputs/{main,ablations}/revision_20260909/<study>/<origin-or-component>',
        source_repositories='source_repositories.json (root-owned); code snapshots are separate from result transfers',
        in_place_reference_checks='in_place_reference_checks.json (generated separately by verify_in_place_references.py): all 316 unchanged ablation tensors and 150 original Burgers FM random-sensor result files under server197/FM4PDE/outputs',
        shared_paper_assets=dict(status='pending',owner='root',sources=[str(PAPER/'source_data'),str(PAPER/'figures')],
            destination='/research_data/users/zhangxifeng/C01Python/FM4PDE/reproducibility/revision_20260909/paper_assets',
            note='Shared main and ablation publication assets are not misclassified as main results; each result study also retains its own final exports.'),
        disk_observation=dict(filesystem='/research_data', available_approx_TiB=4.0, whole_baseline_evaluations_bytes=3059996475392,
            whole_fm_pretrained_bytes=248534016000, unfiltered_parents_are_not_copy_sources=True),
        measured_primary_dependency_logical_bytes=budget,
        conditional_scaling_expected=dict(canonical_draws=96000,executed_draws=106944,
            prediction_tensor_bytes=106944*2*128*128*4, canonical_prediction_bytes=96000*2*128*128*4,
            expected_result_bytes_approx=14500000000, note='Current pending directory measurements are lower bounds; add about 14.5 GB for complete raw results including tensor/container overhead.'),
        ns_owner_inventory_sha256=hashlib.sha256((AUDIT/'ns_main_revision_0909/ARCHIVE_INVENTORY.json').read_bytes()).hexdigest(),
        ns_code=ns['code'], ns_environments=ns['environments'],
        exclusions=['Do not copy the 3.06 TB baseline evaluation parent.', 'Do not copy the 248.5 GB historical checkpoint library.',
                    'Training 50000-example datasets remain external dependencies; selected inference truths/weights are archived.',
                    'No active checkout is synchronized, no source is moved/deleted, and pending entries are not executable.'],
        entries=entries)
    (HERE/'archive_inventory.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    summary = {key: value for key, value in report.items() if key not in ['entries', 'ns_code', 'ns_environments']}
    summary['entries'] = [{**{k: e.get(k) for k in ['id','status','role','source_host','source_path','destination_relative','purpose']},
        'logical_bytes':e['observed'].get('logical_bytes'), 'disk_bytes':e['observed'].get('disk_bytes'),
        'files':e['observed'].get('files'), 'metadata_hash_count':len(e['observed'].get('metadata_sha256',{}))} for e in entries]
    summary['inventory_sha256'] = hashlib.sha256((HERE/'archive_inventory.json').read_bytes()).hexdigest()
    summary['ready_primary_dependency_bytes'] = sum(e['observed'].get('logical_bytes',0) for e in entries if e['status']=='ready' and e['role'] in ['primary','dependency'])
    summary['pending_primary_dependency_bytes'] = sum(e['observed'].get('logical_bytes',0) for e in entries if e['status']=='pending' and e['role'] in ['primary','dependency'])
    (HERE/'archive_inventory.summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'entries':len(entries),'bytes':budget,'status_counts':{s:sum(e['status']==s for e in entries) for s in sorted({e['status'] for e in entries})}},indent=2))


if __name__ == '__main__':main()

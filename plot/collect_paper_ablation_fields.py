"""Combine the 1,060 main ablation configurations, reporting a and u separately.

The default export requires all 744 requested reruns. --allow-pending writes
a development snapshot with explicit source/coverage fields; it must not be
represented as a finished rerun. Original special controls and three
unchanged PDEs remain part of the complete inventory.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_ablation_study import PDES, EXCLUDED_GROUPS, digest
from run_ablation_queue import REVISED


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def original_path(item, original_root):
    relative = item['source_result'].split('/FM4PDE/', 1)[1]
    return original_root / relative


def collect(args):
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    archived_rows = list(csv.DictReader(args.archive.open()))
    assert len(archived_rows) == 1060
    protocols = {pde: json.loads((args.inputs/pde/'protocol.json').read_text()) for pde in PDES}
    complete_pdes = set()
    for pde in REVISED:
        marker = args.results/pde/'rerun_complete.json'
        expected = [i['id'] for i in protocols[pde]['archived'] if i['config']['ablation_group'] not in EXCLUDED_GROUPS]
        if marker.exists() and all((args.results/pde/'main'/ident/'receipt.json').exists() for ident in expected):
            assert json.loads(marker.read_text())['protocol_sha256'] == digest(args.inputs/pde/'protocol.json')
            complete_pdes.add(pde)
    all_items = {item['id']: (pde, item) for pde, protocol in protocols.items() for item in protocol['archived']}
    assert len(all_items) == len(archived_rows)
    records, metrics, pending = [], [], []
    revised = 0
    for index, old in enumerate(archived_rows):
        ident = f'archive_{index:04d}'
        pde, item = all_items[ident]
        assert old['pde'] == pde
        for field in ['a', 'u']:
            assert float(old['rel_l2_'+field]) == item['old_errors'][field]
        required = pde in REVISED and item['config']['ablation_group'] not in EXCLUDED_GROUPS
        folder = args.results/pde/'main'/ident
        config = item['config'].copy()
        values = {field: float(old['rel_l2_'+field]) for field in ['a', 'u']}
        diagnostics = {key:float(old[key]) for key in ['L_pde','pde_residual_norm','L_obs_a','L_obs_u',
                                                     'obs_rel_l2_a','obs_rel_l2_u']}
        result_path = original_path(item, args.original_root)
        source = 'unchanged'
        receipt_path = folder/'receipt.json'
        eligible = not args.completed_pdes_only or pde in complete_pdes
        if required and eligible and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt['protocol_sha256'] == digest(args.inputs/pde/'protocol.json')
            assert receipt['sample_ids'] == [0] and receipt['stage'] == 'main'
            selection = json.loads((args.results/pde/'selection.json').read_text())
            intended = dict(config, **selection['task_updates'][config['task']])
            config = receipt['config'].copy()
            runtime = {'checkpoint_path','device','output_dir','batch_size','offset',
                       'save_plots','save_intermediate','save_per_sample_curves',
                       'ablation_name','allow_synthetic_data'}
            assert {k:v for k,v in config.items() if k not in runtime} == {k:v for k,v in intended.items() if k not in runtime}, ident
            candidates = list(folder.rglob('result.pt'))
            assert len(candidates) == 1, ident
            result_path = candidates[0]
            assert digest(result_path) == receipt['result_sha256'], ident
            payload = torch.load(result_path, map_location='cpu', weights_only=False)
            for field, prefix in [('a','coef'), ('u','sol')]:
                truth = payload[prefix+'_ground_truth'].double().numpy()
                prediction = payload[prefix+'_final'].double().numpy()
                error = np.linalg.norm(prediction-truth)/max(np.linalg.norm(truth), 1e-12)
                recorded = receipt['errors'][field][0]
                assert (np.isfinite(error) and recorded is not None and np.isclose(error, recorded, rtol=1e-11, atol=1e-12)) or (not np.isfinite(error) and recorded is None)
                values[field] = float(error)
            terminal = list(csv.DictReader((result_path.parent/'curves.csv').open()))[-1]
            assert np.isclose(float(terminal['t_next']), 1.0)
            for field in ['a', 'u']:
                assert np.isclose(float(terminal['rel_l2_'+field]), values[field], rtol=3e-6, atol=1e-9)
            diagnostics = {key:float(terminal[key]) for key in diagnostics}
            source = 'revised'
            revised += 1
        elif required:
            source = 'pending'
            pending.append(ident)
        record = dict(id=ident, pde=pde, source=source, required_rerun=required,
                      old_rel_l2_a=float(old['rel_l2_a']), old_rel_l2_u=float(old['rel_l2_u']),
                      rel_l2_a=values['a'], rel_l2_u=values['u'], config=config,
                      diagnostics=diagnostics,
                      result_path=str(result_path) if result_path.exists() else None,
                      source_result=item['source_result'], receipt=str(receipt_path) if source=='revised' else None)
        records.append(record)
        for field in (['u'] if pde=='burger' else ['a','u']):
            metrics.append(dict(id=ident,pde=pde,source=source,field=field,error_percent=100*values[field],
                                old_error_percent=100*float(old['rel_l2_'+field]),
                                **{k:config[k] for k in ['task','ablation_group','guidance_components','loss_state',
                                   'sampler_phase','switch_ratio','num_steps','time_grid','step_method','num_obs',
                                   'sensor_mode','noise_level','sample_seed','zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold']}))
    assert sum(r['required_rerun'] for r in records) == 744
    if pending and not args.allow_pending:
        raise RuntimeError(f'{len(pending)} of 744 main reruns still pending; no finished export written')
    (args.output/'records.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    write_csv(args.output/'field_errors.csv', metrics)
    (args.output/'manifest.json').write_text(json.dumps(dict(
        configurations=len(records),required_reruns=744,completed_reruns=revised,pending_ids=pending,
        included_revised_pdes=sorted({r['pde'] for r in records if r['source']=='revised'}),
        completed_pdes_only=args.completed_pdes_only,
        final_ready=not pending, source_archive_sha256=digest(args.archive),
        protocols={pde:digest(args.inputs/pde/'protocol.json') for pde in PDES},
        collector_sha256=digest(Path(__file__)),
        scope='Main-ablation physical input ID0; five inference/mask seeds only in stability group. '
              'Both fields are reported separately. This inventory excludes the old 32-ID repeated controls.'
    ),indent=2)+'\n')
    print('COLLECTED',len(records),'configurations;',revised,'revised;',len(pending),'pending',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--original-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-pending',action='store_true')
    parser.add_argument('--completed-pdes-only',action='store_true',
                        help='During a draft revision, replace a PDE only after its whole rerun is complete')
    collect(parser.parse_args())

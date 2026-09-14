"""Check all original elliptic full-observation physical inputs, CPU only.

The plan binds 18 cells/900 archived batches to nine unique source files.
The audit executes the saved historical loader extraction/casting functions
against every physical index 0..999 and compares both fields to actual saved
baseline targets. It does not sample or modify any prediction.
"""
from __future__ import annotations
import argparse
import ast
import csv
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_inputs import read, write, PDES, TASK_NAME
from input_sources import COUNT, GRID, file_sha256, load_cell, tensor_sha256

def full_source_plan(args):
    """Freeze 18 original full-observation cells and their nine source paths."""
    rows = list(csv.DictReader((args.source_data / 'main_hyperparameters_verified.csv').open()))
    full = [r for r in rows if r['pde'] in PDES[:3] and r['observations'] == 'full']
    assert len(full) == 18
    archive_path = args.source_data / 'main_configs_archive.json.gz'
    with gzip.open(archive_path, 'rt') as f:
        archive = json.load(f)['records']
    groups = {}
    for row in full:
        records = [r for r in archive if r['sheet'] == row['sheet'] and str(r['workbook_row']) == row['workbook_row']]
        assert records and all((r['config']['pde'] == row['pde'] for r in records))
        keys = ['pde', 'data_path', 'loadby', 'coef_name', 'solution_name', 'dtype']
        source_spec = {tuple((r['config'].get(k) for k in keys)) for r in records}
        assert len(source_spec) == 1
        offsets = sorted((i for r in records for i in range(r['config']['offset'], r['config']['offset'] + r['config']['batch_size'])))
        assert offsets == list(range(COUNT))
        cfg = {k: records[0]['config'].get(k) for k in keys}
        assert cfg['dtype'] == 'float32'
        key = (row['pde'], row['distribution'].lower(), cfg['data_path'])
        group = groups.setdefault(key, dict(pde=row['pde'], distribution=row['distribution'].lower(), config=cfg, cells=[]))
        group['cells'].append(dict(task=row['task'], sheet=row['sheet'], workbook_row=int(row['workbook_row']), batch_count=len(records), examples=COUNT, source_group=row['source_root'], representative_config_source=records[0]['source'], config_artifact_sha256s=[r['sha256'] for r in records]))
    assert len(groups) == 9 and all((len(g['cells']) == 2 for g in groups.values()))
    source = args.historical_loader.read_text()
    names = {'_load_raw_data', '_extract_single_sample', '_ensure_bchw', '_torch_dtype'}
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    loader = 'from __future__ import annotations\n' + ast.unparse(ast.Module(body=functions, type_ignores=[])) + '\n'
    value = dict(version=1, task_name=TASK_NAME, datasets=list(groups.values()), cells=18, intended_reused_predictions=18000, physical_indices=list(range(COUNT)), source_archive=str(archive_path), source_archive_sha256=file_sha256(archive_path), main_hyperparameters_sha256=file_sha256(args.source_data / 'main_hyperparameters_verified.csv'), historical_loader_source=str(args.historical_loader), historical_loader_sha256=file_sha256(args.historical_loader), historical_loader_functions=loader, historical_loader_functions_sha256=__import__('hashlib').sha256(loader.encode()).hexdigest(), scope='CPU physical-field comparison only; exact historical extraction/casting functions; no inference')
    if args.output.exists():
        assert read(args.output) == value, 'Cannot replace a different frozen full-source plan'
    else:
        write(args.output, value)
    print('FULL_SOURCE_PLAN_FROZEN', len(groups), 'sources', 18, 'cells', flush=True)

def audit_full_sources(args):
    plan = read(args.plan)
    ph = file_sha256(args.plan)
    assert plan['cells'] == 18 and len(plan['datasets']) == 9
    loader = plan['historical_loader_functions']
    assert __import__('hashlib').sha256(loader.encode()).hexdigest() == plan['historical_loader_functions_sha256']
    env = {}
    exec(compile(loader, '<saved historical FM loader>', 'exec'), env)
    reports = []
    dest = args.output.parent / 'full_reuse_source_checks'
    dest.mkdir(parents=True, exist_ok=True)
    for dataset in plan['datasets']:
        pde, dist = (dataset['pde'], dataset['distribution'])
        cfg = SimpleNamespace(**dataset['config'])
        if args.data_root:
            cfg.data_path = str(args.data_root / pde / Path(cfg.data_path).name)
        path = Path(cfg.data_path)
        stat = path.stat()
        ir = read(args.inputs / 'receipts' / f'supervised_{pde}_{dist}.json')
        assert ir['status'] == 'complete'
        cell = next((c for c in ir['cells'] if c['setting'] == 'sparse_joint'))
        target = dest / f'{pde}_{dist}.json'
        if target.exists():
            previous = read(target)
            assert previous['plan_sha256'] == ph and previous['baseline_truth_sha256'] == cell['truth_sha256']
            assert previous['source_size'] == stat.st_size and previous['source_mtime_ns'] == stat.st_mtime_ns
            reports.append(previous)
            print('REUSED_FULL_SOURCE_CHECK', pde, dist, previous['status'], flush=True)
            continue
        print('READING_FULL_SOURCE', pde, dist, cfg.data_path, flush=True)
        baseline = load_cell(args.inputs, cell)
        raw = env['_load_raw_data'](cfg)
        if cfg.loadby == 'h5py':
            handle = raw['__h5__']
            raw = {'__h5__': {name: handle[name][:, :, :COUNT] for name in [cfg.coef_name, cfg.solution_name]}}
            handle.close()
        counts = {'coef': 0, 'sol': 0, 'pair': 0}
        max_abs = {'coef': 0.0, 'sol': 0.0}
        row_hashes = []
        mismatches = []
        for i in range(COUNT):
            ar, ur = env['_extract_single_sample'](cfg, raw, i)
            a = env['_ensure_bchw'](ar, pde, 'coef', 'cpu', cfg.dtype, 1)
            u = env['_ensure_bchw'](ur, pde, 'sol', 'cpu', cfg.dtype, 1)
            ba, bu = (baseline['coef_ground_truth'][i:i + 1], baseline['sol_ground_truth'][i:i + 1])
            assert a.shape == u.shape == ba.shape == bu.shape == (1, 1, GRID, GRID)
            ea, eu = (torch.equal(a, ba), torch.equal(u, bu))
            counts['coef'] += int(ea)
            counts['sol'] += int(eu)
            counts['pair'] += int(ea and eu)
            max_abs['coef'] = max(max_abs['coef'], float((a - ba).abs().max()))
            max_abs['sol'] = max(max_abs['sol'], float((u - bu).abs().max()))
            if not (ea and eu):
                mismatches.append(i)
            row_hashes.append(dict(index=i, fm_source_coef_sha256=tensor_sha256(a), fm_source_sol_sha256=tensor_sha256(u), baseline_coef_sha256=tensor_sha256(ba), baseline_sol_sha256=tensor_sha256(bu), coef_equal=ea, sol_equal=eu))
        del raw, baseline
        rows_path = dest / f'{pde}_{dist}_per_input.json'
        write(rows_path, row_hashes)
        report = dict(status='pass' if counts['pair'] == COUNT else 'MISMATCH', pde=pde, distribution=dist, examples=COUNT, exactly_equal_fields=counts, max_absolute_difference=max_abs, mismatched_physical_indices=mismatches, source=str(path), source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns, baseline_truth_file=cell['truth_file'], baseline_truth_sha256=cell['truth_sha256'], plan_sha256=ph, original_full_cells=dataset['cells'], per_input_file=str(rows_path), per_input_sha256=file_sha256(rows_path), historical_loader_sha256=plan['historical_loader_sha256'], historical_loader_functions_sha256=plan['historical_loader_functions_sha256'])
        write(target, report)
        reports.append(report)
        print('FULL_SOURCE_CHECK', pde, dist, report['status'], counts, flush=True)
    result = dict(status='pass' if all((r['status'] == 'pass' for r in reports)) else 'MISMATCH', task_name=TASK_NAME, plan_sha256=ph, datasets=reports, original_full_cells=18, unique_physical_inputs_compared=9000, reusable_full_prediction_cells=sum((len(r['original_full_cells']) for r in reports if r['status'] == 'pass')), no_inference_performed=True, scope='Actual current source values using saved historical FM extraction functions versus all1000 actual baseline target tensors; does not reread every historical FM prediction')
    if args.output.exists():
        assert read(args.output) == result, 'Cannot replace a different complete full-source audit'
    else:
        write(args.output, result)
    print('FULL_REUSE_AUDIT', result['status'], result['reusable_full_prediction_cells'], 'of18cells', flush=True)

def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("plan")
    p.add_argument("--source-data",type=Path,required=True)
    p.add_argument("--historical-loader",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("run")
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--data-root",type=Path)
    args=parser.parse_args()
    {"plan":full_source_plan,"run":audit_full_sources}[args.mode](args)

if __name__=="__main__": main()

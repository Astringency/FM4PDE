"""CPU-only archive replay: rebuild read paths, recompute statistics, and compare.

Original raw files and publication snapshots are never written. The new view
contains relative symlinks; its target map is recorded separately from receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PDES = ['poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger',
        'reaction_diffusion', 'shallow_water', 'heat', 'wave',
        'advection_diffusion', 'steady_heat_conduction']
CONTRACTS = Path(__file__).parent / 'legacy_scripts/contracts'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path,
                        help='Existing local study layout; used for default paths only')
    parser.add_argument('--output', required=True, type=Path, help='Fresh verification directory')
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--result-map', type=Path,
                        help='JSON mapping each of 11 PDE names to its authoritative raw directory')
    parser.add_argument('--original-root', type=Path)
    parser.add_argument('--snapshot', type=Path, help='Existing baseline ablation snapshot')
    parser.add_argument('--ensemble', type=Path, help='Existing baseline three-draw export')
    parser.add_argument('--paper-figures', type=Path, help='Optional existing PNGs for exact pixel checks')
    parser.add_argument('--plots', choices=['all', 'none'], default='all')
    args = parser.parse_args()
    args.study = args.study.resolve()
    args.output = args.output.absolute()
    if args.output.exists():
        raise FileExistsError(f'Use a fresh directory: {args.output}')
    args.output.mkdir(parents=True)
    view = args.output/'view'
    historical_snapshot = args.snapshot or args.study/'ablation_publication_snapshot'
    historical_ensemble = args.ensemble or args.study/'ensemble_complete'
    inputs = args.inputs or args.study/'inputs'
    original = args.original_root or args.study/'original_predictions'
    result_map = (json.loads(args.result_map.read_text()) if args.result_map else
                  {pde: str(args.study/'output'/pde) for pde in PDES})
    assert set(result_map) == set(PDES), 'The results view must contain all eleven PDEs'
    mapping = []

    def link(source, destination):
        target = Path(source).resolve(strict=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        relative = os.path.relpath(target, destination.parent)
        destination.symlink_to(relative, target_is_directory=target.is_dir())
        assert destination.resolve(strict=True) == target
        mapping.append(dict(view_path=str(destination), relative_target=relative,
                            source_path=str(source), resolved_source=str(target)))

    for pde in PDES:
        link(inputs/pde, view/'inputs'/pde)
        link(result_map[pde], view/'results'/pde)
        assert (view/'results'/pde/'selection.json').is_file()
        assert (view/'results'/pde/'ensemble_complete.json').is_file()
    link(original, view/'original')
    link(args.study/'archived_ablation_summary.csv', view/'archived_ablation_summary.csv')
    write(args.output/'view_manifest.json', dict(
        no_raw_copies=True, links_are_relative=True, mappings=mapping,
        original_root_on_server197='/research_data/users/zhangxifeng/C01Python/FM4PDE',
        original_root_rule="append source_result suffix following '/FM4PDE/'"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
               MPLBACKEND='Agg')
    report = dict(status='running', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  repo_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  command=shlex.join([sys.executable, *sys.argv]), cwd=str(ROOT),
                  environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS',
                      'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','MPLBACKEND']},
                  original_inputs_preserved=True, plots_requested=args.plots, commands=[], checks={})
    report_path = args.output/'relocation_validation.json'
    write(report_path, report)

    def run(name, script, *arguments):
        command = [sys.executable, str(ROOT/'plot'/script), *map(str, arguments)]
        print('RUN', name, flush=True)
        started = time.monotonic()
        with (args.output/(name+'.log')).open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        entry = dict(name=name, command=shlex.join(command), script_sha256=digest(ROOT/'plot'/script),
                     seconds=round(time.monotonic()-started, 3), returncode=result.returncode,
                     peak_children_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                     log_sha256=digest(args.output/(name+'.log')))
        report['commands'].append(entry)
        write(report_path, report)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {args.output/(name+".log")}')
        print('DONE', name, entry['seconds'], 'seconds', flush=True)

    baseline = args.output/'baseline_snapshot'
    snapshot = args.output/'snapshot'
    # Hold numerical libraries and thread counts constant for the path-only test.
    # Historical exports may have used another BLAS reduction order.
    source_view = args.study/'output'
    if args.result_map:
        source_view = view/'results'
    run('baseline_collect', 'collect_paper_ablation_fields.py', '--archive', args.study/'archived_ablation_summary.csv',
        '--inputs', inputs, '--results', source_view, '--original-root', original,
        '--output', baseline, '--completed-pdes-only')
    run('collect', 'collect_paper_ablation_fields.py', '--archive', view/'archived_ablation_summary.csv',
        '--inputs', view/'inputs', '--results', view/'results', '--original-root', view/'original',
        '--output', snapshot, '--completed-pdes-only')
    old_records = json.loads((baseline/'records.json').read_text())
    new_records = json.loads((snapshot/'records.json').read_text())
    assert len(old_records) == len(new_records) == 1060
    ignored = {'result_path', 'receipt'}
    for old, new in zip(old_records, new_records):
        assert {k:v for k,v in old.items() if k not in ignored} == {k:v for k,v in new.items() if k not in ignored}, new['id']
        for key in ignored:
            if old[key] is None:
                assert new[key] is None, (new['id'], key, 'baseline lacks this raw path')
            else:
                assert Path(new[key]).is_relative_to(view), (new['id'], key)
                assert Path(new[key]).resolve(strict=True) == Path(old[key]).resolve(strict=True)
    assert digest(baseline/'field_errors.csv') == digest(snapshot/'field_errors.csv')
    historical_records = json.loads((historical_snapshot/'records.json').read_text())
    roundoff = []
    for old, new in zip(historical_records, new_records):
        for key in old.keys() | new.keys():
            if key in ignored or old[key] == new[key]:
                continue
            assert key in {'rel_l2_a', 'rel_l2_u'}, (new['id'], key)
            delta = abs(old[key]-new[key])
            relative = delta/max(abs(old[key]), 1e-300)
            assert delta <= 1e-12*max(abs(old[key]), abs(new[key]), 1e-12), (new['id'], key)
            roundoff.append(dict(id=new['id'], field=key, historical=old[key], recomputed=new[key],
                                 absolute_difference=delta, relative_difference=relative))
    manifest = json.loads((snapshot/'manifest.json').read_text())
    assert manifest['final_ready'] and manifest['completed_reruns'] == 744 and not manifest['pending_ids']
    report['checks']['ablation'] = dict(configurations=1060, revised=744, unchanged=316,
        all_nonpath_record_values_exactly_equal=True,
        field_errors_csv_byte_identical=True, field_errors_sha256=digest(snapshot/'field_errors.csv'),
        historical_snapshot_comparison=dict(changed_float_values=len(roundoff),
            max_absolute_difference=max((x['absolute_difference'] for x in roundoff), default=0),
            max_relative_difference=max((x['relative_difference'] for x in roundoff), default=0),
            cause='Different historical BLAS reduction order; path-only replay uses fixed two-thread libraries.'),
        new_read_paths_inside_view=True, available_raw_records=sum(x['result_path'] is not None for x in new_records),
        unchanged_missing_raw_by_pde=dict(Counter(x['pde'] for x in new_records if x['result_path'] is None)))
    write(args.output/'historical_ablation_roundoff.json', roundoff)
    for name, source in [('baseline_tables', baseline), ('relocated_tables', snapshot)]:
        run(name, 'export_paper_ablation_tables.py', '--source', source, '--output', args.output/name)
    old_tables = {p.name:digest(p) for p in (args.output/'baseline_tables').glob('*.tex')}
    new_tables = {p.name:digest(p) for p in (args.output/'relocated_tables').glob('*.tex')}
    assert old_tables == new_tables and len(new_tables) == 20
    run('historical_tables', 'export_paper_ablation_tables.py', '--source', historical_snapshot,
        '--output', args.output/'historical_tables')
    assert new_tables == {p.name:digest(p) for p in (args.output/'historical_tables').glob('*.tex')}
    report['checks']['tables'] = dict(count=20, all_tex_byte_identical=True, sha256=new_tables)
    old_ensemble = args.output/'baseline_ensemble'
    ensemble = args.output/'ensemble'
    run('baseline_ensemble_export', 'export_paper_seed_ensemble.py', '--inputs', inputs,
        '--results', source_view, '--output', old_ensemble, '--plan', CONTRACTS/'ENSEMBLE_ANALYSIS_PLAN.md')
    run('ensemble_export', 'export_paper_seed_ensemble.py', '--inputs', view/'inputs',
        '--results', view/'results', '--output', ensemble, '--plan', CONTRACTS/'ENSEMBLE_ANALYSIS_PLAN.md')
    csv_checks = {}
    for name in ['summary.csv', 'per_input.csv', 'frequency_per_input.csv']:
        assert digest(old_ensemble/name) == digest(ensemble/name), name
        csv_checks[name] = dict(rows=len(rows(ensemble/name)), byte_identical=True, sha256=digest(ensemble/name))
        old_rows, new_rows = rows(historical_ensemble/name), rows(ensemble/name)
        assert len(old_rows) == len(new_rows)
        differences = []
        for old, new in zip(old_rows, new_rows):
            assert old.keys() == new.keys()
            for key in old:
                if old[key] == new[key]:
                    continue
                a, b = float(old[key]), float(new[key])
                delta = abs(a-b)
                assert delta <= 1e-12*max(abs(a), abs(b))+1e-12, (name, key, a, b)
                differences.append((delta, delta/max(abs(a), 1e-300)))
        csv_checks[name]['historical_comparison'] = dict(changed_numeric_values=len(differences),
            max_absolute_difference=max((x[0] for x in differences), default=0),
            max_relative_difference=max((x[1] for x in differences), default=0))
    old_verified, new_verified = rows(old_ensemble/'verified_predictions.csv'), rows(ensemble/'verified_predictions.csv')
    assert len(old_verified) == len(new_verified) == 1056
    for old, new in zip(old_verified, new_verified):
        assert {k:v for k,v in old.items() if k!='receipt'} == {k:v for k,v in new.items() if k!='receipt'}
        assert Path(new['receipt']).is_relative_to(view)
        assert Path(new['receipt']).resolve(strict=True) == Path(old['receipt']).resolve(strict=True)
    import numpy as np
    with np.load(old_ensemble/'spectral_shells.npz') as old, np.load(ensemble/'spectral_shells.npz') as new:
        assert set(old.files) == set(new.files)
        array_checks = {key: bool(np.array_equal(old[key], new[key])) for key in old.files}
        assert all(array_checks.values())
    with np.load(historical_ensemble/'spectral_shells.npz') as old, np.load(ensemble/'spectral_shells.npz') as new:
        assert set(old.files) == set(new.files)
        historical_arrays = {}
        for key in old.files:
            assert np.allclose(old[key], new[key], rtol=1e-12, atol=1e-14), key
            historical_arrays[key] = dict(exactly_equal=bool(np.array_equal(old[key], new[key])),
                max_absolute_difference=float(np.max(np.abs(old[key]-new[key]))))
    report['checks']['three_draw'] = dict(pdes=11, predictions=1056, fields=21, csv=csv_checks,
        verified_predictions_exact_except_receipt_path=True, spectral_arrays_exactly_equal=array_checks,
        historical_spectral_arrays=historical_arrays)
    write(report_path, report)
    if args.plots == 'all':
        run('fields_plot', 'plot_paper_ablation_fields.py', '--source', snapshot,
            '--output', args.output/'field_figures', '--contract', CONTRACTS/'ABLATION_CHART_CONTRACT.md')
        run('sweeps_plot', 'plot_paper_ablation_sweeps.py', '--source', snapshot,
            '--output', args.output/'sweep_figures')
        run('ensemble_plot', 'plot_paper_seed_ensemble.py', '--source', ensemble,
            '--output', args.output/'ensemble_figures', '--contract', CONTRACTS/'ENSEMBLE_CHART_CONTRACT.md')
        from PIL import Image
        plots = []
        for folder in ['field_figures', 'sweep_figures', 'ensemble_figures']:
            for png in sorted((args.output/folder).glob('*.png')):
                pdf = png.with_suffix('.pdf')
                assert pdf.read_bytes().startswith(b'%PDF-')
                with Image.open(png) as img:
                    img.verify()
                item = dict(name=png.stem, folder=folder, png_sha256=digest(png), pdf_sha256=digest(pdf))
                if args.paper_figures:
                    old = args.paper_figures/png.name
                    item['existing_png_found'] = old.exists()
                    if old.exists():
                        with Image.open(old) as a, Image.open(png) as b:
                            item['pixels_exactly_equal'] = bool(np.array_equal(np.asarray(a), np.asarray(b)))
                plots.append(item)
        assert len(plots) == 80, len(plots)
        report['checks']['plots'] = dict(count=len(plots), valid_pdf_and_png=True, figures=plots)
    report['status'] = 'pass'
    report['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    report['scope_limit'] = ('All 744 revised and 1056 three-draw raw predictions were reread. '
        'The 316 unchanged controls retain exact archived summary values; 271 original raw records '
        'are absent from this local subset. This test does not certify their raw recomputation or server197 transfer.')
    write(report_path, report)
    print('PASS', report_path, flush=True)


if __name__ == '__main__':
    main()

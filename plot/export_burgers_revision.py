"""Recompute archived Burgers random observations and export the two-mode table."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from run_paper_ablation_revision import digest, write

DISTRIBUTIONS = ['id', 'smooth', 'rough']
METHODS = [('RecFNO', 0), ('Senseiver', 0), ('VoronoiCNN', 0),
           ('4D-Var', 0), ('VIVID', 0), ('FM4PDE', 100),
           ('DiffusionPDE', 100), ('DiffusionPDE', 1000)]


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def error(prediction, truth):
    assert prediction.shape == truth.shape
    assert torch.isfinite(prediction).all() and torch.isfinite(truth).all()
    return float(torch.linalg.vector_norm(prediction.double()-truth.double()) /
                 torch.linalg.vector_norm(truth.double()))


def freeze_fm(args):
    """Read all physical trajectories, rather than reuse rounded table means."""
    target = args.output
    target.mkdir(parents=True, exist_ok=True)
    assert not (target/'fm_random_manifest.json').exists(), 'Already exported'
    rows, run_sources, geometries = [], {}, {}
    for distribution in DISTRIBUTIONS:
        cell = torch.load(args.inputs/f'{distribution}_random.pt', map_location='cpu', weights_only=False)
        parent = args.archive/f'outputs/main/MAIN1000_100_TEST_{distribution}/burger/both'
        folders = list(parent.glob('*global_norm50_random500_noise0_uniform100_euler*'))
        assert len(folders) == 1, folders
        seen = set()
        matched_masks = 0
        for run in sorted(folders[0].glob('*/result.pt')):
            payload = torch.load(run, map_location='cpu', weights_only=False)
            masks = torch.load(run.parent/'masks.pt', map_location='cpu', weights_only=False)
            metrics = list(csv.DictReader((run.parent/'metrics_per_sample.csv').open()))
            metrics.sort(key=lambda r: int(r['sample_index']))
            assert len(metrics) == len(payload['sol_final'])
            assert torch.equal(masks['coef'], masks['sol']), (run, 'different trajectory masks')
            assert (masks['sol'].flatten(1).sum(1) == 500).all(), run
            sh = digest(run)
            run_sources[str(run)] = sh
            for j, row in enumerate(metrics):
                assert int(row['sample_index']) == j
                i = int(row['sample_id'])
                assert i not in seen and 0 <= i < 1000, (distribution, i)
                seen.add(i)
                truth = payload['sol_ground_truth'][j]
                assert torch.equal(truth, cell['truth'][i]), (distribution, i, 'truth mismatch')
                e = error(payload['sol_final'][j], truth)
                assert np.isclose(e, float(row['rel_l2_u']), rtol=3e-6, atol=1e-9)
                same_mask = torch.equal(masks['sol'][j], cell['mask'][i].to(masks['sol'].dtype))
                matched_masks += int(same_mask)
                rows.append(dict(distribution=distribution, mode='random', method='FM4PDE', steps=100,
                    sample_id=i, rel_l2_u=e, source=str(run), source_sha256=sh,
                    same_reference_masks=same_mask))
        assert seen == set(range(1000)), (distribution, len(seen))
        geometries[distribution] = dict(samples=1000, count=500, reference_masks_equal=matched_masks)
        print('EXPORTED FM', distribution, len(seen), 'mask matches', matched_masks, flush=True)
    rows.sort(key=lambda r: (r['distribution'], r['sample_id']))
    path = target/'fm_random_per_input.csv'
    write_csv(path, rows)
    write(target/'fm_random_manifest.json', dict(input_protocol_sha256=digest(args.inputs/'protocol.json'),
        per_input_sha256=digest(path), runs=run_sources, observations=geometries,
        exporter_sha256=digest(Path(__file__))))


def export(args):
    source = json.loads((args.inputs/'protocol.json').read_text())
    protocol = json.loads(args.sampling_protocol.read_text())
    ph = digest(args.sampling_protocol)
    assert protocol['input_protocol_sha256'] == digest(args.inputs/'protocol.json')
    assert digest(args.inputs/'baseline_per_input.csv') == source['baseline_metrics_sha256']
    frozen = json.loads((args.fm_random/'fm_random_manifest.json').read_text())
    assert frozen['input_protocol_sha256'] == digest(args.inputs/'protocol.json')
    assert frozen['per_input_sha256'] == digest(args.fm_random/'fm_random_per_input.csv')
    cells = {}
    for name, sha in source['artifacts'].items():
        assert digest(args.inputs/name) == sha
        cells[name[:-3]] = torch.load(args.inputs/name, map_location='cpu', weights_only=False)
    rows = []
    for path in [args.inputs/'baseline_per_input.csv', args.fm_random/'fm_random_per_input.csv']:
        for row in csv.DictReader(path.open()):
            row.update(sample_id=int(row['sample_id']), steps=int(row.get('steps', 0)),
                       rel_l2_u=float(row['rel_l2_u']))
            rows.append(row)
    pending = []
    for job in protocol['jobs']:
        relative = Path('results')/job['cell']/f'{job["method"]}_{job["steps"]}'/f'sample{job["sample_id"]}.json'
        paths = [root/relative for root in args.results if (root/relative).exists()]
        if not paths:
            pending.append(job)
            continue
        assert len(paths) == 1, ('duplicate result', paths)
        path = paths[0]
        receipt = json.loads(path.read_text())
        assert receipt['protocol_sha256'] == ph
        assert all(receipt[k] == v for k, v in job.items()), path
        tensor_path = path.with_suffix('.pt')
        assert digest(tensor_path) == receipt['tensor_sha256'], path
        payload = torch.load(tensor_path, map_location='cpu', weights_only=False)
        assert payload['job'] == job and payload['protocol_sha256'] == ph
        i = job['sample_id']; cell = cells[job['cell']]
        assert torch.equal(payload['truth'], cell['truth'][i:i+1]), path
        assert torch.equal(payload['mask'], cell['mask'][i:i+1].float()), path
        for field in ['truth', 'mask']:
            assert hashlib.sha256(payload[field].numpy().tobytes()).hexdigest() == receipt[field+'_sha256']
        e = error(payload['prediction'], payload['truth'])
        assert np.isclose(e, receipt['rel_l2_u'], rtol=1e-11, atol=1e-12), path
        assert receipt['nfe'] == (job['steps'] if job['method'] == 'FM4PDE' else 2*job['steps']-1)
        distribution, mode = job['cell'].split('_', 1)
        rows.append(dict(distribution=distribution, mode=mode, method=job['method'], steps=job['steps'],
            sample_id=i, rel_l2_u=e, source=str(tensor_path), source_sha256=receipt['tensor_sha256'],
            same_reference_masks=True))
    if pending and not args.development:
        raise RuntimeError(f'{len(pending)} Burgers predictions remain; no final table written')
    summary = []
    index = {}
    for mode in ['random', 'structured']:
        for distribution in DISTRIBUTIONS:
            for method, steps in METHODS:
                found = [r for r in rows if (r['distribution'], r['mode'], r['method'], r['steps']) ==
                         (distribution, mode, method, steps)]
                supported = method != 'DiffusionPDE' or distribution == 'smooth'
                ids = [r['sample_id'] for r in found]
                assert len(ids) == len(set(ids)) and set(ids) <= set(range(1000))
                assert supported or not found
                complete = len(ids) == 1000
                if not args.development: assert complete == supported
                vals = np.array([r['rel_l2_u']*100 for r in found])
                assert np.isfinite(vals).all()
                r = dict(distribution=distribution, mode=mode, method=method, steps=steps, n=len(found),
                    supported=supported, complete=complete,
                    mean_percent=float(vals.mean()) if complete else None,
                    sd_percent=float(vals.std(ddof=1)) if complete else None)
                summary.append(r); index[(mode, distribution, method, steps)] = r
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'per_input.csv', rows)
    write_csv(args.output/'summary.csv', summary)
    lines = [r'\begin{table}[!htbp]', r'\FMTableMark{start}{tab:burgers-results}',
        r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',
        r'\caption{Burgers trajectory reconstruction under two observation patterns. Each entry is the mean $\pm$ sample SD of full-trajectory relative $L^2$ error (\%) over 1,000 inputs. Random observations contain 500 time--space points; structured observations contain 640 values at five physical-time levels. DiffusionPDE is evaluated on Smooth.}',
        r'\label{tab:burgers-results}', r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',
        r'& \multicolumn{3}{c}{Random} & \multicolumn{3}{c}{Structured} \\',
        r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
        r'Method & ID & Smooth & Rough & ID & Smooth & Rough \\\midrule']
    for method, steps in METHODS:
        values = []
        for mode in ['random', 'structured']:
            for distribution in DISTRIBUTIONS:
                row = index[(mode, distribution, method, steps)]
                values.append('--' if not row['supported'] else r'\textit{Pending}' if not row['complete'] else
                    f"${row['mean_percent']:.2f}\\pm{row['sd_percent']:.2f}$")
        label = method + (f' ({steps})' if steps else '')
        lines.append(label+' & '+' & '.join(values)+r' \\')
    lines += [r'\bottomrule\end{tabular}', r'\FMTableMark{end}{tab:burgers-results}', r'\end{table}']
    (args.output/'burgers_two_modes.tex').write_text('\n'.join(lines)+'\n')
    write(args.output/'manifest.json', dict(final_ready=not pending, protocol_sha256=ph,
        expected_new_calls=len(protocol['jobs']), completed_new_calls=len(protocol['jobs'])-len(pending),
        pending_jobs=pending, rows=len(rows), exporter_sha256=digest(Path(__file__)),
        random_fm_observations=frozen['observations']))
    print('EXPORTED BURGERS', len(rows), 'input errors;', len(pending), 'pending', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze-fm', 'export'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--sampling-protocol', type=Path)
    parser.add_argument('--results', nargs='+', type=Path)
    parser.add_argument('--fm-random', type=Path)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == 'freeze-fm':
        assert args.archive is not None
        freeze_fm(args)
    else:
        assert args.sampling_protocol and args.results and args.fm_random
        export(args)

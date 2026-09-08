"""Independently score the four-input inverse step probe, retaining both budgets."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from run_ns_loss_study import sha, write
from report_ns_checkpoint_comparison import csv_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--diffusion-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    protocol = json.loads((args.results / 'protocol.json').read_text())
    ph = sha(args.results / 'protocol.json')
    fields = np.load(args.inputs / 'fields_masks.npz')
    assert sha(args.inputs / 'fields_masks.npz') == protocol['fields_sha256']
    variants = [v['name'] for v in protocol['variants']] + ['DiffusionPDE_100', 'DiffusionPDE_1000']
    rows, hashes = [], {str(args.results / 'protocol.json'): ph}
    for variant in variants:
        for i in protocol['evaluation_ids']:
            for seed in protocol['seeds']:
                is_dm = variant.startswith('DiffusionPDE')
                root = args.diffusion_results / 'results/inverse' / variant / 'original' if is_dm else args.results / 'evaluation/inverse' / variant
                path = root / f'sample{i}_seed{seed}.json'
                r = json.loads(path.read_text())
                assert (r['task'],r['sample_id'],r['seed']) == ('inverse',i,seed)
                if not is_dm:
                    assert r['protocol_sha256'] == ph
                    assert r['worker'] == protocol['evaluation_ids'].index(i) % 2
                hashes[str(path)] = sha(path)
                if r['status'] != 'complete':
                    rows.append(dict(variant=variant, sample_id=i, seed=seed, status=r['status'], error=r.get('error')))
                    continue
                pt = path.with_suffix('.pt')
                assert sha(pt) == r['prediction_sha256']
                hashes[str(pt)] = r['prediction_sha256']
                d = torch.load(pt, map_location='cpu', weights_only=False)
                errors = []
                for j, field in enumerate(['a', 'u']):
                    truth = torch.from_numpy(fields[f'{field}_{i}']).double()
                    pred = d['prediction'][j].double()
                    assert torch.equal(d['truth'][j].double(), truth)
                    mask = torch.from_numpy(fields[f'mask_u_{i}']).double() if field == 'u' else torch.zeros_like(truth)
                    assert torch.equal(d['masks'][j].double(), mask)
                    assert bool(torch.isfinite(pred).all())
                    error = float((pred-truth).norm()/truth.norm())
                    saved = r['relative_l2'][j] if is_dm else r[f'rel_l2_{field}']
                    assert abs(error-saved) < 1e-12
                    errors.append(error)
                steps = r['steps']
                assert r['nfe'] == (2*steps-1 if is_dm else steps)
                rows.append(dict(variant=variant, sample_id=i, seed=seed, status='complete',
                                 steps=steps, nfe=r['nfe'], rel_l2_a=errors[0], rel_l2_u=errors[1]))
    assert len(rows) == 96
    csv_write(args.output / 'per_call.csv', rows)
    summary, per_input = [], []
    for variant in variants:
        rr = [r for r in rows if r['variant'] == variant]
        complete = all(r['status'] == 'complete' for r in rr)
        if not complete:
            summary.append(dict(variant=variant, finite_calls=sum(r['status']=='complete' for r in rr), mean_pct=None, sd_pct=None))
            continue
        values = []
        for i in protocol['evaluation_ids']:
            value = np.mean([r['rel_l2_a'] for r in rr if r['sample_id'] == i]) * 100
            per_input.append(dict(variant=variant,sample_id=i,seed_mean_a_pct=value))
            values.append(value)
        summary.append(dict(variant=variant, inputs=4, finite_calls=12, mean_pct=np.mean(values), sd_pct=np.std(values,ddof=1)))
    csv_write(args.output / 'per_input.csv', per_input)
    csv_write(args.output / 'summary.csv', summary)
    write(args.output / 'audit_manifest.json', dict(status='complete', source_hashes=hashes,
          new_calls=72, new_finite_calls=sum(r['status']=='complete' for r in rows[:72]),
          diffusion_reference_calls=24, input_ids=protocol['evaluation_ids'], seeds=protocol['seeds'],
          scope=protocol['scope'], source_script_sha256=sha(Path(__file__)),
          checks='All 96 identities, saved tensor hashes, matched truth and masks, independent float64 errors and actual NFE verified.'))
    lines = ['# Exploratory inverse step-budget probe', '', 'Four inputs fixed by the prior frozen ordering; three seeds per input. Mean ± input sample SD (%). This diagnostic does not establish a 32- or 1000-input ranking.', '', '| Configuration | Relative L2(a), % |', '|---|---:|']
    for r in summary:
        value = f"{r['mean_pct']:.2f} ± {r['sd_pct']:.2f}" if r['mean_pct'] is not None else f"{r['finite_calls']}/12 finite"
        lines.append(f"| {r['variant']} | {value} |")
    lines += ['', 'Both budgets and every specified checkpoint are retained. FM 1000 uses 1000 model evaluations; DiffusionPDE 1000 uses 1999. These are sampling-step interventions, with no additional training. Diffusion references are identical-input archived results, with a separate execution environment.']
    (args.output / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

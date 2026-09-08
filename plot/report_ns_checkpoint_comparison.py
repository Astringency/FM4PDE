"""Audit saved checkpoint comparisons and export input-level tables and figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_ns_loss_study import sha, write

LABELS = {'current_common100': 'FM current / common guidance / 100',
          'v260904_common100': 'FM 260904 / common guidance / 100',
          'bak_common100': 'FM bak / common guidance / 100',
          'bak_legacy100': 'FM bak / legacy guidance / 100',
          'DiffusionPDE_100': 'DiffusionPDE / native / 100',
          'DiffusionPDE_1000': 'DiffusionPDE / native / 1000'}


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def font_setup():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    for name in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']:
        path = Path('/mnt/c/Windows/Fonts') / name
        if path.exists():
            font_manager.fontManager.addfont(path)
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    matplotlib.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
                               'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
                               'axes.spines.right': False})


def save(fig, stem):
    import matplotlib.pyplot as plt
    fig.canvas.draw()
    fig.canvas.draw()
    fig.set_layout_engine('none')
    for ext in ['pdf', 'png']:
        fig.savefig(stem.with_suffix('.' + ext), dpi=170, bbox_inches='tight')
    plt.close(fig)


def training(args, protocol):
    import matplotlib.pyplot as plt
    histories, summary = {}, []
    for label in ['current', 'v260904', 'bak']:
        path = args.results / 'evidence' / label / 'nsnonbounded_log.txt'
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]
        histories[label] = rows
        for metric in ['train_loss', 'val_loss', 'lr']:
            rr = [r for r in rows if metric in r]
            if not rr:
                continue
            summary.append(dict(checkpoint=label, metric=metric, rows=len(rr), first_epoch=rr[0]['epoch'],
                                last_epoch=rr[-1]['epoch'], first=rr[0][metric], last=rr[-1][metric],
                                last20_mean=np.mean([r[metric] for r in rr[-20:]]),
                                preceding20_mean=np.mean([r[metric] for r in rr[-40:-20]]),
                                min=min(r[metric] for r in rr), max=max(r[metric] for r in rr),
                                source_sha256=sha(path)))
    csv_write(args.output / 'training_summary.csv', summary)
    csv_write(args.output / 'training_history.csv', [dict(checkpoint=k, **r) for k, rr in histories.items() for r in rr])
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7), layout='constrained')
    names = {'current': 'Current (387.5M parameters)', 'v260904': '260904 (44.1M parameters)', 'bak': 'Backup (98.8M parameters)'}
    for j, label in enumerate(histories):
        rr = histories[label]
        for metric, color, style in [('train_loss', '#2878A5', '-'), ('val_loss', '#C87932', '--')]:
            r = [r for r in rr if metric in r]
            if r:
                axes[0, j].plot([x['epoch'] for x in r], [x[metric] for x in r], color=color,
                                linestyle=style, linewidth=1., label=metric.replace('_', ' '))
        axes[0, j].set(title=names[label], xlabel='Recorded epoch', ylabel='Flow matching loss (log scale)', yscale='log')
        axes[0, j].legend(frameon=False, fontsize=9)
        axes[0, j].grid(axis='y', color='.9', linewidth=.5)
        if label != 'bak':
            axes[1, j].plot([x['epoch'] for x in rr], [x['lr'] for x in rr], color='#2878A5', linewidth=1.)
            axes[1, j].set(xlabel='Recorded epoch', ylabel='Learning rate (log scale)', yscale='log')
        else:
            axes[1, j].axis('off')
            axes[1, j].text(0, .9, 'Backup: train loss only.\nNo saved validation curve or\ntraining normalization statistics.\n\nLoss magnitudes across different\nnormalizations are not comparable.\n\nCurrent log begins at epoch 150\nafter a resumed training run.', va='top', linespacing=1.7, fontsize=11)
    fig.suptitle('NS checkpoint training histories\nRecorded losses and learning rates; separate training protocols', fontsize=14)
    save(fig, args.output / 'ns_training_histories')


def audit(args, protocol):
    import torch
    torch.set_num_threads(2)
    source = json.loads((args.inputs / 'source.json').read_text())
    assert sha(args.inputs / 'source.json') == protocol['source_sha256']
    assert sha(args.inputs / 'fields_masks.npz') == protocol['fields_sha256']
    fields = np.load(args.inputs / 'fields_masks.npz')
    ph = sha(args.results / 'protocol.json')
    rows, hashes, paths = [], {}, {}
    variants = [v['name'] for v in protocol['variants']]
    expected = [(t, v, i, s) for t in protocol['tasks'] for v in variants
                for i in protocol['evaluation_ids'] for s in protocol['seeds']]
    for task, variant, i, seed in expected:
        path = args.results / 'evaluation' / task / variant / f'sample{i}_seed{seed}.json'
        assert path.exists(), f'Missing expected call: {path}'
        r = json.loads(path.read_text())
        assert r['protocol_sha256'] in {ph, protocol.get('upstream_protocol_sha256')}
        assert (r['task'], r['variant'], r['sample_id'], r['seed']) == (task, variant, i, seed)
        assert r['worker'] == protocol['evaluation_ids'].index(i) % protocol['workers']
        hashes[str(path)] = sha(path)
        if r['status'] != 'complete':
            rows.append(r)
            continue
        pt = path.with_suffix('.pt')
        assert sha(pt) == r['prediction_sha256']
        hashes[str(pt)] = r['prediction_sha256']
        data = torch.load(pt, map_location='cpu', weights_only=False)
        assert r['nfe'] == 100
        errors = []
        for j, field in enumerate(['a', 'u']):
            truth = torch.from_numpy(fields[f'{field}_{i}'])
            mask = torch.from_numpy(fields[f'mask_{"a" if task == "both" else field}_{i}'])
            if (task, field) in [('forward', 'u'), ('inverse', 'a')]:
                mask = torch.zeros_like(mask)
            assert torch.equal(data['truth'][j], truth)
            assert torch.equal(data['masks'][j], mask)
            pred = data['prediction'][j].double()
            assert pred.shape == truth.shape and bool(torch.isfinite(pred).all())
            error = float((pred - truth.double()).norm() / truth.double().norm())
            assert abs(error - r[f'rel_l2_{field}']) < 1e-12
            errors.append(error)
        primary = errors[1] if task == 'forward' else errors[0] if task == 'inverse' else max(errors)
        assert abs(primary - r['primary_error']) < 1e-12
        rows.append(r)
        paths[task, variant, i, seed] = pt
    assert len(rows) == protocol['formal_calls']
    new_count = len(rows)
    for task in protocol['tasks']:
        for variant in ['DiffusionPDE_100', 'DiffusionPDE_1000']:
            for i in protocol['evaluation_ids']:
                for seed in protocol['seeds']:
                    path = args.diffusion_results / 'results' / task / variant / 'original' / f'sample{i}_seed{seed}.json'
                    r = json.loads(path.read_text())
                    assert r['status'] == 'complete' and not r['exchange']
                    pt = path.with_suffix('.pt')
                    assert sha(pt) == r['prediction_sha256']
                    hashes[str(path)] = sha(path)
                    hashes[str(pt)] = r['prediction_sha256']
                    data = torch.load(pt, map_location='cpu', weights_only=False)
                    errors = []
                    for j, field in enumerate(['a', 'u']):
                        truth = torch.from_numpy(fields[f'{field}_{i}']).double()
                        pred = data['prediction'][j].double()
                        truth_key = 'truths' if 'truths' in data else 'truth'
                        assert torch.equal(data[truth_key][j].double(), truth)
                        mask = torch.from_numpy(fields[f'mask_{"a" if task == "both" else field}_{i}']).double()
                        if (task, field) in [('forward', 'u'), ('inverse', 'a')]:
                            mask = torch.zeros_like(mask)
                        assert torch.equal(data['masks'][j].double(), mask)
                        errors.append(float((pred - truth).norm() / truth.norm()))
                    r.update(variant=variant, rel_l2_a=errors[0], rel_l2_u=errors[1],
                             primary_error=errors[1] if task == 'forward' else errors[0] if task == 'inverse' else max(errors))
                    rows.append(r)
                    paths[task, variant, i, seed] = pt
    csv_write(args.output / 'per_call.csv', rows)
    per_input = []
    for task in protocol['tasks']:
        for variant in LABELS:
            for i in protocol['evaluation_ids']:
                rr = [r for r in rows if r['task'] == task and r['variant'] == variant and r['sample_id'] == i]
                assert len(rr) == 3 and sorted(r['seed'] for r in rr) == protocol['seeds']
                finite = sum(r['status'] == 'complete' for r in rr)
                item = dict(task=task, variant=variant, sample_id=i, calls=3, finite=finite)
                for metric in ['primary_error', 'rel_l2_a', 'rel_l2_u']:
                    item[metric] = float(np.mean([r[metric] for r in rr])) if finite == 3 else None
                per_input.append(item)
    csv_write(args.output / 'per_input.csv', per_input)
    summaries = []
    for task in protocol['tasks']:
        for variant in LABELS:
            rr = [r for r in per_input if r['task'] == task and r['variant'] == variant]
            n = sum(r['finite'] == 3 for r in rr)
            row = dict(task=task, variant=variant, expected_inputs=32, complete_inputs=n,
                       expected_calls=96, finite_calls=sum(r['finite'] for r in rr))
            for metric in ['primary_error', 'rel_l2_a', 'rel_l2_u']:
                values = np.array([r[metric] for r in rr], dtype=float)
                row[metric + '_mean_pct'] = float(values.mean() * 100) if n == 32 else None
                row[metric + '_sd_pct'] = float(values.std(ddof=1) * 100) if n == 32 else None
            summaries.append(row)
    csv_write(args.output / 'summary.csv', summaries)
    paired = []
    for task in protocol['tasks']:
        for variant in variants:
            for control in ['current_common100', 'DiffusionPDE_100', 'DiffusionPDE_1000']:
                if variant == control:
                    continue
                aa = sorted([r for r in per_input if r['task'] == task and r['variant'] == variant], key=lambda r:r['sample_id'])
                bb = sorted([r for r in per_input if r['task'] == task and r['variant'] == control], key=lambda r:r['sample_id'])
                if not all(r['finite'] == 3 for r in aa + bb):
                    continue
                delta = np.array([a['primary_error'] - b['primary_error'] for a,b in zip(aa,bb)]) * 100
                rng = np.random.default_rng(20260912)
                boot = delta[rng.integers(0, 32, (4000, 32))].mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                paired.append(dict(task=task, variant=variant, control=control, n_inputs=32,
                                   mean_delta_pp=delta.mean(), ci95_low_pp=lo, ci95_high_pp=hi,
                                   input_wins=int((delta < 0).sum()), bootstrap_seed=20260912,
                                   resamples=4000, interval='pointwise paired input percentile bootstrap'))
    csv_write(args.output / 'paired_effects.csv', paired)
    write(args.output / 'audit_manifest.json', dict(status='complete', protocol_sha256=ph,
          new_calls=new_count, new_finite_calls=sum(r['status'] == 'complete' for r in rows[:new_count]),
          diffusion_reference_calls=len(rows)-new_count, per_input_rows=len(per_input),
          summary_rows=len(summaries), paired_effect_rows=len(paired), source_hashes=hashes,
          verified='All planned identities, checkpoint pairing by input/GPU, tensor hashes, truth arrays, 500-point task masks, finite outputs, NFE and independent float64 errors.',
          runtime_caveat='Diffusion outcomes reused from prior multi-GPU accuracy study; runtimes are not a controlled latency comparison.'))
    return summaries, paired, paths


def figures(args, protocol, paths):
    import torch
    import matplotlib.pyplot as plt
    fields = np.load(args.inputs / 'fields_masks.npz')
    i, seed = protocol['evaluation_ids'][0], 0
    variants = ['current_common100', 'v260904_common100', 'bak_legacy100', 'DiffusionPDE_1000']
    short = ['FM current\n100 steps', 'FM 260904\n100 steps',
             'FM bak, legacy\n100 steps', 'DiffusionPDE\n1000 steps']
    for task, field, j in [('forward', 'u', 1), ('inverse', 'a', 0), ('both', 'a', 0), ('both', 'u', 1)]:
        truth = fields[f'{field}_{i}'].squeeze()
        predictions = []
        for variant in variants:
            path = paths.get((task, variant, i, seed))
            predictions.append(torch.load(path, map_location='cpu', weights_only=False)['prediction'][j].numpy().squeeze() if path else None)
        vmax = max(float(np.abs(x).max()) for x in [truth] + predictions if x is not None)
        emax = max(float(np.abs(x - truth).max()) for x in predictions if x is not None)
        fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), layout='constrained')
        im = axes[0, 0].imshow(truth, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        axes[0, 0].set_title('Reference')
        observed = (task, field) not in [('forward', 'u'), ('inverse', 'a')]
        mask = fields[f'mask_{"a" if task == "both" else field}_{i}'].squeeze() if observed else np.zeros_like(truth)
        axes[1, 0].imshow(mask, cmap='Greys', vmin=0, vmax=1, origin='lower')
        axes[1, 0].set_title(f'Observed {field}: {int(mask.sum())} points')
        for k, pred in enumerate(predictions, 1):
            axes[0, k].set_title(short[k-1])
            if pred is None:
                axes[0, k].text(.5,.5,'Nonfinite outcome',ha='center')
                continue
            axes[0, k].imshow(pred, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
            error = np.abs(pred - truth)
            errim = axes[1, k].imshow(error, cmap='magma', vmin=0, vmax=emax, origin='lower')
            rel = np.linalg.norm((pred.astype(float) - truth.astype(float)).ravel()) / np.linalg.norm(truth.astype(float).ravel())
            axes[1, k].set_title(f'Relative L2: {100 * rel:.2f}%')
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[0,:], shrink=.7, pad=.01, label='Field value (shared scale)')
        fig.colorbar(errim, ax=axes[1,1:], shrink=.7, pad=.01, label='Absolute error (shared scale)')
        fig.suptitle(f'NS {task}: {field}, input {i}, seed {seed}\nPrespecified example; identical truths and task observations', fontsize=14)
        save(fig, args.output / f'ns_fields_{task}_{field}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--diffusion-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.results / 'protocol.json').read_text())
    font_setup()
    training(args, protocol)
    if args.training_only:
        return
    summaries, paired, paths = audit(args, protocol)
    figures(args, protocol, paths)
    lines = ['# NS checkpoint comparison', '', '32 matched Smooth inputs × 3 seeds × 3 tasks. Each table cell averages seed-level errors within each input, then reports mean ± sample SD (%) over 32 inputs. These are supplementary results, not the 1000-input main experiment.', '', '| Model / configuration / steps | Forward u | Inverse a | Joint max(a,u) |', '|---|---:|---:|---:|']
    for variant, label in LABELS.items():
        cells = []
        for task in protocol['tasks']:
            r = next(r for r in summaries if r['task'] == task and r['variant'] == variant)
            fmt = '.2e' if (r['primary_error_mean_pct'] or 0) > 10000 else '.2f'
            cells.append(f"{r['primary_error_mean_pct']:{fmt}} ± {r['primary_error_sd_pct']:{fmt}}" if r['complete_inputs'] == 32 else f"Incomplete finite outcomes: {r['finite_calls']}/96")
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'All specified checkpoints and failed/nonfinite outcomes are retained. The common-guidance comparison changes the checkpoint, saved architecture and normalizer. Backup legacy guidance is a separate configuration. No evaluation-score tuning occurred. Current and 260904 differ in architecture, numerical training precision, resume history and learning-rate trajectory; this is not a training-duration ablation.', '', 'DiffusionPDE references use the identical input fields, masks and nominal seeds, but were computed earlier across several GPUs. FM 100 steps uses 100 network evaluations; DiffusionPDE 100 and 1000 steps use 199 and 1999 respectively. Runtime values must not be pooled as a controlled speed benchmark.', '', 'Paired differences and pointwise 95% bootstrap intervals are in `paired_effects.csv`; all raw tensor and source hashes are in `audit_manifest.json`.']
    (args.output / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]))


if __name__ == '__main__':
    main()

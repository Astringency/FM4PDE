"""Complete-only NS guidance calibration tables and publication figures."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from audit_ns_guidance_calibration import csvwrite
from export_ns_loss_spectra import summarize, spectral_shape_metrics
from publication_style import use_times_new_roman
from run_ns_loss_study import sha, write
from spectral_diagnostics import paired_bootstrap

TASKS = ['forward', 'inverse', 'both']
BASELINES = ['RecFNO', 'Senseiver', 'VoronoiCNN']
METRICS = ['primary_error', 'rel_l2_a', 'rel_l2_u', 'secant_rms', 'spatial_loss',
           'obs_rel_l2_a', 'obs_rel_l2_u']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.audit / 'calibration_audit_manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['selection_independently_verified']
    assert sum(manifest['outcomes']['calibration'].values()) == 540
    assert sum(manifest['outcomes']['evaluation'].values()) == 576
    for name, h in manifest['outputs'].items():
        assert sha(args.audit / name) == h, name
    read = lambda name: list(csv.DictReader((args.audit / name).open()))
    calls, grid = read('calibration_per_call.csv'), read('calibration_grid.csv')
    ensembles = read('calibration_mean3.csv') if (args.audit / 'calibration_mean3.csv').exists() else []
    baselines = read('calibration_baselines.csv')
    evaluation = [r for r in calls if r['stage'] == 'evaluation']
    ids = sorted({int(r['sample_id']) for r in evaluation})
    assert len(evaluation) == 576 and len(ids) == 32 and len(grid) == 45
    groups = defaultdict(list)
    for row in evaluation:
        groups[row['task'], row['variant'], int(row['sample_id'])].append(row)
    values, outcomes = {}, []
    for task in TASKS:
        for variant in ['reference', 'selected']:
            rr = [r for r in evaluation if r['task'] == task and r['variant'] == variant]
            assert len(rr) == 96
            good = [i for i in ids if len(groups[task, variant, i]) == 3
                    and all(r['status'] == 'complete' for r in groups[task, variant, i])]
            outcomes.append(dict(task=task, variant=variant, total_calls=96,
                                 finite_calls=sum(r['status'] == 'complete' for r in rr),
                                 nonfinite_calls=sum(r['status'] == 'nonfinite' for r in rr),
                                 unstable_calls=sum(r['status'] == 'unstable' for r in rr),
                                 complete_inputs=len(good), total_inputs=32))
            for metric in METRICS:
                per = {}
                for i in good:
                    rows = groups[task, variant, i]
                    assert {int(r['seed']) for r in rows} == {0, 1, 2}
                    if all(r[metric] for r in rows):
                        per[i] = float(np.mean([float(r[metric]) for r in rows]))
                values[task, variant, 'single', metric] = per
                er = [r for r in ensembles if r['task'] == task and r['variant'] == variant and r[metric]]
                values[task, variant, 'mean3', metric] = {int(r['sample_id']): float(r[metric]) for r in er}
        for method in BASELINES:
            for field in ['a', 'u']:
                values[task, method, 'deterministic', 'rel_l2_' + field] = {
                    int(r['sample_id']): float(r['rel_l2']) for r in baselines
                    if r['task'] == task and r['method'] == method and r['field'] == field}
            aa = values[task, method, 'deterministic', 'rel_l2_a']
            uu = values[task, method, 'deterministic', 'rel_l2_u']
            values[task, method, 'deterministic', 'primary_error'] = (
                uu if task == 'forward' else aa if task == 'inverse' else {i: max(aa[i], uu[i]) for i in ids})
    summary, effects = [], []
    for (task, variant, estimator, metric), per in values.items():
        summary.append(dict(task=task, variant=variant, estimator=estimator, metric=metric,
                            **summarize(list(per.values()))))
    for task in TASKS:
        for estimator in ['single', 'mean3']:
            for metric in METRICS:
                selected = values[task, 'selected', estimator, metric]
                comparisons = [('reference', estimator)]
                if metric in ['primary_error', 'rel_l2_a', 'rel_l2_u']:
                    comparisons += [(m, 'deterministic') for m in BASELINES]
                if estimator == 'mean3':
                    comparisons.append(('selected', 'single'))
                for method, other_estimator in comparisons:
                    other = values.get((task, method, other_estimator, metric), {})
                    common = sorted(selected.keys() & other.keys())
                    if common:
                        effects.append(dict(task=task, variant='selected', estimator=estimator,
                                            comparator=method, comparator_estimator=other_estimator, metric=metric,
                                            **summarize([selected[i] - other[i] for i in common])))
    args.output.mkdir(parents=True, exist_ok=True)
    csvwrite(args.output / 'guidance_evaluation_summary.csv', summary)
    csvwrite(args.output / 'guidance_paired_effects.csv', effects)
    csvwrite(args.output / 'guidance_outcomes.csv', outcomes)
    csvwrite(args.output / 'guidance_grid.csv', grid)
    def stat(task, variant, estimator, metric='primary_error'):
        return next(r for r in summary if (r['task'], r['variant'], r['estimator'], r['metric']) ==
                    (task, variant, estimator, metric))
    def number(r):
        if r['mean'] is None:
            return '---'
        return f"${100*r['mean']:.2f}\\pm{100*r['sd']:.2f}$" if r['sd'] is not None else f"${100*r['mean']:.2f}$"
    tex = [r'\begin{table}[!htbp]\centering\footnotesize',
           r'\caption{Exploratory NS guidance calibration at 100 steps. Primary relative error (\%, mean $\pm$ SD across 32 input-level seed averages) is $e_u$ for forward, $e_a$ for inverse, and $\max(e_a,e_u)$ per call for joint. Selection uses four disjoint development inputs from the same test source, not the 32 evaluation targets. Mean3 averages three predicted fields before scoring and costs three sampling calls. Failed calls are retained in Table~\ref{tab:ns-calibration-outcomes}; error statistics use inputs with all three finite calls.}',
           r'\label{tab:ns-calibration}', r'\begin{tabular}{@{}lrrr@{}}\toprule',
           r'Method / estimator & Forward & Inverse & Joint \\\midrule']
    entries = [('FM original / single', 'reference', 'single'), ('FM selected / single', 'selected', 'single'),
               ('FM original / mean3', 'reference', 'mean3'), ('FM selected / mean3', 'selected', 'mean3')]
    entries += [(m, m, 'deterministic') for m in BASELINES]
    for label, variant, estimator in entries:
        tex.append(label + ' & ' + ' & '.join(number(stat(t, variant, estimator)) for t in TASKS) + r' \\')
    tex += [r'\bottomrule\end{tabular}\end{table}',
            r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
            r'\caption{Field-specific errors for the same NS guidance evaluation (\%, mean $\pm$ SD across complete inputs). The two joint fields are scored separately here, so changes in the larger-error selection score cannot conceal a tradeoff between fields. Single averages three individual seed errors within each input; mean3 scores the average of three predicted fields.}',
            r'\label{tab:ns-calibration-fields}', r'\begin{tabular}{@{}lrrrr@{}}\toprule',
            r'Method / estimator & Forward $e_u$ & Inverse $e_a$ & Joint $e_a$ & Joint $e_u$ \\\midrule']
    for label, variant, estimator in entries:
        cells = [number(stat(task, variant, estimator, 'rel_l2_' + field))
                 for task, field in [('forward', 'u'), ('inverse', 'a'), ('both', 'a'), ('both', 'u')]]
        tex.append(label + ' & ' + ' & '.join(cells) + r' \\')
    tex += [r'\bottomrule\end{tabular}\end{table}',
            r'\begin{table}[!htbp]\centering\footnotesize',
            r'\caption{All NS calibration-evaluation outcomes. Finite counts include arbitrarily large finite errors; no error threshold is applied. Complete inputs have three finite inference seeds. The 540 grid calls and their candidate-level outcomes are reported in the accompanying source data.}',
            r'\label{tab:ns-calibration-outcomes}', r'\begin{tabular}{@{}llrrrr@{}}\toprule',
            r'Task & Setting & Finite / 96 & Nonfinite & Unstable & Inputs / 32 \\\midrule']
    for r in outcomes:
        tex.append(' & '.join([('Joint' if r['task'] == 'both' else r['task'].capitalize()), r['variant'],
                               str(r['finite_calls']), str(r['nonfinite_calls']), str(r['unstable_calls']),
                               str(r['complete_inputs'])]) + r' \\')
    tex += [r'\bottomrule\end{tabular}\end{table}']
    tex += [r'\begin{table}[!htbp]\centering\footnotesize',
            r'\caption{Guidance weights selected using the four reserved development inputs. Multipliers apply to the recipient FM weights in Appendix~\ref{app:ns-loss-exchange}. Calibration errors are percentages, mean $\pm$ SD over four input-level seed averages. Selection minimizes the primary error and disqualifies any candidate with a failed call. These development scores are separate from the 32-input evaluation in Table~\ref{tab:ns-calibration}.}',
            r'\label{tab:ns-calibration-selection}', r'\begin{tabular}{@{}lrrrr@{}}\toprule',
            r'Task & Obs. multiplier & PDE multiplier & Original error & Selected error \\\midrule']
    for task in TASKS:
        chosen = manifest['selected'][task]
        rr = [next(r for r in grid if r['task'] == task and r['name'] == name)
              for name in ['obs1_pde1', chosen['name']]]
        displays = [number(dict(mean=float(r['mean_primary']), sd=float(r['sd_input_primary'])))
                    if r['mean_primary'] else 'Ineligible' for r in rr]
        tex.append(' & '.join(['Joint' if task == 'both' else task.capitalize(),
                               f"{chosen['observation_multiplier']:g}", f"{chosen['pde_multiplier']:g}", *displays]) + r' \\')
    tex += [r'\bottomrule\end{tabular}\end{table}']
    (args.output / 'guidance_calibration_tables.tex').write_text('\n'.join(tex) + '\n')

    font = use_times_new_roman()
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        for ext in ['pdf', 'png']:
            fig.savefig(args.output / (name + '.' + ext), dpi=190, bbox_inches='tight')
        plt.close(fig)
    # Retain the entire prospective grid. Log colors show all finite scales;
    # an ineligible cell explicitly displays its failure count.
    finite_grid = [100 * float(r['mean_primary']) for r in grid if r['mean_primary']]
    assert finite_grid and min(finite_grid) > 0
    norm = LogNorm(vmin=min(finite_grid), vmax=max(finite_grid))
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), layout='constrained')
    for ax, task in zip(axes, TASKS):
        values_grid = np.full((5, 3), np.nan)
        task_rows = [r for r in grid if r['task'] == task]
        for j, obs in enumerate([.1, .3, 1., 3., 10.]):
            for k, pde in enumerate([0., 1., 1000.]):
                r = next(r for r in task_rows if float(r['observation_multiplier']) == obs and float(r['pde_multiplier']) == pde)
                selected = r['name'] == manifest['selected'][task]['name']
                if r['mean_primary']:
                    v = 100 * float(r['mean_primary'])
                    values_grid[j, k] = v
                    label = f'{v:.1f}' if v < 1000 else f'{v:.1e}'
                    color = 'white' if norm(v) < .55 else '#111111'
                else:
                    label, color = f"{12-int(r['finite_calls'])}/12 failed", '#111111'
                if selected:
                    label += '\nselected'
                ax.text(k, j, label, ha='center', va='center', fontsize=10, color=color)
        im = ax.imshow(values_grid, cmap='viridis', norm=norm, aspect='auto')
        ax.set(xticks=[0, 1, 2], xticklabels=['0', '1', '1000'], yticks=range(5),
               yticklabels=['0.1', '0.3', '1', '3', '10'], xlabel='PDE-weight multiplier',
               title='Joint' if task == 'both' else task.capitalize())
        if ax is axes[0]:
            ax.set_ylabel('Observation-weight multiplier')
    fig.colorbar(im, ax=axes, label='Calibration primary error (%) · log color scale', shrink=.85)
    fig.suptitle('Frozen guidance grid · four development inputs, three seeds\nAll 15 candidates per task; original weights at (1, 1)', fontsize=12)
    save(fig, 'ns_guidance_calibration_grid')

    # Paired differences give the intervention a direct interpretation. The
    # bootstrap resamples physical inputs, keeping all seeds paired within each.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), layout='constrained')
    comparisons = [('reference', 'single', 'Original FM'), ('RecFNO', 'deterministic', 'RecFNO'),
                   ('VoronoiCNN', 'deterministic', 'VoronoiCNN'), ('Senseiver', 'deterministic', 'Senseiver')]
    for ax, task in zip(axes, TASKS):
        for j, (method, estimator, label) in enumerate(comparisons):
            rr = [r for r in effects if (r['task'], r['estimator'], r['comparator'], r['comparator_estimator'], r['metric']) ==
                  (task, 'single', method, estimator, 'primary_error')]
            if not rr:
                ax.text(0, j, 'No complete pairs', fontsize=9)
                continue
            r = rr[0]
            mean = 100 * r['mean']
            err = np.array([[100 * (r['mean'] - r['ci_low'])], [100 * (r['ci_high'] - r['mean'])]]) if r['ci_low'] is not None else None
            ax.errorbar(mean, j, xerr=err, fmt='o', color='#256493', capsize=3, markersize=5)
        ax.axvline(0, color='.45', lw=.8)
        ax.set(yticks=range(4), yticklabels=[c[2] for c in comparisons],
               xlabel='Selected FM − comparator (pp)', title='Joint' if task == 'both' else task.capitalize())
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Paired primary-error difference · 32 evaluation inputs\nPoints: input means; bars: 95% paired-bootstrap CI; negative favors selected FM', fontsize=12)
    save(fig, 'ns_guidance_paired_effects')

    with gzip.open(args.audit / 'calibration_spectra.json.gz', 'rt') as f:
        frequency = json.load(f)
    band_rows, band_groups = [], defaultdict(list)
    for r in frequency:
        meta = {k: r[k] for k in ['task', 'variant', 'estimator', 'field', 'sample_id', 'seed']}
        for cutoffs, bands in [('8/32', r['bands'])] + list(r['sensitivity_bands'].items()):
            for band, raw_metrics in bands.items():
                metrics = spectral_shape_metrics(raw_metrics, r['reference_total'])
                band_rows.append(dict(**meta, cutoffs=cutoffs, band=band, **metrics))
                for metric in ['relative_error', 'predicted_reference_energy_ratio', 'global_error_contribution',
                               'reference_fraction', 'absolute_energy_ratio_mismatch', 'global_normalized_error',
                               'coefficient_alignment']:
                    if metrics[metric] is not None:
                        band_groups[r['task'], r['variant'], r['estimator'], r['field'], cutoffs, band, metric, r['sample_id']].append(metrics[metric])
    band_values = defaultdict(dict)
    for key, vv in band_groups.items():
        expected_seeds = 3 if key[2] == 'single' else 1
        if len(vv) == expected_seeds:
            band_values[key[:-1]][key[-1]] = float(np.mean(vv))
    band_summary = [dict(zip(['task', 'variant', 'estimator', 'field', 'cutoffs', 'band', 'metric'], key),
                         **summarize(list(vv.values()))) for key, vv in band_values.items()]
    band_effects = []
    for (task, variant, estimator, field, cutoffs, band, metric), selected in band_values.items():
        if variant != 'selected':
            continue
        comparisons = [('reference', estimator)] + [(m, 'deterministic') for m in BASELINES]
        if estimator == 'mean3':
            comparisons.append(('selected', 'single'))
        for comparator, other_estimator in comparisons:
            other = band_values.get((task, comparator, other_estimator, field, cutoffs, band, metric), {})
            common = sorted(selected.keys() & other.keys())
            if common:
                band_effects.append(dict(task=task, variant=variant, estimator=estimator, field=field,
                                         cutoffs=cutoffs, band=band, metric=metric, comparator=comparator,
                                         comparator_estimator=other_estimator,
                                         **summarize([selected[i] - other[i] for i in common])))
    csvwrite(args.output / 'guidance_frequency_per_call.csv', band_rows)
    csvwrite(args.output / 'guidance_frequency_summary.csv', band_summary)
    csvwrite(args.output / 'guidance_frequency_paired_effects.csv', band_effects)
    settings = [('reference', 'single', 'Original FM (100)'), ('selected', 'single', 'Selected FM (100)'),
                ('selected', 'mean3', 'Selected FM (3 × 100)')] + [(m, 'deterministic', m) for m in BASELINES]
    colors = ['#256493', '#aa4b32', '#c3943b', '#5a8055', '#805f95', '#697c85']
    spectrum_counts = {}
    for task, field in [('forward', 'u'), ('inverse', 'a'), ('both', 'a'), ('both', 'u')]:
        per = {}
        for variant, estimator, label in settings:
            grouped = defaultdict(list)
            for r in frequency:
                if (r['task'], r['field'], r['variant'], r['estimator']) == (task, field, variant, estimator):
                    grouped[r['sample_id']].append(r)
            per[variant, estimator] = {i: rr for i, rr in grouped.items() if len(rr) == (3 if estimator == 'single' else 1)}
        common = sorted(set.intersection(*(set(v) for v in per.values())))
        spectrum_counts[f'{task}_{field}'] = len(common)
        if not common:
            continue  # Outcomes table still explicitly reports every failed call.
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout='constrained')
        reference = None
        for j, (variant, estimator, label) in enumerate(settings):
            rr = per[variant, estimator]
            curves = {k: np.stack([np.mean([np.asarray(r[k]) / r['reference_total'] for r in rr[i]], axis=0)
                                    for i in common]) for k in ['reference_power', 'prediction_power', 'error_power']}
            if reference is None:
                reference = curves['reference_power']
            else:
                assert np.allclose(reference, curves['reference_power'], rtol=1e-12, atol=1e-14)
            radius = np.arange(reference.shape[1])
            for ax, k in zip(axes, ['prediction_power', 'error_power']):
                ax.plot(radius[1:], curves[k].mean(0)[1:], color=colors[j], lw=1.4,
                        ls=['-', '--', '-.', ':', '--', ':'][j], label=label)
            if len(common) > 1:
                ci = paired_bootstrap(curves['error_power'])
                axes[1].fill_between(radius[1:], ci[0, 1:], ci[1, 1:], color=colors[j], alpha=.1, lw=0)
        axes[0].plot(radius[1:], reference.mean(0)[1:], color='#222222', lw=1.7, label='Ground truth')
        for ax, title, ylabel in zip(axes, ['Reference and prediction energy', 'Reconstruction error energy'],
                                     ['Shell energy / full reference energy', 'Shell error energy / full reference energy']):
            ax.set(xscale='log', yscale='log', xlabel='Radial Fourier-mode index', ylabel=ylabel, title=title)
            ax.grid(alpha=.2)
            for cut in [8, 32]:
                ax.axvline(cut, color='.6', lw=.6)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=9, frameon=False)
        label = 'joint' if task == 'both' else task
        field_label = 'initial vorticity' if field == 'a' else 'final vorticity'
        fig.suptitle(f'NS {label} · {field_label} · held-out guidance evaluation\n'
                     f'{len(common)} common inputs with finite seeds; shading: 95% input-bootstrap CI', fontsize=12)
        save(fig, f'ns_guidance_spectrum_{task}_{field}')
    write(args.output / 'guidance_report_manifest.json', dict(
        status='complete', audit_manifest_sha256=sha(args.audit / 'calibration_audit_manifest.json'),
        protocol_sha256=manifest['protocol_sha256'], selection_sha256=manifest['selection_sha256'],
        selected=manifest['selected'], font_path=font, script_sha256=sha(Path(__file__)),
        report_dependency_sha256={name: sha(Path(__file__).with_name(name)) for name in
                                  ['export_ns_loss_spectra.py', 'spectral_diagnostics.py', 'publication_style.py']},
        call_seconds={stage: float(sum(float(r['seconds']) for r in calls if r['stage'] == stage))
                      for stage in ['calibration', 'evaluation']},
        timing_scope='Sum of per-call measured seconds on two A100 workers; development cost, not a controlled cross-method timing benchmark.',
        numerical_outcomes=manifest['outcomes'], visual_review_pending=True,
        spectrum_common_input_counts=spectrum_counts,
        outputs={p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'guidance_report_manifest.json'}))


if __name__ == '__main__':
    main()

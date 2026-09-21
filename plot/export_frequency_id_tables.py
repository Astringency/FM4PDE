"""Manuscript tables and figures for the FM4PDE-only frequency diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'plot'))
from run_ns_loss_study import sha, write

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_style import use_times_new_roman

from analyze_frequency_id_tasks import SELECTION, TASKS, OBSERVED, TARGET

COLORS = {'forward': '#27638b', 'inverse': '#bf922f', 'both': '#282828'}
MARKERS = {'forward': 'o', 'inverse': 's', 'both': '^'}
BAND_ORDER = ['dc', 'low8', 'mid1', 'mid2', 'hi1', 'hi2', 'hi3', 'mid8', 'high8']
BAND_LABEL = {'dc': 'DC', 'low8': '(0, 8]', 'mid1': '(8, 16]', 'mid2': '(16, 32]',
              'hi1': '(32, 64]', 'hi2': '(64, 96]', 'hi3': '(96, $\\infty$)', 'mid8': '(8, 32]',
              'high8': '$(32, \\infty)$'}
FIELD_LABEL = {'coef': '$\\mathbf a$', 'sol': '$\\mathbf u$'}
FIGURE_FIELD = {'coef': 'a', 'sol': 'u'}
PDE_LABEL = {'poisson': 'Poisson', 'darcy': 'Darcy'}
CLUSTER_LABEL = {'poisson': '$\\boldsymbol k$', 'darcy': '$\\boldsymbol k$'}


def read_rows(path):
    return list(csv.DictReader(Path(path).open()))


def number(row, key, digits=3, factor=1.0):
    value = row.get(key)
    if value in (None, '', 'None'):
        return '---'
    return f'{factor * float(value):.{digits}f}'


def select_summary(rows, pde, task, field, collection=None):
    collection = collection or SELECTION[pde, task]
    for row in rows:
        if (row['pde'], row['task'], row['field'], row['band'], row['collection']) \
                == (pde, task, field, 'high8', collection):
            return row
    raise KeyError((pde, task, field, collection))


def band_table(rows, output, high_only=True):
    lines = [r'\begin{table}[!htbp]', r'\FMTableStyle', r'\smallskip']
    if high_only:
        lines += [r'\caption{High-frequency content and alignment of the FM4PDE coefficient field '
                  r'$\mathbf a$ on the ID split, 1000 inputs per task, 100 sampling steps. '
                  r'$q_H$ is the mean ratio of predicted to reference energy in '
                  r'$H=\{\boldsymbol k:\|\boldsymbol k\|_2>32\}$, $A_H$ the mean coefficient alignment, '
                  r'and $\operatorname{RelL2}_H$ the relative error within $H$. '
                  r'Chance $A_H$ is the mean $\pm$ sample SD of alignments obtained by pairing each '
                  r'prediction with the reference of another input, and $(A_H-\text{chance})/\text{SD}$ '
                  r'expresses the observed alignment in units of that SD. '
                  r'$\dagger$ marks the task in which $\mathbf a$ carries the observations. '
                  r'The reference field carries $0.11\%$ (Poisson) and $0.87\%$ (Darcy) of its energy in '
                  r'$H$.}']
    else:
        lines += [r'\caption{Radial cosine-mode decomposition of FM4PDE reconstruction on the ID split, '
                  r'1000 inputs per task and field. Bands are open on the left and closed on the right; '
                  r'the zero mode is excluded. Ref.\ share is the fraction of reference field energy in '
                  r'the band.}']
    lines += [r'\label{tab:frequency-bands' + ('' if high_only else '-detail') + '}',
              r'\begin{tabular}{@{}lllrrrrrr@{}}' if high_only else r'\begin{tabular}{@{}llllrrrr@{}}',
              r'\toprule']
    if high_only:
        lines.append(r'PDE & Task & Field & $q_H$ & $A_H$ & $A_H^2$ & Chance $A_H$ & '
                     r'$(A_H-\text{chance})/\text{SD}$ & $\operatorname{RelL2}_H$ \\')
    else:
        lines.append(r'PDE & Task & Field & Band & Ref.\ share & $q_B$ & $A_B$ & $\operatorname{RelL2}_B$ \\')
    lines.append(r'\midrule')
    for pde in ('poisson', 'darcy'):
        body = []
        if high_only:
            for task in TASKS:
                row = select_summary(rows, pde, task, 'coef')
                mark = ' $\\dagger$' if 'coef' in OBSERVED[task] else ''
                chance, spread = float(row['chance_alignment_mean']), float(row['chance_alignment_sd'])
                deviation = (float(row['alignment_mean']) - chance) / spread
                body.append(' & '.join([
                    task.capitalize(), FIELD_LABEL['coef'] + mark,
                    f"${number(row, 'energy_ratio_mean')}$", f"${number(row, 'alignment_mean')}$",
                    f"${number(row, 'matched_fraction_mean')}$",
                    f"${chance:.3f}\\pm{spread:.3f}$", f"${deviation:.1f}$",
                    f"${number(row, 'band_relative_error_mean')}$"]) + r' \\')
        else:
            for task in TASKS:
                for field in ('coef', 'sol'):
                    mark = ' $\\dagger$' if field in OBSERVED[task] else ''
                    for band in ['low8', 'mid1', 'mid2', 'hi1', 'hi2', 'hi3']:
                        match = [r for r in rows if (r['pde'], r['task'], r['field'], r['band'], r['collection'])
                                 == (pde, task, field, band, SELECTION[pde, task])]
                        assert len(match) == 1, (pde, task, field, band)
                        row = match[0]
                        if row.get('energy_ratio_mean') in (None, ''):
                            continue
                        body.append(' & '.join([
                            task.capitalize(), FIELD_LABEL[field] + mark, BAND_LABEL[band],
                            f"${number(row, 'reference_fraction_mean', 4)}$",
                            f"${number(row, 'energy_ratio_mean')}$", f"${number(row, 'alignment_mean')}$",
                            f"${number(row, 'band_relative_error_mean')}$"]) + r' \\')
        lines.append(r' & '.join([r'\multirow{' + str(len(body)) + r'}{*}{' + PDE_LABEL[pde] + r'}',
                                  body[0].replace(r' \\', '')]) + r' \\')
        lines.extend(body[1:])
        lines.append(r'\midrule' if pde == 'poisson' else r'\bottomrule')
    lines += [r'\end{tabular}', r'\end{table}']
    (output / ('frequency_id_band_table.tex' if high_only else 'frequency_id_band_detail.tex')) \
        .write_text('\n'.join(lines) + '\n')


def radial_figure(rows, pde, output, stem):
    use_times_new_roman()
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': .5})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), layout='constrained')
    for task in TASKS:
        field = 'coef'
        series = sorted([r for r in rows if (r['pde'], r['task'], r['field'], r['collection'])
                         == (pde, task, field, SELECTION[pde, task])], key=lambda r: int(r['radius']))
        radius = np.array([int(r['radius']) for r in series])
        reference = np.array([float(r['reference_power']) for r in series])
        prediction = np.array([float(r['prediction_power']) for r in series])
        error = np.array([float(r['error_power']) for r in series])
        alignment = np.array([float(r['pooled_alignment']) if r['pooled_alignment'] not in ('', 'None')
                              else np.nan for r in series])
        keep = radius > 0
        style = dict(color=COLORS[task], lw=1.0, marker=MARKERS[task], markersize=2.4, markevery=3,
                     label=task.capitalize())
        axes[0].plot(radius[keep], prediction[keep], **style)
        axes[1].plot(radius[keep], error[keep], **style)
        axes[2].plot(radius[keep], alignment[keep], **style)
        if task == 'forward':
            axes[0].plot(radius[keep], reference[keep], color='#999999', lw=1.0, ls=':', label='Reference')
    axes[0].set(yscale='log', xscale='log', title=f'{PDE_LABEL[pde]} coefficient a',
                xlabel='Radial cosine-mode index', ylabel='Shell energy')
    axes[1].set(yscale='log', xscale='log', title='High-band error energy', xlabel='Radial cosine-mode index',
                ylabel='Shell error energy')
    axes[2].set(xscale='log', title='Coefficient alignment', xlabel='Radial cosine-mode index',
                ylabel='$A(r)$')
    axes[2].axhline(0.0, color='#999999', lw=.5)
    axes[2].axhline(0.5, color='#999999', lw=.5, ls=':')
    for axis in axes:
        axis.axvline(8, color='#bbbbbb', lw=.5)
        axis.axvline(32, color='#bbbbbb', lw=.5)
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=7, frameon=False)
    for extension in ('pdf', 'png'):
        fig.savefig(output / f'{stem}.{extension}', dpi=210, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


def chance_figure(rows, output, stem):
    use_times_new_roman()
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': .5})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), layout='constrained')
    bands = [(0.0, 8.0, 'low8'), (8.0, 16.0, 'mid1'), (16.0, 32.0, 'mid2'), (32.0, 64.0, 'hi1'),
             (64.0, 96.0, 'hi2'), (96.0, 181.0, 'hi3')]
    for axis, pde in zip(axes, ('poisson', 'darcy')):
        centers = np.array([np.sqrt(max(low, 1.0) * high) for low, high, _ in bands])
        for task in TASKS:
            row_lookup = {r['band']: r for r in rows if (r['pde'], r['task'], r['field'], r['collection'])
                          == (pde, task, 'coef', SELECTION[pde, task])}
            values = np.array([float(row_lookup[band]['alignment_mean']) for _, _, band in bands])
            chance = np.array([float(row_lookup[band]['chance_alignment_mean']) for _, _, band in bands])
            spread = np.array([float(row_lookup[band]['chance_alignment_sd']) for _, _, band in bands])
            axis.plot(centers, values, color=COLORS[task], lw=1.1, marker=MARKERS[task], markersize=3,
                      label=task.capitalize())
            axis.fill_between(centers, chance - 2 * spread, chance + 2 * spread, color='#dddddd', zorder=0)
            axis.plot(centers, chance, color='#888888', lw=.6, ls='--')
        axis.set(xscale='log', ylim=(-0.25, 1.05), xlabel='Band centre radius',
                 ylabel='$A_B$', title=PDE_LABEL[pde])
        axis.axhline(0.0, color='#999999', lw=.5)
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=7, frameon=False, title='Shaded: chance $\\pm2$ SD', title_fontsize=6)
    for extension in ('pdf', 'png'):
        figure.savefig(output / f'{stem}.{extension}', dpi=210, bbox_inches='tight', pad_inches=.03)
    plt.close(figure)


def ensemble_figure(rows, output, stem):
    use_times_new_roman()
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': .5})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), layout='constrained')
    ks = sorted({int(r['K']) for r in rows})
    for task in TASKS:
        for field, style in (('coef', '-'), ('sol', '--')):
            series = {int(r['K']): r for r in rows if (r['task'], r['field'], r['band'])
                      == (task, field, 'high8')}
            if not series:
                continue
            ratio, alignment = [], []
            for k in ks:
                row = series[k]
                tr, pr, er = (float(row['reference_energy']), float(row['prediction_energy']),
                              float(row['error_energy']))
                ratio.append(pr / tr)
                alignment.append((pr + tr - er) / (2.0 * np.sqrt(pr * tr)) if pr > 0 else np.nan)
            label = f"{task.capitalize()} · {FIGURE_FIELD[field]}"
            axes[0].plot(ks, ratio, color=COLORS[task], ls=style, lw=1.0, marker=MARKERS[task],
                         markersize=3, label=label)
            axes[1].plot(ks, alignment, color=COLORS[task], ls=style, lw=1.0, marker=MARKERS[task],
                         markersize=3)
    axes[0].set(xscale='log', xlabel='Averaged draws $K$', ylabel='$q_H(K)$', title='Retained high-band energy')
    axes[1].set(xscale='log', xlabel='Averaged draws $K$', ylabel='$A_H(K)$', title='Alignment with the reference')
    axes[1].axhline(0.0, color='#999999', lw=.5)
    for axis in axes:
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=6.5, frameon=False)
    for extension in ('pdf', 'png'):
        figure.savefig(output / f'{stem}.{extension}', dpi=210, bbox_inches='tight', pad_inches=.03)
    plt.close(figure)


def ensemble_table(rows, output):
    summary = read_rows(output / 'ensemble_band_summary.csv')
    lines = [r'\begin{table}[!htbp]', r'\FMTableStyle', r'\smallskip',
             r'\caption{Decomposition of the retained high-frequency energy of FM4PDE under repeated '
             r'sampling. Each of the 32 Poisson ID inputs is reconstructed with 1000 independent draws under '
             r'fixed observations; $\widehat{\mathbf c}_H$ is the ensemble mean of the high-band coefficients. '
             r'$q_H$ is the mean energy ratio of individual draws, $q_H^{\det}$ and $q_H^{\mathrm{sto}}$ its '
             r'deterministic and stochastic parts, $A_H^{\det}$ the alignment of the ensemble mean, and '
             r'chance $A_H^{\det}$ the alignment of a shuffled pairing.}',
             r'\label{tab:frequency-ensemble}',
             r'\begin{tabular}{@{}llrrrrrr@{}}', r'\toprule',
             r'Task & Field & $q_H$ & $q_H^{\det}$ & $q_H^{\mathrm{sto}}$ & $A_H^{\det}$ & Chance & $\operatorname{RelL2}_H^{\det}$ \\',
             r'\midrule']
    for task in TASKS:
        for field in ('coef', 'sol'):
            match = [r for r in summary if (r['task'], r['field'], r['band']) == (task, field, 'high8')]
            assert len(match) == 1, (task, field)
            row = match[0]
            values = {key: float(row[key]) if row[key] not in (None, '') else None for key in
                      ('draw_energy_ratio_mean', 'deterministic_energy_ratio', 'stochastic_energy_ratio',
                       'deterministic_alignment', 'chance_alignment_mean', 'chance_alignment_sd')}
            rel = None
            if values['deterministic_energy_ratio'] is not None and values['deterministic_alignment'] is not None:
                rel = np.sqrt(1.0 + values['deterministic_energy_ratio']
                              - 2.0 * np.sqrt(values['deterministic_energy_ratio'])
                              * values['deterministic_alignment'])
            cells = [task.capitalize(),
                     FIELD_LABEL[field] + (' $\\dagger$' if field in OBSERVED[task] else ''),
                     f"${values['draw_energy_ratio_mean']:.3f}$",
                     f"${values['deterministic_energy_ratio']:.3f}$",
                     f"${values['stochastic_energy_ratio']:.3f}$",
                     f"${values['deterministic_alignment']:.3f}$",
                     f"${values['chance_alignment_mean']:.3f}\\pm{values['chance_alignment_sd']:.3f}$",
                     f"${rel:.3f}$"]
            lines.append(' & '.join(cells) + r' \\')
        lines.append(r'\addlinespace' if task != TASKS[-1] else r'\bottomrule')
    lines += [r'\end{tabular}', r'\end{table}']
    (output / 'frequency_id_ensemble_table.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--ensemble', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows = read_rows(args.main / 'main_band_summary.csv')
    radial_rows = read_rows(args.main / 'main_radial.csv')
    band_table(summary_rows, args.output, high_only=True)
    band_table(summary_rows, args.output, high_only=False)
    ensemble_rows = read_rows(args.ensemble / 'ensemble_per_input.csv')
    ensemble_table(ensemble_rows, args.ensemble)
    for pde in ('poisson', 'darcy'):
        radial_figure(radial_rows, pde, args.output, f'frequency_id_radial_{pde}')
    chance_figure(summary_rows, args.output, 'frequency_id_alignment')
    ensemble_figure(ensemble_rows, args.output, 'frequency_id_ensemble')
    manifest = dict(script_sha256=sha(Path(__file__)),
                    sources={name: sha(args.main / name) for name in
                             ('main_band_summary.csv', 'main_radial.csv')},
                    ensemble_source=sha(args.ensemble / 'ensemble_per_input.csv'),
                    outputs={path.name: sha(path) for path in sorted(args.output.iterdir())
                             if path.suffix in ('.tex', '.pdf', '.png')})
    if args.summary:
        write(args.summary, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

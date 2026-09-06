"""Freeze a complete, tensor-audited four-PDE report for the manuscript.

This exports all predeclared variants, without selecting results or changing
the manuscript's interpretation. Partial-PDE and pilot reports are refused.
Run summarize_revision_sampling.py and audit_revision_predictions.py first.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil


PDES = ['poisson', 'darcy', 'nsnonbounded', 'burger']
LABELS = {'poisson': 'Poisson', 'darcy': 'Darcy',
          'nsnonbounded': 'Navier--Stokes', 'burger': 'Burgers'}
VARIANTS = {
    'guidance_noguide': 'No guide',
    'guidance_pde_only': 'PDE only',
    'guidance_obs_only': 'Observation only',
    'guidance_obs_pde': 'Observation + PDE',
    'steps_25_fixed': '25 steps, fixed',
    'steps_25_normalized': '25 steps, normalized',
    'steps_50_fixed': '50 steps, fixed',
    'steps_50_normalized': '50 steps, normalized',
    'steps_200_fixed': '200 steps, fixed',
    'steps_200_normalized': '200 steps, normalized',
    'phase_deterministic': 'Deterministic',
    'phase_hybrid_d2s': r'D$\to$S (0.2)',
    'phase_hybrid_s2d': r'S$\to$D (0.2)',
}
FIGURES = ['controlled_time_accuracy', 'sampling_confirmation_guidance',
           'sampling_confirmation_phase', 'sampling_confirmation_trajectories']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.report/'sampling_confirmation_manifest.json').read_text())
    audit = json.loads((args.report/'prediction_tensor_audit.json').read_text())
    assert manifest['stage'] == 'evaluation', 'Pilot reports cannot enter the paper.'
    protocol = manifest['protocol']
    assert protocol['physical_examples'] == 32 and protocol['inference_seeds'] == [0, 1, 2]
    assert {v['name'] for v in protocol['variants']} == set(VARIANTS)
    assert {r['pde'] for r in manifest['validation']} == set(PDES), 'All four PDEs are required.'
    assert {r['pde'] for r in audit['reports']} == set(PDES), 'All four tensor audits are required.'
    for result in manifest['validation'] + audit['reports']:
        assert result['verified_batches'] == 312 and result['verified_example_runs'] == 1248
    for result in audit['reports']:
        reference = next(r for r in manifest['validation'] if r['pde'] == result['pde'])
        assert result['protocol_sha256'] == reference['protocol_sha256']
        assert result['prediction_batches_checked'] == 312
        assert result['field_errors_recomputed'] == 2496
        assert result['observation_errors_recomputed'] == 2496

    rows = list(csv.DictReader((args.report/'sampling_confirmation_summary.csv').open()))
    keys = [(r['pde'], r['variant']) for r in rows]
    assert len(keys) == len(set(keys)) == 52
    assert set(keys) == {(pde, variant) for pde in PDES for variant in VARIANTS}
    for row in rows:
        assert row['stage'] == 'evaluation' and int(row['n_examples']) == 32
        assert int(row['inference_seeds']) == 3

    # Check that the audited tensors and receipt exports describe the same runs.
    recomputed = list(csv.DictReader((args.report/'prediction_errors_recomputed.csv').open()))
    archived = list(csv.DictReader((args.report/'sampling_confirmation_per_example.csv').open()))
    actual = {(r['pde'], r['variant'], int(r['sample_id']), int(r['seed'])): r
              for r in recomputed}
    assert len(actual) == len(recomputed) == len(archived) == 4992
    for row in archived:
        key = (row['pde'], row['variant'], int(row['sample_id']), int(row['inference_seed']))
        other = actual[key]
        for field in ['a', 'u']:
            for prefix in ['', 'obs_']:
                metric = f'{prefix}rel_l2_{field}'
                x, y = float(row[metric]), float(other[f'{metric}_f64'])
                assert abs(x-y) <= 2e-8 + 1e-6*abs(y), (key, metric, x, y)

    source_dir, figure_dir = args.paper/'source_data', args.paper/'figures'
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    names = [f'sampling_confirmation_{suffix}.csv'
             for suffix in ['summary', 'pairs', 'per_example', 'trajectories', 'budgets']]
    names += ['sampling_confirmation_manifest.json', 'sampling_confirmation_receipts.json.gz',
              'prediction_errors_recomputed.csv', 'prediction_tensor_audit.json']
    for name in names:
        destination = source_dir/name
        shutil.copy2(args.report/name, destination)
        copied[str(destination.relative_to(args.paper))] = sha256(destination)
    for name in FIGURES:
        for extension in ['pdf', 'png']:
            destination = figure_dir/f'{name}.{extension}'
            shutil.copy2(args.report/destination.name, destination)
            copied[str(destination.relative_to(args.paper))] = sha256(destination)

    lines = [r'\section{Complete Results for the Repeated Sampling Study}',
             r'\label{app:sampling-confirmation-results}',
             r'All 13 predeclared variants are reported for each PDE. The protocol is',
             r'given in Appendix~\ref{app:sampling-confirmation-protocol}. Field errors',
             r'and the primary score are percentages. The primary score is computed',
             r'on each run before averaging seeds within physical example; brackets',
             r'give a 95\% bootstrap interval over 32 physical examples. $R$ is the',
             r'mean per-example physical-residual RMS, not a normalized cross-PDE score.',
             r'$T$ is median instrumented sampling time per example in seconds at batch',
             r'size four; its IQR and both observation errors are retained in the CSV',
             r'exports. NFE counts model forward calls. No rank marks are assigned.', '']
    for pde in PDES:
        lines += [r'\begin{table}[!htbp]', r'\centering\scriptsize',
                  r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.13}',
                  rf'\caption{{Repeated sampling results for {LABELS[pde]}.}}',
                  rf'\label{{tab:sampling-confirmation-{pde}}}',
                  r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',
                  r'Variant & NFE & $e_a$ & $e_u$ & Primary [95\% CI] & $R$ & $T$ \\\midrule']
        for name, label in VARIANTS.items():
            row = next(r for r in rows if r['pde'] == pde and r['variant'] == name)
            a = '---' if pde == 'burger' else f"{100*float(row['mean_rel_l2_a']):.2f}"
            u = f"{100*float(row['mean_rel_l2_u']):.2f}"
            primary = (f"{100*float(row['mean_error']):.2f} "
                       f"[{100*float(row['ci95_low']):.2f}, {100*float(row['ci95_high']):.2f}]")
            residual = f"{float(row['mean_pde_residual']):.3g}"
            timing = f"{float(row['median_amortized_seconds']):.3f}"
            lines.append(f"{label} & {row['model_forward_calls']} & {a} & {u} & {primary} & {residual} & {timing}" + r' \\')
        lines += [r'\bottomrule\end{tabular}', r'\end{table}', r'\FloatBarrier', '']
    table = source_dir/'sampling_confirmation_tables.tex'
    table.write_text('\n'.join(lines)+'\n')
    copied[str(table.relative_to(args.paper))] = sha256(table)
    (args.paper/'audit/sampling_paper_export.json').write_text(json.dumps({
        'protocol_sha256': manifest['validation'][0]['protocol_sha256'],
        'summary_rows': len(rows), 'example_runs': len(archived), 'files': copied,
        'exporter_sha256': sha256(Path(__file__)),
        'scope': 'Complete report and tensor-audit consistency checks before export; manuscript interpretation and visual QA are separate.',
    }, indent=2)+'\n')
    print('EXPORTED 52 variant summaries and 4992 audited example-runs to', args.paper)


if __name__ == '__main__':
    main()

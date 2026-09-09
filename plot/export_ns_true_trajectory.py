"""Export completed oracle-trajectory diagnostics and the viscous linearization."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from publication_style import use_times_new_roman


def viscous_amplification_figure(radius, exact, secant, threshold):
    """Render the analytic curves at the supplement's 5.4-inch display width."""
    font = use_times_new_roman()
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10,
                         'axes.labelsize': 10, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'mathtext.fontset': 'stix',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(5.4, 3.35), layout='constrained')
    ax.plot(radius, exact, color='#256493', lw=1.8, label=r'Exact viscous decay: $e^{-z}$')
    ax.plot(radius, secant, color='#aa4b32', lw=1.8, ls='--', label=r'Endpoint secant: $(1-z/2)/(1+z/2)$')
    ax.axhline(0, color='.5', lw=.7)
    ax.axvline(threshold, color='.6', lw=.8, ls=':')
    ax.annotate(f'Sign change at mode radius {threshold:.2f}', xy=(threshold, 0), xytext=(18, .38),
                arrowprops=dict(arrowstyle='-', color='.4', lw=.8), fontsize=10)
    ax.set(xlim=(0, 64), ylim=(-1.08, 1.05), xlabel='Radial Fourier-mode index', ylabel='Amplification factor',
           title='Linear viscous term in the endpoint constraint\n' + r'$\nu=10^{-3},\quad\mathcal{T}=1$')
    ax.grid(alpha=.15)
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    return fig, font


def main():
    from run_ns_loss_study import sha, write

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['audit', 'inputs', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.audit / 'true_trajectory_audit.json').read_text())
    assert manifest['status'] == 'complete' and manifest['examples'] == 32
    assert all(manifest['checks'].values())
    assert sha(args.inputs / 'source.json') == manifest['source_sha256']
    for name, h in manifest['outputs'].items():
        assert sha(args.audit / name) == h, name
    source = json.loads((args.inputs / 'source.json').read_text())
    T, nu = [float(source['pde_params'][k]) for k in ['T', 'nu']]
    assert T == 1 and nu == .001
    rows = list(csv.DictReader((args.audit / 'true_trajectory_summary.csv').open()))
    assert len(rows) == 9 and all(int(r['n']) == 32 for r in rows)
    stats = {r['metric']: (float(r['mean']), float(r['sd'])) for r in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    table = [r'\begin{table}[!htbp]\centering\footnotesize',
             r'\caption{Residual diagnostics on 32 true NS trajectories (mean $\pm$ SD across inputs). Only the first row uses the two endpoints alone. The other rows access archived true intermediate states and are diagnostic checks, not feasible endpoint-only conditioning rules or sampling-accuracy results. Both centered-difference rows use the same four center times, $0.2,0.4,0.6,0.8$. Integral rows use all 11 archived frames.}',
             r'\label{tab:ns-true-trajectory}', r'\begin{tabular}{@{}lr@{}}\toprule',
             r'Temporal diagnostic & RMS residual \\\midrule']
    labels = [('endpoint_secant_rms', r'Endpoint secant at $(\mathbf a+\mathbf u)/2$'),
              ('endpoint_drift_true_midpoint_rms', r'Endpoint drift at true $\omega(0.5)$'),
              ('centered_2h_common_4_times_rms', r'Centered difference, $\Delta\tau=0.2$'),
              ('centered_h_common_4_times_rms', r'Centered difference, $\Delta\tau=0.1$'),
              ('trapezoid_integral_rms', r'Trapezoidal RHS integral, $\Delta\tau=0.1$'),
              ('simpson_integral_rms', r'Simpson RHS integral, $\Delta\tau=0.1$')]
    for metric, label in labels:
        mean, sd = stats[metric]
        exponent = int(np.floor(np.log10(mean)))
        scale = 10. ** exponent
        table.append(label + f' & $({mean/scale:.3f}\\pm{sd/scale:.3f})\\times10^{{{exponent}}}$' + r' \\')
    table += [r'\bottomrule\end{tabular}\end{table}']
    (args.output / 'ns_true_trajectory_table.tex').write_text('\n'.join(table) + '\n')

    # This is the exact amplification of the linear viscous component, not
    # an amplification formula for the nonlinear NS dynamics or neural sampler.
    radius = np.linspace(0, 64, 1601)
    z = nu * (2 * np.pi * radius) ** 2 * T
    exact = np.exp(-z)
    secant = (1 - z / 2) / (1 + z / 2)
    threshold = np.sqrt(2 / (nu * (2 * np.pi) ** 2 * T))
    assert np.isclose((1 - nu*(2*np.pi*threshold)**2*T/2), 0, atol=1e-14)
    assert exact[0] == secant[0] == 1 and np.all(exact > 0)
    assert np.all(np.diff(exact) <= 0) and np.all(np.diff(secant) < 0)
    fig, font = viscous_amplification_figure(radius, exact, secant, threshold)
    for ext in ['pdf', 'png']:
        fig.savefig(args.output / f'ns_viscous_amplification.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)
    points = []
    for k in [0, 4, 8, 16, 32, 64]:
        zz = nu * (2 * np.pi * k) ** 2 * T
        points.append(dict(radius=k, z=zz, exact_viscous_amplification=float(np.exp(-zz)),
                           endpoint_secant_amplification=(1 - zz / 2) / (1 + zz / 2)))
    write(args.output / 'ns_temporal_diagnostic_report.json', dict(
        status='complete', examples=32, audit_sha256=sha(args.audit / 'true_trajectory_audit.json'),
        source_sha256=manifest['source_sha256'], script_sha256=sha(Path(__file__)), font_path=font,
        sign_change_radius=threshold, amplification_points=points,
        interpretation='Linear viscous-mode illustration only. It excludes advection and forcing and does not describe the neural sampling transition. Oracle trajectory diagnostics cannot be substituted for endpoint-only inference results.',
        visual_review_pending=True,
        outputs={p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'ns_temporal_diagnostic_report.json'}))


if __name__ == '__main__':
    main()

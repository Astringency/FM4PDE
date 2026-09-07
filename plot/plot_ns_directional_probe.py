"""Analytic Fourier probes of the two archived NS residual implementations.

These are operator diagnostics, not learned predictions or accuracy results.
The frozen sampling inputs, checkpoints, and running experiments are untouched.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sampling.config import AblationConfig
from sampling.pde_residuals import compute_pde_residual
from ns_loss_exchange import diffusion_spatial_residual, diffusion_pde_loss, fm_pde_loss
from publication_style import use_times_new_roman
from run_ns_loss_study import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.inputs / 'source.json').read_text())
    params = dict(source['pde_params'], enforce_boundary_conditions=False)
    cfg = AblationConfig(**source['fm_configs']['both'])
    assert params['nu'] == .001 and params['T'] == 1.
    torch.set_num_threads(2)
    n = 128
    axis = torch.arange(n, dtype=torch.float64) / n
    x, y = torch.meshgrid(axis, axis, indexing='ij')
    zero = torch.zeros((1, 1, n, n), dtype=torch.float64)

    def residual(a, u):
        return compute_pde_residual('nsnonbounded', a, u, pde_params=params,
                                    residual_mode='endpoint_secant').residual

    background = residual(zero, zero)
    rows, panels = [], {}
    for k in [4, 8, 16, 32, 48, 63]:
        for sign in [1, -1]:
            p, q = k, sign*k
            phase = 2*torch.pi*(p*x + q*y)
            probe = torch.cos(phase)[None, None]
            rd = diffusion_spatial_residual(probe, probe)
            coefficient = np.sin(2*np.pi*p/n) + np.sin(2*np.pi*q/n)
            expected_d = -coefficient * torch.sin(phase)[None, None]
            expected_d[..., (0, -1), :] = 0
            expected_d[..., :, (0, -1)] = 0
            rf = residual(probe, probe)
            # A single Fourier vorticity mode has zero self-advection.
            # Subtract the unchanged forcing background to isolate its response.
            delta_f = rf - background
            coefficient_f = params['nu']*(2*np.pi)**2*(p*p+q*q)
            expected_f = coefficient_f*probe
            error_d = float((rd-expected_d).abs().max())
            error_f = float((delta_f-expected_f).abs().max())
            assert error_d < 2e-13, (p, q, error_d)
            assert error_f < 2e-9, (p, q, error_f)
            ld = float(diffusion_pde_loss(probe, probe))
            lf = float(fm_pde_loss(probe, probe, cfg, params))
            assert np.isclose(ld, float(rd.norm())/(n*n), rtol=1e-14, atol=1e-30)
            assert np.isclose(lf, float(rf.square().mean()), rtol=1e-12, atol=1e-12)
            if sign == -1:
                assert float(rd.abs().max()) < 2e-13
            rows.append(dict(k_x=p, k_y=q, mode_radius=float(np.hypot(p,q)),
                directional_stencil_symbol=float(coefficient),
                viscous_response_symbol=float(coefficient_f),
                diffusion_residual_rms=float(rd.square().mean().sqrt()),
                fm_forcing_subtracted_rms=float(delta_f.square().mean().sqrt()),
                diffusion_loss=ld, fm_full_loss=lf,
                stencil_max_abs_error=error_d, fm_response_max_abs_error=error_f))
            if k == 8:
                panels[sign] = [v[0,0].numpy() for v in [probe, rd, delta_f]]

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'ns_directional_probe.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    font = use_times_new_roman()
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11})
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.2), layout='constrained')
    titles = ['Stationary probe: $a=u=\\phi$',
              'Archived directional residual $r_D$',
              'FM response $r_F(\\phi,\\phi)-r_F(0,0)$']
    for col in range(3):
        maximum = max(float(np.abs(panels[s][col]).max()) for s in [1,-1])
        for row, sign in enumerate([1,-1]):
            ax = axes[row,col]
            im = ax.imshow(panels[sign][col].T, origin='lower', cmap='RdBu_r',
                           vmin=-maximum, vmax=maximum, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(titles[col])
            if col == 0: ax.set_ylabel(f'$\\mathbf{{k}}=(8,{sign*8})$')
            if col > 0:
                rms = np.sqrt(np.mean(panels[sign][col]**2))
                ax.set_xlabel(f'RMS = {rms:.3g}')
        bar = fig.colorbar(im, ax=axes[:,col], orientation='horizontal', shrink=.8, pad=.04)
        bar.set_label(['Probe amplitude', 'Directional residual scale', 'Forcing-subtracted residual scale'][col])
    fig.suptitle('Equal-radius Fourier probes expose directional cancellation\n'
                 'Analytic operator diagnostic; independent color scales across columns', fontsize=12)
    for ext in ['pdf','png']:
        fig.savefig(args.output/f'ns_directional_probe.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)
    (args.output/'ns_directional_probe.tex').write_text(r'''% Analytic diagnostic; no learned sampling results are represented here.
\paragraph{Directional cancellation at equal mode radius.}
For the stationary Fourier probe $a=u=\phi_{p,q}$, where
$\phi_{p,q}(i,j)=\cos[2\pi(pi+qj)/128]$, the interior stencil in
Equation~\eqref{eq:ns-shared-spatial} gives
\[
 r_D=-\left[\sin\left(\frac{2\pi p}{128}\right)
          +\sin\left(\frac{2\pi q}{128}\right)\right]
       \sin\left[\frac{2\pi(pi+qj)}{128}\right].
\]
It vanishes for $q=-p$, including after the implementation zeros the outer
grid ring. This is a directional null space, not evidence that the field
satisfies the NS equation. The displayed multiplier describes the interior
stencil; zeroing the boundary ring prevents interpreting it as the Fourier
eigenvalue of the entire finite-array operator.

A single Fourier vorticity mode has zero self-advection. With the original
forcing retained, subtracting the zero-field forcing background gives
\[
 r_F(\phi_{p,q},\phi_{p,q})-r_F(0,0)
   =\nu(2\pi)^2(p^2+q^2)\phi_{p,q}.
\]
This response has equal magnitude for the two displayed orientations.
Substitution into the actual implementations verifies both formulas for
$p\in\{4,8,16,32,48,63\}$ and $q=\pm p$. The probe pairs are deliberately
stationary fields, not time-evolved reference solutions. This check
illustrates a structural distinction between the constraints; it does not
establish lower reconstruction error, remove the endpoint temporal
approximation, or describe a learned sampler's frequency response.

\begin{figure}[!htbp]
\centering\includegraphics[width=\linewidth]{figures/ns_directional_probe.pdf}
\caption{Analytic equal-radius Fourier probes evaluated with the archived
NS residual implementations on a $128\times128$ grid. The directional
residual cancels for $(8,-8)$, while the forcing-subtracted FM residual
responds to the viscous term for both orientations. Color scales are shared
between rows within each column and differ across columns; raw residual
magnitudes have different definitions and are not accuracy scores. No
neural predictions or training/evaluation inputs are used in this figure.}
\label{fig:ns-directional-probe}
\end{figure}
''')
    write(args.output/'ns_directional_probe_manifest.json', dict(
        status='complete_analytic_diagnostic', source_sha256=sha(args.inputs/'source.json'),
        params=params, grid_shape=[n,n], probes=len(rows), displayed_modes=[[8,8],[8,-8]],
        font_path=font, script_sha256=sha(Path(__file__)),
        dependency_sha256={name:sha(ROOT/name) for name in ['plot/ns_loss_exchange.py',
            'sampling/pde_residuals.py','sampling/losses.py','plot/publication_style.py']},
        maximum_stencil_abs_error=max(r['stencil_max_abs_error'] for r in rows),
        maximum_fm_response_abs_error=max(r['fm_response_max_abs_error'] for r in rows),
        visual_review_pending=True,
        scope='Analytic stationary Fourier probes only. Original forcing is retained and subtracted as a zero-field background for the FM response illustration. Not a reconstruction comparison, learned transfer function, or new sampler.',
        outputs={f.name:sha(f) for f in args.output.iterdir() if f.is_file()
                 and f.name!='ns_directional_probe_manifest.json'}))
    print(json.dumps(dict(status='complete_analytic_diagnostic', probes=len(rows),
                         output=str(args.output))))


if __name__ == '__main__':
    main()

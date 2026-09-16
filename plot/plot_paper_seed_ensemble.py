"""Draw fieldwise paired effects and spatial spectra of conditional averages."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

from publication_style import use_times_new_roman, error_number
from export_paper_seed_ensemble import NAMES, NONPERIODIC
from run_ablation_study import digest, write

BLUE, GOLD, INK = '#246591', '#C99732', '#2B2B2B'


def plot(args):
    assert args.contract.is_file()
    manifest = json.loads((args.source/'manifest.json').read_text())
    assert manifest['full_study'] or args.development
    for name in ['summary.csv', 'spectral_shells.npz']:
        assert digest(args.source/name) == manifest['outputs'][name]
    rows = list(csv.DictReader((args.source/'summary.csv').open()))
    shells = np.load(args.source/'spectral_shells.npz')
    args.output.mkdir(parents=True, exist_ok=True)
    use_times_new_roman()
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,
        'xtick.labelsize':9,'ytick.labelsize':9,'font.family':'serif',
        'font.serif':['Times New Roman'],'mathtext.fontset':'custom',
        'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
        'mathtext.bf':'Times New Roman:bold','pdf.fonttype':42,'ps.fonttype':42})
    outputs, captions, tick_checks = {}, [], []
    def save(fig, stem, caption):
        for ext in ['pdf', 'png']:
            path=args.output/f'{stem}.{ext}'
            fig.savefig(path,dpi=240,bbox_inches='tight',pad_inches=.04)
            outputs[path.name]=digest(path)
        plt.close(fig)
        captions.extend([r'\begin{figure}[!htbp]',r'\centering',
            r'\includegraphics[width=\linewidth]{figures/'+stem+'.pdf}',
            r'\caption{'+caption+'}',r'\label{fig:'+stem.replace('_','-')+'}',r'\end{figure}',''])
    fig, ax = plt.subplots(figsize=(6.0, max(2.7, .24*len(rows)+.8)))
    for i, row in enumerate(rows):
        point, lo, hi = [float(row[k]) for k in ['paired_delta_pp','family21_low','family21_high']]
        color = BLUE if row['field']=='a' else GOLD
        ax.plot([lo, hi], [i, i],color=color,lw=1.5)
        ax.plot(point,i,'o' if row['field']=='a' else 's',color=color,ms=4)
        ax.text(1.015,i,'$'+error_number(point,signed=True)+'$',transform=ax.get_yaxis_transform(),
                ha='left',va='center',fontsize=9,color=INK,clip_on=False)
    ax.axvline(0,color=INK,lw=.7,ls='--')
    ax.set_yticks(range(len(rows)),[NAMES[r['pde']].replace('--','–')+rf"  ${r['field']}$" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('Change in relative error (percentage points)')
    ax.set_title('Mean of three predictions − single prediction')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.14)
    fig.tight_layout()
    save(fig,'ensemble_paired_fields',
        'Effect of averaging three 100-step conditional predictions. Points show the mean paired change in field error relative to the prespecified single prediction, over 32 inputs per PDE. '
        'Negative values indicate lower error; the right-hand labels give mean changes in percentage points. Intervals use the paired bootstrap with a Bonferroni adjustment for the 21 field comparisons; blue circles and gold squares denote $a$ and $u$, respectively.')
    for pde in manifest['pdes']:
        fields=['u'] if pde=='burger' else ['a','u']
        fig, axes=plt.subplots(len(fields),2,figsize=(6.0,2.15*len(fields)+.3),squeeze=False)
        for i, field in enumerate(fields):
            prefix=pde+'__'+field+'__'
            reference=shells[prefix+'reference']
            assert reference.shape[0]==32
            k=np.arange(reference.shape[-1])[1:]
            for column in range(2):
                ax=axes[i,column]
                series=[]
                if column==0:series.append(('Truth',reference.mean(0)[1:],INK,'-'))
                for estimator,label,color,style in [('seed0','Single',BLUE,'--'),('mean3','Mean of 3',GOLD,'-')]:
                    values=shells[prefix+estimator]
                    assert values.shape==(32,2,len(k)+1)
                    series.append((label,values[:,column,:].mean(0)[1:],color,style))
                for label,values,color,style in series:
                    assert np.isfinite(values).all() and (values>=0).all()
                    ax.plot(k,values,label=label,color=color,ls=style,lw=1.15)
                ax.set_yscale('symlog',linthresh=1e-14,linscale=.3)
                ax.set_ylim(bottom=max(0.0,ax.get_ylim()[0]))
                # Keep the original symlog scale and limits, but avoid crowded
                # labels in the short linear region near zero.
                lower, upper = ax.get_ylim()
                first = int(np.ceil(np.log10(max(lower, 1e-12))))
                last = int(np.floor(np.log10(upper)))
                stride = max(1, int(np.ceil((last - first + 1) / 6)))
                ticks = ([0.0] if lower == 0 else []) + [10.0**e for e in range(first, last + 1, stride)]
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_xlabel('Spatial mode' if pde=='burger' else 'Radial mode index' if pde in NONPERIODIC else 'Radial wavenumber')
                ax.set_ylabel('Normalized shell energy' if column==0 else 'Normalized shell error energy')
                ax.set_title(rf'${field}$: '+('energy' if column==0 else 'error'))
                ax.spines[['top','right']].set_visible(False)
                ax.grid(alpha=.12)
                if i==0:ax.legend(frameon=False,fontsize=9)
        fig.suptitle(NAMES[pde].replace('--','–'),fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes.flat:
            boxes = sorted((label.get_window_extent(renderer) for label in ax.get_yticklabels()
                            if label.get_visible() and label.get_text()), key=lambda box: box.y0)
            gaps = [b.y0-a.y1 for a,b in zip(boxes, boxes[1:])]
            assert not gaps or min(gaps) >= 2.0, (pde, ax.get_title(), gaps)
            tick_checks.append(dict(pde=pde, panel=ax.get_title(), major_ticks=len(boxes),
                                    minimum_label_gap_pt=min(gaps)*72/fig.dpi if gaps else None))
        transform=('Fourier modes along space, summed across physical time' if pde=='burger' else
                   'radially grouped DCT-II modes' if pde in NONPERIODIC else 'radially grouped spatial Fourier modes')
        save(fig,'ensemble_spectra_'+pde,
            NAMES[pde]+' spectra for the single prediction and the average of three predictions, using '+transform+'. '
            'Curves average 32 input spectra, each normalized by its full reference-field energy. The left column shows reference and predicted energy; the right shows prediction-error energy. '
            'The vertical scale is logarithmic above $10^{-14}$ and linear below it. '
            'All channels of a field contribute to its spectrum. The constant mode is excluded from the horizontal axis.')
    (args.output/'ensemble_figures.tex').write_text('\n'.join(captions))
    write(args.output/'figure_manifest.json',dict(full_study=manifest['full_study'],pdes=manifest['pdes'],
        analysis_sha256=digest(args.source/'manifest.json'),contract_sha256=digest(args.contract),
        plotter_sha256=digest(Path(__file__)),tick_label_checks=tick_checks,outputs=outputs))
    print('PLOTTED ensemble effects and',len(manifest['pdes']),'PDE spectra',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--development',action='store_true')
    plot(p.parse_args())

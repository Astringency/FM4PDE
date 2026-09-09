"""Render the verified matched-input frequency study at JMLR column width."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_style import use_times_new_roman
from spectral_diagnostics import paired_bootstrap,filtered_field
from run_paper_ablation_revision import digest
from plot_paper_ablation_fields import FIELD_CMAP,ERROR_CMAP

COLORS={'FM4PDE':'#27638b','RecFNO':'#bf922f','Senseiver':'#282828','VoronoiCNN':'#27638b'}
STYLES={'FM4PDE':'-','RecFNO':'--','Senseiver':'-.','VoronoiCNN':':'}
METHODS=list(COLORS)


def main(args):
    records=json.load(gzip.open(args.source/'frequency_records.json.gz','rt'))
    examples=np.load(args.source/'frequency_example_fields.npz')
    summaries=list(csv.DictReader((args.source/'frequency_summary.csv').open()))
    source_manifest=json.loads((args.source/'spectra_manifest.json').read_text())
    assert source_manifest['record_count']==len(records)==1216
    groups=defaultdict(list)
    for record in records:
        for key in ['reference_power','prediction_power','error_power']:
            record[key]=np.asarray(record[key],dtype=float)
        groups[record['pde'],record['method'],record['steps'],record['ensemble']].append(record)
    use_times_new_roman()
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,
        'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.5})
    args.output.mkdir(parents=True,exist_ok=True);outputs={}
    def save(fig,stem):
        fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
        for ext in ['pdf','png']:
            path=args.output/(stem+'.'+ext);fig.savefig(path,dpi=210,bbox_inches='tight',pad_inches=.03);outputs[path.name]=digest(path)
        plt.close(fig)
    for pde in ['poisson','darcy']:
        fig,axes=plt.subplots(1,2,figsize=(6,3.2),layout='constrained')
        for j,method in enumerate(METHODS):
            rows=groups[pde,method,100 if method=='FM4PDE' else 1,False]
            ids=list(dict.fromkeys(r['sample_id'] for r in rows));assert len(ids)==32
            per_id=[[r for r in rows if r['sample_id']==i] for i in ids]
            power=np.stack([np.mean([r['prediction_power']/r['reference_total'] for r in rr],axis=0) for rr in per_id])
            error=np.stack([np.mean([r['error_power']/r['reference_total'] for r in rr],axis=0) for rr in per_id])
            radius=np.arange(power.shape[1])
            for ax,values in zip(axes,[power,error]):ax.plot(radius[1:],values.mean(0)[1:],label=method,color=COLORS[method],ls=STYLES[method],lw=1.1)
            ci=paired_bootstrap(error)
            axes[1].fill_between(radius[1:],ci[0,1:],ci[1,1:],color=COLORS[method],alpha=.10,linewidth=0)
            if j==0:
                truth=np.stack([rr[0]['reference_power']/rr[0]['reference_total'] for rr in per_id]).mean(0)
                axes[0].plot(radius[1:],truth[1:],color='#282828',lw=1.3,label='Truth')
        for ax,title in zip(axes,['Reference and prediction','Reconstruction error']):
            ax.set(xscale='log',yscale='log',xlabel='Radial cosine-mode index',title=title)
            ax.grid(alpha=.15);ax.axvline(8,color='#888888',lw=.5);ax.axvline(32,color='#888888',lw=.5)
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel('Normalized shell energy');axes[1].set_ylabel('Normalized shell error energy')
        axes[0].legend(fontsize=9,frameon=False)
        fig.suptitle(f'{pde.title()} inverse reconstruction · 32 ID inputs',fontsize=10)
        save(fig,'spectrum_'+pde)

        fig,axes=plt.subplots(4,5,figsize=(6,6.5),layout='constrained')
        truth=examples[pde+'_truth']
        for k,band in enumerate(['low','high']):
            t=filtered_field(truth,'cosine',band)
            preds=[filtered_field(examples[pde+'_'+m],'cosine',band) for m in METHODS]
            vmax=max(np.max(np.abs(v)) for v in [t]+preds)
            errors=[np.abs(v-t) for v in preds];emax=max(np.max(v) for v in errors)
            for j,v in enumerate([t]+preds):
                field_im=axes[2*k,j].imshow(v,origin='lower',cmap=FIELD_CMAP,vmin=-vmax,vmax=vmax)
                if k==0:axes[0,j].set_title((['Truth']+METHODS)[j],fontsize=9)
            axes[2*k+1,0].axis('off')
            axes[2*k+1,0].text(.5,.5,'Absolute\nband error',ha='center',va='center',transform=axes[2*k+1,0].transAxes,fontsize=9)
            for j,v in enumerate(errors,1):
                error_im=axes[2*k+1,j].imshow(v,origin='lower',cmap=ERROR_CMAP,vmin=0,vmax=max(emax,1e-14))
                error=np.linalg.norm(v)/max(np.linalg.norm(t),1e-30)
                axes[2*k+1,j].set_xlabel(r'$\mathrm{RelL2}_{a,B}$'+f'\n${100*error:.2f}\\%$',fontsize=9)
            axes[2*k,0].set_ylabel('DC–8' if band=='low' else '>32',fontsize=9)
            for im,axrow in [(field_im,axes[2*k,:]),(error_im,axes[2*k+1,:])]:
                cb=fig.colorbar(im,ax=axrow,orientation='horizontal',fraction=.07,pad=.02,aspect=40,shrink=.8)
                cb.ax.tick_params(labelsize=9)
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'{pde.title()} frequency-separated reconstruction · ID 425',fontsize=10)
        save(fig,'frequency_fields_'+pde)

    fig,axes=plt.subplots(1,2,figsize=(6,3.8),layout='constrained')
    for ax,pde in zip(axes,['poisson','darcy']):
        for ensemble,style in [('False','o-'),('True','s--')]:
            rows=sorted([r for r in summaries if r['pde']==pde and r['method']=='FM4PDE' and r['ensemble']==ensemble],key=lambda r:int(r['steps']))
            x=np.array([int(r['steps']) for r in rows]);y=np.array([float(r['mean_rel_l2']) for r in rows])*100
            lo=np.array([float(r['ci_low']) for r in rows])*100;hi=np.array([float(r['ci_high']) for r in rows])*100
            ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=style,color=COLORS['FM4PDE'],capsize=3,
                markerfacecolor='white' if ensemble=='True' else COLORS['FM4PDE'],label='Mean of 3' if ensemble=='True' else 'Single sample')
        for method in METHODS[1:]:
            row=next(r for r in summaries if r['pde']==pde and r['method']==method)
            ax.axhline(100*float(row['mean_rel_l2']),color=COLORS[method],ls=STYLES[method],label=method)
        ax.set(xlabel='Steps per prediction',ylabel=r'$\mathrm{RelL2}_a$ (%)',title=pde.title());ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    fig.suptitle('Sampling budget and averaging · 32 inverse inputs',fontsize=10)
    save(fig,'frequency_sampling_improvement')
    (args.output/'frequency_figure_manifest.json').write_text(json.dumps(dict(
        source_hashes={name:digest(args.source/name) for name in ['frequency_records.json.gz','frequency_example_fields.npz','frequency_summary.csv','spectra_manifest.json']},
        script_sha256=digest(Path(__file__)),outputs=outputs,records=len(records),physical_inputs=32,pdes=['poisson','darcy'],font_size=9),indent=2)+'\n')
    print('PLOTTED',len(outputs)//2,'matched frequency figures',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

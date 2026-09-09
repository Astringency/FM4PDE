#!/usr/bin/env python3
"""Merge fully audited host exports; render the Poisson sample-scaling study."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'plot'))
from publication_style import use_times_new_roman,error_number
from run_paper_ablation_revision import digest,write
KS=[1,3,10,100,1000]
FIELDS=[('forward','u','Forward'),('inverse','a','Inverse'),('both','a','Joint'),('both','u','Joint')]
COLORS={'a':'#A77522','u':'#255F85'}
TASKS={'forward':'Forward','inverse':'Inverse','both':'Joint'}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.figures.mkdir(parents=True,exist_ok=True)
    rows=[];arrays={};manifests=[]
    for folder in args.exports:
        manifest=json.loads((folder/'conditional_scaling_manifest.json').read_text())
        assert manifest['complete'],folder
        manifests.append(manifest)
        for row in csv.DictReader((folder/'conditional_scaling_per_input.csv').open()):
            rows.append({k:v if k=='task' else int(v) if k in {'offset','K','peak_bytes'} else float(v) for k,v in row.items()})
        for key,value in np.load(folder/'conditional_scaling_fields.npz').items():
            assert key not in arrays,key
            arrays[key]=value
    index={(r['task'],r['offset'],r['K']):r for r in rows}
    expected={(t,i,k) for t in TASKS for i in range(1500,1532) for k in KS}
    assert len(rows)==480 and set(index)==expected
    assert sum(m['conditional_trajectories'] for m in manifests)==96000
    assert sum(m['independent_timing_trajectories'] for m in manifests)==106944
    rows=sorted(rows,key=lambda x:(x['task'],x['offset'],x['K']))
    write_csv(args.output/'conditional_scaling_per_input.csv',rows)
    np.savez_compressed(args.output/'conditional_scaling_fields.npz',**arrays)
    rng=np.random.default_rng(20260909)
    draws=rng.integers(0,32,(100000,32))
    summary=[];effects=[];timing=[]
    for task,field,name in FIELDS:
        baseline=np.array([index[task,i,1][f'rel_l2_{field}']*100 for i in range(1500,1532)])
        for k in KS:
            values=np.array([index[task,i,k][f'rel_l2_{field}']*100 for i in range(1500,1532)])
            boot=values[draws].mean(1)
            summary.append(dict(task=task,field=field,K=k,n=32,mean_percent=values.mean(),sd_percent=values.std(ddof=1),
                                mean_ci_low=np.quantile(boot,.025),mean_ci_high=np.quantile(boot,.975)))
            if k>1:
                delta=values-baseline;bd=delta[draws].mean(1)
                effects.append(dict(task=task,field=field,K=k,n=32,mean_delta_pp=delta.mean(),
                    simultaneous_ci_low=np.quantile(bd,.05/(2*16)),simultaneous_ci_high=np.quantile(bd,1-.05/(2*16)),
                    improved_inputs=int((delta<0).sum()),relative_mean_error_change_percent=100*(values.mean()/baseline.mean()-1)))
    for task in TASKS:
        for k in KS:
            values=np.array([index[task,i,k]['seconds'] for i in range(1500,1532)])
            ratios=np.array([index[task,i,k]['seconds']/index[task,i,1]['seconds'] for i in range(1500,1532)])
            timing.append(dict(task=task,K=k,n=32,median_seconds=np.median(values),mean_seconds=values.mean(),sd_seconds=values.std(ddof=1),
                               q25_seconds=np.quantile(values,.25),q75_seconds=np.quantile(values,.75),
                               median_ratio_to_K1=np.median(ratios),median_speedup_vs_serial=np.median(k/ratios)))
    write_csv(args.output/'conditional_scaling_summary.csv',summary)
    write_csv(args.output/'conditional_scaling_paired_effects.csv',effects)
    write_csv(args.output/'conditional_scaling_timing.csv',timing)
    si={(r['task'],r['field'],r['K']):r for r in summary}
    ti={(r['task'],r['K']):r for r in timing}
    font=use_times_new_roman()
    plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
                         'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
    generated=[]
    def save(fig,name):
        for ext in ['pdf','png']:
            path=args.figures/f'{name}.{ext}'
            fig.savefig(path,dpi=240,bbox_inches='tight',facecolor='white')
            generated.append(dict(path=str(path),sha256=digest(path)))
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.6),layout='constrained')
    for ax,(task,field,name) in zip(axes.flat,FIELDS):
        vals=[si[task,field,k] for k in KS]
        mean=np.array([r['mean_percent'] for r in vals])
        ci=np.array([[r['mean_ci_low'] for r in vals],[r['mean_ci_high'] for r in vals]])
        ax.errorbar(KS,mean,yerr=np.vstack([mean-ci[0],ci[1]-mean]),color=COLORS[field],fmt='o-',markersize=4,lw=1.1,capsize=2)
        ax.set_xscale('log');ax.set_xticks(KS,labels=[str(k) for k in KS]);ax.minorticks_off()
        ax.set_title(f'{name}: '+rf'$\operatorname{{RelL2}}_{field}$')
        ax.set_xlabel('Conditional samples, $K$');ax.set_ylabel('Relative error (%)')
        ax.grid(axis='y',color='#DDDDDD',lw=.4);ax.set_ylim(bottom=0)
    save(fig,'conditional_scaling_accuracy')
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.65),layout='constrained')
    for ax,(task,name) in zip(axes,TASKS.items()):
        vals=[ti[task,k] for k in KS];med=np.array([r['median_seconds'] for r in vals])
        ax.plot(KS,med,'o-',color='#255F85',markersize=4,lw=1.1,label='Batched')
        ax.fill_between(KS,[r['q25_seconds'] for r in vals],[r['q75_seconds'] for r in vals],color='#255F85',alpha=.13)
        ax.plot(KS,np.array(KS)*med[0],'--',color='#777777',lw=.9,label=r'$K$ times $K=1$')
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(KS,labels=[str(k) for k in KS]);ax.minorticks_off()
        ax.set_title(name);ax.set_xlabel('Conditional samples, $K$');ax.set_ylabel('Estimate latency (s)')
        ax.grid(axis='y',color='#DDDDDD',lw=.4)
    axes[0].legend(loc='upper left',frameon=False,handlelength=1.6)
    save(fig,'conditional_scaling_time')
    # The first predeclared evaluation offset is used without inspecting errors.
    fig,axes=plt.subplots(4,6,figsize=(7.1,5.75))
    fig.subplots_adjust(left=.065,right=.94,top=.95,bottom=.025,wspace=.12,hspace=.28)
    for row,(task,field,name) in enumerate(FIELDS):
        j=0 if field=='a' else 1;key=f'{task}_1500'
        vals=[arrays[key+'_truth'][j]]+[arrays[key+f'_K{k}'][j] for k in KS]
        lo,hi=min(float(v.min()) for v in vals),max(float(v.max()) for v in vals)
        for col,(ax,v) in enumerate(zip(axes[row],vals)):
            im=ax.imshow(v,origin='lower',extent=[0,1,0,1],cmap='cividis',norm=Normalize(lo,hi),interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title('Truth' if col==0 else f'$K={KS[col-1]}$',fontsize=10,pad=5)
            if col==0:
                ax.set_ylabel(name+'\n'+rf'$\mathbf{{{field}}}$',fontsize=9)
            else:
                err=index[task,1500,KS[col-1]][f'rel_l2_{field}']*100
                ax.set_xlabel(f'{err:.2f}%',fontsize=9,labelpad=2)
        pos=axes[row,-1].get_position()
        cax=fig.add_axes([.95,pos.y0,.012,pos.height])
        cb=fig.colorbar(im,cax=cax);cb.ax.tick_params(labelsize=9,pad=2,width=.4);cb.outline.set_linewidth(.4)
        cb.locator=matplotlib.ticker.MaxNLocator(nbins=3);cb.update_ticks()
    save(fig,'conditional_scaling_reconstructions')
    lines=[r'\begin{table}[!htbp]\centering\small',r'\caption{Conditional-sample averaging for Poisson over 32 ID inputs. Relative errors are mean $\pm$ sample standard deviation in percent. Boldface marks the lowest mean and $\dagger$ the second-lowest within each field.}',
        r'\label{tab:conditional-scaling}',r'\begin{tabular}{rcccc}\toprule',
        r'$K$ & Forward $\operatorname{RelL2}_u$ & Inverse $\operatorname{RelL2}_a$ & Joint $\operatorname{RelL2}_a$ & Joint $\operatorname{RelL2}_u$ \\\midrule']
    ranks={(t,f):sorted(KS,key=lambda k:si[t,f,k]['mean_percent']) for t,f,_ in FIELDS}
    for k in KS:
        cells=[]
        for t,f,_ in FIELDS:
            r=si[t,f,k];mean=error_number(r['mean_percent'])
            if k==ranks[t,f][0]:mean=r'\mathbf{'+mean+'}'
            elif k==ranks[t,f][1]:mean+=r'^{\dagger}'
            cells.append('$'+mean+r'\,\pm\,'+error_number(r['sd_percent'])+'$')
        lines.append(str(k)+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (args.output/'conditional_scaling_table.tex').write_text('\n'.join(lines)+'\n')
    figure_lines=[]
    captions={
      'accuracy':r'Poisson reconstruction error versus the number of averaged conditional samples. Means and pointwise 95\% bootstrap intervals are computed over the same 32 ID inputs. Every draw uses 100 stochastic Euler steps; field averages are formed before evaluating $\operatorname{RelL2}$.',
      'time':r'Latency of averaged Poisson estimates, including batch preparation, generation, physical-field conversion, transfer, and averaging. Curves show medians over 32 inputs and shaded bands the interquartile range. Samples are processed in batches of at most 64 on an A100 or A800 GPU; all values of $K$ for an input are measured on the same device. The dashed line is the serial reference $K$ times the measured $K=1$ median, not a separately timed serial experiment. This latency definition differs from the FM4PDE--DiffusionPDE sampling-time comparison, which isolates the sampling computation.',
      'reconstructions':r'Poisson conditional-sample averages for the first input in the evaluation cohort. Columns compare the reference fields with averages of $K=1,3,10,100,1000$ predictions. The four rows show forward $\mathbf{u}$, inverse $\mathbf{a}$, and joint $\mathbf{a}$ and $\mathbf{u}$. Colors share one scale within each row. Labels below reconstructed fields give $\operatorname{RelL2}$ in percent. Observations and guidance parameters are fixed across columns.'}
    for name,caption in captions.items():
        figure_lines += [r'\begin{figure}[!htbp]\centering',r'\includegraphics[width=\linewidth]{figures/conditional_scaling_'+name+'.pdf}',r'\caption{'+caption+'}',r'\label{fig:conditional-scaling-'+name+'}',r'\end{figure}']
    (args.output/'conditional_scaling_figures.tex').write_text('\n'.join(figure_lines)+'\n')
    write(args.output/'conditional_scaling_final_manifest.json',dict(complete=True,physical_inputs=32,tasks=list(TASKS),K=KS,
          canonical_trajectories=96000,timed_trajectories=106944,rows=len(rows),bootstrap_resamples=100000,
          simultaneous_interval_comparisons=16,plot_font=font,figure_files=generated,source_manifests=manifests,
          plotter_sha256=digest(__file__)))
    print('COMPLETE: 32 offsets, three tasks, five K, 96000 conditional trajectories; 480 timings',flush=True)


if __name__=='__main__':main()

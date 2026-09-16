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
from matplotlib.colors import Normalize,LinearSegmentedColormap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'plot'))
from publication_style import use_times_new_roman,error_number
from run_ablation_study import digest,write
from audit_conditional_timing import build as audit_timing, save_report as save_timing
KS=[1,3,10,100,1000]
FIELDS=[('forward','u','Forward'),('inverse','a','Inverse'),('both','a','Joint'),('both','u','Joint')]
COLORS={'a':'#A77522','u':'#255F85'}
FIELD_CMAP=LinearSegmentedColormap.from_list('field_blue_gold',['#27638b','#f7f7f3','#d5a246'])
TASKS={'forward':'Forward','inverse':'Inverse','both':'Joint'}
FIGURE_WIDTH_IN=6.0
ORDINARY_FONT_PT=9.3


def figure_layout(kind):
    """Fixed-width canvases for the JMLR six-inch text block."""
    if kind=='accuracy':
        return plt.subplots(2,2,figsize=(FIGURE_WIDTH_IN,4.7),layout='constrained')
    if kind=='time':
        fig,axes=plt.subplots(1,3,figsize=(FIGURE_WIDTH_IN,2.85))
        fig.subplots_adjust(left=.105,right=.955,bottom=.235,top=.78,wspace=.39)
        fig.supxlabel('Conditional samples, $K$',y=.025,fontsize=ORDINARY_FONT_PT)
        fig.supylabel('Cumulative sampling time (s)',x=.012,fontsize=ORDINARY_FONT_PT)
        return fig,axes
    if kind=='reconstructions':
        fig,axes=plt.subplots(4,6,figsize=(FIGURE_WIDTH_IN,4.9))
        fig.subplots_adjust(left=.105,right=.865,top=.935,bottom=.035,wspace=.10,hspace=.34)
        return fig,axes
    raise ValueError(kind)


def ranked_mean(value,means):
    """Rank unrounded values; exact ties share the same distinct-value rank."""
    ordered=sorted(set(means))
    shown=error_number(value)
    if value==ordered[0]:
        return r'\mathbf{'+shown+'}'
    if len(ordered)>1 and value==ordered[1]:
        return shown+r'^{\dagger}'
    return shown


def publication_style():
    font=use_times_new_roman()
    plt.rcParams.update({'font.size':ORDINARY_FONT_PT,'axes.titlesize':ORDINARY_FONT_PT,
        'axes.labelsize':ORDINARY_FONT_PT,'xtick.labelsize':ORDINARY_FONT_PT,
        'ytick.labelsize':ORDINARY_FONT_PT,'legend.fontsize':ORDINARY_FONT_PT,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
    return font


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    p.add_argument('--figure-prefix',default='revision_0915_conditional_fixed')
    p.add_argument('--timing-results',type=Path,
                   help='Pool root for occupancy analysis; defaults to the single export parent.')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.figures.mkdir(parents=True,exist_ok=True)
    rows=[];arrays={};manifests=[]
    for folder in args.exports:
        manifest=json.loads((folder/'conditional_scaling_manifest.json').read_text())
        assert manifest['complete'] and manifest['status']=='pass' and manifest['fixed_observations_verified_all_draws'],folder
        for name,expected_hash in manifest['export_files'].items():
            assert digest(folder/name)==expected_hash
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
    assert sum(m['independent_timing_trajectories'] for m in manifests)==0
    timing_root=args.timing_results
    if timing_root is None:
        assert len(args.exports)==1,'Specify --timing-results for merged exports'
        timing_root=args.exports[0].parent
    timing_audit,timing_input_rows,timing_cohorts=audit_timing(timing_root,
        timing_root/'provenance/external_gpu_occupancy_initial.json',
        timing_root/'provenance/external_gpu_occupancy_after_handoffs.json')
    assert timing_audit['complete'] and timing_audit['completed_pools']==96
    for r in timing_input_rows:
        assert abs(r['seconds']-index[r['task'],r['offset'],r['K']]['seconds'])<1e-6
    save_timing(args.output,timing_audit,timing_input_rows,timing_cohorts)
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
                               median_ratio_to_K1=np.median(ratios)))
    write_csv(args.output/'conditional_scaling_summary.csv',summary)
    write_csv(args.output/'conditional_scaling_paired_effects.csv',effects)
    write_csv(args.output/'conditional_scaling_timing.csv',timing)
    si={(r['task'],r['field'],r['K']):r for r in summary}
    ti={(r['task'],r['K']):r for r in timing}
    font=publication_style()
    generated=[]
    def save(fig,name):
        for ext in ['pdf','png']:
            path=args.figures/f"{name.replace('conditional_scaling',args.figure_prefix)}.{ext}"
            # Preserve the exact six-inch PDF width: a tight bounding box can
            # silently shrink all text when the PDF is included at \linewidth.
            fig.savefig(path,dpi=240,bbox_inches=None,facecolor='white')
            generated.append(dict(path=str(path),sha256=digest(path)))
        plt.close(fig)
    fig,axes=figure_layout('accuracy')
    for ax,(task,field,name) in zip(axes.flat,FIELDS):
        vals=[si[task,field,k] for k in KS]
        mean=np.array([r['mean_percent'] for r in vals])
        ci=np.array([[r['mean_ci_low'] for r in vals],[r['mean_ci_high'] for r in vals]])
        ax.errorbar(KS,mean,yerr=np.vstack([mean-ci[0],ci[1]-mean]),color=COLORS[field],fmt='o-',markersize=4,lw=1.1,capsize=2)
        ax.set_xscale('log');ax.set_xticks(KS,labels=[str(k) for k in KS]);ax.minorticks_off()
        ax.set_title(name)
        ax.set_xlabel('Conditional samples, $K$')
        ax.set_ylabel(rf'$\operatorname{{RelL2}}_{{{field}}}$ (%)')
        ax.grid(axis='y',color='#DDDDDD',lw=.4);ax.set_ylim(bottom=0)
    save(fig,'conditional_scaling_accuracy')
    fig,axes=figure_layout('time')
    for ax,(task,name) in zip(axes,TASKS.items()):
        vals=[ti[task,k] for k in KS];med=np.array([r['median_seconds'] for r in vals])
        ax.plot(KS,med,'o-',color='#255F85',markersize=4,lw=1.1,label='Cumulative measured time')
        ax.fill_between(KS,[r['q25_seconds'] for r in vals],[r['q75_seconds'] for r in vals],color='#255F85',alpha=.13)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(KS,labels=[str(k) for k in KS]);ax.minorticks_off()
        ax.set_title(name)
        ax.grid(axis='y',color='#DDDDDD',lw=.4)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,1.0),
               ncols=2,frameon=False,handlelength=1.6,columnspacing=1.2)
    save(fig,'conditional_scaling_time')
    # The first predeclared evaluation offset is used without inspecting errors.
    fig,axes=figure_layout('reconstructions')
    for row,(task,field,name) in enumerate(FIELDS):
        j=0 if field=='a' else 1;key=f'{task}_1500'
        vals=[arrays[key+'_truth'][j]]+[arrays[key+f'_K{k}'][j] for k in KS]
        lo,hi=min(float(v.min()) for v in vals),max(float(v.max()) for v in vals)
        for col,(ax,v) in enumerate(zip(axes[row],vals)):
            im=ax.imshow(v,origin='lower',extent=[0,1,0,1],cmap=FIELD_CMAP,norm=Normalize(lo,hi),interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title('Truth' if col==0 else f'$K={KS[col-1]}$',fontsize=ORDINARY_FONT_PT,pad=5)
            if col==0:
                ax.set_ylabel(name+'\n'+rf'$\mathbf{{{field}}}$',fontsize=ORDINARY_FONT_PT)
            else:
                err=index[task,1500,KS[col-1]][f'rel_l2_{field}']*100
                ax.set_xlabel(f'{err:.2f}%',fontsize=ORDINARY_FONT_PT,labelpad=2)
        pos=axes[row,-1].get_position()
        cax=fig.add_axes([.89,pos.y0,.014,pos.height])
        cb=fig.colorbar(im,cax=cax,format='%.2g');cb.ax.tick_params(labelsize=ORDINARY_FONT_PT,pad=2,width=.4);cb.outline.set_linewidth(.4)
        cb.ax.set_title(rf'$\mathbf{{{field}}}$',fontsize=ORDINARY_FONT_PT,pad=3)
        cb.locator=matplotlib.ticker.MaxNLocator(nbins=3);cb.update_ticks()
    save(fig,'conditional_scaling_reconstructions')
    lines=[r'\begin{table}[!htbp]\centering\small',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}rcccc@{}}\toprule',
        r'$K$ & Forward & Inverse & Joint & Joint \\',
        r' & $\operatorname{RelL2}_u$ & $\operatorname{RelL2}_a$ & $\operatorname{RelL2}_a$ & $\operatorname{RelL2}_u$ \\\midrule']
    for k in KS:
        cells=[]
        for t,f,_ in FIELDS:
            r=si[t,f,k];mean=ranked_mean(r['mean_percent'],[si[t,f,j]['mean_percent'] for j in KS])
            cells.append('$'+mean+r'\,\pm\,'+error_number(r['sd_percent'])+'$')
        lines.append(str(k)+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Conditional-sample averaging for Poisson over 32 ID inputs. Relative errors are mean $\pm$ sample standard deviation in percent. Boldface marks the lowest mean and $\dagger$ the second-lowest distinct mean within each field. Ranking uses unrounded values; exact ties share a rank.}',
        r'\label{tab:conditional-scaling}',r'\end{table}']
    (args.output/'conditional_scaling_table.tex').write_text('\n'.join(lines)+'\n')
    figure_lines=[]
    captions={
      'accuracy':r'Poisson reconstruction error versus the number $K$ of averaged conditional samples. Means and pointwise 95\% bootstrap intervals are computed over the same 32 ID inputs. The same 500 observations per observed field are used for all draws of an input and task. Every draw uses 100 stochastic Euler steps; the physical-field average $\overline{\mathbf z}_K$ is formed before evaluating $\operatorname{RelL2}_a$ or $\operatorname{RelL2}_u$.',
      'time':r'Cumulative sampling time for averaged Poisson estimates under fixed observations. Each input and task uses one sequence of 1000 predictions; a mean is evaluated when the first $K$ predictions are available. Curves show medians over 32 inputs and shaded bands the interquartile range. Batches contain at most 64 samples on an A800 GPU and end at the reported values of $K$. Times include generation, physical-field conversion, transfer, and all preceding prefix averages; model loading and file I/O are excluded. All values of $K$ are measured along the same nested sequence, rather than through separate sampling runs.',
      'reconstructions':r'Poisson conditional-sample averages for the first input in the evaluation cohort. Columns compare the reference fields with the corresponding components of $\overline{\mathbf z}_K$ for $K=1,3,10,100,1000$. The four rows show forward $\mathbf{u}$, inverse $\mathbf{a}$, and joint $\mathbf{a}$ and $\mathbf{u}$. Colors share one scale within each row. Labels below reconstructed fields give $\operatorname{RelL2}_a$ or $\operatorname{RelL2}_u$, according to the field, in percent. Observations and guidance parameters are fixed across columns.'}
    captions['time'] += ' Measurements use A800 GPUs, with concurrent workloads in some runs.'
    for name,caption in captions.items():
        figure_lines += [r'\begin{figure}[!htbp]\centering',r'\includegraphics[width=\linewidth]{figures/'+args.figure_prefix+'_'+name+'.pdf}',r'\caption{'+caption+'}',r'\label{fig:conditional-scaling-'+name+'}',r'\end{figure}']
    (args.output/'conditional_scaling_figures.tex').write_text('\n'.join(figure_lines)+'\n')
    (args.output/'settings.tex').write_text(r'''We evaluate conditional-sample averaging on 32 Poisson ID inputs (indices 1500--1531) for forward, inverse, and joint recovery. For each input and task, one set of 500 observations per observed field, including their values, is held fixed across all draws. A single pool of 1000 predictions is generated using 100 stochastic Euler steps per draw, and each estimate is the physical-space arithmetic mean of the first $K\in\{1,3,10,100,1000\}$ predictions. Guidance parameters remain fixed. Uncertainty intervals resample the 32 physical inputs. Sampling times are cumulative costs measured when each prefix estimate becomes available.'''+'\n')
    write(args.output/'conditional_scaling_final_manifest.json',dict(complete=True,physical_inputs=32,tasks=list(TASKS),K=KS,
          canonical_trajectories=96000,timed_trajectories=96000,cumulative_prefix_timings=480,timing_mode='cumulative_prefix',rows=len(rows),bootstrap_resamples=100000,
          simultaneous_interval_comparisons=16,plot_font=font,figure_width_inches=FIGURE_WIDTH_IN,
          ordinary_font_points=ORDINARY_FONT_PT,figure_files=generated,source_manifests=manifests,
          field_colormap='field_blue_gold',
          timing_occupancy_audit_sha256=digest(args.output/'conditional_timing_audit.json'),
          timing_occupancy_summary=timing_audit['task_impact'],
          plotter_sha256=digest(__file__)))
    print('COMPLETE: 32 offsets, three tasks, five K, 96000 conditional trajectories; 480 cumulative prefix timings',flush=True)


if __name__=='__main__':main()

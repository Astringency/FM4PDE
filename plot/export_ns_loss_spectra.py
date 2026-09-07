"""Tables and plots for the complete paired NS loss-exchange study.

--preview permits incomplete engineering plots, stamped as incomplete and
kept separate from manuscript exports. Production requires all 1,728 calls.
"""
from __future__ import annotations
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_style import use_times_new_roman
from spectral_diagnostics import paired_bootstrap
from run_ns_loss_study import sha,write

TASKS=['forward','inverse','both']
SETTINGS=[('FM4PDE',100),('DiffusionPDE',100),('DiffusionPDE',1000)]
BASELINES=['RecFNO','Senseiver','VoronoiCNN']
COLORS=['#256493','#aa4b32','#c3943b','#5a8055','#805f95','#697c85']


def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(values):
    x=np.asarray(values,float)
    if len(x)==0:return dict(n=0,mean=None,sd=None,ci_low=None,ci_high=None)
    ci=paired_bootstrap(x) if len(x)>1 else [None,None]
    return dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else None,
                ci_low=float(ci[0]) if ci[0] is not None else None,ci_high=float(ci[1]) if ci[1] is not None else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preview',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    m=json.loads((args.audit/'ns_audit_manifest.json').read_text())
    if not args.preview:
        assert m['status']=='complete' and m['calls_verified']==1728
        assert m['outcome_counts']=={'complete':1728}, 'Numerical failures require an explicit outcome table; do not silently drop them.'
    for name,h in m['outputs'].items():assert sha(args.audit/name)==h,name
    assert sha(args.inputs/'source.json')==m['source_sha256']
    source=json.loads((args.inputs/'source.json').read_text());ids=source['evaluation_ids']
    rows=list(csv.DictReader((args.audit/'ns_loss_per_run.csv').open()))
    reference_rows=list(csv.DictReader((args.audit/'ns_reference_residuals.csv').open()))
    groups=defaultdict(list)
    for r in rows:groups[r['task'],r['method'],int(r['steps']),r['exchange']=='True',int(r['sample_id'])].append(r)
    summaries=[];paired=[];per_input={}
    metrics=['rel_l2_a','rel_l2_u','primary','secant_rms','spatial_loss','obs_rel_l2_a','obs_rel_l2_u']
    def value(r,k):
        if k=='primary':return float(r['rel_l2_u']) if r['task']=='forward' else float(r['rel_l2_a']) if r['task']=='inverse' else max(float(r['rel_l2_a']),float(r['rel_l2_u']))
        if k=='secant_rms':return np.sqrt(float(r['fm_residual_mse_f64']))
        if k=='spatial_loss':return float(r['diffusion_spatial_loss_f64'])
        return float(r[k]) if r[k] else None
    for t in TASKS:
        for method,n in SETTINGS:
            for e in [False,True]:
                for metric in metrics:
                    vals={}
                    for i in ids:
                        rr=groups[t,method,n,e,i]
                        if len(rr)!=3:
                            assert args.preview;continue
                        assert {int(r['seed']) for r in rr}=={0,1,2}
                        vv=[value(r,metric) for r in rr]
                        if all(v is not None for v in vv):vals[i]=float(np.mean(vv))
                    per_input[t,method,n,e,metric]=vals
                    summaries.append(dict(task=t,method=method,steps=n,exchange=e,metric=metric,**summarize(list(vals.values()))))
            for metric in metrics:
                original=per_input[t,method,n,False,metric];exchanged=per_input[t,method,n,True,metric]
                common=sorted(original.keys()&exchanged.keys())
                if common:paired.append(dict(task=t,method=method,steps=n,metric=metric,**summarize([exchanged[i]-original[i] for i in common])))
    csvwrite(args.output/'ns_loss_summary.csv',summaries);csvwrite(args.output/'ns_loss_paired_effects.csv',paired)
    def stat(t,method,n,e,metric):return next(r for r in summaries if (r['task'],r['method'],r['steps'],r['exchange'],r['metric'])==(t,method,n,e,metric))
    def display(r,factor=1):
        if r['mean'] is None:return '---'
        return f"${factor*r['mean']:.2f}\\pm{factor*r['sd']:.2f}$" if r['sd'] is not None else f"${factor*r['mean']:.2f}$"
    tex=[]
    for t in TASKS:
        tex += [r'\begin{table}[!htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
            r'\caption{NS '+('joint' if t=='both' else t)+r' loss exchange on 32 common inputs and three seeds. Field errors are percentages (mean $\pm$ SD over input-level seed averages). $L_F$ is endpoint-secant MSE; $L_D$ is the original directional spatial loss. $R_F$ is mean endpoint-secant RMS and $L_D^{\rm final}$ is the common final spatial diagnostic. Weights and solvers remain fixed within recipient pairs.}',
            r'\label{tab:ns-loss-'+t+'}',r'\begin{tabular}{@{}llrrrrr@{}}\toprule',
            r'Method & Steps / loss & $e_a$ (\%) & $e_u$ (\%) & $R_F$ & $L_D^{\rm final}$ & $n$ \\\midrule']
        for method,n in SETTINGS:
            for e in [False,True]:
                loss='F' if (method=='FM4PDE')!=e else 'D'
                ra,ru=stat(t,method,n,e,'rel_l2_a'),stat(t,method,n,e,'rel_l2_u')
                residual=stat(t,method,n,e,'secant_rms')['mean'];spatial=stat(t,method,n,e,'spatial_loss')['mean']
                vals=[method,f'{n} / $L_{loss}$',display(ra,100),display(ru,100),f'{residual:.4f}' if residual is not None else '---',f'{spatial:.3g}' if spatial is not None else '---',str(ra['n'])]
                tex.append(' & '.join(vals)+r' \\')
        tex += [r'\bottomrule\end{tabular}',r'\end{table}']
    if args.preview:tex.insert(0,'% INCOMPLETE ENGINEERING PREVIEW: not a manuscript result.')
    (args.output/'ns_loss_tables.tex').write_text('\n'.join(tex)+'\n')
    with gzip.open(args.audit/'ns_frequency_records.json.gz','rt') as f:freq=json.load(f)
    powers=['reference_power','prediction_power','error_power']
    fg=defaultdict(list);bands=[]
    for r in freq:
        for k in powers:r[k]=np.asarray(r[k],float)
        fg[r['task'],r['field'],r['method'],r['steps'],r['exchange']].append(r)
        for label,bb in [('8/32',r['bands'])]+list(r['sensitivity_bands'].items()):
            for name,v in bb.items():bands.append(dict(task=r['task'],field=r['field'],method=r['method'],steps=r['steps'],exchange=r['exchange'],sample_id=r['sample_id'],seed=r['seed'],cutoffs=label,band=name,**v))
    csvwrite(args.output/'ns_frequency_bands.csv',bands)
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False})
    font=use_times_new_roman();outputs=[]
    def save(fig,name):
        if args.preview:fig.text(.5,.5,'INCOMPLETE PREVIEW',fontsize=30,color='crimson',alpha=.25,ha='center',rotation=20)
        for ext in ['pdf','png']:
            out=args.output/(name+'.'+ext);fig.savefig(out,dpi=190,bbox_inches='tight');outputs.append(out)
        plt.close(fig)
    fields=np.load(args.audit/'ns_example_fields.npz')
    for task,field in [('forward','u'),('inverse','a'),('both','a'),('both','u')]:
        task_label='joint' if task=='both' else task
        field_label='Initial vorticity ($a$)' if field=='a' else 'Final vorticity ($u$)'
        fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
        settings=SETTINGS+[(b,1) for b in BASELINES]
        truth_drawn=False;counts=[]
        for j,(method,n) in enumerate(settings):
            rr=fg[task,field,method,n,False];per=defaultdict(list)
            for r in rr:per[r['sample_id']].append(r)
            expected=3 if method in ['FM4PDE','DiffusionPDE'] else 1
            per={i:rs for i,rs in per.items() if len(rs)==expected}
            if not per:continue
            assert len(per)==32 or args.preview
            curves={k:np.stack([np.mean([r[k]/r['reference_total'] for r in rs],axis=0) for rs in per.values()]) for k in powers}
            radius=np.arange(curves['error_power'].shape[1]);label=method+(f' ({n})' if n>1 else '')
            for ax,k in zip(axes,['prediction_power','error_power']):
                ax.plot(radius[1:],curves[k].mean(0)[1:],color=COLORS[j],lw=1.4,ls=['-','--','-.',':','--',':'][j],label=label)
            if len(per)>1:
                ci=paired_bootstrap(curves['error_power']);axes[1].fill_between(radius[1:],ci[0,1:],ci[1,1:],color=COLORS[j],alpha=.1,lw=0)
            if not truth_drawn:
                axes[0].plot(radius[1:],curves['reference_power'].mean(0)[1:],color='#222222',label='Ground truth',lw=1.7);truth_drawn=True
            counts.append(len(per))
        for ax,title,ylabel in zip(axes,['Reference and prediction energy','Reconstruction error energy'],['Shell energy / full reference energy','Shell error energy / full reference energy']):
            ax.set(xscale='log',yscale='log',xlabel='Radial Fourier-mode index',title=title)
            ax.set_ylabel(ylabel);ax.grid(alpha=.2)
            for cut in [8,32]:ax.axvline(cut,color='.6',lw=.6)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9,frameon=False)
        fig.suptitle(f'NS {task_label} · {field_label} · original PDE losses\n32 common inputs; 3 generative seeds; shading: 95% input-bootstrap CI',fontsize=12)
        save(fig,f'ns_spectrum_{task}_{field}')
        truth_key=f'{task}_truth_{field}'
        if truth_key not in fields:
            assert args.preview;continue
        truth=fields[truth_key];names=[];preds=[];meta=[]
        for method,n in SETTINGS:
            for e in [False,True]:
                key=f'{task}_{method}_{n}_{int(e)}_{field}'
                pred=fields[key] if key in fields else None
                assert pred is not None or args.preview
                loss='F' if (method=='FM4PDE')!=e else 'D'
                names.append(f'{method} {n}\n$L_{loss}$');preds.append(pred);meta.append((method,n,e))
        actual=[v for v in preds if v is not None];vmax=max(np.max(np.abs(v)) for v in [truth]+actual)
        emax=max([np.max(np.abs(v-truth)) for v in actual]+[1e-12])
        fig,axes=plt.subplots(2,7,figsize=(10,4.4),layout='constrained')
        im=axes[0,0].imshow(truth,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax);axes[0,0].set_title('Ground truth',fontsize=10)
        axes[1,0].imshow(fields[f'{task}_mask_{field}'],origin='lower',cmap='Greys',vmin=0,vmax=1);axes[1,0].set_title('Observed mask',fontsize=10)
        gt_residual=np.sqrt(float(next(r for r in reference_rows if int(r['sample_id'])==ids[0])['fm_residual_mse_f64']))
        axes[1,0].set_xlabel(f'True-pair $R_F$\n{gt_residual:.4f}',fontsize=10)
        err=None
        for j,(name,pred) in enumerate(zip(names,preds),1):
            axes[0,j].set_title(name,fontsize=10)
            if pred is None:
                for ax in axes[:,j]:ax.text(.5,.5,'Pending',ha='center',va='center',transform=ax.transAxes)
                continue
            axes[0,j].imshow(pred,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
            err=axes[1,j].imshow(np.abs(pred-truth),origin='lower',cmap='magma',vmin=0,vmax=emax)
            rel=np.linalg.norm(pred-truth)/np.linalg.norm(truth)
            method,n,e=meta[j-1]
            r=next(r for r in rows if (r['task'],r['method'],int(r['steps']),r['exchange']=='True',int(r['sample_id']),int(r['seed']))==(task,method,n,e,ids[0],0))
            residual=np.sqrt(float(r['fm_residual_mse_f64']))
            axes[1,j].set_xlabel(f'Error {100*rel:.2f}%\n$R_F$ = {residual:.4f}',fontsize=10)
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=axes[0,:],orientation='horizontal',shrink=.65,aspect=45)
        if err is not None:fig.colorbar(err,ax=axes[1,1:],orientation='horizontal',shrink=.7,aspect=45)
        fig.suptitle(f'NS {task_label} · {field_label} · predeclared input {ids[0]}, seed 0\nTop: truth and prediction; bottom: absolute error; shared scales within rows',fontsize=12)
        save(fig,f'ns_loss_fields_{task}_{field}')
    write(args.output/'ns_report_manifest.json',dict(status='preview' if args.preview else 'complete',calls_verified=m['calls_verified'],
        protocol_sha256=m['protocol_sha256'],audit_manifest_sha256=sha(args.audit/'ns_audit_manifest.json'),font_path=font,
        source_script_sha256=sha(Path(__file__)),outputs={f.name:sha(f) for f in args.output.iterdir() if f.is_file() and f.name!='ns_report_manifest.json'}))


if __name__=='__main__':main()

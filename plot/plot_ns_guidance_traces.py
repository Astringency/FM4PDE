"""Report the measured scale of original and exchanged NS physical guidance."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.audit/'ns_audit_manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['calls_verified']==1728
    path=args.audit/'ns_guidance_traces.json.gz';assert sha(path)==manifest['outputs'][path.name]
    with gzip.open(path,'rt') as f:rows=json.load(f)
    by_run=defaultdict(list)
    for r in rows:by_run[r['task'],r['method'],r['steps'],r['exchange'],r['sample_id'],r['seed']].append(r)
    assert len(by_run)==1728
    per_input=defaultdict(list);summary_rows=[];run_rows=[]
    for key,rr in by_run.items():
        rr.sort(key=lambda r:r['step']);assert len(rr)==key[2]
        assert [r['step'] for r in rr]==list(range(key[2]))
        active=[r for r in rr if r['zeta_pde']>0]
        assert active
        ratios=np.array([r['component_norm_ratio'] for r in active],float)
        assert np.isfinite(ratios).all()
        full=np.array([r['component_norm_ratio'] or 0. for r in rr],float)
        assert np.isfinite(full).all()
        per_input[key[:-1]].append(full)
        run_rows.append(dict(task=key[0],method=key[1],steps=key[2],exchange=key[3],sample_id=key[4],seed=key[5],
            active_steps=len(active),active_mean_component_ratio=float(ratios.mean()),active_median_component_ratio=float(np.median(ratios)),
            final_component_ratio=float(ratios[-1]),active_mean_physical_norm=float(np.mean([r['weighted_pde_norm'] for r in active])),
            active_mean_observation_norm_sum=float(np.mean([r['sum_weighted_observation_norms'] for r in active]))))
    del rows,by_run
    curves=defaultdict(list)
    for key,v in per_input.items():
        assert len(v)==3
        curves[key[:-1]].append(np.mean(v,axis=0))
    plt.rcParams.update({'font.size':11,'axes.titlesize':11,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False})
    font=use_times_new_roman()
    fig,axes=plt.subplots(3,3,figsize=(9,7),layout='constrained',sharex=True)
    for row,task in enumerate(['forward','inverse','both']):
        for col,(method,n) in enumerate([('FM4PDE',100),('DiffusionPDE',100),('DiffusionPDE',1000)]):
            ax=axes[row,col]
            for e,color,style in [(False,'#28628F','-'),(True,'#B76738','--')]:
                x=np.stack(curves[task,method,n,e]);assert x.shape==(32,n)
                median=np.median(x,axis=0);lo,hi=np.quantile(x,[.25,.75],axis=0)
                active=median>0;t=np.arange(n)/n
                loss='F' if (method=='FM4PDE')!=e else 'D'
                ax.plot(t[active],median[active],color=color,ls=style,lw=1.4,label=f'$L_{loss}$')
                ax.fill_between(t[active],lo[active],hi[active],color=color,alpha=.15,lw=0)
                rr=[r for r in run_rows if (r['task'],r['method'],r['steps'],r['exchange'])==(task,method,n,e)]
                ids=sorted({r['sample_id'] for r in rr});assert len(ids)==32
                vals=np.array([np.mean([r['active_mean_component_ratio'] for r in rr if r['sample_id']==i]) for i in ids])
                ci=paired_bootstrap(vals)
                summary_rows.append(dict(task=task,method=method,steps=n,exchange=e,n=32,
                    mean_active_component_ratio=float(vals.mean()),sd_active_component_ratio=float(vals.std(ddof=1)),ci_low=float(ci[0]),ci_high=float(ci[1])))
            ax.set(yscale='log',xlim=(.77,1.0),title=f'{method}, {n} steps · {task}')
            if row==2:ax.set_xlabel('Step index / step budget')
            if col==0:ax.set_ylabel('Weighted component-norm ratio')
            ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=10)
    fig.suptitle('Scale of physical guidance under fixed recipient weights\nMedian and interquartile range of 32 input-level seed averages',fontsize=12)
    for ext in ['pdf','png']:fig.savefig(args.output/f'ns_guidance_component_ratios.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    for name,data in [('ns_guidance_per_run.csv',run_rows),('ns_guidance_summary.csv',summary_rows)]:
        with (args.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(args.output/'ns_guidance_plot_manifest.json',dict(status='complete',calls_verified=1728,font_path=font,
        trace_source_sha256=sha(path),script_sha256=sha(Path(__file__)),
        definition='Weighted PDE-component norm divided by the sum of weighted observation-component norms before common global clipping; not a norm of summed vectors or final updates.',
        uncertainty='Plot: pointwise median and IQR across 32 seed-averaged inputs. Summary CSV: mean, SD and 95% paired-input bootstrap interval for the active-step mean.',
        outputs={f.name:sha(f) for f in args.output.iterdir() if f.is_file() and f.name!='ns_guidance_plot_manifest.json'}))


if __name__=='__main__':main()

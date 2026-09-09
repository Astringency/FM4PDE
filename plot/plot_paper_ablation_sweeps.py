"""Complete fieldwise plots of the single-input ablation factor sweeps.

The publication snapshot is authoritative: no values are inferred from maps,
rounded LaTeX, or unfinished reruns. Every planned PDE and every factor level
is checked before plotting. The manifest records all source record IDs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_paper_ablation_revision import PDES, digest
from export_paper_seed_ensemble import NAMES
from publication_style import use_times_new_roman

PHASES=[('stochastic','S','#27638b','o','-'),('deterministic','D','#282828','s','--'),
        ('hybrid_d2s','D → S','#bf922f','^',':'),('hybrid_s2d','S → D','#27638b','v','-.')]


def main(args):
    records=json.loads((args.source/'records.json').read_text())
    assert json.loads((args.source/'manifest.json').read_text())['final_ready']
    use_times_new_roman()
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,
        'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.5,
        'text.color':'#282828','axes.labelcolor':'#282828','savefig.facecolor':'white'})
    args.output.mkdir(parents=True,exist_ok=True)
    inventory=[]; captions=[]

    def select(pde,group,**filters):
        found=[r for r in records if r['pde']==pde and r['config']['task']=='both'
            and r['config']['ablation_group']==group
            and all(r['config'].get(k)==v for k,v in filters.items())]
        assert len(found)==1,(pde,group,filters,len(found))
        assert found[0]['source']!='pending'
        return found[0]

    families=[
        ('loss_state','loss_state_by_sampler','loss_state',['xt','x_next','endpoint'],
         [r'$x_t$',r'$\widetilde{x}_{k+1}$','Endpoint'],PHASES[:2],
         'Loss evaluation state','Guidance loss evaluation on ID input 0 for each PDE. The current state, the next proposal, and the predicted endpoint are compared at 100 steps, for stochastic (S) and deterministic (D) sampling.'),
        ('switch','sampler_phase','switch_ratio',[.2,.5,.8],['0.2','0.5','0.8'],PHASES[2:],
         'Switch flow time','Hybrid switch times on ID input 0 for each PDE, using 100 steps. Both deterministic-to-stochastic and stochastic-to-deterministic orderings are evaluated at flow times 0.2, 0.5, and 0.8.'),
        ('grid','time_grid_by_sampler','time_grid',['uniform','cosine','geometric'],['Uniform','Cosine','Geometric'],PHASES,
         'Flow-time grid','Uniform, cosine, and geometric flow-time grids on ID input 0 for each PDE. Each sampler uses 100 steps and the same guidance parameters within the sweep.'),
        ('integrator','step_method_by_sampler','step_method',['euler','midpoint'],['Euler','Midpoint'],PHASES,
         'Numerical update','Euler and midpoint updates on ID input 0 for each PDE, using 100 steps. S and D denote stochastic and deterministic sampling; both hybrid orderings are included.'),
        ('layout','sensor_mode','sensor_mode',['random','per_sample_random','fixed','grid','sensor_column'],
         ['Random','Per-input','Fixed','Grid','Columns'],PHASES[:1],
         'Observation layout','Observation layouts on ID input 0 for each PDE, using 100 stochastic Euler steps. Random, per-input random, fixed, and grid layouts use 500 locations per field; five complete sensor columns use 640 locations.'),
        ('noise','noise_robustness','noise_level',[0.,.01,.05,.1],['0','0.01','0.05','0.10'],PHASES[:1],
         'Noise standard-deviation level','Observation-noise levels on ID input 0 for each PDE, using 500 locations per field and 100 stochastic Euler steps. Errors are evaluated against the clean reference; noise amplitudes are relative to the prescribed physical field scale.'),
        ('seed','statistics_stability','sample_seed',list(range(5)),[str(i) for i in range(5)],PHASES[:1],
         'Inference seed','Inference and mask sensitivity on ID input 0 for each PDE. Inference seeds 0--4 are paired with mask seeds 0, 11, 22, 33, and 44; each reconstruction uses 500 observations per field and 100 stochastic Euler steps.'),
        ('temporal','temporal_residual_mode','residual_mode',['endpoint_secant','hermite_bridge','near_endpoint_temporal'],
         ['Secant','Hermite','Near-endpoint'],PHASES[:1],
         'Temporal residual','Temporal-residual comparisons on ID input 0 for the six endpoint-pair evolution equations, using 100 stochastic Euler steps and 500 endpoint observations per field. The near-endpoint condition additionally observes auxiliary physical-time slices and therefore changes the available information.'),
    ]
    for name,group,key,levels,labels,phases,xlabel,caption in families:
        pdes=PDES if name!='temporal' else ['nsnonbounded','reaction_diffusion','shallow_water','heat','wave','advection_diffusion']
        for start in range(0,len(pdes),3):
            subset=pdes[start:start+3]
            fig,axes=plt.subplots(len(subset),2,figsize=(6,2.15*len(subset)+.3),squeeze=False,layout='constrained')
            used=[]
            for i,pde in enumerate(subset):
                for j,field in enumerate(['a','u']):
                    ax=axes[i,j]
                    if pde=='burger' and field=='a':ax.set_axis_off();continue
                    all_values=[]
                    for phase,label,color,marker,style in phases:
                        selected=[select(pde,group,**{key:level},**({'sampler_phase':phase} if name in {'loss_state','switch','grid','integrator'} else {})) for level in levels]
                        values=[100*r['rel_l2_'+field] for r in selected]
                        all_values.extend(values)
                        assert np.isfinite(values).all(),(group,pde,field)
                        used.extend(r['id'] for r in selected)
                        ax.plot(range(len(levels)),values,color=color,marker=marker,ls=style,lw=1,ms=4,
                                markerfacecolor='white' if style!='-' else color,label=label)
                    ax.set_xticks(range(len(levels)),labels,rotation=25 if name=='layout' else 0,
                                  ha='right' if name=='layout' else 'center')
                    ax.set(title=NAMES[pde].replace('--','–')+f' · ${field}$',
                           xlabel=xlabel,ylabel=rf'$\mathrm{{RelL2}}_{field}$ (%)',yscale='log')
                    # Do not magnify negligible numerical changes into a steep curve.
                    if min(all_values)>0 and max(all_values)/min(all_values)<1.25:
                        ax.set_ylim(min(all_values)*.9,max(all_values)*1.1)
                    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18,lw=.5)
                    if i==0 and j==1 and len(phases)>1:
                        handles,legend_labels=ax.get_legend_handles_labels()
                        fig.legend(handles,legend_labels,ncol=len(phases),fontsize=9,frameon=False,loc='outside upper center')
            stem=f'ablation_{name}_sweep_{start//3+1}'
            fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
            hashes={}
            for ext in ['pdf','png']:
                path=args.output/(stem+'.'+ext);fig.savefig(path,dpi=210,bbox_inches='tight',pad_inches=.02);hashes[path.name]=digest(path)
            plt.close(fig)
            fullcaption=caption+' Coefficient and solution errors are reported separately, including all vector components; Burgers is scored over its complete trajectory. Lines connect the tested conditions, and no population uncertainty is implied.'
            captions.extend([r'\begin{figure}[!htbp]',rf'\centering\includegraphics[width=\linewidth]{{figures/{stem}.pdf}}',
                r'\caption{'+fullcaption+'}',rf'\label{{fig:{stem.replace("_","-")}}}',r'\end{figure}',''])
            inventory.append(dict(stem=stem,group=group,pdes=subset,levels=levels,source_ids=sorted(set(used)),outputs=hashes))
    (args.output/'ablation_sweeps.tex').write_text('\n'.join(captions))
    (args.output/'sweep_figure_manifest.json').write_text(json.dumps(dict(
        source_sha256=digest(args.source/'records.json'),plotter_sha256=digest(Path(__file__)),
        figures=inventory,font_size=9,scope='ID input 0; joint fields or Burgers trajectory'),indent=2)+'\n')
    print('PLOTTED',len(inventory),'complete sweep figures',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())

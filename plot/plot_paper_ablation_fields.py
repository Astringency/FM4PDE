"""Plot main-ablation physical fields and separate a/u errors for the paper.

Chart contract: audit/paper_revision_20260908/ABLATION_CHART_CONTRACT.md.
Every reconstruction is the main ID sample, not a 32-ID repeated control.
Whole vector-field errors include every channel; component maps are shown
individually. All displayed values share full color ranges within each row.
"""
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
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.specs import get_pde_spec
from publication_style import use_times_new_roman
from export_paper_seed_ensemble import NAMES
from run_paper_ablation_revision import PDES,digest

BLUE='#27638b'; GOLD='#bf922f'; INK='#282828'
FIELD_CMAP=LinearSegmentedColormap.from_list('field_blue_gold',[BLUE,'#f7f7f3','#d5a246'])
ERROR_CMAP=LinearSegmentedColormap.from_list('error_blue',['#ffffff','#bed0df',BLUE,'#143747'])


def main(args):
    torch.set_num_threads(2)
    use_times_new_roman()
    plt.rcParams.update({'font.size':8.5,'axes.titlesize':8.5,'axes.labelsize':8.5,
                         'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.5,
                         'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,
                         'savefig.facecolor':'white'})
    manifest=json.loads((args.source/'manifest.json').read_text())
    assert manifest['final_ready'] or args.development
    records=json.loads((args.source/'records.json').read_text())
    for pde in args.pdes:
        assert not any(r['source']=='pending' for r in records if r['pde']==pde), (pde,'rerun incomplete')
    args.output.mkdir(parents=True,exist_ok=True)
    cache={}; metric_rows=[]; sources={}; outputs={}; captions=[]

    def select(pde,group,**query):
        rows=[r for r in records if r['pde']==pde and r['config']['task']=='both' and
              r['config']['ablation_group']==group and all(r['config'][k]==v for k,v in query.items())]
        assert len(rows)==1,(pde,group,query,len(rows))
        return rows[0]

    def load(record):
        ident=record['id']
        if ident not in cache:
            path=Path(record['result_path'])
            payload=torch.load(path,map_location='cpu',weights_only=False)
            masks=torch.load(path.parent/'masks.pt',map_location='cpu',weights_only=False)
            data={}
            for field,prefix in [('a','coef'),('u','sol')]:
                truth=payload[prefix+'_ground_truth'][0].double().numpy()
                pred=payload[prefix+'_final'][0].double().numpy()
                mask=masks[prefix][0].double().numpy()
                assert truth.shape==pred.shape==mask.shape and truth.shape[-2:]==(128,128)
                assert np.isfinite(pred).all(),ident
                norm=np.linalg.norm(truth)
                error=np.linalg.norm(pred-truth)/max(norm,1e-12)
                assert np.isclose(error,record['rel_l2_'+field],rtol=3e-6,atol=3e-8),(ident,field,error,record['rel_l2_'+field])
                obs_norm=np.linalg.norm(truth*mask)
                observed=np.linalg.norm((pred-truth)*mask)/obs_norm if obs_norm>1e-14 else None
                data[field]=dict(truth=truth,pred=pred,mask=mask,error=error,observed=observed)
                if record['pde']!='burger' or field=='u':
                    metric_rows.append(dict(id=ident,pde=record['pde'],field=field,
                        error_percent=100*error,observed_error_percent=100*observed if observed is not None else '',
                        channels=truth.shape[0],result_sha256=digest(path)))
            cache[ident]=data;sources[str(path)]=digest(path)
        return cache[ident]

    def save(fig,stem,caption):
        fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
        for ext in ['pdf','png']:
            path=args.output/(stem+'.'+ext)
            fig.savefig(path,dpi=210,bbox_inches='tight')
            outputs[path.name]=digest(path)
        plt.close(fig)
        label='fig:'+stem.replace('_','-')
        captions.extend([r'\begin{figure}[!htbp]',
            rf'\centering\includegraphics[width=\linewidth]{{figures/{stem}.pdf}}',
            r'\caption{'+caption+'}',rf'\label{{{label}}}',r'\end{figure}',''])

    def compare(rows,field):
        values=[load(r)[field] for r in rows]
        truth,mask=values[0]['truth'],values[0]['mask']
        assert all(np.array_equal(v['truth'],truth) and np.array_equal(v['mask'],mask) for v in values)
        return truth,values

    def map_axis(ax,array,norm,cmap,title='',label=''):
        im=ax.imshow(array,origin='lower',interpolation='nearest',norm=norm,cmap=cmap)
        ax.set(xticks=[],yticks=[])
        if title:ax.set_title(title,pad=4)
        if label:ax.set_xlabel(label,labelpad=3)
        return im

    def metric_label(field,data):
        label=f'$e_{field}={100*data["error"]:.2f}\\%$'
        if data['observed'] is not None:
            label+=f'\n$e_{{{field},\\mathrm{{obs}}}}={100*data["observed"]:.2f}\\%$'
        return label

    def colorbar(fig,im,axes):
        bar=fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.05,pad=.035,aspect=40,shrink=.88)
        bar.locator=MaxNLocator(4)
        bar.formatter=ScalarFormatter(useMathText=False)
        bar.formatter.set_powerlimits((-2,3));bar.update_ticks()
        bar.ax.tick_params(length=2,pad=1,labelsize=7)

    def components(pde):
        spec=get_pde_spec(pde)
        return [(f,k,name) for f,names in ([('u',spec.sol_channel_names)] if pde=='burger' else [('a',spec.coef_channel_names),('u',spec.sol_channel_names)])
                for k,name in enumerate(names)]

    def field_scale(arrays):
        low=min(x.min() for x in arrays);high=max(x.max() for x in arrays)
        if low<0<high:
            return TwoSlopeNorm(vmin=low,vcenter=0,vmax=high),FIELD_CMAP
        return Normalize(low,high),ERROR_CMAP

    def field_label(pde,field,component):
        if pde=='burger':return r'Trajectory $u$'
        spec=get_pde_spec(pde)
        count=spec.coef_channels if field=='a' else spec.sol_channels
        return f'${field}_{component+1}$' if count>1 else f'${field}$'

    # Every PDE appears in the phase figures, with all components of a and u.
    for pde in args.pdes:
        rows=[select(pde,'sampler_phase',sampler_phase=phase,**({'switch_ratio':.2} if 'hybrid' in phase else {}))
              for phase in ['stochastic','deterministic','hybrid_d2s','hybrid_s2d']]
        parts=components(pde)
        batches=[parts] if len(parts)<=4 else [parts[:3],parts[3:]]
        for part_index,part in enumerate(batches):
            fig,axes=plt.subplots(len(part),5,figsize=(6.2,1.65*len(part)+.35),squeeze=False,layout='constrained')
            for i,(field,component,name) in enumerate(part):
                truth,values=compare(rows,field)
                arrays=[truth[component]]+[v['pred'][component] for v in values]
                norm,cmap=field_scale(arrays)
                for j,array in enumerate(arrays):
                    im=map_axis(axes[i,j],array,norm,cmap,
                        ['Truth','S','D','D → S','S → D'][j] if i==0 else '',
                        metric_label(field,values[j-1]) if j else '')
                axes[i,0].set_ylabel(field_label(pde,field,component),fontsize=8)
                colorbar(fig,im,axes[i,:])
            fig.suptitle(NAMES[pde].replace('--','–')+' · sampler phases at 100 steps',fontsize=10)
            stem='ablation_phase_reconstruction_'+pde+(f'_{part_index+1}' if len(batches)>1 else '')
            caption=(NAMES[pde]+' sampler phases on the main ID sample. S and D denote stochastic and deterministic updates; both hybrids switch at flow time 0.2. '
                     r'Labels give the relative error of the entire field, $e_a$ or $e_u$, and its observed values, in percent. '
                     'Each component is shown separately, with a common color range across settings within its row. ')
            if pde=='burger':caption+='The horizontal and vertical axes represent space and physical time, respectively.'
            save(fig,stem,caption)

    # Retain the earlier reconstruction style for four prescribed main PDEs.
    for pde in ['poisson','darcy','nsnonbounded','burger']:
        if pde not in args.pdes:continue
        rows=[select(pde,'guidance_components',guidance_components=mode) for mode in ['obs_only','obs_pde']]
        parts=components(pde)
        fig,axes=plt.subplots(len(parts),5,figsize=(6.2,1.75*len(parts)+.35),squeeze=False,layout='constrained')
        for i,(field,component,name) in enumerate(parts):
            truth,values=compare(rows,field)
            arrays=[truth[component]]+[v['pred'][component] for v in values]
            errors=[np.abs(x-arrays[0]) for x in arrays[1:]]
            norm,cmap=field_scale(arrays)
            error_norm=Normalize(0,max(max(x.max() for x in errors),1e-12))
            for j,array in enumerate(arrays):
                im=map_axis(axes[i,j],array,norm,cmap,
                    ['Truth','Obs.','Obs. + PDE'][j] if i==0 else '',
                    metric_label(field,values[j-1]) if j else '')
            for j,error in enumerate(errors):
                err=map_axis(axes[i,j+3],error,error_norm,ERROR_CMAP,['Obs. error','Obs. + PDE error'][j] if i==0 else '')
            axes[i,0].set_ylabel(field_label(pde,field,component),fontsize=8)
            colorbar(fig,im,axes[i,:3]);colorbar(fig,err,axes[i,3:])
        fig.suptitle(NAMES[pde].replace('--','–')+' · guidance at 100 stochastic steps',fontsize=10)
        save(fig,'ablation_guidance_reconstruction_'+pde,
             NAMES[pde]+r' observation and physical guidance on the main ID sample. Both physical fields are shown separately; Burgers has one trajectory field. '
             r'Labels give whole-field and observed relative errors in percent. Predictions share a color range within each row; absolute errors use a separate range starting at zero.')

    # Ordered experimental conditions: marker shapes/styles distinguish phases.
    for family,group,key,levels in [('budget','num_steps_by_sampler','num_steps',[10,50,100,200,500,1000,2000]),
                                    ('coverage','sensor_sparsity','num_obs',[50,100,250,500,1000])]:
        nrows=len(args.pdes)
        # Three PDEs per page keep both field labels readable at paper width.
        for start in range(0,nrows,3):
            pdes=args.pdes[start:start+3]
            fig,axes=plt.subplots(len(pdes),2,figsize=(6.2,1.9*len(pdes)+.3),squeeze=False,layout='constrained')
            for i,pde in enumerate(pdes):
                for j,field in enumerate(['a','u']):
                    ax=axes[i,j]
                    if pde=='burger' and field=='a':ax.axis('off');continue
                    phases=[('stochastic','S',BLUE,'o','-'),('deterministic','D',INK,'s','--'),
                            ('hybrid_d2s','D → S',GOLD,'^',':'),('hybrid_s2d','S → D',BLUE,'v','-.')] if family=='budget' else [('stochastic','S',BLUE,'o','-')]
                    for phase,label,color,marker,style in phases:
                        y=[100*select(pde,group,**{key:x},sampler_phase=phase)['rel_l2_'+field] for x in levels]
                        ax.plot(levels,y,color=color,marker=marker,ls=style,lw=1,ms=3,label=label,markerfacecolor='white' if style!='-' else color)
                    ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(levels,labels=[str(x) for x in levels])
                    ax.set(title=NAMES[pde].replace('--','–')+f' · {field}',ylabel='Relative error (%)',xlabel='Sampling steps' if family=='budget' else 'Observed locations')
                    ax.grid(alpha=.18,lw=.5);ax.spines[['top','right']].set_visible(False)
                    if i==0 and j==1 and family=='budget':ax.legend(ncol=2,fontsize=7,frameon=False)
            stem=f'ablation_{family}_fields_{start//3+1}'
            caption=('Sampling steps' if family=='budget' else 'Observation counts')+r' on the main ID samples. Coefficient and solution errors are separate; Burgers uses the full trajectory error. '
            caption+='Markers show the tested settings. Guidance coefficients remain fixed within each sweep; the lines connect discrete experimental conditions.'
            save(fig,stem,caption)

    # Actual updated-state error trajectories, no population uncertainty bands.
    for start in range(0,len(args.pdes),3):
        pdes=args.pdes[start:start+3]
        fig,axes=plt.subplots(len(pdes),2,figsize=(6.2,1.8*len(pdes)+.3),squeeze=False,layout='constrained')
        for i,pde in enumerate(pdes):
            for j,field in enumerate(['a','u']):
                ax=axes[i,j]
                if pde=='burger' and field=='a':ax.axis('off');continue
                for mode,label,color,style in [('obs_only','Obs.',BLUE,'-'),('obs_pde','Obs. + PDE',GOLD,'--')]:
                    record=select(pde,'guidance_components',guidance_components=mode)
                    path=Path(record['result_path']).parent/'curves.csv'
                    curve=list(csv.DictReader(path.open()))
                    x=np.array([float(r['t_next']) for r in curve]);y=np.array([100*float(r['rel_l2_'+field]) for r in curve])
                    assert len(curve)==100 and np.isclose(x[-1],1)
                    assert np.isclose(y[-1],100*record['rel_l2_'+field],rtol=3e-6,atol=1e-5)
                    ax.plot(x,y,color=color,ls=style,lw=1,label=label);sources[str(path)]=digest(path)
                ax.set(title=NAMES[pde].replace('--','–')+f' · {field}',xlabel='Flow time',ylabel='Relative error (%)',yscale='log')
                ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18,lw=.5)
                if i==0 and j==1:ax.legend(fontsize=7,frameon=False)
        save(fig,f'ablation_trajectories_fields_{start//3+1}',
             r'Updated-state relative errors during 100 stochastic steps on the main ID samples. Solid and dashed curves use observation-only and combined guidance. Each coefficient or solution field is scored separately; these trajectories describe one physical sample per PDE.')

    (args.output/'ablation_figures.tex').write_text('\n'.join(captions))
    with (args.output/'reconstruction_metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
    (args.output/'figure_manifest.json').write_text(json.dumps(dict(
        final_ready=manifest['final_ready'],pdes=args.pdes,sample_id=0,
        source_inventory_sha256=digest(args.source/'records.json'),sources=sources,outputs=outputs,
        chart_contract_sha256=digest(args.contract),plotter_sha256=digest(Path(__file__)),font='Times New Roman'),indent=2)+'\n')
    print('PLOTTED',len(outputs)//2,'figures for',args.pdes,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--pdes',nargs='+',choices=PDES,default=PDES)
    parser.add_argument('--development',action='store_true')
    main(parser.parse_args())

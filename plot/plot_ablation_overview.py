"""Plot guidance, sampling-budget, observation-density and noise ablations.

Contract: four preselected PDE families; every tested level retained. Ordered
budget/density/noise sweeps share panels with colors AND markers for PDEs.
Categorical interventions use actual physical predictions and sensor masks.
Output is PDF/PNG for the LaTeX paper; no population claims from one input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from publication_style import use_times_new_roman
from plot_paper_ablation_fields import FIELD_CMAP, ERROR_CMAP

PDES = ['poisson', 'helmholtz', 'nsnonbounded', 'burger']
NAMES = ['Poisson', 'Helmholtz', 'Navier–Stokes', 'Burgers']
COLORS = ['#27638b', '#bf922f', '#737d38', '#b06483']
MARKERS = ['o', 's', '^', 'D']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    torch.set_num_threads(2)
    use_times_new_roman()
    plt.rcParams.update({'font.size':9.1, 'axes.titlesize':9.1,
        'axes.labelsize':9.1, 'xtick.labelsize':9.1, 'ytick.labelsize':9.1,
        'legend.fontsize':9.1, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':.5, 'pdf.fonttype':42})
    records = json.loads(args.records.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    used, sources, outputs, metrics = {}, {}, {}, []

    def select(pde, group, **kw):
        rows = [r for r in records if r['pde']==pde and
                r['config']['task']=='both' and r['config']['ablation_group']==group
                and all(r['config'][k]==v for k,v in kw.items())]
        assert len(rows)==1, (pde,group,kw,len(rows))
        r=rows[0]; assert r['source']!='pending'
        used[r['id']] = {'pde':pde, 'group':group, 'config':r['config'],
                         'rel_l2_a':r['rel_l2_a'], 'rel_l2_u':r['rel_l2_u']}
        return r

    def save(fig, stem):
        fig.canvas.draw(); fig.canvas.draw()
        fig.set_layout_engine('none')
        for ext in ['pdf','png']:
            out=args.output/f'{stem}.{ext}'
            fig.savefig(out,dpi=200,bbox_inches='tight',pad_inches=.04)
            outputs[out.name]=sha(out)
        plt.close(fig)

    def lines(ax, group, key, levels, field, **kw):
        for pde,label,color,marker in zip(PDES,NAMES,COLORS,MARKERS):
            if pde=='burger' and field=='a': continue
            rows=[select(pde,group,**{key:v},**kw) for v in levels]
            values=[100*r['rel_l2_'+field] for r in rows]
            assert np.isfinite(values).all() and min(values)>0
            ax.plot(levels,values,color=color,marker=marker,markersize=4,
                    lw=1,label=label,markerfacecolor='white' if pde=='burger' else color)
        ax.set_ylabel(rf'$\operatorname{{RelL2}}_{field}$ (%)')
        ax.set_yscale('log');ax.grid(alpha=.2,lw=.5)

    fig,axes=plt.subplots(2,2,figsize=(6.4,4.8),layout='constrained')
    for i,phase in enumerate(['stochastic','deterministic']):
        for j,field in enumerate(['a','u']):
            ax=axes[i,j]
            lines(ax,'num_steps_by_sampler','num_steps',[10,50,100,200,500,1000,2000],field,sampler_phase=phase)
            ax.set(xscale='log',xlabel='Sampling steps $N$',
                   title=('Stochastic' if i==0 else 'Deterministic')+rf' · $\mathbf{{{field}}}$')
            ax.set_xticks([10,100,1000],[10,100,1000])
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc='outside upper center',ncol=4,frameon=False)
    save(fig,'ablation_compact_budget')

    fig,axes=plt.subplots(2,2,figsize=(6.4,4.8),layout='constrained')
    for i,(group,key,levels,label) in enumerate([
        ('sensor_sparsity','num_obs',[50,100,250,500,1000],'Observed locations per field'),
        ('noise_robustness','noise_level',[0.,.01,.05,.1],'Relative noise level')]):
        for j,field in enumerate(['a','u']):
            ax=axes[i,j];lines(ax,group,key,levels,field)
            ax.set(xlabel=label,title=('Density' if i==0 else 'Noise')+rf' · $\mathbf{{{field}}}$')
            if i==0: ax.set_xscale('log');ax.set_xticks([50,100,500,1000],[50,100,500,1000])
            else: ax.set_xticks(levels,['0','0.01','0.05','0.10'])
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc='outside upper center',ncol=4,frameon=False)
    save(fig,'ablation_compact_observations')

    def load(r,field):
        path=Path(r['result_path']); sources[str(path)]=sha(path)
        maskfile=path.parent/'masks.pt';sources[str(maskfile)]=sha(maskfile)
        p=torch.load(path,map_location='cpu',weights_only=False)
        m=torch.load(maskfile,map_location='cpu',weights_only=False)
        prefix={'a':'coef','u':'sol'}[field]
        truth=p[prefix+'_ground_truth'][0].double().numpy()
        pred=p[prefix+'_final'][0].double().numpy()
        mask=m[prefix][0].double().numpy()
        assert truth.shape==pred.shape==mask.shape
        error=np.linalg.norm(pred-truth)/np.linalg.norm(truth)
        assert np.isclose(error,r['rel_l2_'+field],rtol=5e-6,atol=1e-8),(r['id'],field,error)
        observed=np.linalg.norm((pred-truth)*mask)/np.linalg.norm(truth*mask)
        assert np.isfinite([error,observed]).all()
        metrics.append(dict(id=r['id'],pde=r['pde'],field=field,
                            rel_l2=error,obs_rel_l2=observed,observations=int(mask.sum())))
        return truth[0],pred[0],mask[0],error,observed

    def norm(arrays):
        lo=min(a.min() for a in arrays);hi=max(a.max() for a in arrays)
        return TwoSlopeNorm(vmin=lo,vcenter=0,vmax=hi) if lo<0<hi else Normalize(lo,hi)

    def maprow(fig,axes,arrays,labels,field,cmap=FIELD_CMAP):
        scale=norm(arrays)
        for ax,a,label in zip(axes,arrays,labels):
            im=ax.imshow(a,origin='lower',cmap=cmap,norm=scale,interpolation='nearest')
            ax.set(xticks=[],yticks=[],xlabel=label)
            for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.3)
        axes[0].set_ylabel(field)
        cb=fig.colorbar(im,ax=list(axes),orientation='horizontal',fraction=.05,pad=.025,aspect=40,shrink=.8)
        cb.locator=MaxNLocator(4);cb.update_ticks();cb.ax.tick_params(length=2,pad=1)

    # Two sampler phases, all three loss-state choices on the same trajectory.
    fig,axes=plt.subplots(2,4,figsize=(6.4,4.5),layout='constrained')
    reference=None
    for i,phase in enumerate(['stochastic','deterministic']):
        rows=[select('burger','loss_state_by_sampler',loss_state=s,sampler_phase=phase)
              for s in ['xt','x_next','endpoint']]
        values=[load(r,'u') for r in rows]
        for v in values:
            if reference is None:reference=v[0]
            assert np.array_equal(reference,v[0])
            assert np.array_equal(values[0][2],v[2])
        arrays=[reference]+[v[1] for v in values]
        labels=['']+[f'{100*v[3]:.2f}% / {100*v[4]:.2f}%' for v in values]
        maprow(fig,axes[i],arrays,labels,('S' if i==0 else 'D')+r' · $\mathbf{u}_{\rm traj}$')
        if i==0:
            for ax,title in zip(axes[i],['Truth',r'Current $\mathbf{x}_k$',r'Next $\widetilde{\mathbf{x}}_{k+1}$','Endpoint']):ax.set_title(title)
    fig.suptitle('Burgers · guidance evaluation state',fontsize=10)
    save(fig,'ablation_compact_state_burgers')

    # Joint-layout comparison: distinguish the two actual masks by color in a
    # shared top row. Full color ranges are retained even for poorer settings.
    rows=[select('helmholtz','sensor_mode',sensor_mode=s) for s in ['random','fixed','grid','sensor_column']]
    values={f:[load(r,f) for r in rows] for f in ['a','u']}
    fig,axes=plt.subplots(4,5,figsize=(6.4,6.7),layout='constrained')
    axes[0,0].set_axis_off()
    axes[0,0].text(.5,.55,'Observed\nlocations',ha='center',va='center')
    axes[0,0].text(.5,.2,r'$\mathbf{a}$: gold'+'\n'+r'$\mathbf{u}$: blue',ha='center',va='center')
    for j,title in enumerate(['Random','Fixed','Grid','Columns'],start=1):
        ma=values['a'][j-1][2]>0;mu=values['u'][j-1][2]>0
        rgb=np.ones(ma.shape+(3,))
        rgb[ma]=[.75,.57,.18];rgb[mu]=[.15,.39,.55];rgb[ma&mu]=[.16,.16,.16]
        axes[0,j].imshow(rgb,origin='lower',interpolation='nearest')
        axes[0,j].set(title=title,xticks=[],yticks=[],xlabel=f'$m_a=m_u={int(ma.sum())}$')
        assert ma.sum()==mu.sum()
    for i,f in enumerate(['a','u'],start=1):
        vals=values[f];assert all(np.array_equal(vals[0][0],v[0]) for v in vals)
        arrays=[vals[0][0]]+[v[1] for v in vals]
        labels=['Truth']+[f'{100*v[3]:.2f}% / {100*v[4]:.2f}%' for v in vals]
        maprow(fig,axes[i],arrays,labels,rf'$\mathbf{{{f}}}$')
    vals=values['u'];arrays=[np.zeros_like(vals[0][0])]+[np.abs(v[1]-v[0]) for v in vals]
    maprow(fig,axes[3],arrays,['']*5,r'$|\widehat{\mathbf{u}}-\mathbf{u}|$',ERROR_CMAP)
    fig.suptitle('Helmholtz · observation layouts and reconstructions',fontsize=10)
    save(fig,'ablation_compact_layout_helmholtz')
    (args.output/'compact_manifest.json').write_text(json.dumps({
        'records_sha256':sha(args.records),'plotter_sha256':sha(__file__),
        'scope':'ID input 0; joint fields; Burgers full trajectory',
        'records':used,'tensor_sources':sources,'verified_metrics':metrics,
        'outputs':outputs},indent=2)+'\n')
    print('Wrote four compact figures with source and tensor checks.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())

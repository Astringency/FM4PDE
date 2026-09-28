"""Reproduce the two switching-time figures from public-script outputs.

python -m plot.switching_time --inputs outputs/switching --output figures
Sharded runs: pass all shard output directories after --inputs.
"""
import argparse,json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,LogLocator,NullFormatter


def load_runs(folders):
    import torch
    rows=[]
    for folder in folders:
        for receipt in json.loads((folder/'index.json').read_text()):
            result=torch.load(receipt['result'],map_location='cpu',weights_only=False)
            errors={}
            for f,name in [('a','coef'),('u','sol')]:
                truth=result[name+'_ground_truth'].double()
                errors[f]=((result[name+'_final'].double()-truth).flatten(1).norm(dim=1)
                           /truth.flatten(1).norm(dim=1)).tolist()
            rows.append(dict(id=receipt['id'],config=receipt['identity']['config'],
                             errors=errors,terminal=result['metrics']))
    return rows


def render(rows,out):
    out.mkdir(parents=True,exist_ok=True)
    # Match the manuscript's other ablation figures, including mathematical
    # lettering; the error symbols are defined in its evaluation section.
    for font in Path('/mnt/c/Windows/Fonts').glob('times*.ttf'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,
                         'axes.titlesize':9,'axes.labelsize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':9,
                         'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
                         'mathtext.it':'Times New Roman:italic',
                         'mathtext.bf':'Times New Roman:bold',
                         'mathtext.bfit':'Times New Roman:bold:italic',
                         'mathtext.cal':'Times New Roman:italic',
                         'mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman',
                         'axes.linewidth':.5,'pdf.fonttype':42,'ps.fonttype':42})
    def select(pde,phase,grid,ratio=None):
        matches=[r for r in rows if r['config']['pde']==pde and r['config']['sampler_phase']==phase
                 and r['config']['time_grid']==grid
                 and (ratio is None or r['config']['switch_ratio']==ratio)]
        if len(matches)!=1:raise ValueError((pde,phase,grid,ratio,len(matches)))
        return matches[0]
    def value(row,field):
        if len(row['errors']['a'])!=1:raise ValueError('The switching figure uses one fixed input')
        return row['terminal']['pde_loss_component_total'] if field=='pde' else row['errors'][field][0]
    for pde in ['nsnonbounded','poisson']:
        fig,axes=plt.subplots(1,3,figsize=(7.2,2.85),layout='constrained')
        for ax,field in zip(axes,['a','u','pde']):
            for phase,title,color,marker in [('hybrid_d2s','D → S','#27638b','o'),
                                              ('hybrid_s2d','S → D','#b88928','s')]:
                for grid,style,description in [('uniform','-','uniform'),('geometric','--','D geometric')]:
                    x=[.2,.5,.8];y=[value(select(pde,phase,grid,r),field) for r in x]
                    if not np.isfinite(y).all() or min(y)<=0:raise ValueError('Nonpositive/nonfinite logarithmic value')
                    ax.plot(x,y,label=f'{title} ({description})',color=color,ls=style,
                            marker=marker,ms=4,lw=1.1,mfc=color if grid=='uniform' else 'white',mew=.9)
            for phase,title,color,style,grid in [('deterministic','D (geometric)','#404040','-.','geometric'),
                                                 ('stochastic','S (uniform)','#858585',':','uniform')]:
                ax.axhline(value(select(pde,phase,grid),field),label=title,color=color,ls=style,lw=1)
            ax.set(xlabel=r'Switch time $t_{\mathrm{sw}}$',xticks=[.2,.5,.8],xlim=(.15,.85),yscale='log')
            ax.set_ylabel(r'$\mathcal{L}_{\mathrm{PDE},h}$' if field=='pde'
                          else rf'$\operatorname{{RelL2}}_{{{field}}}$')
            lo,hi=ax.get_ylim()
            ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5) if hi/lo>30 else (1,2,3,4,6,8)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:np.format_float_positional(v,precision=4,unique=False,fractional=False,trim='-')))
            ax.yaxis.set_minor_formatter(NullFormatter());ax.grid(which='major',alpha=.18,lw=.5)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,
                   frameon=False,columnspacing=1.5,handlelength=2.6)
        name=f'switch_sensitivity_{pde}_A_decimal'
        fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
        for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.04)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--inputs',nargs='+',type=Path)
    group.add_argument('--records-json',type=Path,help='Audited portable export of the same run records')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    rows=json.loads(args.records_json.read_text()) if args.records_json else load_runs(args.inputs)
    render(rows,args.output)


if __name__=='__main__':main()

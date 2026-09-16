"""Validate and render the complete layout and temporal controls from saved fields."""
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

ROOT=Path('/home/tat512/C01Python/FM4PDE/outputs/ablations/revision_0912_full')
REMOTE=Path('/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ablations/revision_0912_full')
PAPER=Path('/home/tat512/C04Papers/fm4pde_jmlr')
DATA=PAPER/'source_data/revision_0912_full'
FIGS=PAPER/'figures/revision_0912_full'
LAYOUTS=['random','fixed','grid','columns']
PDES=['nsnonbounded','reaction_diffusion','shallow_water','heat','wave','advection_diffusion']
NAMES=['Navier--Stokes','Reaction--Diffusion','Shallow Water','Heat','Wave','Advection--Diffusion']
MODES=['endpoint_secant','hermite_bridge','near_endpoint_temporal']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def key(r):
    return f"{r['pde']}_{r['condition']}_{r['seed']}_"+('shared' if r['shared'] else 'separate')

def prepare():
    arrays={};meta={}
    for mode,count in [('layouts',40),('temporal',18)]:
        complete=json.loads((ROOT/mode/'complete.json').read_text())
        assert len(complete['rows'])==count
        for r in complete['rows']:
            k=key(r);assert k not in meta
            assert r['status']=='ok',r['status']
            p=ROOT/Path(r['result_path']).relative_to(REMOTE)
            assert sha(p)==r['result_sha256']
            assert sha(p.parent/'masks.pt')==r['mask_sha256']
            payload=torch.load(p,map_location='cpu',weights_only=False)
            masks=torch.load(p.parent/'masks.pt',map_location='cpu',weights_only=False)
            for field,prefix in [('a','coef'),('u','sol')]:
                truth=payload[prefix+'_ground_truth'][0].cpu().double().numpy()
                pred=payload[prefix+'_final'][0].cpu().double().numpy()
                mask=masks[prefix][0].cpu().double().numpy()
                assert truth.shape==pred.shape==mask.shape
                assert np.isfinite(pred).all() and set(np.unique(mask)) <= {0.,1.}
                for kind,v in [('truth',truth),('pred',pred),('mask',mask)]:arrays[f'{k}__{field}__{kind}']=v
                for name,msk in [('full',np.ones_like(mask)),('observed',mask)]:
                    e=np.linalg.norm((pred-truth)*msk)/np.linalg.norm(truth*msk)
                    assert np.isclose(e,r['errors'][field][name],rtol=1e-10,atol=1e-12)
            r['environment']=complete['environment'];meta[k]=r
    def arr(k,f,v):return arrays[f'{k}__{f}__{v}']
    for layout in LAYOUTS:
        for seed in range(5):
            ks=[f'helmholtz_{layout}_{seed}_{s}' for s in ['separate','shared']]
            ma=arr(ks[0],'a','mask');mu=arr(ks[0],'u','mask')
            count=640 if layout=='columns' else 500
            assert ma.sum()==mu.sum()==count and not np.array_equal(ma,mu)
            assert np.array_equal(ma,arr(ks[1],'a','mask'))
            assert np.array_equal(ma,arr(ks[1],'u','mask'))
            if layout=='fixed':assert ma[...,64:].sum()==mu[...,64:].sum()==0
            if layout=='grid':assert np.array_equal(mu,np.roll(ma,(3,3),(-2,-1)))
            for f in ['a','u']:assert np.array_equal(arr(ks[0],f,'truth'),arr(ks[1],f,'truth'))
            for k in ks:meta[k]['overlap']=int((arr(k,'a','mask')*arr(k,'u','mask')).sum())
    for pde in PDES:
        ks=[f'{pde}_{m}_0_separate' for m in MODES]
        for f in ['a','u']:
            assert all(np.array_equal(arr(ks[0],f,'truth'),arr(k,f,'truth')) for k in ks)
            assert all(np.array_equal(arr(ks[0],f,'mask'),arr(k,f,'mask')) for k in ks)
        # Only the residual and output bookkeeping may differ between controls.
        ignored={'residual_mode','output_dir'}
        configs=[{k:v for k,v in meta[x]['config'].items() if k not in ignored} for x in ks]
        assert all(c==configs[0] for c in configs)
    np.savez_compressed(DATA/'fields.npz',**arrays);write(DATA/'metadata.json',meta)

def render():
    meta=json.loads((DATA/'metadata.json').read_text());arrays=np.load(DATA/'fields.npz')
    def arr(k,f,v):return arrays[f'{k}__{f}__{v}']
    use_times_new_roman()
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'axes.linewidth':.4})
    outputs={}
    def save(fig,name):
        fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
        for ext in ['pdf','png']:
            p=FIGS/f'{name}.{ext}';fig.savefig(p,dpi=230,bbox_inches='tight',pad_inches=.03);outputs[p.name]=sha(p)
        plt.close(fig)
    def row(fig,axs,values,labels,error=False):
        lo=min(v.min() for v in values);hi=max(v.max() for v in values)
        norm=(TwoSlopeNorm(0,lo,hi) if lo<0<hi and min(abs(lo),abs(hi))>.05*max(abs(lo),abs(hi)) else Normalize(lo,hi))
        for ax,v,label in zip(axs,values,labels):
            im=ax.imshow(v,origin='lower',interpolation='nearest',norm=norm,cmap=ERROR_CMAP if error else FIELD_CMAP)
            ax.set(xticks=[],yticks=[],xlabel=label)
        cb=fig.colorbar(im,ax=list(axs),orientation='horizontal',fraction=.07,pad=.035,aspect=28,shrink=.95)
        cb.locator=MaxNLocator(3);cb.update_ticks();cb.ax.tick_params(length=2,pad=1)
    for layouts,name in [(LAYOUTS[:2],'layouts_random_fixed'),(LAYOUTS[2:],'layouts_grid_columns')]:
        fig,axs=plt.subplots(4,6,figsize=(6.9,6.0),layout='constrained')
        for b,layout in enumerate(layouts):
            ax=axs[:,3*b:3*b+3];ks=[f'helmholtz_{layout}_0_{s}' for s in ['separate','shared']]
            ax[0,0].axis('off');ax[0,0].text(.5,.65,layout.title(),ha='center',va='center',fontsize=11,fontweight='bold')
            ax[0,0].text(.5,.28,'$a$: gold; $u$: blue\noverlap: dark gray',ha='center',va='center',fontsize=8)
            for j,k in enumerate(ks,1):
                ma=arr(k,'a','mask')[0]>0;mu=arr(k,'u','mask')[0]>0
                rgb=np.ones(ma.shape+(3,));rgb[ma]=[.75,.57,.18];rgb[mu]=[.15,.39,.55];rgb[ma&mu]=[.16,.16,.16]
                ax[0,j].imshow(rgb,origin='lower',interpolation='nearest')
                ax[0,j].set(xticks=[],yticks=[],title='Separate' if j==1 else 'Shared',xlabel=f'{int(ma.sum())} per field\n{int((ma&mu).sum())} overlapping')
            for i,f in enumerate(['a','u'],1):
                vs=[arr(ks[0],f,'truth')[0]]+[arr(k,f,'pred')[0] for k in ks]
                labels=['Truth']+[f"Full: {100*meta[k]['errors'][f]['full']:.2f}%\nObs.: {100*meta[k]['errors'][f]['observed']:.2f}%" for k in ks]
                row(fig,ax[i],vs,labels);ax[i,0].set_ylabel(rf'$\mathbf{{{f}}}$')
            truth=arr(ks[0],'u','truth')[0]
            row(fig,ax[3],[np.zeros_like(truth)]+[np.abs(arr(k,'u','pred')[0]-truth) for k in ks],['']*3,error=True)
            ax[3,0].set_ylabel(r'$|\widehat{\mathbf{u}}-\mathbf{u}|$')
        save(fig,name)
    summary={}
    rows=[]
    for layout in LAYOUTS:
        cells=[];summary[layout]={}
        for condition in ['separate','shared']:
            summary[layout][condition]={}
            for f in ['a','u']:
                vals=[100*meta[f'helmholtz_{layout}_{seed}_{condition}']['errors'][f]['full'] for seed in range(5)]
                mean,sd=np.mean(vals),np.std(vals,ddof=1)
                summary[layout][condition][f]={'mean_percent':float(mean),'sd_percent':float(sd)}
                cells.append(f'${mean:.2f}\\pm{sd:.2f}$')
        rows.append(layout.title()+' & '+' & '.join(cells)+r' \\')
    (FIGS/'layout_summary_rows.tex').write_text('\n'.join(rows)[:-2]+'\n')
    write(DATA/'layout_summary.json',summary)
    rows=[]
    for layout in LAYOUTS:
        for seed in range(5):
            vals=[f"{100*meta[f'helmholtz_{layout}_{seed}_{condition}']['errors'][f]['full']:.2f}" for condition in ['separate','shared'] for f in ['a','u']]
            rows.append(layout.title()+f' & {seed} & '+' & '.join(vals)+r' \\')
    (FIGS/'layout_seed_rows.tex').write_text('\n'.join(rows)[:-2]+'\n')
    rows=[]
    for pde,name in zip(PDES,NAMES):
        vals=[f"{100*meta[f'{pde}_{m}_0_separate']['errors'][f]['full']:.2f}" for m in MODES for f in ['a','u']]
        rows.append(name+' & '+' & '.join(vals)+r' \\')
    (FIGS/'temporal_rows.tex').write_text('\n'.join(rows)[:-2]+'\n')
    write(DATA/'figure_manifest.json',dict(plotter_sha256=sha(__file__),inputs={p.name:sha(p) for p in [DATA/'metadata.json',DATA/'fields.npz']},outputs=outputs))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render-only',action='store_true');args=p.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);FIGS.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if not args.render_only:prepare()
    render()

"""Plot field reconstructions, guidance states, switching times and shared sensors.

Prepare a compact numeric archive once; --render-only redraws from that archive.
No synthetic fields, averaging, rescaling of predictions, or hidden channels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from publication_style import use_times_new_roman
from plot_paper_ablation_fields import FIELD_CMAP, ERROR_CMAP

HOME = Path('/home/tat512')
ARCHIVE = HOME/'C01Python/audit/paper_revision_20260908'
PAPER = HOME/'C04Papers/fm4pde_jmlr'
DATA = PAPER/'source_data/revision_0912'
FIGS = PAPER/'figures/revision_0912'
EXTRA=['reaction_diffusion','shallow_water','heat','wave','advection_diffusion','steady_heat_conduction']
NAMES=dict(zip(EXTRA,['Reaction–Diffusion','Shallow Water','Heat','Wave','Advection–Diffusion','Steady Heat']))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    cache=DATA/'tensors';cache.mkdir(exist_ok=True)
    records=json.loads((ARCHIVE/'ablation_fields/records.json').read_text())
    arrays={}; metadata={}; candidates=[]

    def add(key,path,config,extra,expected=None):
        path=Path(path)
        if not path.exists():
            local=cache/key;local.mkdir(exist_ok=True)
            for name in ['result.pt','masks.pt','curves.csv']:
                target=local/name
                if not target.exists():
                    subprocess.run(['scp','server197:'+str(path.parent/name),str(target)],check=True)
            path=local/'result.pt'
        p=torch.load(path,map_location='cpu',weights_only=False)
        m=torch.load(path.parent/'masks.pt',map_location='cpu',weights_only=False)
        errors={}
        for field,prefix in [('a','coef'),('u','sol')]:
            truth=p[prefix+'_ground_truth'][0].double().cpu().numpy()
            pred=p[prefix+'_final'][0].double().cpu().numpy()
            mask=m[prefix][0].double().cpu().numpy()
            assert truth.shape==pred.shape==mask.shape and np.isfinite(pred).all()
            full=np.linalg.norm(pred-truth)/np.linalg.norm(truth)
            obs=np.linalg.norm((pred-truth)*mask)/np.linalg.norm(truth*mask)
            if expected is not None:
                assert np.isclose(full,expected[field],rtol=5e-6,atol=1e-9),(key,field,full,expected[field])
            errors[field]={'full':float(full),'observed':float(obs),'channels':truth.shape[0],
                           'locations':int(mask[0].sum())}
            for name,v in [('truth',truth),('pred',pred),('mask',mask)]:
                arrays[f'{key}__{field}__{name}']=v
        last=list(csv.DictReader((path.parent/'curves.csv').open()))[-1]
        metadata[key]=dict(config=config,errors=errors,result_source=str(path),
                           result_sha256=sha(path),masks_sha256=sha(path.parent/'masks.pt'),
                           physical_loss=float(last['L_pde']),**extra)

    for pde in EXTRA:
        unique={}
        for f in sorted(ARCHIVE.glob(f'output*/{pde}/ensemble/seed*/sample*/receipt.json')):
            d=json.loads(f.read_text());identity=(d['sample_ids'][0],d['config']['sample_seed'])
            if identity in unique:
                assert unique[identity][1]['result_sha256']==d['result_sha256']
            unique[identity]=(f,d)
        assert len(unique)==96
        ranked=sorted(unique.values(),key=lambda fd:(max(fd[1]['errors'][v][0] for v in ['a','u']),fd[1]['sample_ids'][0],fd[1]['config']['sample_seed']))
        for f,d in ranked:
            candidates.append(dict(pde=pde,sample_id=d['sample_ids'][0],seed=d['config']['sample_seed'],
                                   error_a=d['errors']['a'][0],error_u=d['errors']['u'][0],
                                   score=max(d['errors'][v][0] for v in ['a','u']),receipt=str(f)))
        f,d=ranked[0];files=list(f.parent.rglob('result.pt'));assert len(files)==1
        assert sha(files[0])==d['result_sha256']
        add(pde,files[0],d['config'],dict(sample_id=d['sample_ids'][0],seed=d['config']['sample_seed'],
            selection='minimum max(RelL2_a,RelL2_u) among 32 inputs x 3 draws; one unaveraged draw',
            receipt=str(f)),{v:d['errors'][v][0] for v in ['a','u']})
    for r in records:
        c=r['config']
        if c['task']!='both':continue
        if r['pde']=='burger' and c['ablation_group']=='loss_state_by_sampler':
            add('burger_'+c['sampler_phase']+'_'+c['loss_state'],r['result_path'] or r['source_result'],c,
                dict(archive_id=r['id']),{v:r['rel_l2_'+v] for v in ['a','u']})
        if r['pde']=='nsnonbounded' and c['ablation_group']=='sampler_phase':
            key='ns_'+c['sampler_phase']+(f"_{c['switch_ratio']:g}" if c['sampler_phase'].startswith('hybrid') else '')
            add(key,r['result_path'],c,dict(archive_id=r['id']),{v:r['rel_l2_'+v] for v in ['a','u']})
    layoutroot=HOME/'C01Python/FM4PDE/outputs/ablations/shared_sensors_20260912'
    complete=json.loads((layoutroot/'complete.json').read_text())
    for r in complete['rows']:
        key=f"layout_{r['seed']}_"+('shared' if r['shared'] else 'independent')
        path=layoutroot/Path(r['result_path']).relative_to('/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ablations/shared_sensors_20260912')
        assert sha(path)==r['result_sha256']
        add(key,path,r['config'],dict(seed=r['seed'],shared=r['shared'],overlap=r['overlap'],
            environment=complete['environment']),{v:r['errors'][v]['full'] for v in ['a','u']})
    np.savez_compressed(DATA/'fields.npz',**arrays)
    write(DATA/'metadata.json',metadata);write(DATA/'selection_candidates.json',candidates)


def render():
    FIGS.mkdir(parents=True,exist_ok=True)
    meta=json.loads((DATA/'metadata.json').read_text());arr=np.load(DATA/'fields.npz')
    use_times_new_roman()
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':9,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5})
    outputs={}
    def get(key,f,kind):return arr[f'{key}__{f}__{kind}']
    def save(fig,name):
        fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
        for ext in ['pdf','png']:
            p=FIGS/f'{name}.{ext}';fig.savefig(p,dpi=210,bbox_inches='tight',pad_inches=.04);outputs[p.name]=sha(p)
        plt.close(fig)
    def norm(vs):
        lo=min(v.min() for v in vs);hi=max(v.max() for v in vs)
        if lo==hi:lo,hi=lo-.5,hi+.5
        return TwoSlopeNorm(vcenter=0,vmin=lo,vmax=hi) if lo<0<hi and min(abs(lo),abs(hi)) > .05*max(abs(lo),abs(hi)) else Normalize(lo,hi)
    def row(fig,axs,vs,labels,scale=None,cmap=FIELD_CMAP):
        scale=scale or norm(vs)
        for ax,v,label in zip(axs,vs,labels):
            im=ax.imshow(v,origin='lower',interpolation='nearest',cmap=cmap,norm=scale)
            ax.set(xticks=[],yticks=[],xlabel=label)
        cb=fig.colorbar(im,ax=list(axs),orientation='horizontal',fraction=.075,pad=.04,aspect=30,shrink=.9)
        cb.locator=MaxNLocator(3);cb.update_ticks();cb.ax.tick_params(length=2,pad=1)
    def err(x):return f'{100*x:.2f}%' if x>=.0001 else f'{100*x:.4f}%'

    for name,entries in [('additional_multicomponent', [('reaction_diffusion',0,'concentration 1'),('reaction_diffusion',1,'concentration 2'),('shallow_water',0,'depth'),('shallow_water',1,'momentum 1'),('shallow_water',2,'momentum 2')]),
                          ('additional_transport', [('heat',0,''),('wave',0,'displacement'),('wave',1,'velocity'),('advection_diffusion',0,''),('steady_heat_conduction',0,'')])]:
        fig,axs=plt.subplots(len(entries),4,figsize=(6.5,8.0),layout='constrained')
        for i,(pde,ch,component) in enumerate(entries):
            first=i==0 or entries[i-1][0]!=pde
            for j,f in enumerate(['a','u']):
                vs=[get(pde,f,k)[ch] for k in ['truth','pred']]
                label=('All-channel RelL2\n' if len(get(pde,f,'truth'))>1 else 'RelL2: ')+err(meta[pde]['errors'][f]['full']) if first else ''
                row(fig,axs[i,2*j:2*j+2],vs,['',label])
            axs[i,0].set_ylabel(NAMES[pde]+'\n'+(component or 'scalar field'),fontsize=9)
            if i==0:
                for ax,title in zip(axs[i],[r'Truth $\mathbf{a}$',r'FM4PDE $\mathbf{a}$',r'Truth $\mathbf{u}$',r'FM4PDE $\mathbf{u}$']):ax.set_title(title)
        save(fig,name)

    fig,axs=plt.subplots(2,4,figsize=(6.5,4.0),layout='constrained')
    keys=['burger_'+p+'_'+s for p in ['stochastic','deterministic'] for s in ['xt','x_next','endpoint']]
    scale=norm([get(k,'u','pred')[0] for k in keys]+[get(keys[0],'u','truth')[0]])
    for i,phase in enumerate(['stochastic','deterministic']):
        ks=['burger_'+phase+'_'+s for s in ['xt','x_next','endpoint']]
        truth=get(ks[0],'u','truth')[0]
        assert all(np.array_equal(truth,get(k,'u','truth')[0]) for k in ks)
        labels=['']+[f"Full: {err(meta[k]['errors']['u']['full'])}\nObs.: {err(meta[k]['errors']['u']['observed'])}" for k in ks]
        row(fig,axs[i],[truth]+[get(k,'u','pred')[0] for k in ks],labels,scale)
        axs[i,0].set_ylabel(('Stochastic' if i==0 else 'Deterministic')+'\nphysical time ↑')
        if i==0:
            for ax,title in zip(axs[i],['Truth',r'Current $\mathbf{x}_k$',r'Next proposal','Endpoint']):ax.set_title(title)
    fig.supxlabel('Space →    Full: all trajectory grid points; Obs.: 500 measured points',fontsize=9)
    save(fig,'guidance_state_burgers')

    fig,axs=plt.subplots(1,3,figsize=(6.5,2.55),layout='constrained')
    for j,field in enumerate(['a','u','pde']):
        ax=axs[j]
        def val(key):return meta[key]['physical_loss'] if field=='pde' else 100*meta[key]['errors'][field]['full']
        for phase,label,color,marker in [('hybrid_d2s','D→S','#27638b','o'),('hybrid_s2d','S→D','#bf922f','s')]:
            ax.plot([.2,.5,.8],[val(f'ns_{phase}_{x:g}') for x in [.2,.5,.8]],color=color,marker=marker,label=label,lw=1.1,ms=4)
        for phase,label,color,style in [('deterministic','Pure D','#444444','--'),('stochastic','Pure S','#888888',':')]:
            ax.axhline(val('ns_'+phase),label=label,color=color,ls=style,lw=1)
        ax.set(xlabel=r'Switch time $t_s$',xticks=[.2,.5,.8],title=rf'$\operatorname{{RelL2}}_{field}$ (%)' if field!='pde' else r'$\mathcal{L}_{\rm PDE,h}$')
        ax.grid(alpha=.18,lw=.5)
        if field!='pde':ax.set_ylim(bottom=0)
        else:ax.set_yscale('log')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',ncol=4,frameon=False)
    save(fig,'switch_sensitivity_ns')

    ks=['layout_0_independent','layout_0_shared']
    fig,axs=plt.subplots(4,3,figsize=(5.5,6.8),layout='constrained')
    axs[0,0].axis('off');axs[0,0].text(.5,.5,'Sensor locations\n$a$: gold; $u$: blue\noverlap: dark gray',ha='center',va='center')
    for j,k in enumerate(ks,1):
        ma=get(k,'a','mask')[0]>0;mu=get(k,'u','mask')[0]>0
        rgb=np.ones(ma.shape+(3,));rgb[ma]=[.75,.57,.18];rgb[mu]=[.15,.39,.55];rgb[ma&mu]=[.16,.16,.16]
        axs[0,j].imshow(rgb,origin='lower',interpolation='nearest');axs[0,j].set(xticks=[],yticks=[],title='Independent' if j==1 else 'Shared',xlabel=f'500 per field\n{int((ma&mu).sum())} overlapping')
    for i,f in enumerate(['a','u'],1):
        truth=get(ks[0],f,'truth')[0]
        assert np.array_equal(truth,get(ks[1],f,'truth')[0])
        labels=['Truth']+[f"Full: {err(meta[k]['errors'][f]['full'])}\nObs.: {err(meta[k]['errors'][f]['observed'])}" for k in ks]
        row(fig,axs[i],[truth]+[get(k,f,'pred')[0] for k in ks],labels)
        axs[i,0].set_ylabel(rf'$\mathbf{{{f}}}$')
    truth=get(ks[0],'u','truth')[0]
    row(fig,axs[3],[np.zeros_like(truth)]+[np.abs(get(k,'u','pred')[0]-truth) for k in ks],['']*3,cmap=ERROR_CMAP)
    axs[3,0].set_ylabel(r'$|\widehat{\mathbf{u}}-\mathbf{u}|$')
    save(fig,'shared_sensor_reconstruction')
    summary={}
    for shared in ['independent','shared']:
        summary[shared]={f:{'mean_percent':float(np.mean([100*meta[f'layout_{s}_{shared}']['errors'][f]['full'] for s in range(5)])),
                            'sd_percent':float(np.std([100*meta[f'layout_{s}_{shared}']['errors'][f]['full'] for s in range(5)],ddof=1))} for f in ['a','u']}
    write(DATA/'shared_summary.json',summary)
    write(DATA/'figure_manifest.json',dict(plotter_sha256=sha(__file__),inputs={p.name:sha(p) for p in [DATA/'metadata.json',DATA/'fields.npz',DATA/'selection_candidates.json']},outputs=outputs))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--render-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    if not args.render_only:prepare()
    render()

"""Scientific paired-error plots and all sixteen solution-field comparisons."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from experiments.optimizer_diagnostics.hard_sampling import location
from experiments.optimizer_diagnostics.study import sha, write

COLORS={'lr_control_128':'#275D93','selected_128':'#B37624','beta2_128':'#B37624',
        'lr_control_512':'#507968','beta2_512':'#77609B'}


def predictions(folder,seed=0):
    packs=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted(folder.glob(f'seed{seed}_batch*.pt'))]
    return [i for p in packs for i in p['indices']],torch.cat([p['prediction'] for p in packs])


def main(root,pde):
    out=location(root,pde)
    assert json.loads((out/'audit.json').read_text())['status']=='verified'
    summary=json.loads((out/'summary.json').read_text())
    selected=json.loads((out/'selection.json').read_text())
    ids=selected['indices']
    reference=torch.load(out/'selected_reference.pt',map_location='cpu',weights_only=False)
    with (out/'per_sample.csv').open() as stream: rows=list(csv.DictReader(stream))
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    files=[]
    for field in (['u'] if pde=='burger' else ['u','a']):
        fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        for vnum,v in enumerate(summary['variants']):
            table=[r for r in rows if r['variant']==v['variant'] and r['field']==field]
            original=np.array([np.mean([float(r['original']) for r in table if int(r['index'])==i]) for i in ids])
            candidate=np.array([np.mean([float(r['candidate']) for r in table if int(r['index'])==i]) for i in ids])
            if vnum==0: axes[0].plot(range(16),100*original,'o--',color='#555555',label='Original')
            axes[0].plot(range(16),100*candidate,'o-',color=COLORS[v['variant']],label=v['variant'],markersize=4)
            axes[1].plot(range(16),100*(1-candidate/original),'o-',color=COLORS[v['variant']],label=v['variant'],markersize=4)
        for ax in axes:
            ax.set_xticks(range(16),[str(i) for i in ids],rotation=45,ha='right')
            ax.set_xlabel('Fixed case ID (historical difficulty order)');ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Relative L2 error (%)');axes[0].set_ylim(bottom=0)
        axes[1].set_ylabel('Relative error reduction (%)');axes[1].axhline(0,color='#444444',lw=1)
        axes[1].axhline(10,color='#888888',lw=1,linestyle=':')
        axes[0].legend(fontsize=9)
        fig.suptitle(f'{pde} | field {field} | 16 fixed difficult cases, mean of seeds 0 and 1\nSame batch, observations, 100 sampling steps and all noise draws; positive reduction = improvement',fontsize=13)
        name=f'paired_errors_{field}.png';fig.savefig(figures/name);plt.close(fig);files.append(name)
    original_ids,original=predictions(out/'runs/original')
    assert original_ids==ids
    channel=0 if pde=='burger' else 1
    truth=reference['sol_truth'][:,0].numpy()
    baseline=original[:,channel].numpy()
    for v in summary['variants']:
        if not v['variant'].startswith(('selected','beta2')): continue
        current_ids,current=predictions(out/'runs'/v['variant'])
        assert current_ids==ids
        candidate=current[:,channel].numpy()
        pdf_name=f"fields_{v['variant']}_u.pdf"
        with PdfPages(figures/pdf_name) as pdf:
            for start in range(0,16,4):
                fig,axes=plt.subplots(4,5,figsize=(17,12),layout='constrained')
                for row,i in enumerate(range(start,start+4)):
                    arrays=[truth[i],baseline[i],candidate[i],abs(baseline[i]-truth[i]),abs(candidate[i]-truth[i])]
                    lim=max(float(abs(a).max()) for a in arrays[:3]);norm=Normalize(-lim,lim)
                    errmax=max(float(a.max()) for a in arrays[3:]);errnorm=Normalize(0,errmax or 1)
                    for col,array in enumerate(arrays):
                        im=axes[row,col].imshow(array,cmap='RdBu_r' if col<3 else 'magma',norm=norm if col<3 else errnorm,origin='lower',interpolation='nearest')
                        axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                        if col==0: axes[row,col].set_ylabel(f'Case {ids[i]}',fontsize=12)
                        if row==0: axes[row,col].set_title(['Truth','Original',v['variant'],'Original |error|','Candidate |error|'][col],fontsize=11)
                        if col==2: fig.colorbar(im,ax=axes[row,:3].tolist(),shrink=.8,pad=.01)
                        if col==4: fig.colorbar(im,ax=axes[row,3:].tolist(),shrink=.8,pad=.01)
                fig.suptitle(f"{pde} | solution u | {v['variant']} | fixed cases {start+1}-{start+4}/16\nSeed 0 fields; shared value/error scales within each row; quantitative comparisons average seeds 0 and 1",fontsize=14)
                pdf.savefig(fig)
                name=f"fields_{v['variant']}_u_{start//4+1}.png"
                fig.savefig(figures/name);files.append(name);plt.close(fig)
        files.append(pdf_name)
    write(figures/'manifest.json',dict(pde=pde,selection_sha256=sha(out/'selection.json'),
        summary_sha256=sha(out/'summary.json'),audit_sha256=sha(out/'audit.json'),
        all_16_cases_shown=True,physical_fields_seed=0,metrics_seeds=[0,1],
        shared_color_scale='Truth and both predictions share one symmetric range per case; both absolute errors share a zero-based range per case',
        files=[dict(path=name,sha256=sha(figures/name)) for name in files]))
    print('SAMPLING_FIGURES_COMPLETE',pde,len(files),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pde',required=True)
    a=p.parse_args();torch.set_num_threads(4);main(a.root,a.pde)

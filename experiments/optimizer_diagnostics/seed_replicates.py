"""Independent seeds on the same difficult cases, retaining the initial evidence."""
import argparse
import copy
import json
import os
from pathlib import Path
import csv
import numpy as np
import torch
from experiments.optimizer_diagnostics.hard_sampling import location, queue
from experiments.optimizer_diagnostics.sampling_audit import audit
from experiments.optimizer_diagnostics.study import sha, write


def setup(root,pde,screen_root=None):
    target=root/'seed_replicates'
    out=location(target,pde);out.mkdir(parents=True,exist_ok=True)
    for name,path in [('inputs',root/'inputs'),('runs',Path(screen_root) if screen_root else root/'runs')]:
        link=target/name
        if not link.exists(): link.symlink_to(os.path.relpath(path,link.parent),target_is_directory=True)
        assert link.resolve()==path.resolve()
    original=location(root,pde)
    selection=copy.deepcopy(json.loads((original/'selection.json').read_text()))
    selection['parent_selection_sha256']=sha(original/'selection.json')
    selection['seeds']=[2,3]
    selection['cell']['config']['sample_seed']=2
    selection['scope']='Exploratory independent seeds 2 and 3 added after seeing seed 0/1 variability; identical sixteen fixed cases; not a population estimate'
    path=out/'selection.json'
    if path.exists(): assert json.loads(path.read_text())==selection
    else: write(path,selection)
    reference=out/'selected_reference.pt'
    if not reference.exists(): reference.symlink_to(os.path.relpath(original/'selected_reference.pt',reference.parent))
    assert sha(reference)==selection['selected_reference_sha256']
    print('SEED_REPLICATES_READY',pde,selection['indices'],selection['seeds'],flush=True)
    return target


def combine(root,pde):
    other=root/'seed_replicates'
    audit(other,pde)
    original=location(root,pde);followup=location(other,pde)
    first=json.loads((original/'selection.json').read_text())
    second=json.loads((followup/'selection.json').read_text())
    assert first['indices']==second['indices'] and second['parent_selection_sha256']==sha(original/'selection.json')
    assert first['seeds']==[0,1] and second['seeds']==[2,3]
    rows=[]
    for out in [original,followup]:
        assert json.loads((out/'audit.json').read_text())['status']=='verified'
        with (out/'per_sample.csv').open() as stream: rows.extend(list(csv.DictReader(stream)))
    variants=[]
    for name in ['lr_control_128','selected_128']:
        fields={}
        for field in ['u','a']:
            lookup={(int(r['seed']),int(r['index'])):r for r in rows if r['variant']==name and r['field']==field}
            assert len(lookup)==64
            b=np.array([[float(lookup[s,i]['original']) for i in first['indices']] for s in range(4)])
            c=np.array([[float(lookup[s,i]['candidate']) for i in first['indices']] for s in range(4)])
            baseline=b.mean(0);candidate=c.mean(0);diff=candidate-baseline
            bootstrap=diff[np.random.default_rng(20260919).integers(0,16,size=(10000,16))].mean(1)
            improvement=100*(1-candidate/baseline)
            fields[field]=dict(original_mean=float(baseline.mean()),candidate_mean=float(candidate.mean()),
                aggregate_improvement_pct=float(100*(1-candidate.mean()/baseline.mean())),
                median_case_improvement_pct=float(np.median(improvement)),improved=int((improvement>0).sum()),
                improved_ge10=int((improvement>=10).sum()),improved_ge20=int((improvement>=20).sum()),
                paired_difference_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                seed_improvement_pct=(100*(1-c.mean(1)/b.mean(1))).tolist())
        variants.append(dict(variant=name,fields=fields))
    result=dict(status='verified',pde=pde,seeds=[0,1,2,3],cases=16,variants=variants,
        independent_seed_audit_sha256=sha(followup/'audit.json'),
        original_audit_sha256=sha(original/'audit.json'),scope=second['scope'])
    write(original/'four_seed_summary.json',result)
    with (original/'four_seed_per_sample.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    from experiments.optimizer_diagnostics.sampling_figures import plt,COLORS
    for field in ['u','a']:
        fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        for number,name in enumerate(['lr_control_128','selected_128']):
            lookup={(int(r['seed']),int(r['index'])):r for r in rows if r['variant']==name and r['field']==field}
            b=np.array([[float(lookup[s,i]['original']) for i in first['indices']] for s in range(4)]).mean(0)
            c=np.array([[float(lookup[s,i]['candidate']) for i in first['indices']] for s in range(4)]).mean(0)
            if number==0: axes[0].plot(range(16),100*b,'o--',color='#555555',label='Original')
            axes[0].plot(range(16),100*c,'o-',color=COLORS[name],label=name)
            axes[1].plot(range(16),100*(1-c/b),'o-',color=COLORS[name],label=name)
        for ax in axes:
            ax.set_xticks(range(16),[str(i) for i in first['indices']],rotation=45,ha='right')
            ax.set_xlabel('Fixed case ID (historical difficulty order)');ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Relative L2 error (%)');axes[0].set_ylim(bottom=0);axes[0].legend()
        axes[1].set_ylabel('Relative error reduction (%)');axes[1].axhline(0,color='#444444',lw=1)
        axes[1].axhline(10,color='#888888',ls=':',lw=1)
        fig.suptitle(f'Darcy | field {field} | same 16 cases, mean of FOUR seeds (0, 1, 2, 3)\nSeeds 2/3 added after observing seed 0/1 variability; positive reduction = improvement')
        fig.savefig(original/'figures'/f'four_seed_errors_{field}.png',dpi=160);plt.close(fig)
    lines=['# Darcy：追加独立种子的采样复核','',
        '在看到最初两个种子的差异后，追加 seeds 2 和 3；保留全部原先结果，样本仍为预先固定的 16 例。以下先按输入平均四个种子，再做配对 bootstrap；不能代表总体泛化。','',
        '| 候选 | 场 | 原误差 | 新误差 | 相对改善 | 改善 / ≥10% / ≥20% | seeds 0 / 1 / 2 / 3 改善 |',
        '|---|---|---:|---:|---:|---|---|']
    for variant in variants:
        for field,d in variant['fields'].items():
            seeds=' / '.join(f'{v:+.2f}%' for v in d['seed_improvement_pct'])
            lines.append(f"| {variant['variant']} | {field} | {100*d['original_mean']:.3f}% | {100*d['candidate_mean']:.3f}% | {d['aggregate_improvement_pct']:+.3f}% | {d['improved']} / {d['improved_ge10']} / {d['improved_ge20']} | {seeds} |")
    (original/'FOUR_SEEDS.md').write_text('\n'.join(lines)+'\n')
    print('FOUR_SEED_COMPARISON_VERIFIED',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['setup','queue','combine'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--pde',choices=['darcy'],default='darcy');p.add_argument('--screen-root',type=Path)
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.mode=='combine': combine(args.root,args.pde)
    else:
        target=setup(args.root,args.pde,args.screen_root)
        if args.mode=='queue': queue(target,args.pde,args.screen_root)

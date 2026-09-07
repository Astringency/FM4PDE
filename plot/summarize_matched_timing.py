"""Audit all matched-timing tensors, then produce the cross-method figure.

Physical examples are the bootstrap unit. Timing IQR spans the 96 individual
calls. Deterministic repetitions are checked before reducing error to 32 rows.
Refuse partial groups; preserve every solver degeneration in the plot/table.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_revision_sampling import digest, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--dest',type=Path,required=True)
    parser.add_argument('--pdes',nargs='+',choices=['poisson','darcy'],default=['poisson','darcy'])
    args=parser.parse_args()
    assert len(args.pdes)==len(set(args.pdes))
    if len(args.pdes)<2:
        assert 'completed_'+args.pdes[0] in args.dest.name, 'Use a separate completed-PDE report directory'
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(2)
    protocol=json.loads((args.inputs/'protocol.json').read_text());ph=digest(args.inputs/'protocol.json')
    for artifact in protocol['artifacts']:
        assert digest(args.inputs/artifact['path'])==artifact['sha256'],artifact['path']
    env=json.loads((args.results/'environment.json').read_text())
    assert env['protocol_sha256']==ph and env['batch_size']==1 and not env['tf32']
    assert env['deterministic_algorithms']
    args.dest.mkdir(parents=True,exist_ok=True)
    variants=[('fm',n) for n in protocol['fm_steps']]+[(m,1) for m in ['recfno','senseiver','voronoicnn']]+[('pde_opt',n) for n in protocol['pde_opt_steps']]
    expected={(p,m,n,i,s) for p in args.pdes for m,n in variants for i in protocol['evaluation_ids'] for s in protocol['seeds']}
    rows=[];receipts=[];seen=set();per_example=[];summary=[];differences=[]
    max_discrepancy=0.
    truths={};masks={}
    for pde in args.pdes:
        assert json.loads((args.results/pde/'run_complete.json').read_text())['protocol_sha256']==ph
        original=json.loads((args.inputs/protocol['baselines'][pde]['pde_opt']).read_text())
        effective=json.loads((args.results/pde/'pde_opt_effective_config.json').read_text())
        semantic=hashlib.sha256(json.dumps(effective,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
        assert semantic==original['args']['baseline_config_sha256']
        truths[pde]=torch.load(args.inputs/'cache'/f'{pde}_ground_truth.pt',weights_only=False,map_location='cpu')['truths']
        masks[pde]=torch.load(args.inputs/'cache'/f'{pde}_masks.pt',weights_only=False,map_location='cpu')
        for method in ['fm','recfno','senseiver','voronoicnn','pde_opt']:
            pilot=json.loads((args.results/pde/f'pilot_{method}.json').read_text())
            assert pilot['status']=='pass' and pilot['protocol_sha256']==ph
    paths=[path for pde in args.pdes for path in (args.results/pde).glob('*/n*/seed*/sample*/receipt.json')]
    for path in sorted(paths):
        r=json.loads(path.read_text());key=(r['pde'],r['method'],r['budget'],r['sample_id'],r['seed'])
        assert key in expected and key not in seen and r['protocol_sha256']==ph and r['status']=='ok',path
        seen.add(key)
        tensor=path.parent/'prediction.pt';assert digest(tensor)==r['tensor_sha256']
        d=torch.load(tensor,weights_only=False,map_location='cpu');p,m,n,i,s=key
        assert all(d[k].dtype==torch.float32 for k in ['prediction','truth','observations','mask']),path
        assert torch.equal(d['truth'],truths[p][i].coef)
        assert torch.equal(d['mask'],masks[p][i]) and int(d['mask'].sum())==500
        assert torch.equal(d['observations'],truths[p][i].sol*masks[p][i])
        assert hashlib.sha256(d['mask'].numpy().tobytes()).hexdigest()==r['mask_sha256']
        assert d['prediction'].shape==d['truth'].shape and torch.isfinite(d['prediction']).all()
        err=float((d['prediction'].double()-d['truth'].double()).norm()/d['truth'].double().norm())
        max_discrepancy=max(max_discrepancy,abs(err-r['rel_l2_a']))
        assert abs(err-r['rel_l2_a'])<1e-12 and np.isfinite(r['seconds']) and r['seconds']>0
        if m=='fm':assert r['nfe']==n
        statuses=r.get('optimization_status_per_sample',[])
        steps=r.get('optimization_steps_completed',0)
        assert not statuses or 0<=steps<=n
        row=dict(pde=p,method=m,budget=n,sample_id=i,seed=s,rel_l2_a=err,seconds=r['seconds'],
                 peak_allocated_bytes=r['peak_allocated_bytes'],nfe=r.get('nfe',0),
                 actual_optimizer_steps=steps,early_stopped=r.get('optimization_early_stopped',False),
                 zero_prediction=bool(d['prediction'].count_nonzero()==0),
                 prediction_sha256=hashlib.sha256(d['prediction'].numpy().tobytes()).hexdigest(),
                 tensor_sha256=r['tensor_sha256'],mask_sha256=r['mask_sha256'])
        rows.append(row);receipts.append(r)
    assert seen==expected,{'missing':len(expected-seen),'found':len(seen),'expected':len(expected)}
    boot=np.random.default_rng(20260907).integers(0,32,size=(10000,32))
    grouped={}
    for pde in args.pdes:
        for method,n in variants:
            selected=[r for r in rows if (r['pde'],r['method'],r['budget'])==(pde,method,n)]
            errors=[]
            for i in protocol['evaluation_ids']:
                obs=[r for r in selected if r['sample_id']==i];assert len(obs)==3
                if method!='fm':assert len({r['prediction_sha256'] for r in obs})==1,(pde,method,n,i)
                error=float(np.mean([r['rel_l2_a'] for r in obs]));errors.append(error)
                per_example.append(dict(pde=pde,method=method,budget=n,sample_id=i,mean_seed_error=error))
            errors=np.array(errors);grouped[pde,method,n]=errors
            samples=errors[boot].mean(1);lo,hi=np.quantile(samples,[.025,.975])
            times=np.array([r['seconds'] for r in selected]);q1,median,q3=np.quantile(times,[.25,.5,.75])
            summary.append(dict(pde=pde,method=method,budget=n,n_examples=32,calls=96,
                                mean_error=float(errors.mean()),ci95_low=float(lo),ci95_high=float(hi),
                                median_seconds=float(median),q25_seconds=float(q1),q75_seconds=float(q3),
                                maximum_allocated_gib=max(r['peak_allocated_bytes'] for r in selected)/2**30,
                                optimizer_steps_min=min(r['actual_optimizer_steps'] for r in selected),
                                optimizer_steps_max=max(r['actual_optimizer_steps'] for r in selected),
                                zero_prediction_calls=sum(r['zero_prediction'] for r in selected)))
        reference=grouped[pde,'fm',100]
        for method,n in variants:
            diff=grouped[pde,method,n]-reference
            lo,hi=np.quantile(diff[boot].mean(1),[.025,.975])
            differences.append(dict(pde=pde,method=method,budget=n,reference='fm_100',n=32,
                                    mean_difference_pp=float(diff.mean()*100),ci95_low_pp=float(lo*100),ci95_high_pp=float(hi*100)))
    def write_csv(name,data):
        with (args.dest/name).open('w') as out:
            writer=csv.DictWriter(out,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write_csv('matched_timing_calls.csv',rows);write_csv('matched_timing_per_example.csv',per_example)
    write_csv('matched_timing_summary.csv',summary);write_csv('matched_timing_pairs.csv',differences)
    with gzip.open(args.dest/'matched_timing_receipts.json.gz','wt') as out:json.dump(receipts,out,allow_nan=False)
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    from publication_style import use_times_new_roman
    use_times_new_roman()
    fig,axes=plt.subplots(1,len(args.pdes),figsize=(6.2,3.25),squeeze=False,sharey=True)
    axes=axes[0]
    colors={'fm':'#176c9a','recfno':'#a94b23','senseiver':'#886ab5','voronoicnn':'#487d43','pde_opt':'#555555'}
    markers={'fm':'o','recfno':'s','senseiver':'^','voronoicnn':'D','pde_opt':'x'}
    labels={'fm':'FM4PDE','recfno':'RecFNO','senseiver':'Senseiver','voronoicnn':'VoronoiCNN','pde_opt':'PDE-Opt'}
    for ax,pde in zip(axes,args.pdes):
        for method in colors:
            data=[r for r in summary if r['pde']==pde and r['method']==method]
            x=np.array([r['median_seconds'] for r in data]);y=100*np.array([r['mean_error'] for r in data])
            low=100*np.array([r['ci95_low'] for r in data]);high=100*np.array([r['ci95_high'] for r in data])
            q1=np.array([r['q25_seconds'] for r in data]);q3=np.array([r['q75_seconds'] for r in data])
            ax.errorbar(x,y,yerr=[y-low,high-y],xerr=[x-q1,q3-x],color=colors[method],
                        marker=markers[method],linestyle='-' if method=='fm' else ':' if method=='pde_opt' else 'none',
                        markersize=4,capsize=2,linewidth=1,label=labels[method])
            if method=='fm':
                for xx,yy,r in zip(x,y,data):
                    offset=(4,-11) if r['budget']==200 else (4,5)
                    ax.annotate(str(r['budget']),(xx,yy),xytext=offset,textcoords='offset points',fontsize=7)
            elif method=='pde_opt' and all(r['zero_prediction_calls']==96 for r in data):
                lo=min(r['optimizer_steps_min'] for r in data);hi=max(r['optimizer_steps_max'] for r in data)
                steps=str(lo) if lo==hi else f'{lo}–{hi}'
                ax.annotate(f'{steps} iterations; zero output',(float(np.median(x)),float(y[0])),
                            xytext=(0,8),textcoords='offset points',ha='center',fontsize=7,color=colors[method])
        ax.set_xscale('log');ax.set_yscale('log');ax.set_title(pde.capitalize())
        ax.tick_params(axis='y',labelleft=True)
        ax.set_xlabel('Prediction-call latency (s)');ax.grid(True,which='major',alpha=.16)
        ax.margins(x=.15,y=.23)
    axes[0].set_ylabel(r'Inverse field relative $L^2$ error (%)')
    handles,labels_=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels_,loc='upper center',ncol=5,frameon=False,fontsize=8,bbox_to_anchor=(.5,1.01),columnspacing=1.)
    fig.subplots_adjust(left=.1,right=.98,bottom=.18,top=.8,wspace=.28)
    fig.savefig(args.dest/'matched_time_accuracy.pdf');fig.savefig(args.dest/'matched_time_accuracy.png',dpi=220);plt.close(fig)
    outputs={p.name:digest(p) for p in args.dest.iterdir() if p.is_file() and p.name!='matched_timing_manifest.json'}
    write_json(args.dest/'matched_timing_manifest.json',dict(protocol_sha256=ph,environment=env,
               pdes=args.pdes,calls_verified=len(rows),physical_examples_per_pde=32,prediction_error_max_discrepancy=max_discrepancy,
               uncertainty='10000 paired bootstrap resamples of 32 physical examples; three FM seeds averaged within each example',
               timing='Median and IQR of 96 single-example calls; includes preprocessing and transfer, excludes scoring and I/O',
               limitations=protocol['scope'],outputs=outputs))
    print('AUDITED AND PLOTTED',len(rows),'matched prediction calls',flush=True)


if __name__=='__main__':main()

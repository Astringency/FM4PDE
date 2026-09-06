"""Validate a complete sampling stage, then export paired statistics and figures.

Pilot output must go to a separate destination and cannot be confused with the
32-example confirmation. Inference seeds are averaged within physical example
before bootstrapping. Timing is amortized complete-call cost, not single-example
latency or a cross-method benchmark. No incomplete groups are plotted.
"""
from pathlib import Path
import argparse
import collections
import csv
import gzip
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_revision_sampling import check
PDES=['poisson','darcy','nsnonbounded','burger']
LABELS={'poisson':'Poisson','darcy':'Darcy','nsnonbounded':'Navier–Stokes','burger':'Burgers'}
COLORS=['#24649a','#ce8731','#40816b','#935c83']


def csv_write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def actual_run(root,receipt):
    # Receipts retain absolute remote paths; resolve inside the copied archive.
    return root/receipt['run_dir'].split('/revision_results/',1)[1]


def primary(pde,row):
    return float(row['rel_l2_u']) if pde=='burger' else max(float(row['rel_l2_a']),float(row['rel_l2_u']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True);p.add_argument('--stage',choices=['pilot','evaluation'],default='evaluation')
    p.add_argument('--pdes',nargs='+',choices=PDES,default=PDES)
    args=p.parse_args();args.dest.mkdir(parents=True,exist_ok=True)
    if args.stage=='pilot' and 'pilot' not in str(args.dest):
        raise ValueError('Pilot output requires a destination explicitly named pilot.')
    protocol=json.loads((args.inputs/'protocol.json').read_text())
    reports=[check(args.inputs,args.results,pde,args.stage) for pde in args.pdes]
    ids=protocol['pilot_ids'] if args.stage=='pilot' else protocol['evaluation_ids']
    seeds=[0] if args.stage=='pilot' else protocol['inference_seeds']
    variants=protocol['variants'];names=[v['name'] for v in variants];anchor='guidance_obs_pde'
    n=len(ids);rng=np.random.default_rng(20260907);indices=rng.integers(0,n,size=(10000,n))
    summary=[];pairs=[];examples=[];receipt_archive=[];arrays={};batch_groups={};run_hashes=[]
    for pde in args.pdes:
        receipts=[json.loads(f.read_text()) for f in (args.results/pde/args.stage).glob('*/seed*/*/receipt.json')]
        receipt_archive.extend(receipts)
        groups={name:[r for r in receipts if r['variant']['name']==name] for name in names}
        batch_groups[pde]=groups;arrays[pde]={}
        for variant in variants:
            name=variant['name'];records=groups[name]
            values={};residuals={};field_a={};field_u={}
            for r in records:
                for row in r['rows']:
                    key=(r['seed'],int(row['sample_id']));value=primary(pde,row)
                    values[key]=value;residuals[key]=float(row['pde_residual_norm'])
                    field_a[key]=float(row['rel_l2_a']);field_u[key]=float(row['rel_l2_u'])
                    examples.append(dict(pde=pde,variant=name,inference_seed=r['seed'],sample_id=key[1],
                                         primary_error=value,rel_l2_a=field_a[key],rel_l2_u=field_u[key],pde_residual_norm=residuals[key]))
            matrix=lambda vals:np.array([[vals[(s,i)] for i in ids] for s in seeds])
            per_seed=matrix(values);avg=per_seed.mean(axis=0);arrays[pde][name]=avg
            lo,hi=np.quantile(avg[indices].mean(1),[.025,.975])
            times=np.array([r['elapsed_seconds']/len(r['sample_ids']) for r in records])
            nfe=sorted({r['model_evaluations'] for r in records})
            assert len(nfe)==1,(pde,name,nfe)
            summary.append(dict(stage=args.stage,pde=pde,variant=name,family=variant['family'],
                                steps=variant['num_steps'],n_examples=n,inference_seeds=len(seeds),
                                mean_error=float(avg.mean()),ci95_low=float(lo),ci95_high=float(hi),
                                sd_example_seed_means=float(avg.std(ddof=1)),
                                mean_rel_l2_a=float(matrix(field_a).mean()),mean_rel_l2_u=float(matrix(field_u).mean()),
                                mean_pde_residual=float(matrix(residuals).mean()),
                                median_amortized_seconds=float(np.median(times)),q25_seconds=float(np.quantile(times,.25)),
                                q75_seconds=float(np.quantile(times,.75)),model_forward_calls=nfe[0],
                                maximum_allocated_gib=max(r['peak_allocated_bytes'] for r in records)/(1<<30)))
        for name in names:
            ref='guidance_obs_only' if name=='guidance_obs_pde' else anchor
            a=arrays[pde][name];b=arrays[pde][ref]
            diffs=a-b;boot=diffs[indices].mean(1);low,high=np.quantile(boot,[.025,.975])
            ratio_boot=a[indices].mean(1)/b[indices].mean(1)-1;rlo,rhi=np.quantile(ratio_boot,[.025,.975])
            pairs.append(dict(pde=pde,variant=name,reference=ref,n=n,mean_difference_pp=100*float(diffs.mean()),
                              ci95_low_pp=100*float(low),ci95_high_pp=100*float(high),
                              relative_change_pct=100*float(a.mean()/b.mean()-1),
                              relative_ci95_low_pct=100*float(rlo),relative_ci95_high_pct=100*float(rhi),
                              wins=int((a<b).sum())))
    lookup={(r['pde'],r['variant']):r for r in summary}
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        if args.stage=='pilot':fig.suptitle('PILOT — pipeline validation only',fontsize=12)
        for ext in ['pdf','png']:fig.savefig(args.dest/f'{name}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    # Main time figure: all discrete budgets, both nominal coefficient rules.
    fig,axes=plt.subplots(len(args.pdes),1,figsize=(7.9,2.0*len(args.pdes)),layout='constrained',squeeze=False)
    for ax,pde in zip(axes[:,0],args.pdes):
        for rule,color,marker in [('fixed',COLORS[0],'o'),('normalized',COLORS[1],'s')]:
            selected=[lookup[pde,anchor if k==100 else f'steps_{k}_{rule}'] for k in [25,50,100,200]]
            x=np.array([r['median_amortized_seconds'] for r in selected]);y=np.array([100*r['mean_error'] for r in selected])
            low=np.array([100*r['ci95_low'] for r in selected]);high=np.array([100*r['ci95_high'] for r in selected])
            ax.errorbar(x,y,yerr=[y-low,high-y],marker=marker,color=color,capsize=2,lw=1,label=rule.capitalize()+' coefficient')
            for r,xx,yy in zip(selected,x,y):
                if rule=='fixed':ax.annotate(str(r['steps']),(xx,yy),xytext=(4,5),textcoords='offset points',fontsize=8,color=color)
        ax.set_title(LABELS[pde],loc='left');ax.set_ylabel('Primary error (%)');ax.grid(color='.9',lw=.5)
        ax.set_xscale('log');ax.set_yscale('log')
    axes[0,0].legend(frameon=False,fontsize=8);axes[-1,0].set_xlabel('Median amortized time per example (s; batch = 4)')
    save(fig,'controlled_time_accuracy')
    # Guidance and phases keep every declared configuration, with physical-unit intervals.
    for family,selected,labels,filename in [
        ('guidance',['guidance_'+g for g in ['noguide','pde_only','obs_only','obs_pde']],['No guide','PDE only','Obs. only','Obs.+PDE'],'sampling_confirmation_guidance'),
        ('phase',[anchor,'phase_deterministic','phase_hybrid_d2s','phase_hybrid_s2d'],['Stochastic','Deterministic','D→S (0.2)','S→D (0.2)'],'sampling_confirmation_phase')]:
        fig,axes=plt.subplots(1,len(args.pdes),figsize=(8.4,3.15),layout='constrained',squeeze=False)
        for ax,pde in zip(axes[0],args.pdes):
            for j,name in enumerate(selected):
                r=lookup[pde,name];mean=100*r['mean_error']
                ax.errorbar(j,mean,yerr=[[mean-100*r['ci95_low']],[100*r['ci95_high']-mean]],fmt='o',color=COLORS[j],capsize=3)
            ax.set_xticks(range(4),labels,rotation=50,ha='right');ax.set_title(LABELS[pde],loc='left')
            ax.set_yscale('log');ax.grid(axis='y',color='.9',lw=.5)
        axes[0,0].set_ylabel('Primary error (%)');save(fig,filename)
    # Endpoint losses and current-state diagnostics are deliberately kept separate.
    trajectory=[];budget=[]
    for pde in args.pdes:
        for name,records in batch_groups[pde].items():
            for r in records:
                run=actual_run(args.results,r);path=run/'curves.csv';content=path.read_bytes()
                curves=list(csv.DictReader(content.decode().splitlines()))
                assert len(curves)==r['variant']['num_steps']
                run_hashes.append(dict(path=str(path),sha256=hashlib.sha256(content).hexdigest()))
                budget.append(dict(pde=pde,variant=name,seed=r['seed'],batch_start=r['sample_ids'][0],
                                   scalar_update_sum=sum(float(v['guidance_update_scale']) for v in curves),
                                   mean_clip_scale=float(np.mean([float(v['clip_scale']) for v in curves])),
                                   batch_steps_with_any_clipping=sum(float(v['clip_scale'])<.999999 for v in curves),
                                   physical_active_steps=sum(float(v['zeta_pde_t'])>0 for v in curves),
                                   correction_norm_sum=sum(float(v['guidance_correction_norm']) for v in curves)))
            if name not in [anchor,'guidance_obs_only']:continue
            all_steps=collections.defaultdict(list);all_errors={}
            for r in records:
                run=actual_run(args.results,r)
                for row in csv.DictReader((run/'metrics_step_per_sample.csv').open()):
                    all_errors[r['seed'],int(row['sample_id']),int(row['step'])]=primary(pde,row)
                for row in csv.DictReader((run/'curves.csv').open()):all_steps[int(row['step'])].append(row)
            for step in range(100):
                avg=np.array([[all_errors[s,i,step] for i in ids] for s in seeds]).mean(0)
                lo,hi=np.quantile(avg[indices[:3000]].mean(1),[.025,.975]);batch=all_steps[step]
                mean=lambda key:float(np.mean([float(v[key]) for v in batch]))
                weighted_physics=float(np.mean([float(v['zeta_pde_t'])*float(v['grad_norm_pde']) for v in batch]))
                weighted_obs=float(np.mean([float(v['zeta_obs_a_t'])*float(v['grad_norm_obs_a'])+float(v['zeta_obs_u_t'])*float(v['grad_norm_obs_u']) for v in batch]))
                trajectory.append(dict(pde=pde,variant=name,step=step,t=mean('t_next'),mean_error=float(avg.mean()),
                                       ci95_low=float(lo),ci95_high=float(hi),
                                       state_residual=mean('eval_pde_residual_norm'),endpoint_residual=mean('guidance_pde_residual_norm'),
                                       weighted_physical_gradient=weighted_physics,weighted_observation_norm_sum=weighted_obs,
                                       mean_clip_scale=mean('clip_scale')))
    fig,axes=plt.subplots(len(args.pdes),3,figsize=(8.5,2.15*len(args.pdes)),layout='constrained',squeeze=False)
    for axes_row,pde in zip(axes,args.pdes):
        for name,color,label in [(anchor,COLORS[0],'Obs.+PDE'),('guidance_obs_only',COLORS[1],'Obs. only')]:
            data=[r for r in trajectory if (r['pde'],r['variant'])==(pde,name)];x=[r['t'] for r in data]
            axes_row[0].plot(x,[100*r['mean_error'] for r in data],color=color,label=label,lw=1)
            axes_row[0].fill_between(x,[100*r['ci95_low'] for r in data],[100*r['ci95_high'] for r in data],color=color,alpha=.13,lw=0)
            if name==anchor:
                axes_row[1].plot(x,[r['state_residual'] for r in data],color=color,label='Updated state',lw=1)
                axes_row[1].plot(x,[r['endpoint_residual'] for r in data],color=COLORS[2],ls='--',label='Guidance endpoint',lw=1)
                axes_row[2].plot(x,[r['weighted_physical_gradient'] for r in data],color=COLORS[2],label='Weighted physics',lw=1)
                axes_row[2].plot(x,[r['weighted_observation_norm_sum'] for r in data],color=color,ls='--',label='Sum of weighted obs. norms',lw=1)
        axes_row[0].set_title(LABELS[pde],loc='left')
        for ax in axes_row:ax.set_yscale('log');ax.grid(color='.9',lw=.5)
        for ax,label in zip(axes_row,['Primary error (%)','Residual diagnostic','Gradient norm']):ax.set_ylabel(label)
    handles=[];legend_labels=[]
    for ax in axes[0]:
        h,l=ax.get_legend_handles_labels();handles.extend(h);legend_labels.extend(l)
    fig.legend(handles,legend_labels,loc='outside upper center',ncols=3,frameon=False,fontsize=7)
    for ax in axes[-1]:ax.set_xlabel('Flow time')
    save(fig,'sampling_confirmation_trajectories')
    for filename,rows in [('sampling_confirmation_summary.csv',summary),('sampling_confirmation_pairs.csv',pairs),
                          ('sampling_confirmation_per_example.csv',examples),('sampling_confirmation_trajectories.csv',trajectory),
                          ('sampling_confirmation_budgets.csv',budget)]:csv_write(args.dest/filename,rows)
    manifest=dict(stage=args.stage,protocol=protocol,validation=reports,trajectory_source_hashes=run_hashes,
                  resampling='10,000 paired physical-example bootstrap resamples; 3,000 for time-point ribbons; seed 20260907.',
                  scope='Fixed checkpoints and masks; inference seeds averaged within example. Pointwise intervals, no simultaneous-coverage claim.',
                  timing=protocol['timing_scope'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.dest/'sampling_confirmation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with gzip.open(args.dest/'sampling_confirmation_receipts.json.gz','wt') as f:json.dump(receipt_archive,f,allow_nan=False)
    print(json.dumps(reports,indent=2));print('EXPORTED',args.stage,args.dest)


if __name__=='__main__':main()

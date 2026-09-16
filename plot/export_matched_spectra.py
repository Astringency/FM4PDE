"""Frequency diagnostics of the complete matched Poisson/Darcy inverse study."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from publication_style import use_times_new_roman
from spectral_diagnostics import spectral_record,paired_bootstrap,filtered_field,self_check
from run_ns_loss_study import sha,write

COLORS={'FM4PDE':'#28628F','RecFNO':'#BF6634','Senseiver':'#A77B19','VoronoiCNN':'#697540'}
NAMES={'fm':'FM4PDE','recfno':'RecFNO','senseiver':'Senseiver','voronoicnn':'VoronoiCNN'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(2)
    plt.rcParams.update({'font.size':12,'axes.titlesize':12,'axes.labelsize':12,
                        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
    font=use_times_new_roman()
    protocol=json.loads((args.study/'inputs_v2/protocol.json').read_text())
    ph=sha(args.study/'inputs_v2/protocol.json')
    ids=protocol['evaluation_ids'];seeds=protocol['seeds'];assert len(ids)==32 and len(seeds)==3
    records=[];provenance=[];examples={};sources={};truths={};masks={}
    for pde in ['poisson','darcy']:
        for method,budgets in [('fm',protocol['fm_steps']),('recfno',[1]),('senseiver',[1]),('voronoicnn',[1])]:
            for n in budgets:
                for i in ids:
                    predictions=[]
                    for seed in seeds:
                        folder=args.study/'results_v3'/pde/method/f'n{n}'/f'seed{seed}'/f'sample{i}'
                        receipt=json.loads((folder/'receipt.json').read_text());path=folder/'prediction.pt'
                        assert receipt['protocol_sha256']==ph and sha(path)==receipt['tensor_sha256']
                        d=torch.load(path,map_location='cpu',weights_only=False)
                        pred=d['prediction'][0,0].double().numpy();truth=d['truth'][0,0].double().numpy();mask=d['mask'][0,0].numpy()
                        key=pde,i
                        if key in truths:
                            assert np.array_equal(truths[key],truth) and np.array_equal(masks[key],mask)
                        else:truths[key]=truth;masks[key]=mask
                        r=spectral_record(pred,truth,'cosine')
                        assert np.isclose(r['rel_l2'],receipt['rel_l2_a'],rtol=2e-6,atol=2e-8)
                        sources[str(path)]=receipt['tensor_sha256'];predictions.append(pred)
                        # Deterministic timing repeats are not independent samples.
                        if method!='fm' and seed!=seeds[0]:
                            assert np.array_equal(predictions[0],pred)
                            continue
                        meta=dict(pde=pde,task='inverse',field='a',distribution='ID',method=NAMES[method],steps=n,sample_id=i,seed=seed,ensemble=False,source_path=str(path))
                        records.append(dict(**meta,**r))
                        for cuts in [(4.,16.),(8.,32.),(16.,48.)]:
                            rr=r if cuts==(8.,32.) else spectral_record(pred,truth,'cosine',cuts)
                            for band,values in rr['bands'].items():provenance.append(dict(**meta,low_cutoff=cuts[0],high_cutoff=cuts[1],band=band,rel_l2=r['rel_l2'],**values))
                        if i==ids[0] and seed==0 and (method!='fm' or n==100):examples[pde+'_'+NAMES[method]]=pred
                    if method=='fm':
                        pred=np.mean(predictions,axis=0);r=spectral_record(pred,truth,'cosine')
                        meta=dict(pde=pde,task='inverse',field='a',distribution='ID',method='FM4PDE',steps=n,sample_id=i,seed=-1,ensemble=True,source_path='mean of three verified predictions')
                        records.append(dict(**meta,**r))
                        for band,values in r['bands'].items():provenance.append(dict(**meta,low_cutoff=8.,high_cutoff=32.,band=band,rel_l2=r['rel_l2'],**values))
        examples[pde+'_truth']=truths[pde,ids[0]]
        examples[pde+'_mask']=masks[pde,ids[0]]

    def csvwrite(name,rows):
        with (args.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csvwrite('frequency_band_per_sample.csv',provenance)
    np.savez_compressed(args.output/'frequency_example_fields.npz',**examples)
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    with gzip.open(args.output/'frequency_records.json.gz','wt') as f:json.dump(records,f,default=convert,allow_nan=False)

    groups=defaultdict(list)
    for r in records:groups[r['pde'],r['method'],r['steps'],r['ensemble']].append(r)
    summaries=[]
    for key,rows in groups.items():
        per_id=[]
        for i in ids:per_id.append(np.mean([r['rel_l2'] for r in rows if r['sample_id']==i]))
        ci=paired_bootstrap(per_id)
        summaries.append(dict(pde=key[0],method=key[1],steps=key[2],ensemble=key[3],examples=32,
                             predictions_per_example=3 if key[1]=='FM4PDE' else 1,
                             mean_rel_l2=float(np.mean(per_id)),sd_rel_l2=float(np.std(per_id,ddof=1)),
                             ci_low=float(ci[0]),ci_high=float(ci[1])))
    csvwrite('frequency_summary.csv',summaries)
    primary={pde:[(pde,'FM4PDE',100,False)]+[(pde,m,1,False) for m in ['RecFNO','Senseiver','VoronoiCNN']] for pde in ['poisson','darcy']}
    outputs=[]
    def save(fig,name):
        for ext in ['pdf','png']:
            path=args.output/(name+'.'+ext);fig.savefig(path,dpi=190,bbox_inches='tight');outputs.append(path)
        plt.close(fig)
    for pde in ['poisson','darcy']:
        fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
        for j,key in enumerate(primary[pde]):
            rows=groups[key]
            per_id={i:[r for r in rows if r['sample_id']==i] for i in ids}
            power=np.stack([np.mean([r['prediction_power']/r['reference_total'] for r in rr],axis=0) for rr in per_id.values()])
            error=np.stack([np.mean([r['error_power']/r['reference_total'] for r in rr],axis=0) for rr in per_id.values()])
            label=key[1]+(' (100 steps)' if key[1]=='FM4PDE' else '')
            radius=np.arange(power.shape[1]);style=['-','--','-.',':'][j]
            axes[0].plot(radius[1:],power.mean(0)[1:],label=label,color=COLORS[key[1]],ls=style,lw=1.5)
            axes[1].plot(radius[1:],error.mean(0)[1:],label=label,color=COLORS[key[1]],ls=style,lw=1.5)
            ci=paired_bootstrap(error)
            axes[1].fill_between(radius[1:],ci[0,1:],ci[1,1:],color=COLORS[key[1]],alpha=.12,linewidth=0)
            if j==0:
                truth=np.stack([rr[0]['reference_power']/rr[0]['reference_total'] for rr in per_id.values()]).mean(0)
                axes[0].plot(radius[1:],truth[1:],color='#222222',lw=1.8,label='Ground truth')
        for ax,title in zip(axes,['Reference and prediction energy','Reconstruction error energy']):
            ax.set(xscale='log',yscale='log',xlabel='Radial cosine-mode index',title=title)
            ax.grid(alpha=.2);ax.axvline(8,color='#888888',lw=.6);ax.axvline(32,color='#888888',lw=.6)
        axes[0].set_ylabel('Shell energy / full reference energy')
        axes[1].set_ylabel('Shell error energy / full reference energy')
        axes[0].legend(fontsize=10,frameon=False)
        fig.suptitle(f'{pde.title()} inverse reconstruction · 32 matched ID inputs\nFM: 3 seeds per input; shading: 95% bootstrap CI over inputs; DC reported separately',fontsize=12)
        save(fig,'spectrum_'+pde)

        # A predeclared input supports, rather than substitutes for, the
        # aggregate spectra. Field/error scales are common across methods.
        methods=['FM4PDE','RecFNO','Senseiver','VoronoiCNN']
        fig,axes=plt.subplots(4,5,figsize=(9,7),layout='constrained')
        truth=examples[pde+'_truth']
        for k,band in enumerate(['low','high']):
            t=filtered_field(truth,'cosine',band)
            preds=[filtered_field(examples[pde+'_'+m],'cosine',band) for m in methods]
            vmax=max(np.max(np.abs(v)) for v in [t]+preds)
            errors=[np.abs(v-t) for v in preds];emax=max(np.max(v) for v in errors)
            images=[];error_images=[]
            for j,v in enumerate([t]+preds):
                images.append(axes[2*k,j].imshow(v,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax))
                if k==0:axes[0,j].set_title((['Ground truth']+methods)[j],fontsize=11)
            axes[2*k+1,0].axis('off')
            axes[2*k+1,0].text(.5,.5,'Absolute\nband error',ha='center',va='center',transform=axes[2*k+1,0].transAxes)
            for j,v in enumerate(errors,1):
                error_images.append(axes[2*k+1,j].imshow(v,origin='lower',cmap='magma',vmin=0,vmax=max(emax,1e-14)))
                err=np.linalg.norm(v)/max(np.linalg.norm(t),1e-30)
                axes[2*k+1,j].set_xlabel(f'Band error = {100*err:.2f}%',fontsize=10)
            axes[2*k,0].set_ylabel('Low modes (DC–8)' if band=='low' else 'High modes (>32)',fontsize=11)
            fig.colorbar(images[0],ax=axes[2*k,:],orientation='vertical',shrink=.8,aspect=22)
            fig.colorbar(error_images[0],ax=axes[2*k+1,1:],orientation='vertical',shrink=.8,aspect=22)
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'{pde.title()} frequency-separated reconstruction · ID input {ids[0]}\nFM4PDE: 100 steps, seed 0 · shared field and error scales within each row',fontsize=12)
        save(fig,'frequency_fields_'+pde)

    fig,axes=plt.subplots(1,2,figsize=(8,4.2),layout='constrained')
    for ax,pde in zip(axes,['poisson','darcy']):
        for ensemble,style in [(False,'o-'),(True,'s--')]:
            rs=sorted([r for r in summaries if r['pde']==pde and r['method']=='FM4PDE' and r['ensemble']==ensemble],key=lambda r:r['steps'])
            x=np.array([r['steps'] for r in rs]);y=np.array([r['mean_rel_l2'] for r in rs])*100
            lo=np.array([r['ci_low'] for r in rs])*100;hi=np.array([r['ci_high'] for r in rs])*100
            ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=style,color=COLORS['FM4PDE'],capsize=3,
                        markerfacecolor='white' if ensemble else COLORS['FM4PDE'],label='3-sample mean (3× cost)' if ensemble else 'Single sample')
        for method in ['RecFNO','Senseiver','VoronoiCNN']:
            r=next(r for r in summaries if r['pde']==pde and r['method']==method)
            ax.axhline(100*r['mean_rel_l2'],color=COLORS[method],ls=':',label=method)
        ax.set(xlabel='FM steps per sample',ylabel='Relative field error (%)',title=pde.title());ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=10,frameon=False)
    fig.suptitle('Step budget and sample averaging · 32 matched inverse inputs\nError bars: 95% bootstrap CI over physical inputs',fontsize=12)
    save(fig,'frequency_sampling_improvement')
    write(args.output/'manifest.json',dict(status='computed',examples_per_pde=32,pdes=['poisson','darcy'],
        original_prediction_files_verified=len(sources),record_count=len(records),transform='Orthonormal DCT-II; discrete basis diagnostic, not a PDE eigenbasis claim.',
        cutoffs=[8,32],sensitivity_cutoffs=[[4,16],[16,48]],dc_separate=True,
        aggregation='Average FM errors/powers over inference seeds within input; resample physical inputs; deterministic repeats checked equal and counted once.',
        font_path=font,checks=self_check(),source_hashes=sources,script_sha256=sha(Path(__file__)),
        spectrum_core_sha256=sha(Path(__file__).with_name('spectral_diagnostics.py')),
        outputs={p.name:sha(p) for p in outputs},visual_review_pending=True))
    print(json.dumps(dict(records=len(records),source_files=len(sources),summaries=summaries),indent=2))


if __name__=='__main__':main()

"""Audit all controlled timing receipts and draw grouped mean/SD bars."""
import argparse,csv,gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PDES=['poisson','helmholtz','darcy','nsnonbounded','burger']
LABELS=['Poisson','Helmholtz','Darcy','Navier–Stokes','Burgers']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--results',type=Path,required=True);parser.add_argument('--paper',type=Path,required=True);args=parser.parse_args()
 import torch
 torch.set_num_threads(2)
 protocol=json.loads((args.inputs/'protocol.json').read_text());ph=sha(args.inputs/'protocol.json')
 for a in protocol['artifacts']:assert sha(args.inputs/a['path'])==a['sha256'],a['path']
 data=np.load(args.inputs/'source/timing_truths.npz');masks=np.load(args.inputs/'masks.npz')
 all_rows=[];summary=[];telemetry={};audits=[];output_files={}
 for pde,label in zip(PDES,LABELS):
  target=args.results/pde;complete=json.loads((target/'complete.json').read_text())
  assert complete['protocol_sha256']==ph and complete['calls']==80
  pilot=json.loads((target/'pilot.json').read_text());assert pilot['status']=='pass' and pilot['protocol_sha256']==ph
  assert all(r['repeat_max_abs']==r['hidden_max_abs']==r['reference_max_abs']==0 and r['observation_change_max_abs']>0 for r in pilot['checks'])
  warmups=json.loads((target/'warmups.json').read_text());assert len(warmups)==4 and all(r['finite'] for r in warmups)
  uuids=set()
  for method in ['FM4PDE','DiffusionPDE']:
   for n in [100,1000]:
    rows=[]
    for i in protocol['evaluation_ids']:
     stem=f'{method}_{n}_{i}';path=target/(stem+'.json');r=json.loads(path.read_text())
     assert (r['pde'],r['method'],r['steps'],r['sample_id'])==(pde,method,n,i)
     assert r['protocol_sha256']==ph and r['uncontended'] and r['seconds']>0
     assert r['nfe']==(n if method=='FM4PDE' else 2*n-1)
     pred=target/(stem+'.pt');assert sha(pred)==r['prediction_sha256'];d=torch.load(pred,map_location='cpu',weights_only=False)
     assert d['coef'].dtype==d['sol'].dtype==torch.float32,(pde,stem,'output dtype')
     idx=protocol['evaluation_ids'].index(i)
     for field,key in [('coef','a'),('sol','u')]:
      assert np.array_equal(d[field+'_truth'].numpy(),data[pde+'_'+key][idx:idx+1])
      assert np.array_equal(d['mask_'+key].numpy()[0,0],masks[f'{pde}_{i}_{key}'])
      if r['finite']:
       v=d[field].double();t=d[field+'_truth'].double();err=float((v-t).norm()/t.norm())
       assert np.isclose(err,r['relative_l2'][0 if field=='coef' else 1],rtol=1e-6,atol=2e-8)
     uuid=r['uuid'];uuids.add(uuid)
     if uuid not in telemetry:
      telemetry[uuid]=[json.loads(x) for x in (args.results/f'telemetry_{uuid}.jsonl').read_text().splitlines()]
     before=max((x for x in telemetry[uuid] if x['monotonic']<=r['start_monotonic']),key=lambda x:x['monotonic'])
     after=min((x for x in telemetry[uuid] if x['monotonic']>=r['end_monotonic']),key=lambda x:x['monotonic'])
     samples=[x for x in telemetry[uuid] if before['monotonic']<=x['monotonic']<=after['monotonic']]
     assert len(samples)>=3 and all(not s['foreign'] for s in samples),(pde,stem,'telemetry including call boundaries')
     rows.append(r);all_rows.append(r)
    assert len(rows)==20 and len({r['sample_id'] for r in rows})==20
    x=np.array([r['seconds'] for r in rows])
    summary.append(dict(pde=pde,method=method,steps=n,n=20,mean_seconds=float(x.mean()),sd_seconds=float(x.std(ddof=1)),min_seconds=float(x.min()),max_seconds=float(x.max()),median_seconds=float(np.median(x)),nfe=rows[0]['nfe'],finite_predictions=sum(r['finite'] for r in rows),uuid=rows[0]['uuid']))
  assert len(uuids)==1,(pde,uuids)
  audits.append(dict(pde=pde,pilot=pilot,warmups=warmups,calls_verified=80,uuid=next(iter(uuids))))
 assert len(all_rows)==400 and len(summary)==20
 assert all(r['finite'] for r in all_rows), 'Non-finite predictions retained in raw results; explicit failure analysis is required before manuscript export'
 source=args.paper/'source_data';figdir=args.paper/'figures'
 def csvwrite(name,rows):
  f=source/name
  with f.open('w') as s:
   w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  output_files[str(f.relative_to(args.paper))]=sha(f)
 csvwrite('diffusion_fm_timing_summary.csv',summary);csvwrite('diffusion_fm_timing_calls.csv',all_rows)
 copy=source/'diffusion_fm_timing_protocol.json';shutil.copy2(args.inputs/'protocol.json',copy);output_files[str(copy.relative_to(args.paper))]=sha(copy)
 archive=source/'diffusion_fm_timing_audit.json.gz'
 with gzip.open(archive,'wt') as f:json.dump(dict(audits=audits,receipts=all_rows,telemetry=telemetry,environments=[json.loads(p.read_text()) for p in args.results.glob('environment_*.json')]),f)
 output_files[str(archive.relative_to(args.paper))]=sha(archive)
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.labelsize':10,'ytick.labelsize':9})
 from publication_style import use_times_new_roman
 use_times_new_roman()
 fig,axes=plt.subplots(2,1,figsize=(8.2,6.4),layout='constrained')
 for ax,n in zip(axes,[100,1000]):
  x=np.arange(5)
  for j,method in enumerate(['FM4PDE','DiffusionPDE']):
   rs=[next(r for r in summary if r['pde']==p and r['method']==method and r['steps']==n) for p in PDES]
   means=[r['mean_seconds'] for r in rs];sd=[r['sd_seconds'] for r in rs]
   bars=ax.bar(x+(j-.5)*.34,means,width=.32,yerr=sd,color=['#28628F','#BF6634'][j],label=method,capsize=4,error_kw={'elinewidth':1,'capthick':1},zorder=3)
   ceiling=max(r['mean_seconds']+r['sd_seconds'] for r in summary if r['steps']==n)
   for b,m,s in zip(bars,means,sd):ax.text(b.get_x()+b.get_width()/2,m+s+.025*ceiling,f'{m:.2f} s',ha='center',va='bottom',fontsize=9)
  ax.set_ylim(0,ceiling*(1.30 if n==100 else 1.21));ax.set_xticks(x,LABELS);ax.set_ylabel('Time per sample (s)')
  ax.set_title(f'{n:,} sampling steps  ·  NFE: FM4PDE {n:,}, DiffusionPDE {2*n-1:,}',loc='left',fontsize=11,pad=10)
  ax.grid(axis='y',alpha=.25,zorder=0);ax.set_axisbelow(True)
 axes[0].legend(loc='upper left',frameon=False,ncols=2)
 fig.suptitle('Single-sample sampling time\nMean ± sample SD over 20 inputs · RTX 4090 · float32',fontsize=12)
 for ext in ['pdf','png']:
  p=figdir/f'diffusion_fm_timing.{ext}';fig.savefig(p,dpi=190,bbox_inches='tight');output_files[str(p.relative_to(args.paper))]=sha(p)
 plt.close(fig)
 lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\setlength{\tabcolsep}{5pt}',r'\caption{Controlled sampling time: mean $\pm$ sample SD in seconds over 20 single-example calls per setting. All calls use the frozen protocol in Appendix~\ref{app:diffusion-fm-timing}.}',r'\label{tab:diffusion-fm-timing}',r'\begin{tabular}{@{}lrrrr@{}}\toprule',r' & \multicolumn{2}{c}{100 steps} & \multicolumn{2}{c}{1,000 steps} \\',r'PDE & FM4PDE & DiffusionPDE & FM4PDE & DiffusionPDE \\\midrule']
 for pde,label in zip(PDES,LABELS):
  rr=[next(r for r in summary if r['pde']==pde and r['steps']==n and r['method']==m) for n in [100,1000] for m in ['FM4PDE','DiffusionPDE']]
  lines.append(label.replace('–','--')+' & '+' & '.join(f"${r['mean_seconds']:.3f} \\pm {r['sd_seconds']:.3f}$" for r in rr)+r' \\')
 lines.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
 p=source/'diffusion_fm_timing_table.tex';p.write_text('\n'.join(lines)+'\n');output_files[str(p.relative_to(args.paper))]=sha(p)
 manifest=dict(exporter_sha256=sha(Path(__file__)),protocol_sha256=ph,calls_verified=400,settings_verified=20,examples_per_setting=20,sd_ddof=1,source_prediction_errors_recomputed=True,source_truths_masks_verified=True,telemetry_uncontended=True,outputs=output_files,summary=summary)
 (source/'diffusion_fm_timing_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('AUDITED AND EXPORTED 400 calls, 20 mean/SD settings')

if __name__=='__main__':main()

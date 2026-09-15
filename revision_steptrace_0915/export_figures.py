"""Scientific figures from verified physical-unit summaries and step traces."""
import argparse,csv,hashlib,json,math,pathlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'FM4PDE':'#0072B2','DiffusionPDE':'#D55E00','CoCoGen':'#CC79A7'}
METRICS=['relative_l2_a','relative_l2_u','observed_relative_l2_a','observed_relative_l2_u','L_pde']
TITLES=[r'Full-field $a$',r'Full-field $u$',r'Observed $a$',r'Observed $u$',r'Physical PDE loss $L_{\mathrm{pde}}$']

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def style():
 sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'plot'))
 from publication_style import use_times_new_roman
 use_times_new_roman()
 plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':150,'savefig.dpi':220})

def cocogen(root,summary,out):
 audit=json.load(open(root/'cocogen/reuse_audit.json'));assert audit['status']=='pass' and audit['samples_verified']==9000
 fm={x['cell_id']:x for x in csv.DictReader(open(summary))};co={x['cell']:x for x in audit['cells']};rows=[]
 fig,axes=plt.subplots(2,2,figsize=(6.4,4.4),layout='constrained');axes=axes.ravel()
 for ax,(setting,field,title) in zip(axes,[('sparse_forward','u',r'Forward: $u$'),('sparse_inverse','a',r'Inverse: $a$'),('sparse_joint','a',r'Joint: $a$'),('sparse_joint','u',r'Joint: $u$')]):
  for dist in ['id','smooth','rough']:
   cell=f'darcy/{dist}/{setting}';s=co[cell]['summary'];f=fm['supervised/'+cell];assert int(f['n'])==1000
   for method,mean,sd in [('FM4PDE',float(f[f'error_{field}_percent_mean']),float(f[f'error_{field}_percent_sd'])),('CoCoGen',100*np.mean(s['errors'][field]),100*np.std(s['errors'][field],ddof=1))]:
    rows.append({'pde':'darcy','distribution':dist,'setting':setting,'field':field,'method':method,'n':1000,'mean_percent':mean,'sd_percent':sd,'mean_ci95_halfwidth_percent':1.962341*sd/math.sqrt(1000),'comparison':'same physical fields and sensor budget; different sensor locations'})
  for method,offset,marker in [('FM4PDE',-.09,'o'),('CoCoGen',.09,'s')]:
   sub=[r for r in rows if r['setting']==setting and r['field']==field and r['method']==method]
   ax.errorbar(np.arange(3)+offset,[r['mean_percent'] for r in sub],yerr=[r['mean_ci95_halfwidth_percent'] for r in sub],fmt=marker,color=COLORS[method],capsize=3,markersize=5,label=method)
  ax.set_title(title);ax.set_xticks(range(3),['ID','Smooth','Rough']);ax.set_ylim(bottom=0);ax.set_ylabel('Relative $L_2$ error (%)')
 axes[0].legend(frameon=False);fig.savefig(out/'cocogen_darcy_main_comparison.pdf');fig.savefig(out/'cocogen_darcy_main_comparison.png');plt.close(fig)
 with (out/'cocogen_darcy_main_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'cocogen_darcy_main_comparison.provenance.json').write_text(json.dumps({'fm_summary':str(summary),'fm_summary_sha256':sha(summary),'cocogen_audit_sha256':sha(root/'cocogen/reuse_audit.json'),'caption':'Darcy sparse reconstruction on 1,000 test instances per distribution with 500 observations per available field. Symbols show mean relative L2 error and bars show 95% confidence intervals of the mean (sample SD, Student t with 999 degrees of freedom). FM4PDE uses 100 sampling steps; CoCoGen uses 100 intervals with four additional repaint cycles (500 score evaluations).','training_and_sampling':'A pretrained Darcy score model with a constant-label ControlNet is used through an adaptation of CoCoGen data imputation. The frozen model has 23.40M parameters. Sampling uses 100 Euler intervals with four additional repaint cycles per interval (500 score evaluations), solution-only physical corrections in the last 50 intervals and ten postprocessing steps. All test predictions and original files are retained.'},indent=2)+'\n')

def traces(root,out,allow_partial):
 manifest=json.load(open(root/'manifest.json'));aggregate=[];sources=[];axis_scales={}
 for pde,info in manifest['cells'].items():
  matrices={};ids=info['evaluation_ids']
  for method in ['FM4PDE','DiffusionPDE']:
   values=[]
   for i in ids:
    stem=root/'traces'/pde/f'{method}_1000_{i}';receipt=stem.with_suffix('.json')
    if not receipt.exists():continue
    rec=json.load(open(receipt));assert rec['status']=='complete' and rec['steps']==1000;assert sha(stem.with_suffix('.csv'))==rec['trace_sha256']
    raw=list(csv.DictReader(open(stem.with_suffix('.csv'))));assert [int(x['step']) for x in raw]==list(range(1001))
    values.append({k:np.asarray([float(r[k]) if r[k] else np.nan for r in raw]) for k in METRICS+['sampling_seconds_excluding_diagnostics','elapsed_seconds_including_diagnostics']})
    sources.append({'path':str(receipt),'sha256':sha(receipt),'trace_sha256':rec['trace_sha256']})
   if len(values)!=20:
    if allow_partial:return
    raise ValueError(f'{pde}/{method}: {len(values)}/20 complete')
   matrices[method]={k:np.stack([v[k] for v in values]) for k in values[0]}
  for xaxis,suffix in [('step','steps'),('sampling_seconds_excluding_diagnostics','sampling_time')]:
   fig,axes=plt.subplots(2,3,figsize=(6.4,4.8),layout='constrained');axes=axes.ravel();axes[-1].set_axis_off()
   for ax,metric,title in zip(axes,METRICS,TITLES):
    ax.set_title(title)
    if pde=='burger' and metric in ['relative_l2_a','observed_relative_l2_a']:
     ax.text(.5,.5,'N/A\nSingle trajectory field',ha='center',va='center',transform=ax.transAxes,color='#666666');ax.set_xticks([]);ax.set_yticks([]);ax.grid(False);continue
    bounds=[]
    for method,ls in [('FM4PDE','-'),('DiffusionPDE','--')]:
     m=matrices[method];factor=1 if metric=='L_pde' else 100;v=m[metric]*factor
     assert np.isfinite(v).all();mean=v.mean(0);sd=v.std(0,ddof=1)
     weights=np.random.default_rng(20260915).multinomial(20,np.repeat(1/20,20),size=5000)/20
     lower,upper=np.quantile(weights@v,[.025,.975],axis=0)
     assert np.all(lower>=0) and np.all(upper>=lower)
     bounds.extend([float(lower.min()),float(upper.min())])
     x=np.arange(1001) if xaxis=='step' else m[xaxis].mean(0)
     assert np.all(np.diff(x)>=0)
     ax.plot(x,mean,color=COLORS[method],ls=ls,lw=1.5,label=method)
     ax.fill_between(x,lower,upper,color=COLORS[method],alpha=.12,lw=0)
     if suffix=='steps':
      for k in range(1001):aggregate.append({'pde':pde,'method':method,'step':k,'n':20,'metric':metric,'mean':mean[k],'sample_sd':sd[k],'mean_ci95_lower':lower[k],'mean_ci95_upper':upper[k],'unit':'physical residual MSE' if metric=='L_pde' else 'percent','mean_sampling_seconds_excluding_diagnostics':m['sampling_seconds_excluding_diagnostics'][:,k].mean(),'mean_elapsed_seconds_including_diagnostics':m['elapsed_seconds_including_diagnostics'][:,k].mean()})
    scale='linear' if min(bounds)==0 else 'log'
    axis_scales[f'{pde}/{metric}']={'scale':scale,'minimum_interval_bound':min(bounds),'reason':'Exact zero values or interval bounds must remain visible.' if scale=='linear' else 'All means and interval bounds are strictly positive.'}
    ax.set_yscale(scale);ax.set_xlabel('Sampling step' if suffix=='steps' else 'Elapsed time (s)');ax.set_ylabel('Weighted residual MSE sum' if metric=='L_pde' else 'Relative $L_2$ error (%)')
    if suffix=='steps':ax.set_xlim(0,1000)
   legend_ax=axes[1] if pde=='burger' else axes[0];legend_ax.legend(frameon=False)
   name=f'fm_diffusion_{pde}_{suffix}';fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'));plt.close(fig)
 with (out/'step_trace_summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
 (out/'step_trace_provenance.json').write_text(json.dumps({'manifest_sha256':sha(root/'manifest.json'),'selected_root':str(root),'exporter_sha256':sha(__file__),'final_audit_sha256':sha(root/'audit/final_audit.json'),'summary_sha256':sha(out/'step_trace_summary.csv'),'numpy_version':np.__version__,'matplotlib_version':matplotlib.__version__,'axis_scales':axis_scales,'sources':sources,'uncertainty':'pointwise 95% percentile bootstrap CI of mean over 20 fixed examples, 5000 resamples, fixed seed20260915; no paired significance tests','semantics':'Steps0..999 use a clean endpoint estimate at the current native state (FM endpoint, first Diffusion denoiser evaluation); step1000 uses the final output. A point at step100 belongs to the 1000-step schedule and is not a standalone100-step run.','metric':'Identical physical-unit relative L2 and masked relative L2; PDE MSE is evaluated with the same frozen FM evaluator for both methods within each PDE. Residual mode and boundary normalization are recorded per receipt; NS endpoint-secant is an approximate residual.','timing':'Elapsed time excludes metric evaluation but includes the additional field conversions and synchronization used to record the trajectories. This is not a replacement for the original independent resident-model sampling timing. Error-time coordinates are mean elapsed and mean error at the same sampling step; no interpolation or extrapolation. The original producer column named sampling_seconds_excluding_diagnostics includes this recording overhead, and is interpreted only under this definition.','step0':'Actual network estimate from the initial random state; not a fabricated zero-error point.','burgers':'Only u denotes the full time-space trajectory; coefficient panels are not applicable.'},indent=2)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--fm-summary',type=pathlib.Path);p.add_argument('--traces',action='store_true');p.add_argument('--allow-partial',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);style()
 if a.fm_summary:cocogen(a.root,a.fm_summary,a.output)
 if a.traces:traces(a.root,a.output,a.allow_partial)
if __name__=='__main__':main()

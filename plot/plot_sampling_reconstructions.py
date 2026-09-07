"""Plot actual paired sampling outputs; selection fixed before visual inspection.

Contract: first predeclared evaluation ID, seed zero, identical masks/noise
across controls. Physical fields retain their units. Each comparison uses a
common field range and a common zero-based absolute-error range, without
percentile clipping. Labels report independently recomputed full-grid and
observed relative L2 errors. Aggregate uncertainty remains in separate plots.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

LABELS={'poisson':'Poisson','darcy':'Darcy','nsnonbounded':'Navier–Stokes','burger':'Burgers'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9,
 'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
 'axes.linewidth':.5,'savefig.facecolor':'white'})

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--paper',type=Path,required=True);args=p.parse_args()
 import torch
 torch.set_num_threads(2)
 protocol=json.loads((args.study/'inputs/protocol.json').read_text());sample_id=protocol['evaluation_ids'][0];seed=0
 output=args.paper/'figures';rows=[];sources={};outputs={};cache={};frozen={}
 def load(pde,variant):
  key=(pde,variant)
  if key not in cache:
   receipt_path=args.study/'results'/pde/'evaluation'/variant/'seed0'/f'batch{sample_id}'/'receipt.json'
   receipt=json.loads(receipt_path.read_text());run=args.study/'results'/receipt['run_dir'].split('/revision_results/',1)[1]
   path=run/'result.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
   ids=list(map(int,payload['ground_truth_metadata']['sample_ids']));j=ids.index(sample_id)
   row=receipt['rows'][j];assert int(row['sample_id'])==sample_id
   value={f:payload[f][j,0].numpy().astype('float64') for f in ['coef_final','sol_final','coef_ground_truth','sol_ground_truth']}
   value.update({f+'_mask':payload['masks'][f][j,0].numpy().astype('float64') for f in ['coef','sol']})
   for field in ['coef','sol']:
    y,t,m=value[field+'_final'],value[field+'_ground_truth'],value[field+'_mask']
    e=np.linalg.norm(y-t)/np.linalg.norm(t);eo=np.linalg.norm((y-t)*m)/np.linalg.norm(t*m)
    label='a' if field=='coef' else 'u'
    assert np.isclose(e,float(row['rel_l2_'+label]),rtol=1e-6,atol=2e-8)
    rows.append(dict(pde=pde,variant=variant,sample_id=sample_id,seed=seed,field=field,rel_l2=e,obs_rel_l2=eo,residual_rms=float(row['pde_residual_norm']),result_sha256=sha(path)))
    value[field+'_metric']=(e,eo)
   cache[key]=value;sources[str(path)]=sha(path)
   frozen.update({pde+'_'+variant+'_'+k:v for k,v in value.items() if isinstance(v,np.ndarray)})
  return cache[key]
 def comparison(pde,variants,field):
  ds=[load(pde,v) for v in variants];truth=ds[0][field+'_ground_truth'];mask=ds[0][field+'_mask']
  for d in ds:
   assert np.array_equal(truth,d[field+'_ground_truth']) and np.array_equal(mask,d[field+'_mask'])
  vals=[truth]+[d[field+'_final'] for d in ds];vmin=min(v.min() for v in vals);vmax=max(v.max() for v in vals)
  cmap='viridis'
  if vmin<0<vmax:vmax=max(-vmin,vmax);vmin=-vmax;cmap='RdBu_r'
  errors=[np.abs(v-truth) for v in vals[1:]];emax=max(e.max() for e in errors)
  return ds,truth,mask,Normalize(vmin,vmax),cmap,errors,Normalize(0,max(emax,1e-12))
 def draw(ax,x,norm,cmap,title='',metric=None):
  im=ax.imshow(x,origin='lower',norm=norm,cmap=cmap,interpolation='nearest',extent=[0,127,0,127])
  ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,pad=5)
  if metric is not None:ax.set_xlabel(f'$e={metric[0]*100:.2f}\\%$\n$e_{{obs}}={metric[1]*100:.2f}\\%$',labelpad=4)
  return im
 def save(fig,stem):
  for ext in ['pdf','png']:
   f=output/(stem+'.'+ext);fig.savefig(f,dpi=180,bbox_inches='tight');outputs[str(f.relative_to(args.paper))]=sha(f)
  plt.close(fig)
 tex=[]
 for pde in LABELS:
  fields=['sol'] if pde=='burger' else ['coef','sol']
  fig,axes=plt.subplots(len(fields),5,figsize=(8.5,2.55*len(fields)),squeeze=False,layout='constrained')
  for row,field in enumerate(fields):
   ds,truth,mask,norm,cmap,errors,enorm=comparison(pde,['guidance_obs_only','guidance_obs_pde'],field)
   names=['Ground truth','Obs. guidance','Obs. + PDE guidance','Absolute error\nObs. guidance','Absolute error\nObs. + PDE guidance']
   im=draw(axes[row,0],truth,norm,cmap,names[0]);axes[row,0].set_ylabel('Trajectory $u$' if pde=='burger' else ('$a$' if field=='coef' else '$u$'))
   for j in range(2):
    draw(axes[row,j+1],ds[j][field+'_final'],norm,cmap,names[j+1],ds[j][field+'_metric'])
    err=draw(axes[row,j+3],errors[j],enorm,'magma',names[j+3])
   cb=fig.colorbar(im,ax=axes[row,:3],orientation='horizontal',shrink=.78,pad=.03,aspect=35);cb.ax.tick_params(labelsize=8);cb.locator=MaxNLocator(4);cb.formatter=ScalarFormatter(useMathText=True);cb.formatter.set_powerlimits((-2,3));cb.update_ticks()
   cb=fig.colorbar(err,ax=axes[row,3:],orientation='horizontal',shrink=.9,pad=.03,aspect=23);cb.ax.tick_params(labelsize=8);cb.locator=MaxNLocator(4);cb.formatter=ScalarFormatter(useMathText=True);cb.formatter.set_powerlimits((-2,3));cb.update_ticks()
  fig.suptitle(f'{LABELS[pde]} · ID example {sample_id}, seed {seed} · 100 stochastic steps',fontsize=11)
  stem='reconstruction_guidance_'+pde;save(fig,stem)
  tex += [r'\begin{figure}[!htbp]',rf'\centering\includegraphics[width=\linewidth]{{figures/{stem}.pdf}}',
   r'\caption{'+LABELS[pde].replace('–','--')+r' reconstruction with observation-only and combined guidance (100 stochastic steps; ID '+str(sample_id)+r', seed 0). Labels give full-grid error $e$ and observed error $e_{\mathrm{obs}}$. Field and absolute-error ranges are shared within each row. '+('The vertical and horizontal axes index physical time and space, respectively.' if pde=='burger' else 'Rows show the two physical fields.')+r' Aggregate results appear in Figure~\ref{fig:sampling-confirmation-guidance}.}',rf'\label{{fig:{stem.replace("_","-")}}}',r'\end{figure}','']
 guidance_tex='\n'.join(tex);tex=[]
 designs=[('nsnonbounded','sol','phase'),('burger','sol','phase'),('darcy','coef','steps'),('burger','sol','steps')]
 for pde,field,family in designs:
  variants=['guidance_obs_pde','phase_deterministic','phase_hybrid_d2s','phase_hybrid_s2d'] if family=='phase' else ['steps_25_normalized','guidance_obs_pde','steps_200_normalized']
  labels=['Stochastic','Deterministic','D → S at 0.2','S → D at 0.2'] if family=='phase' else ['25 steps, normalized','100 steps','200 steps, normalized']
  ds,truth,mask,norm,cmap,errors,enorm=comparison(pde,variants,field)
  fig,axes=plt.subplots(2,len(variants)+1,figsize=(8.5,4.8),layout='constrained')
  im=draw(axes[0,0],truth,norm,cmap,'Ground truth')
  draw(axes[1,0],mask,Normalize(0,1),'Greys','Observation mask')
  for j,d in enumerate(ds):
   draw(axes[0,j+1],d[field+'_final'],norm,cmap,labels[j],d[field+'_metric'])
   err=draw(axes[1,j+1],errors[j],enorm,'magma','Absolute error')
  fig.colorbar(im,ax=axes[0,:],orientation='horizontal',shrink=.75,pad=.02,aspect=45)
  fig.colorbar(err,ax=axes[1,1:],orientation='horizontal',shrink=.9,pad=.02,aspect=35)
  fig.suptitle(f'{LABELS[pde]} · '+('Trajectory $u$' if pde=='burger' else ('$a$' if field=='coef' else '$u$'))+f' · ID example {sample_id}, seed {seed}',fontsize=11)
  stem=f'reconstruction_{family}_{pde}';save(fig,stem)
  description=('All four phases use 100 steps and combined guidance; both hybrids switch at flow time 0.2.' if family=='phase' else r'The 25- and 200-step runs use $c_N=0.1(101)/(N+1)$; the 100-step reference uses $c_{100}=0.1$.')
  tex += [r'\begin{figure}[!htbp]',rf'\centering\includegraphics[width=\linewidth]{{figures/{stem}.pdf}}',
   r'\caption{'+LABELS[pde].replace('–','--')+' '+family+r' controls (ID 425, seed 0). '+description+r' Top: truth and reconstructions with full-grid and observed relative errors. Bottom: the common mask and absolute errors. Prediction and error color limits are shared; values retain physical units.}',rf'\label{{fig:{stem.replace("_","-")}}}',r'\end{figure}','']
 for name,text in [('guidance',guidance_tex),('controls','\n'.join(tex))]:
  f=args.paper/'figures'/f'reconstructions_{name}.tex';f.write_text(text);outputs[str(f.relative_to(args.paper))]=sha(f)
 npz=args.paper/'source_data/reconstruction_fields.npz';np.savez_compressed(npz,**frozen)
 f=args.paper/'source_data/reconstruction_metrics.csv'
 with f.open('w') as stream:
  w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 outputs[str(f.relative_to(args.paper))]=sha(f);outputs[str(npz.relative_to(args.paper))]=sha(npz)
 manifest=dict(sample_id=sample_id,seed=seed,selection='First predeclared evaluation ID, chosen before visual inspection; no ranking or outcome filtering',source_files=sources,outputs=outputs,metrics=rows,plot_contract=__doc__)
 (args.paper/'source_data/reconstruction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('EXPORTED',len(designs)+4,'figures;',len(rows),'independently checked field metrics')

if __name__=='__main__':main()

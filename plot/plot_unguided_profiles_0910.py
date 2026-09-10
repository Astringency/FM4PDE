"""Physical-channel galleries for direct unguided FM draws, without paired truth."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from publication_style import use_times_new_roman
from plot_paper_ablation_fields import FIELD_CMAP
from run_paper_ablation_revision import digest

NAMES={'poisson':'Poisson','helmholtz':'Helmholtz','darcy':'Darcy','burger':'Burgers',
 'heat':'Heat','advection_diffusion':'Advection–Diffusion','reaction_diffusion':'Reaction–Diffusion',
 'ns_main':'Navier–Stokes (main)','nsnonbounded':'Navier–Stokes (ablation)',
 'wave':'Wave','shallow_water':'Shallow Water','steady_heat_conduction':'Steady Heat'}
PAIRS=[['poisson','helmholtz'],['darcy','burger'],['heat','advection_diffusion'],
       ['reaction_diffusion','ns_main'],['wave','shallow_water'],['nsnonbounded','steady_heat_conduction']]

def main(args):
 torch.set_num_threads(2);use_times_new_roman()
 plt.rcParams.update({'font.size':9.1,'axes.titlesize':9.1,'axes.labelsize':9.1,
     'xtick.labelsize':9.1,'ytick.labelsize':9.1,'pdf.fonttype':42,'axes.linewidth':.5})
 profiles=json.loads((args.source/'profiles.json').read_text());assert len(profiles)==12
 args.output.mkdir(parents=True,exist_ok=True);manifest=[]
 for record in profiles:
  name=record['name'];path=args.source/name/'sample.pt';assert digest(path)==record['sample_sha256']
  sample=torch.load(path,map_location='cpu',weights_only=False)
  a=sample['coef'][0].numpy();u=sample['sol'][0].numpy();assert np.isfinite(a).all() and np.isfinite(u).all()
  nrows=len(a);ncols=1 if name=='burger' else 2
  fig,axes=plt.subplots(nrows,ncols,figsize=(3.0,1.8*nrows+.32),squeeze=False,layout='constrained')
  for i in range(nrows):
   arrays=[u[i]] if name=='burger' else [a[i],u[i]]
   shared=name not in ['poisson','helmholtz','darcy','steady_heat_conduction']
   for j,array in enumerate(arrays):
    lo=min(x.min() for x in arrays) if shared else array.min()
    hi=max(x.max() for x in arrays) if shared else array.max()
    norm=TwoSlopeNorm(vmin=lo,vcenter=0,vmax=hi) if lo<0<hi else Normalize(lo,hi)
    ax=axes[i,j];im=ax.imshow(array,origin='lower',cmap=FIELD_CMAP,norm=norm,interpolation='nearest')
    label=r'$\mathbf u_{\rm traj}$' if name=='burger' else '$\\mathbf{'+('a' if j==0 else 'u')+'}'+(f'_{i+1}' if nrows>1 else '')+'$'
    label=label.replace(r'\mathbf u',r'\mathbf{u}')
    ax.set(xticks=[],yticks=[],title=label)
    if name=='burger':
     ax.set(xlabel=r'$\xi$',ylabel=r'$\tau$')
    bar=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.05,pad=.035,aspect=24)
    bar.locator=MaxNLocator(3);bar.update_ticks();bar.ax.tick_params(length=2,pad=1)
  fig.suptitle(NAMES[name],fontsize=10)
  fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
  files={}
  for ext in ['pdf','png']:
   out=args.output/f'prior_{name}.{ext}';fig.savefig(out,dpi=200,bbox_inches='tight',pad_inches=.025);files[out.name]=digest(out)
  plt.close(fig);manifest.append(dict(name=name,source_sha256=digest(path),outputs=files))
 snippets=[]
 for i,pair in enumerate(PAIRS):
  snippets.append(r'\begin{figure}[!htbp]\centering')
  for name in pair:
   snippets.append(r'\begin{minipage}[t]{0.48\linewidth}\vspace{0pt}\centering'+
       r'\includegraphics[width=\linewidth]{figures/revision_0910/prior_'+name+r'.pdf}\end{minipage}')
  snippets.extend([r'\caption{Direct unguided FM samples from the '+NAMES[pair[0]].replace('–','--')+' and '+NAMES[pair[1]].replace('–','--')+
       r' pretrained models. '+(r'The protocol is shared by Figures~\ref{fig:prior-gallery-1}--\ref{fig:prior-gallery-6}: 100 Euler steps from Gaussian noise, without observation or PDE guidance, retaining known scalar parameters where required. These unpaired draws have no reference-field reconstruction error. ' if i==0 else '')+r'All physical channels are shown; subscripts distinguish components. '+(r'Burgers displays the full time--space trajectory. ' if 'burger' in pair else '')+r'Initial and terminal fields share a color scale within each temporal component; static coefficients and solutions use separate scales.}',
       r'\label{fig:prior-gallery-'+str(i+1)+r'}',r'\end{figure}',''])
 (args.output/'prior_galleries.tex').write_text('\n'.join(snippets))
 (args.output/'prior_manifest.json').write_text(json.dumps(dict(script_sha256=digest(Path(__file__)),profiles_sha256=digest(args.source/'profiles.json'),figures=manifest),indent=2)+'\n')
 print('Plotted 12 model samples in six paired galleries.')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 main(p.parse_args())

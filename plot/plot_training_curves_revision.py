"""Render verified archived training histories at JMLR text width."""
from pathlib import Path
import argparse,csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_style import use_times_new_roman
PDES=['poisson','helmholtz','darcy','nsnonbounded','burger','reaction_diffusion','shallow_water','heat','wave','advection_diffusion','steady_heat_conduction','ns_main']
LABELS=['Poisson','Helmholtz','Darcy','Navier–Stokes: ablation model','Burgers','Reaction–Diffusion','Shallow Water','Heat','Wave','Advection–Diffusion','Steady Heat Conduction','Navier–Stokes: main model']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--paper',type=Path,required=True);parser.add_argument('--new-ns-log',type=Path,required=True);args=parser.parse_args();P=args.paper;A=P/'audit/revision_0909';out=A/'training_figures';out.mkdir(exist_ok=True);folder=P/'source_data/training_logs'
 manifest=json.loads((folder/'manifest.json').read_text());by_pde={r['pde']:r for r in manifest};sources={};allrows=[]
 for pde in PDES:
  path=args.new_ns_log if pde=='ns_main' else folder/f'{pde}_log.txt'
  if pde!='ns_main':assert sha(path)==by_pde[pde]['sha256'],pde
  rows=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')];epochs=[r['epoch']+1 for r in rows];assert epochs==sorted(set(epochs)),pde
  for r in rows:
   assert r['train_loss']>0 and r['val_loss']>0
   allrows.append(dict(pde=pde,epoch=r['epoch']+1,train_loss=r['train_loss'],val_loss=r['val_loss'],lr=r['lr']))
  sources[pde]=dict(path=str(path),sha256=sha(path),first_epoch=epochs[0],last_epoch=epochs[-1],epochs=len(epochs))
 with plt.rc_context({'font.size':9.1,'axes.titlesize':9.1,'axes.labelsize':9.1,'xtick.labelsize':9.1,'ytick.labelsize':9.1,'legend.fontsize':9.1,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
  use_times_new_roman();outputs={};snippets=[]
  for page in range(3):
   fig,axes=plt.subplots(2,2,figsize=(6,4.9),layout='constrained')
   for ax,pde,title in zip(axes.flat,PDES[page*4:page*4+4],LABELS[page*4:page*4+4]):
    rows=[r for r in allrows if r['pde']==pde];epochs=[r['epoch'] for r in rows]
    for key,color,ls,label in [('train_loss','#28628F','-','Training'),('val_loss','#A77B19','--','Validation')]:ax.plot(epochs,[r[key] for r in rows],color=color,lw=1,ls=ls,label=label)
    ax.set(title=title,xlabel='Epoch',ylabel='Flow-matching MSE',xlim=(0,300),yscale='log');ax.set_xticks([0,100,200,300]);ax.grid(alpha=.25)
   fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False)
   stem=f'training_curves_{page+1}'
   for ext in ['pdf','png']:
    path=out/f'{stem}.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight',pad_inches=.03);outputs[path.name]=sha(path)
   plt.close(fig)
   label='fig:training-curves' if page==0 else f'fig:training-curves-{page+1}'
   caption='Training and validation flow-matching losses, part '+str(page+1)+r' of 3. Epoch-level losses are shown without smoothing, with separate logarithmic scales and gaps where pre-resume records are unavailable. Each panel represents one training run.'
   if page==0:caption+=' The Navier--Stokes panel shows the original model retained for the ablation and three-seed studies.'
   if page==2:caption+=' The final panel shows the replacement Navier--Stokes model used for the main comparisons; it is distinct from the ablation model in Figure~\\ref{fig:training-curves}.'
   snippets.append('\\begin{figure}[!htbp]\n\\centering\\includegraphics[width=\\linewidth]{figures/'+stem+'.pdf}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{figure}\n')
  (out/'training_figures.tex').write_text('\n'.join(snippets));(out/'figure_manifest.json').write_text(json.dumps(dict(sources=sources,outputs=outputs),indent=2)+'\n')
 with (out/'training_curves.csv').open('w') as f:
  writer=csv.DictWriter(f,list(allrows[0]));writer.writeheader();writer.writerows(allrows)
 print('PLOTTED 3 figures: 11 original histories and the replacement Navier–Stokes model')
if __name__=='__main__':main()

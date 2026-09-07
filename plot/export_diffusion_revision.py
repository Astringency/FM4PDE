"""Export all completed 100/1000-step Smooth evaluations beside FM means/SDs."""
import argparse,csv,gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np

LABELS={'poisson':'Poisson','helmholtz':'Helmholtz','darcy':'Darcy','nsnonbounded':'Navier--Stokes','burger':'Burgers'}
PDEMAP={'Poisson':'poisson','Helmholtz':'helmholtz','Darcy':'darcy','NS':'nsnonbounded','nsnonbounded':'nsnonbounded','burger':'burger','Burgers':'burger'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
 source=a.paper/'source_data';snapshot=source/'diffusion_snapshot_20260907.json.gz';shutil.copy2(a.source,snapshot)
 data=json.load(gzip.open(snapshot,'rt'));records=data['records'];assert len(records)==26000
 fm=list(csv.DictReader((a.paper/'audit/table_value_provenance.csv').open(encoding='utf-8-sig')))
 rows=[];files={}
 for task in ['forward','inverse','both','burger']:
  table={'forward':'sparse-forward-results','inverse':'sparse-inverse-results','both':'sparse-joint-results','burger':'burgers-results-structured'}[task]
  fields={'forward':['u'],'inverse':['a'],'both':['a','u'],'burger':['u']}[task]
  pdes=['burger'] if task=='burger' else list(LABELS)[:4]
  name={'forward':'sparse forward','inverse':'sparse inverse','both':'sparse joint','burger':'Burgers trajectory'}[task]
  lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\setlength{\tabcolsep}{5pt}',r'\renewcommand{\arraystretch}{1.15}',
   rf'\caption{{Smooth {name} reconstruction: mean $\pm$ sample SD of relative $L^2$ error (\%) over 1,000 examples per entry. '+
   {'burger':'Both methods use five spatial sensor columns (640 values).','forward':r'Each method observes 500 input-field values.','inverse':r'Each method observes 500 solution values.','both':r'Each method observes 500 values of each field.'}[task]+
   ' Results use the archived method-specific Smooth files and sensor locations.}',
   rf'\label{{tab:diffusion-{task}}}',r'\begin{tabular}{@{}llrrr@{}}\toprule',r'PDE & Field & FM4PDE & DiffusionPDE & DiffusionPDE \\',r' & & 100 steps & 100 steps & 1,000 steps \\\midrule']
  for pde in pdes:
   for field in fields:
    selected=[r for r in fm if r['table']=='tab:'+table and r['method']=='FM4PDE' and r['distribution']=='Smooth' and (task=='burger' or PDEMAP.get(r['PDE'],r['PDE'])==pde) and (task=='burger' or r['metric']==f'rel L2({field})')]
    assert len(selected)==1,(task,pde,field,len(selected));r=selected[0]
    values=[dict(task=task,pde=pde,field=field,method='FM4PDE',steps=100,n=1000,mean=float(r['mean']),sd=float(r['sd']),source_file=r['file'],source_sheet=r['sheet'],source_row=r['row'])]
    for n in [100,1000]:
     rr=[r for r in records if r['pde']==pde and r['problem']==('both' if task=='burger' else task) and r['steps']==n]
     assert len(rr)==1000 and {r['offset'] for r in rr}==set(range(1000))
     x=np.array([r['rel_l2_'+field] for r in rr]);assert np.isfinite(x).all()
     values.append(dict(task=task,pde=pde,field=field,method='DiffusionPDE',steps=n,n=1000,mean=float(x.mean()),sd=float(x.std(ddof=1)),source_file=snapshot.name,source_sheet='',source_row=''))
    rows.extend(values)
    text=[f"${100*r['mean']:.2f} \\pm {100*r['sd']:.2f}$" for r in values]
    label=r'$\mathbf{u}_{\mathrm{traj}}$' if pde=='burger' else '$\\mathbf{'+field+'}$'
    lines.append(LABELS[pde]+' & '+label+' & '+' & '.join(text)+r' \\')
  lines.extend([r'\bottomrule\end{tabular}',r'\end{table}',''])
  path=source/f'diffusion_comparison_{task}.tex';path.write_text('\n'.join(lines));files[path.name]=sha(path)
 path=source/'diffusion_comparison_summary.csv'
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 files[path.name]=sha(path)
 manifest=dict(snapshot_sha256=sha(snapshot),records=26000,complete_cells=26,displayed_cells=len(rows),rows=rows,outputs=files,scope=data['scope'])
 (source/'diffusion_comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('EXPORTED',len(rows),'mean/SD entries from 26 complete Diffusion cells and verified FM source rows')

if __name__=='__main__':main()

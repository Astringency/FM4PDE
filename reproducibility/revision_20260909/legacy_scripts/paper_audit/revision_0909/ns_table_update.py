"""Prepare and export NS main-table replacements only beneath an audit directory.

Preparation freezes the unchanged source statistics and records affected claims.
Export requires complete collection, a passing independent raw-tensor audit,
15 complete summary cells, and agreement with all 15,000 per-sample records.
This tool never changes manuscript, source_data, or canonical figure files.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

PDES=('poisson','helmholtz','darcy','nsnonbounded')
DISTS=('ID','Smooth','Rough')
NAMES=dict(zip(PDES,('Poisson','Helmholtz','Darcy','Navier--Stokes')))
SPECS={
 'tab:full-forward-results':('full_forward',('FNO','DeepONet','IFNO','FM4PDE'),('u',)),
 'tab:full-inverse-results':('full_inverse',('IFNO','FM4PDE'),('a',)),
 'tab:sparse-forward-results':('sparse_forward',('RecFNO','Senseiver','VoronoiCNN','FM4PDE'),('u',)),
 'tab:sparse-inverse-results':('sparse_inverse',('RecFNO','Senseiver','VoronoiCNN','FM4PDE'),('a',)),
 'tab:sparse-joint-results':('sparse_joint',('RecFNO','Senseiver','VoronoiCNN','FM4PDE'),('a','u')),
}
DIFF_TASKS={'forward':('sparse_forward',('u',)), 'inverse':('sparse_inverse',('a',)), 'both':('sparse_joint',('a','u'))}
LABELS=tuple(SPECS)+tuple('tab:diffusion-'+x for x in DIFF_TASKS)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readcsv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writejson(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def writecsv(path,rows):
 assert rows
 keys=list(rows[0])
 for row in rows:
  for key in row:
   if key not in keys:keys.append(key)
 with Path(path).open('w',newline='') as f:
  writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
def pde_name(value):
 return {'Poisson':'poisson','Helmholtz':'helmholtz','Darcy':'darcy','NS':'nsnonbounded','Navier--Stokes':'nsnonbounded'}.get(value,value)
def main_key(r):return (r['table'],pde_name(r['PDE']),r['distribution'],r['method'],r['metric'][-2])
def diff_key(r):return (r['task'],r['pde'],r['field'],r['method'],int(r['steps']))
def unique_index(rows,key):
 indexed={key(r):r for r in rows}
 assert len(indexed)==len(rows),'Duplicate source keys'
 return indexed

def table_blocks(text):
 blocks={}
 for block in re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',text,re.S):
  labels=re.findall(r'\\label\{([^}]+)\}',block)
  for label in labels:
   assert label not in blocks,label
   blocks[label]=block
 assert set(LABELS)<=set(blocks),'Missing target table environment'
 return {label:blocks[label] for label in LABELS}

def rank_means(values):
 distinct=sorted(set(values))
 return [distinct.index(x)+1 for x in values]
def formatted_mean(mean,rank):
 value=f'{100*float(mean):.2f}'
 if rank==1:return r'\mathbf{'+value+'}'
 if rank==2:return '{'+value+r'}^{\dagger}'
 return value

def cell(row,rank,stacked=False):
 mean=formatted_mean(float(row['mean']),rank);sd=f"{100*float(row['sd']):.2f}"
 if stacked:return r'\shortstack[r]{$'+mean+r'$\\[-1pt]{\footnotesize$\pm\,'+sd+'$}}'
 return '$'+mean+r'\,\pm\,'+sd+'$'

def main_body(label,index):
 setting,methods,fields=SPECS[label];lines=[]
 for pde in PDES:
  for dist in DISTS:
   ranks={f:rank_means([float(index[label,pde,dist,m,f]['mean']) for m in methods]) for f in fields}
   cells=[cell(index[label,pde,dist,m,f],ranks[f][j],len(fields)==2) for j,m in enumerate(methods) for f in fields]
   suffix=r' \\'
   if dist=='Rough' and pde!=PDES[-1]:suffix+=r'\addlinespace[3pt]'
   lines.append(NAMES[pde]+' & '+dist+' & '+' & '.join(cells)+suffix)
 return '\n'.join(lines)+'\n'

def diffusion_body(task,index):
 lines=[]
 for pde in PDES:
  for field in DIFF_TASKS[task][1]:
   rows=[index[task,pde,field,m,n] for m,n in [('FM4PDE',100),('DiffusionPDE',100),('DiffusionPDE',1000)]]
   ranks=rank_means([float(r['mean']) for r in rows])
   lines.append(NAMES[pde]+r' & $\mathbf{'+field+'}$ & '+' & '.join(cell(r,rank) for r,rank in zip(rows,ranks))+r' \\')
 return '\n'.join(lines)+'\n'

def replace_body(block,body):
 start=block.index(r'\midrule')+len(r'\midrule');end=block.index(r'\bottomrule',start)
 return block[:start]+'\n'+body+block[end:]

def validate_rendered_values(block,body):
 """Require existing two-decimal values to agree before replacing rankings."""
 def rows(text):
  return {tuple(line.split(' & ')[:2]):re.findall(r'\d+\.\d+',line)
          for line in text.splitlines() if any(line.startswith(name+' & ') for name in NAMES.values())}
 assert rows(block)==rows(body),'Current rendered values disagree with frozen source statistics'

def prepare(a):
 a.output.mkdir(parents=True,exist_ok=True)
 sources={
  'main_values':a.paper/'source_data/main_figure_values.csv',
  'provenance':a.paper/'audit/table_value_provenance.csv',
  'diffusion_values':a.paper/'source_data/diffusion_comparison_summary.csv',
  'diffusion_manifest':a.paper/'source_data/diffusion_comparison_manifest.json',
  'manuscript':a.paper/'fm4pde_jmlr_revision_0906.tex',
 }
 main=readcsv(sources['main_values']);provenance=readcsv(sources['provenance']);diffusion=readcsv(sources['diffusion_values'])
 assert len(main)==264 and all(r['table'] in SPECS for r in main)
 ix=unique_index(main,main_key);pi=unique_index([r for r in provenance if r['table'] in SPECS],main_key)
 assert set(ix)==set(pi)
 for key,row in ix.items():
  for col in ('mean','sd'):assert float(row[col])==float(pi[key][col]),(key,col)
  assert math.isfinite(float(row['mean'])) and float(row['mean'])>=0 and math.isfinite(float(row['sd'])) and float(row['sd'])>=0
 manifest=json.loads(sources['diffusion_manifest'].read_text());di=unique_index(diffusion,diff_key)
 assert manifest['outputs']['diffusion_comparison_summary.csv']==sha(sources['diffusion_values'])
 for row in diffusion:
  assert int(row['n'])==1000
  if row['task'] in DIFF_TASKS and row['method']=='FM4PDE':
   label=next(label for label,(s,_,_) in SPECS.items() if s==DIFF_TASKS[row['task']][0])
   old=ix[label,row['pde'],'Smooth','FM4PDE',row['field']]
   assert float(row['mean'])==float(old['mean']) and float(row['sd'])==float(old['sd'])
 blocks=table_blocks(sources['manuscript'].read_text())
 for label in SPECS:validate_rendered_values(blocks[label],main_body(label,ix))
 for task in DIFF_TASKS:validate_rendered_values(blocks['tab:diffusion-'+task],diffusion_body(task,di))
 prepared=dict(sources={k:dict(path=str(v),sha256=sha(v)) for k,v in sources.items()},main=main,provenance=provenance,diffusion=diffusion,templates=blocks)
 target=a.output/'prepared_sources.json'
 if target.exists():
  existing=json.loads(target.read_text())
  assert all(existing[k]==prepared[k] for k in ['main','provenance','diffusion']),'Frozen baseline changed; choose a fresh audit output directory'
 else:writejson(target,prepared)
 writejson(a.output/'pending_updates.json',dict(status='awaiting_complete_NS_results',required_cells=15,examples_per_cell=1000,example_ids=[2000,2999],required_files=[str(a.study/'collection_complete.json')]+[str(a.study/'tables'/x) for x in ['ns_main_summary.csv','ns_main_per_sample.csv','ns_main_validation.json']],table_labels=LABELS,main_fm_target_entries=18,diffusion_fm_entries=4,ranking='All methods in each displayed PDE/distribution/field row; smaller unrounded mean is better; exact ties share rank.',statistics_to_recompute=['Macro mean over 12 PDE-distribution means for each of the six setting/field combinations','Best and second-best counts and cell lists for every main-table method','FM4PDE versus DiffusionPDE at 100 and 1000 steps: mean and SD ordering, per field and PDE','Rough-is-largest statements across each method and PDE','Any overall macro mean or rank-count derivative in figures, summaries, reviewer responses'],source_files_to_update=[str(sources['main_values']),str(sources['provenance']),str(sources['diffusion_values']),str(sources['diffusion_manifest'])],derived_files_to_regenerate=['source_data/main_tables/{full-forward,full-inverse,sparse-forward,sparse-inverse,sparse-joint}-results.tex','source_data/complete_main_tables.tex','source_data/diffusion_comparison_{forward,inverse,both}.tex','figures/main_comparisons.pdf and .png (archived mean-error charts; currently not referenced by the main manuscript)','Main/reviewer-map/response text assertions identified by table label'],limitations='New NS examples are distinct from archived baseline examples. Do not infer paired test significance. No ablation, original three-seed, Burgers, non-NS, or physics-baseline data change.'))
 return prepared

def require_complete(a):
 required=[a.study/'collection_complete.json']+[a.study/'tables'/x for x in ['ns_main_summary.csv','ns_main_per_sample.csv','ns_main_validation.json']]
 missing=[str(x) for x in required if not x.is_file()]
 if missing:raise RuntimeError('Export blocked: required complete-result files are absent: '+', '.join(missing))
 complete=json.loads(required[0].read_text());validation=json.loads(required[3].read_text())
 assert complete['status']=='complete'
 assert validation['status']=='pass' and validation['examples']==15000
 assert all(validation[k] is True for k in ['all_predictions_finite','all_nfe_100','all_result_hashes_match'])
 assert 'float64' in validation['reductions'] and 'ddof=1' in validation['reductions']
 wanted={(dist.lower(),spec[0]) for dist in DISTS for spec in SPECS.values()}
 checks=unique_index(validation['cells'],lambda r:(r['dist'],r['setting']))
 assert set(checks)==wanted and all(r['n']==1000 and r['exact_ids'] is True for r in checks.values())
 summary=unique_index(readcsv(required[1]),lambda r:(r['dist'],r['setting']))
 assert set(summary)==wanted and all(int(r['n'])==1000 for r in summary.values())
 per=readcsv(required[2]);assert len(per)==15000
 cells={key:[] for key in wanted}
 for row in per:
  key=(row['dist'],row['setting']);assert key in cells and int(row['nfe'])==100
  assert all(math.isfinite(float(row[m])) for m in ['error_a','error_u','obs_a','obs_u','pde_mse'])
  cells[key].append(row)
 for key,rows in cells.items():
  assert sorted(int(r['offset']) for r in rows)==list(range(2000,3000)),key
  for metric in ['error_a','error_u','obs_a','obs_u','pde_mse']:
   values=[float(r[metric]) for r in rows]
   for suffix,actual in [('mean',statistics.fmean(values)),('sd',statistics.stdev(values))]:
    expected=float(summary[key][metric+'_'+suffix]);assert math.isclose(actual,expected,rel_tol=2e-12,abs_tol=2e-14),(key,metric,suffix,actual,expected)
 return summary,{str(path):sha(path) for path in required}

def main_statistics(rows):
 ix=unique_index(rows,main_key);out=[]
 for label,(setting,methods,fields) in SPECS.items():
  for field in fields:
   group={m:[] for m in methods};best={m:[] for m in methods};second={m:[] for m in methods}
   for pde in PDES:
    for dist in DISTS:
     values=[float(ix[label,pde,dist,m,field]['mean']) for m in methods]
     for m,value,rank in zip(methods,values,rank_means(values)):
      group[m].append(value)
      if rank==1:best[m].append([pde,dist])
      if rank==2:second[m].append([pde,dist])
   for method in methods:
    rough_largest=[pde for pde in PDES if float(ix[label,pde,'Rough',method,field]['mean'])==max(float(ix[label,pde,d,method,field]['mean']) for d in DISTS)]
    out.append(dict(table=label,setting=setting,field=field,method=method,n_cells=12,macro_mean_percent=100*statistics.fmean(group[method]),best_count=len(best[method]),best_cells=best[method],second_count=len(second[method]),second_cells=second[method],rough_is_largest_pdes=rough_largest))
 return out

def diffusion_statistics(rows):
 ix=unique_index(rows,diff_key);out=[]
 for task,(_,fields) in DIFF_TASKS.items():
  for field in fields:
   for steps in [100,1000]:
    comparisons=[]
    for pde in PDES:
     fm=ix[task,pde,field,'FM4PDE',100];diff=ix[task,pde,field,'DiffusionPDE',steps]
     comparisons.append(dict(pde=pde,fm_mean_percent=100*float(fm['mean']),diffusion_mean_percent=100*float(diff['mean']),fm_lower_mean=float(fm['mean'])<float(diff['mean']),mean_tie=float(fm['mean'])==float(diff['mean']),fm_lower_sd=float(fm['sd'])<float(diff['sd'])))
    out.append(dict(task=task,field=field,diffusion_steps=steps,fm_lower_mean_count=sum(r['fm_lower_mean'] for r in comparisons),n_pdes=4,comparisons=comparisons))
 return out

def export(a,prepared):
 summary,gate_hashes=require_complete(a);main=copy.deepcopy(prepared['main']);provenance=copy.deepcopy(prepared['provenance']);diffusion=copy.deepcopy(prepared['diffusion']);changes=[]
 for rows,source in [(main,'source_data/main_figure_values.csv'),(provenance,'audit/table_value_provenance.csv')]:
  changed=0
  for r in rows:
   if r['table'] not in SPECS or pde_name(r['PDE'])!='nsnonbounded' or r['method']!='FM4PDE':continue
   setting=SPECS[r['table']][0];field=r['metric'][-2];s=summary[r['distribution'].lower(),setting];old={k:r[k] for k in ['mean','sd']}
   r.update(mean=s[f'error_{field}_mean'],sd=s[f'error_{field}_sd'],n='1000',sd_status='independently_recomputed_sample_sd',source_sd=s[f'error_{field}_sd'],file=str(a.study/'tables/ns_main_summary.csv'),sheet='ns_main_summary',row=str(list(summary).index((r['distribution'].lower(),setting))+2),result_source='ns_main_revision_0909',provenance_flag='complete_15_cells_raw_verified',rank='',display='')
   changes.append(dict(source=source,key=list(main_key(r)),before=old,after={k:r[k] for k in ['mean','sd']},summary_cell=[r['distribution'].lower(),setting],summary_field=field));changed+=1
  assert changed==18,(source,changed)
 for r in diffusion:
  if r['pde']!='nsnonbounded' or r['method']!='FM4PDE' or r['task'] not in DIFF_TASKS:continue
  s=summary['smooth',DIFF_TASKS[r['task']][0]];old={k:r[k] for k in ['mean','sd']};field=r['field']
  r.update(mean=s[f'error_{field}_mean'],sd=s[f'error_{field}_sd'],n='1000',source_file=str(a.study/'tables/ns_main_summary.csv'),source_sheet='ns_main_summary',source_row=str(list(summary).index(('smooth',DIFF_TASKS[r['task']][0]))+2))
  changes.append(dict(source='source_data/diffusion_comparison_summary.csv',key=list(diff_key(r)),before=old,after={k:r[k] for k in ['mean','sd']}))
 assert len(changes)==40 # 18 changes in each duplicated main source plus four generative-comparison cells.
 ix=unique_index(main,main_key);di=unique_index(diffusion,diff_key);blocks=table_blocks((a.paper/'fm4pde_jmlr_revision_0906.tex').read_text());out=a.output/'exported';out.mkdir(exist_ok=True)
 # Refuse to erase a concurrent manuscript data change. Captions/styles may evolve.
 for label in SPECS:validate_rendered_values(blocks[label],main_body(label,unique_index(prepared['main'],main_key)))
 for task in DIFF_TASKS:validate_rendered_values(blocks['tab:diffusion-'+task],diffusion_body(task,unique_index(prepared['diffusion'],diff_key)))
 for label in LABELS:
  body=main_body(label,ix) if label in SPECS else diffusion_body(label.split('-')[-1],di)
  (out/(label.removeprefix('tab:')+'.tex')).write_text(replace_body(blocks[label],body)+'\n')
 writecsv(out/'main_figure_values.csv',main);writecsv(out/'table_value_provenance.csv',provenance);writecsv(out/'diffusion_comparison_summary.csv',diffusion)
 writejson(out/'source_changes.json',changes)
 writejson(out/'statistics_to_integrate.json',dict(main_before=main_statistics(prepared['main']),main_after=main_statistics(main),diffusion_before=diffusion_statistics(prepared['diffusion']),diffusion_after=diffusion_statistics(diffusion),macro_definition='Unweighted arithmetic mean of the 12 PDE-distribution means, expressed in percent; no pooled SD or paired uncertainty inferred.'))
 preamble=(a.paper/'fm4pde_jmlr_revision_0906.tex').read_text().split(r'\begin{document}')[0]
 preview=out/'table_fragments_preview.tex'
 preview.write_text(preamble+r'\begin{document}'+'\n'+'\n'.join(r'\clearpage'+'\n'+(out/(label.removeprefix('tab:')+'.tex')).read_text() for label in LABELS)+r'\end{document}')
 build=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(out),str(preview)],cwd=a.paper,text=True,capture_output=True)
 (out/'table_fragments_preview_build.log').write_text(build.stdout+build.stderr)
 assert build.returncode==0,'Table-fragment LaTeX compilation failed'
 assert 'Overfull '+chr(92)+'hbox' not in build.stdout and 'Float too large' not in build.stdout,'Table layout exceeds JMLR text area'
 outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='export_manifest.json'}
 writejson(out/'export_manifest.json',dict(status='pass',input_gate_hashes=gate_hashes,prepared_sources_sha256=sha(a.output/'prepared_sources.json'),tables=8,table_fragment_compilation='pass; no horizontal overflow or oversized floats',manuscript_template_sha256=sha(a.paper/'fm4pde_jmlr_revision_0906.tex'),main_fm_entries_changed=18,diffusion_fm_entries_changed=4,all_row_ranks_recomputed=True,unchanged_scope='All non-NS data, all baselines, archived NS ablations and three-seed data, physics baselines and Burgers.',outputs=outputs))
 pending=json.loads((a.output/'pending_updates.json').read_text());pending.update(status='complete_verified_export',export_manifest=str(out/'export_manifest.json'));writejson(a.output/'pending_updates.json',pending)
 print('EXPORTED 8 reviewable tables after the complete 15 x 1000 raw-audit gate:',out)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','export']);parser.add_argument('--paper',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--study',type=Path,default=Path('/home/tat512/C01Python/audit/ns_main_revision_0909'));parser.add_argument('--output',type=Path);a=parser.parse_args();a.paper=a.paper.resolve();a.study=a.study.resolve();a.output=(a.output or a.paper/'audit/revision_0909/ns_table_updates').resolve()
 assert (a.paper/'audit').resolve() in a.output.parents,'Output must remain beneath paper/audit'
 prepared=prepare(a)
 if a.mode=='export':export(a,prepared)
 else:print('PREPARED source-backed update plan; no new result tables exported:',a.output)
if __name__=='__main__':main()

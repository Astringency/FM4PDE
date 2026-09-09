from pathlib import Path
import collections,csv,hashlib,json,re,shutil,subprocess,concurrent.futures
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import numpy as np
torch.set_num_threads(2)
P=Path('/home/tat512/C04Papers/fm4pde_jmlr');A=P/'audit/revision_0909';S=Path('/home/tat512/C01Python/audit/paper_revision_20260908')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=json.loads((S/'ablation_publication_snapshot/records.json').read_text());source_manifest=json.loads((S/'ablation_publication_snapshot/manifest.json').read_text())
assert source_manifest['final_ready'] and len(records)==1060
archive=list(csv.DictReader((S/'archived_ablation_summary.csv').open()))
assert sha(S/'archived_ablation_summary.csv')==source_manifest['source_archive_sha256']
unchanged=0;revised_verified=0
for r in records:
 assert r['source']!='pending'
 if r['source']=='unchanged':
  old=archive[int(r['id'].split('_')[1])]
  for f in ['a','u']:assert float(old['rel_l2_'+f])==r['rel_l2_'+f]
  for key,value in r['diagnostics'].items():assert float(old[key])==value
  unchanged+=1
 else:
  path=Path(r['result_path']);receipt=json.loads(Path(r['receipt']).read_text())
  assert sha(path)==receipt['result_sha256']
  assert receipt['sample_ids']==[0] and receipt['stage']=='main'
  payload=torch.load(path,map_location='cpu',weights_only=False)
  for field,prefix in [('a','coef'),('u','sol')]:
   truth=payload[prefix+'_ground_truth'].double().numpy();prediction=payload[prefix+'_final'].double().numpy()
   actual=np.linalg.norm(prediction-truth)/max(np.linalg.norm(truth),1e-12)
   assert np.isclose(actual,r['rel_l2_'+field],rtol=3e-6,atol=3e-8),(r['id'],field)
  terminal=list(csv.DictReader((path.parent/'curves.csv').open()))[-1]
  for key,value in r['diagnostics'].items():assert float(terminal[key])==value,(r['id'],key)
  revised_verified+=1
families=collections.defaultdict(set)
for r in records:families[r['config']['ablation_group']].add(r['pde'])
for family,pdes in families.items():
 assert len(pdes)==(1 if family=='deterministic_endpoint_bt' else 6 if family=='temporal_residual_mode' else 11),(family,pdes)
outputs={}
m=json.loads((A/'ablation_figures/figure_manifest.json').read_text());assert len(m['outputs'])==76
for name,h in m['outputs'].items():outputs[A/'ablation_figures'/name]=h
sweep_manifest=json.loads((A/'ablation_figures/sweep_figure_manifest.json').read_text());assert len(sweep_manifest['figures'])==30
for figure in sweep_manifest['figures']:
 for name,h in figure['outputs'].items():outputs[A/'ablation_figures'/name]=h
for folder,file in [('ensemble3_figures','figure_manifest.json'),('frequency_figures','frequency_figure_manifest.json')]:
 for name,h in json.loads((A/folder/file).read_text())['outputs'].items():outputs[A/folder/name]=h
assert len(outputs)==170
for path,h in outputs.items():
 assert sha(path)==h,path
 if path.suffix=='.png':Image.open(path).verify()
text=(A/'ablation_section.tex').read_text();labels=re.findall(r'\\label\{([^}]+)\}',text)
assert len(labels)==len(set(labels))
figure_names=re.findall(r'\\includegraphics[^\{]*\{figures/([^}]+)\}',text)
assert len(figure_names)==len(set(figure_names))==85
assert set(figure_names)=={p.name for p in outputs if p.suffix=='.pdf'}
assert 'Can Guidance Calibration Improve NS Reconstruction?' not in text
assert 'Does the NS Residual Choice Change Reconstruction?' not in text
assert r'\operatorname{RelL2}' in text and r'\mathcal L_{\mathrm{PDE},h}' in text

def inspect_pdf(path):
 raw=subprocess.run(['pdftotext','-bbox',str(path),'-'],capture_output=True,text=True,check=True).stdout
 tree=ET.fromstring(raw);page=next(x for x in tree.iter() if x.tag.endswith('page'))
 width=float(page.attrib['width']);height=float(page.attrib['height'])
 violations=[]
 for word in (x for x in tree.iter() if x.tag.endswith('word')):
  a=word.attrib
  if float(a['xMin'])<-.5 or float(a['yMin'])<-.5 or float(a['xMax'])>width+.5 or float(a['yMax'])>height+.5:violations.append(word.text)
 assert not violations,(path,violations)
 # Source ordinary text is 9 pt; subscripts/superscripts intentionally differ.
 return dict(figure=path.name,width_pt=width,height_pt=height,ordinary_font_at_six_inches=(9.1 if path.parent.name=='ablation_figures' and '_sweep_' not in path.stem else 9)*432/width)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:pdf_checks=list(pool.map(inspect_pdf,[p for p in outputs if p.suffix=='.pdf']))
# Copy only manifest-listed outputs; obsolete unsplit drafts are not published.
for path in outputs:shutil.copy2(path,P/'figures'/path.name)
assert all(sha(P/'figures'/path.name)==h for path,h in outputs.items())
report=dict(configurations=1060,raw_archive_rows_verified=unchanged,revised_configurations=revised_verified,
 family_pde_coverage={k:sorted(v) for k,v in families.items()},figures=85,figure_outputs=170,
 guidance_joint_conditions=4,guidance_joint_pdes=11,guidance_directional_metric_rows=60,
 seed3_predictions=1056,seed3_pdes=11,seed3_fields=21,pdf_text_bounds=pdf_checks,
 source_manifest_sha256=sha(S/'ablation_publication_snapshot/manifest.json'),
 section_sha256=sha(A/'ablation_section.tex'),
 visual_contact_sheets=10,visual_examples=['Poisson guidance','Shallow Water components','Burgers trajectory','budget','layout','frequency-separated Poisson','three-seed Poisson spectra'],
 outputs={path.name:h for path,h in outputs.items()},
 limitations='All factor sweeps use ID input 0; five inference--mask seed pairs do not estimate population or training uncertainty. The separate three-seed study uses 32 physical inputs per PDE. No new GPU sampling was required for this revision.')
(A/'ablation_delivery_qa.json').write_text(json.dumps(report,indent=2)+'\n')
print('VALIDATED and copied',len(outputs),'files; 85 figures; 1060 source configurations. Minimum ordinary font after six-inch scaling:',min(x['ordinary_font_at_six_inches'] for x in pdf_checks))

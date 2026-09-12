"""Independent numerical, source, and document checks for the full revision."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

PAPER=Path('/home/tat512/C04Papers/fm4pde_jmlr')
DATA=PAPER/'source_data/revision_0912_full'
FIGS=PAPER/'figures/revision_0912_full'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def table_rows(name):
    return [[cell.strip().strip('$') for cell in line.rstrip().removesuffix(r'\\').strip().split('&')]
            for line in (FIGS/name).read_text().splitlines() if line.strip()]

def main():
    meta=json.loads((DATA/'metadata.json').read_text());data=np.load(DATA/'fields.npz')
    assert len(meta)==58
    for key,r in meta.items():
        assert r['sample_ids']==[0] and r['config']['num_steps']==100 and r['status']=='ok'
        assert r['environment']['tf32'] is False and r['environment']['batch_size']==1
        for f in ['a','u']:
            truth=data[f'{key}__{f}__truth'];pred=data[f'{key}__{f}__pred'];mask=data[f'{key}__{f}__mask']
            diff=pred-truth
            # Squared sums independently reproduce the norm-based renderer.
            for metric,m in [('full',1),('observed',mask)]:
                score=float(np.sqrt(np.sum((diff*m)**2)/np.sum((truth*m)**2)))
                assert np.isclose(score,r['errors'][f][metric],rtol=1e-12,atol=1e-14)
    summary=json.loads((DATA/'layout_summary.json').read_text())
    summary_rows=[]
    seed_rows=[]
    for layout in ['random','fixed','grid','columns']:
        cells=[layout.title()]
        for cond in ['separate','shared']:
            for f in ['a','u']:
                scores=[100*meta[f'helmholtz_{layout}_{s}_{cond}']['errors'][f]['full'] for s in range(5)]
                mean=sum(scores)/5;sd=(sum((v-mean)**2 for v in scores)/4)**.5
                assert np.isclose(mean,summary[layout][cond][f]['mean_percent'])
                assert np.isclose(sd,summary[layout][cond][f]['sd_percent'])
                cells.append(f'{mean:.2f}\\pm{sd:.2f}')
        summary_rows.append(cells)
        for seed in range(5):
            seed_rows.append([layout.title(),str(seed)]+[
                f"{100*meta[f'helmholtz_{layout}_{seed}_{cond}']['errors'][f]['full']:.2f}"
                for cond in ['separate','shared'] for f in ['a','u']])
    assert table_rows('layout_summary_rows.tex')==summary_rows
    assert table_rows('layout_seed_rows.tex')==seed_rows
    temporal=[r for r in meta.values() if r['pde']!='helmholtz']
    assert len(temporal)==18 and len({r['pde'] for r in temporal})==6
    temporal_rows=[]
    for pde,name in [('nsnonbounded','Navier--Stokes'),('reaction_diffusion','Reaction--Diffusion'),
                     ('shallow_water','Shallow Water'),('heat','Heat'),('wave','Wave'),
                     ('advection_diffusion','Advection--Diffusion')]:
        temporal_rows.append([name]+[
            f"{100*meta[f'{pde}_{mode}_0_separate']['errors'][f]['full']:.2f}"
            for mode in ['endpoint_secant','hermite_bridge','near_endpoint_temporal'] for f in ['a','u']])
    assert table_rows('temporal_rows.tex')==temporal_rows
    manifest=json.loads((DATA/'figure_manifest.json').read_text())
    for f,h in manifest['inputs'].items():assert sha(DATA/f)==h
    for f,h in manifest['outputs'].items():assert sha(FIGS/f)==h
    docs={}
    for stem in ['fm4pde_jmlr_revision_0906','supplement']:
        p=PAPER/f'{stem}.pdf';info=subprocess.check_output(['pdfinfo',str(p)],text=True)
        pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
        if stem!='supplement':assert pages<=100
        log=(PAPER/f'{stem}.log').read_text()
        for error in ['Overfull','undefined','multiply defined','Float too large','! LaTeX Error',
                      'referenced but does not exist']:
            assert error not in log,(stem,error)
        txt=subprocess.check_output(['pdftotext',str(p),'-'],text=True)
        assert 'REVISION PENDING' not in txt and '[PENDING' not in txt
        if stem!='supplement':
            for phrase in ['Temporal Residuals','Separate locations','Near-endpoint','Shared locations']:assert phrase in txt
        docs[stem]={'pages':pages,'sha256':sha(p)}
    result={'validated_runs':len(meta),'temporal_runs':18,'layout_runs':40,'documents':docs,
            'table_hashes':{name:sha(FIGS/name) for name in ['layout_summary_rows.tex','layout_seed_rows.tex','temporal_rows.tex']},
            'checks':'raw scores, all channels, source/output hashes, seed mean/SD, exact ordered table cells, page cap, LaTeX references and overflow'}
    (PAPER/'audit/revision_0912_full/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

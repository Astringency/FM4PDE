"""Check the evidence and document bindings of the September 12 revision."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

PAPER=Path('/home/tat512/C04Papers/fm4pde_jmlr')
DATA=PAPER/'source_data/revision_0912'


def main():
    m=json.loads((DATA/'metadata.json').read_text())
    a=np.load(DATA/'fields.npz')
    candidates=json.loads((DATA/'selection_candidates.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    checked=0
    for key,record in m.items():
        for f in ['a','u']:
            t,p,mask=[a[f'{key}__{f}__{kind}'] for kind in ['truth','pred','mask']]
            assert t.shape==p.shape==mask.shape and np.isfinite(p).all()
            assert set(np.unique(mask)) <= {0,1}
            assert np.all(mask.sum(axis=(-2,-1))==500)
            diff=(p-t).ravel();tf=t.ravel();mf=mask.ravel()
            # Squared sums are an independent implementation of the reported norms.
            full=np.sqrt(np.dot(diff,diff)/np.dot(tf,tf))
            obs=np.sqrt(np.sum((diff*mf)**2)/np.sum((tf*mf)**2))
            assert np.isclose(full,record['errors'][f]['full'],rtol=1e-12)
            assert np.isclose(obs,record['errors'][f]['observed'],rtol=1e-12)
            checked+=1
    extra=['reaction_diffusion','shallow_water','heat','wave','advection_diffusion','steady_heat_conduction']
    for key in extra:
        pool=[c for c in candidates if c['pde']==key]
        assert len(pool)==96 and len({c['sample_id'] for c in pool})==32
        best=min(pool,key=lambda c:(max(c['error_a'],c['error_u']),c['sample_id'],c['seed']))
        assert (best['sample_id'],best['seed'])==(m[key]['sample_id'],m[key]['seed'])
        for c in pool:assert np.isclose(c['score'],max(c['error_a'],c['error_u']))
        assert m[key]['config']['sampler_phase']=='stochastic'
        assert m[key]['config']['num_steps']==100
    ns=[k for k in m if k.startswith('ns_')]
    assert len(ns)==8
    common=m[ns[0]]['config']
    for key in ns:
        diffs={k for k,v in m[key]['config'].items() if v!=common[k]}
        assert diffs <= {'output_dir','sampler_phase','switch_ratio'}
        for f in ['a','u']:
            for kind in ['truth','mask']:
                assert np.array_equal(a[f'{key}__{f}__{kind}'],a[f'{ns[0]}__{f}__{kind}'])
    for seed in range(5):
        i=f'layout_{seed}_independent';s=f'layout_{seed}_shared'
        diffs={k for k,v in m[i]['config'].items() if v!=m[s]['config'][k]}
        assert diffs <= {'output_dir','shared_mask'}
        assert m[i]['config']['sample_seed']==m[s]['config']['sample_seed']==seed
        assert m[i]['config']['zeta_obs_a']==2.5e4 and m[i]['config']['zeta_obs_u']==1e8
        assert m[i]['config']['zeta_pde']==.1
        assert np.array_equal(a[f'{i}__a__mask'],a[f'{s}__a__mask'])
        assert np.array_equal(a[f'{s}__a__mask'],a[f'{s}__u__mask'])
        for key in [i,s]:
            overlap=int(np.sum(a[f'{key}__a__mask']*a[f'{key}__u__mask']))
            assert overlap==m[key]['overlap']
        for f in ['a','u']:
            assert np.array_equal(a[f'{i}__{f}__truth'],a[f'{s}__{f}__truth'])
    for field in ['a','u']:
        for layout in ['independent','shared']:
            values=[100*m[f'layout_{s}_{layout}']['errors'][field]['full'] for s in range(5)]
            summary=json.loads((DATA/'shared_summary.json').read_text())[layout][field]
            assert np.isclose(np.mean(values),summary['mean_percent'])
            assert np.isclose(np.std(values,ddof=1),summary['sd_percent'])
    manifest=json.loads((DATA/'figure_manifest.json').read_text())
    for name,value in manifest['inputs'].items():assert sha(DATA/name)==value
    for name,value in manifest['outputs'].items():assert sha(PAPER/'figures/revision_0912'/name)==value
    main_tex=(PAPER/'fm4pde_jmlr_revision_0906.tex').read_text()
    sup_tex=(PAPER/'supplement.tex').read_text()
    assert 'input{exact_tilt_ablation_0911.tex}' not in main_tex
    assert 'input{exact_tilt_protocol_0911.tex}' not in main_tex
    for name in ['exact_tilt_ablation_0911.tex','exact_tilt_protocol_0911.tex']:
        assert 'input{'+name+'}' in sup_tex
    for name in ['additional_multicomponent','additional_transport','guidance_state_burgers',
                 'switch_sensitivity_ns','shared_sensor_reconstruction']:
        assert 'figures/revision_0912/'+name+'.pdf' in main_tex
    info=subprocess.check_output(['pdfinfo',str(PAPER/'fm4pde_jmlr_revision_0906.pdf')],text=True)
    pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));assert pages<=100
    for stem in ['fm4pde_jmlr_revision_0906','supplement']:
        log=(PAPER/(stem+'.log')).read_text()
        assert not re.search(r'undefined|multiply defined|Overfull|^!',log,re.M)
    result=dict(field_tensor_checks=checked,selection_candidates=len(candidates),additional_pdes=6,
                switch_conditions=len(ns),paired_sensor_seeds=5,main_pages=pages,
                source_hashes_verified=True,exact_tilt_in_supplement=True,
                no_unresolved_references_or_overfull_boxes=True,
                limitation='Post hoc low-error illustrations and single-input sensitivity; no population advantage claimed.')
    (DATA/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

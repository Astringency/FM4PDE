"""Trace main-table FM4PDE cells to all archived batch configurations.

The author-selected workbook rows control selection. No later defaults or
best-performing replacement runs are used. Verify sample-weighted batch means
against every FM4PDE field mean in the paper's value provenance.
"""
from pathlib import Path
import collections
import csv
import gzip
import hashlib
import json
import math
import yaml

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parents[1]/'C04Papers/fm4pde_jmlr'
FIELDS=['zeta_obs_a','zeta_obs_u','zeta_pde','stochastic_guidance_coeff','clip_mode','clip_threshold',
        'residual_mode','pde_guidance_start_ratio','pde_guidance_ramp_ratio','sampler_phase',
        'num_steps','time_grid','step_method','loss_state','sensor_mode','num_obs','num_sensor_columns',
        'shared_mask','mask_seed','sample_seed','dtype','checkpoint_path','guidance_schedule',
        'obs_decay','obs_decay_start_ratio','coef_positive_mode','stochastic_guidance_time']


def main():
    selection=list(csv.DictReader((PAPER/'audit/fm4pde_record_selection_inherited.csv').open(encoding='utf-8-sig')))
    provenance=[r for r in csv.DictReader((PAPER/'audit/table_value_provenance.csv').open(encoding='utf-8-sig'))
                if r['method']=='FM4PDE' and r['file']=='FM4PDE0905_main_summary.xlsx']
    used={(r['sheet'],r['row']) for r in provenance}
    selected=[r for r in selection if r['selected']=='True' and (r['sheet'],r['row']) in used]
    groups={}; records=[]; summaries=[]
    for row in selected:
        name=row['Remark']; folder=ROOT/'outputs/main'/name
        if name not in groups:
            groups[name]=list(csv.DictReader((folder/'summary_all_raw.csv').open()))
        sensor='sensor_column' if row['SENSOR']=='sensor_col' else row['SENSOR']
        batches=[r for r in groups[name] if (r['pde'],r['task'],r['sensor_mode'])==(row['PDE'],row['TASK'],sensor)
                 and Path(r['run_dir']).relative_to('outputs/'+name).parts[0]==row['PDE']]
        assert batches,(name,row)
        counts=[int(r['num_samples']) for r in batches]
        assert sum(counts)==1000,(row,sum(counts))
        offsets=[]; configs=[]
        for batch in batches:
            assert batch['status']=='ok' and batch['synthetic_data']=='False'
            offset=int(batch['offset']); offsets.extend(range(offset,offset+int(batch['num_samples'])))
            relative=Path(batch['run_dir']).relative_to('outputs/'+name)
            path=folder/relative/'resolved_config.yaml'
            text=path.read_bytes(); config=yaml.safe_load(text)
            assert config['batch_size']==int(batch['batch_size'])
            assert config['offset']==offset
            for key in FIELDS:
                if key in batch and batch[key] not in ['',None] and key in config:
                    val=config[key]
                    if isinstance(val,(int,float)) and not isinstance(val,bool):
                        assert math.isclose(float(batch[key]),val,rel_tol=1e-10,abs_tol=1e-12),(path,key)
                    elif isinstance(val,str): assert batch[key]==val,(path,key)
            configs.append(config)
            records.append(dict(sheet=row['sheet'],workbook_row=row['row'],source=str(path),
                                sha256=hashlib.sha256(text).hexdigest(),config=config))
        assert sorted(offsets)==list(range(1000)),row
        for field in provenance:
            if (field['sheet'],field['row'])!=(row['sheet'],row['row']): continue
            metric='rel_l2_a' if '(a)' in field['metric'] else 'rel_l2_u'
            mean=sum(float(b[metric])*n for b,n in zip(batches,counts))/1000
            assert math.isclose(mean,float(field['mean']),rel_tol=1e-8,abs_tol=1e-10),(row,metric,mean,field['mean'])
        varying={key:sorted({json.dumps(c.get(key),sort_keys=True) for c in configs}) for key in FIELDS}
        varying={k:v for k,v in varying.items() if len(v)>1}
        # Batch sizes may differ across resumed runs, but a main-result cell
        # must not silently pool different guidance settings.
        assert not varying,(row,varying)
        summaries.append(dict(pde=row['PDE'],task=row['TASK'],distribution=row['DIST'],
                              observations='full' if row['sheet']=='FM4PDE_FULL' else sensor,
                              source_root=name,sheet=row['sheet'],workbook_row=row['row'],n=1000,
                              batch_sizes=';'.join(map(str,sorted({c['batch_size'] for c in configs}))),
                              archived_batches=len(configs),**{k:configs[0].get(k) for k in FIELDS}))
        print('VERIFIED',row['PDE'],row['TASK'],row['DIST'],row['sheet'],len(configs),flush=True)
    out=PAPER/'source_data'
    with (out/'main_hyperparameters_verified.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    with gzip.open(out/'main_configs_archive.json.gz','wt') as f:
        json.dump(dict(records=records,verification=dict(cells=len(summaries),batches=len(records),
                    source='All archived configurations for author-selected main-table cells; weighted means and offsets checked.',
                    limitations='Offsets/counts are checked from original batch exports; this is not a tensor-level audit of all main predictions.')),f,allow_nan=False)
    print('COMPLETE',len(summaries),'cells',len(records),'batch configs')


if __name__=='__main__':main()

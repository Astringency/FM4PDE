"""Freeze the two existing Burgers baseline observation protocols, read-only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    target=args.output
    target.mkdir(parents=True,exist_ok=True)
    assert not (target/'protocol.json').exists(), 'Already frozen'
    sources=json.loads(args.sources.read_text())
    records=[]; cells={}; source_records=[]
    def read_one(path):
        d=torch.load(path,map_location='cpu',weights_only=False)
        assert d['pde_name']=='burger' and d['target_fields'].shape==(1,128,128)
        truth=d['target_fields'].clone();mask=d['mask'].clone();pred=d['prediction'].clone()
        assert torch.isfinite(truth).all() and torch.isfinite(pred).all()
        e=float(torch.linalg.vector_norm((pred-truth).double())/torch.linalg.vector_norm(truth.double()))
        return dict(sample_id=int(d['sample_index']),truth=truth,mask=mask,error=e,
                    source=str(path),source_sha256=digest(path),
                    geometry=d['metadata']['effective_sensor_mode'])
    for row in sorted(sources['baselines'],key=lambda r:(r['distribution'],r['mode'],r['method']!='RecFNO',r['method'])):
        run=Path(row['run'])
        dirs=list(run.glob('*_samples'));assert len(dirs)==1,(run,dirs)
        paths=sorted(dirs[0].glob('sample_*.pt'));assert len(paths)==1000,(run,len(paths))
        key=row['distribution']+'_'+row['mode']
        current=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for item in pool.map(read_one,paths):
                current.append(item)
        assert [r['sample_id'] for r in current]==list(range(1000)),run
        truth=torch.stack([r['truth'] for r in current])
        masks=torch.stack([r['mask'] for r in current])
        expected=500 if row['mode']=='random' else 640
        assert (masks.flatten(1).sum(1)==expected).all(),(key,expected)
        if row['mode']=='structured':
            assert ((masks.sum(-1)==128).sum(-1)==5).all(),key
        if key not in cells:
            assert row['method']=='RecFNO'
            cells[key]=dict(truth=truth,mask=masks.to(torch.uint8),sample_ids=list(range(1000)),
                            distribution=row['distribution'],mode=row['mode'])
            torch.save(cells[key],target/f'{key}.pt')
        same_truth=torch.equal(truth,cells[key]['truth'])
        same_masks=torch.equal(masks,cells[key]['mask'].to(masks.dtype))
        assert same_truth,(key,row['method'],'truth mismatch')
        for item in current:
            records.append(dict(distribution=row['distribution'],mode=row['mode'],method=row['method'],
                sample_id=item['sample_id'],rel_l2_u=item['error'],source=item['source'],
                source_sha256=item['source_sha256'],same_reference_masks=same_masks,geometry=item['geometry']))
        source_records.append(dict(**row,samples=len(current),same_truth=same_truth,same_masks=same_masks,
            mean_percent=float(np.mean([r['error'] for r in current]))*100,
            sd_percent=float(np.std([r['error'] for r in current],ddof=1))*100))
        print('VERIFIED',key,row['method'],'mask_equal',same_masks,'mean',source_records[-1]['mean_percent'],flush=True)
        del current,truth,masks
    with (target/'baseline_per_input.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    protocol=dict(version=1,baselines=source_records,examples_per_cell=1000,sample_ids=list(range(1000)),
        modes={'random':'500 space-time points','structured':'5 complete time levels; 640 scalar values'},
        truth_layout='N,C,T,X; complete 128-time-level trajectory including initial level',
        fm_steps=[100],diffusion_steps=[100,1000],diffusion_distribution='smooth',
        new_fm_cells=['id_structured','smooth_structured','rough_structured'],
        new_diffusion_cells=['smooth_random','smooth_structured'],
        source_manifest_sha256=digest(args.sources),
        artifacts={p.name:digest(p) for p in target.glob('*.pt')},
        baseline_metrics_sha256=digest(target/'baseline_per_input.csv'),
        source_configs=sources['configs'])
    (target/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('COMPLETE',len(records),'baseline predictions;',len(cells),'input cells',flush=True)


if __name__=='__main__':main()

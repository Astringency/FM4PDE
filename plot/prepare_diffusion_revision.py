"""Freeze legacy Smooth inputs and finish metrics in a NEW output directory.

Read-only with respect to the original DiffusionPDE checkout/results. This
script runs from a separate Git checkout. CUDA tensors inside result pickles
are remapped to CPU; no inference, training, or original output is modified.
"""
from __future__ import annotations
import argparse
import functools
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import re
import sys

PDES = ['poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger']


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diffusion-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metrics',action='store_true')
    args=p.parse_args()
    import numpy as np
    import scipy.io
    import torch
    import yaml
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('diffusion_evaluate_frozen',args.diffusion_root/'evaluate_results.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self,module,name):
            if module=='torch.storage' and name=='_load_from_bytes':
                return lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
            return super().find_class(module,name)
    def cpu_result(path):
        with Path(path).open('rb') as f:return CPUUnpickler(f).load()
    ev.load_result=cpu_result
    # Original scipy loader reads the full MATLAB file for every example.
    # Cache only the current PDE's file; clear between PDEs.
    original_loadmat=scipy.io.loadmat
    scipy.io.loadmat=functools.lru_cache(maxsize=1)(original_loadmat)
    ids=np.random.default_rng(20260907).choice(1000,21,replace=False).tolist()
    info={'evaluation_ids':ids[:20],'pilot_id':ids[20],
          'selection':'20 IDs drawn without replacement before new timing; separate one-example pilot',
          'distribution':'legacy Smooth','data_sources':{},'diffusion_configs':{},
          'evaluator_sha256':digest(args.diffusion_root/'evaluate_results.py')}
    arrays={};records=[];inventory=[]
    for pde in PDES:
        base=args.diffusion_root/'outputs/MAIN1000_100'
        conf=base/'.sample_sweep/configs'/f'{pde}_both.yaml'
        config=yaml.safe_load(conf.read_text())
        info['diffusion_configs'][pde]=config
        source=Path(config['data']['datapath'])
        info['data_sources'][pde]={'path':str(source),'size':source.stat().st_size,
                                  'mtime_ns':source.stat().st_mtime_ns}
        a=[];u=[]
        for i in ids:
            aa,uu,_=ev.load_ground_truth(config,pde,i)
            a.append((uu if aa is None else aa).numpy())
            u.append(uu.numpy())
        arrays[pde+'_a']=np.concatenate(a).astype('float32')
        arrays[pde+'_u']=np.concatenate(u).astype('float32')
        if args.metrics:
            for steps in [100,1000]:
                base=args.diffusion_root/'outputs'/f'MAIN1000_{steps}'
                for task in (['both'] if pde=='burger' else ['forward','inverse','both']):
                    conf=base/'.sample_sweep/configs'/f'{pde}_{task}.yaml'
                    cfg=yaml.safe_load(conf.read_text())
                    assert cfg['test']['iterations']==steps
                    predictions=sorted((base/'samples'/pde/task).glob('*_results.pkl'))
                    actual_ids=[ev.infer_offset(f,-1) for f in predictions]
                    assert len(actual_ids)==1000 and set(actual_ids)==set(range(1000)),(steps,pde,task,len(actual_ids))
                    dest=args.output/'metrics'/str(steps)/pde/task
                    dest.mkdir(parents=True,exist_ok=True)
                    for f in predictions:
                        old=base/'metrics'/pde/task/(f.stem+'_metrics_final.json')
                        if old.exists():
                            row=json.loads(old.read_text());metric_source=old;origin='original_completed_metric'
                        else:
                            own=dest/(f.stem+'_metrics_final.json')
                            row=json.loads(own.read_text()) if own.exists() else ev.compute_metrics(str(conf),str(f),str(dest),None,task)
                            metric_source=own;origin='new_CPU_evaluation_of_existing_prediction'
                        assert row['pde']==pde and row['problem']==task
                        assert int(row['offset'])==ev.infer_offset(f,-1)
                        assert np.isfinite(row['rel_l2_u'])
                        if pde!='burger':assert np.isfinite(row['rel_l2_a'])
                        records.append(dict(row,steps=steps,source_path=str(metric_source),
                                            source_sha256=digest(metric_source),metric_origin=origin,
                                            prediction_path=str(f),prediction_sha256=digest(f),
                                            config_sha256=digest(conf)))
                    inventory.append({'steps':steps,'pde':pde,'task':task,'examples':1000})
                    print('VERIFIED',steps,pde,task,1000,flush=True)
        scipy.io.loadmat.cache_clear()
        print('FROZEN INPUTS',pde,flush=True)
    np.savez_compressed(args.output/'timing_truths.npz',**arrays)
    info['timing_truths_sha256']=digest(args.output/'timing_truths.npz')
    (args.output/'input_inventory.json').write_text(json.dumps(info,indent=2)+'\n')
    if args.metrics:
        with gzip.open(args.output/'diffusion_snapshot_20260907.json.gz','wt') as f:
            json.dump(dict(records=records,inventory=inventory,input_inventory=info,
                           scope='Existing legacy Smooth predictions; original completed metrics plus new CPU metrics. No new accuracy sampling.'),f,allow_nan=False)
        print('COMPLETE',len(records),'metric records',flush=True)


if __name__=='__main__':main()

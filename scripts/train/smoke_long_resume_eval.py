"""CPU end-to-end checks on real checkpoints while both training GPUs are busy."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import torch

from sampling.model_io import load_fm4pde_checkpoint_bundle
from scripts.train.resume_study import file_sha,write
from scripts.train.main_resume_sampling import historical_inputs,validation_inputs,sample_once,paired_identity


def main(study,pdes):
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    out=study/'evaluation_smoke';out.mkdir(exist_ok=True)
    plan=json.loads((study/'study_plan.json').read_text());results=[]
    for pde in pdes:
        job=next(j for j in plan['jobs'] if j['pde']==pde)
        setting='random' if pde=='burger' else 'sparse_joint'
        record=json.loads((study/'evaluation_inputs'/pde/'id'/(setting+'.json')).read_text())
        sample_id=record['hardest_25'][0]['sample_id']
        rows=[r for r in record['rows'] if r['sample_id']==sample_id]
        models=[('source',Path(job['source']))]
        resume=study/pde/'last_resume.pth'
        if resume.exists():
            snapshot=out/pde/'resumed_snapshot.pth';snapshot.parent.mkdir(exist_ok=True)
            if not snapshot.exists():shutil.copyfile(resume,snapshot)
            models.append(('resumed',snapshot))
        receipts={}
        for label,path in models:
            bundle=load_fm4pde_checkpoint_bundle(str(path),pde,'cpu',model_profile='auto')
            digest=file_sha(path)
            gt,masks,ids=historical_inputs(record,rows,'cpu')
            receipt=sample_once(record['config'],bundle,str(path),digest,out/pde/label/'historical',
                gt,masks,ids,rows[0]['noise_source_size'],[rows[0]['noise_source_index']],device='cpu')
            receipts[label]={'historical':receipt}
            cache_path=study/pde/'sampling_validation_inputs.pt'
            if cache_path.exists():
                cache=torch.load(cache_path,map_location='cpu',weights_only=False)
                gt,masks,ids=validation_inputs(cache,record['config'],[0],'cpu')
                receipt=sample_once(record['config'],bundle,str(path),digest,out/pde/label/'validation',
                    gt,masks,ids,32,[0],device='cpu')
                receipts[label]['validation']=receipt
            del bundle;gc.collect()
            print('REAL_MODEL_SMOKE',pde,label,flush=True)
        if 'resumed' in receipts:
            for case in receipts['source']:paired_identity(receipts['source'][case],receipts['resumed'][case])
        results.append(dict(pde=pde,cases={label:{case:dict(result_sha256=row['result_sha256'],fields=row['fields'],seconds=row['seconds'])
            for case,row in cases.items()} for label,cases in receipts.items()}))
        write(out/'progress.json',dict(completed=results,requested=pdes))
    write(out/'complete.json',dict(status='complete',pdes=pdes,results=results,ended_unix=time.time(),
        scope='CPU functional end-to-end tests only, not GPU equivalence or formal accuracy results; immutable early resumed snapshots when available'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--pdes',nargs='+',default=['nsnonbounded','helmholtz','poisson','darcy','burger'])
    args=p.parse_args();main(args.study.resolve(),args.pdes)

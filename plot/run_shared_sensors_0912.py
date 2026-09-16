"""Paired Helmholtz sensor-location control for the September 12 revision.

One physical input, five paired sampling/mask seeds; batch one throughout.
Only shared_mask changes within each pair. Long-term output is explicit.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from sampling.config import AblationConfig
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.runner import run_single_ablation
from sampling.batching import combine_truths


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.output.is_absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads((args.inputs/'protocol.json').read_text())
    config = next(x['config'] for x in protocol['archived']
                  if x['config']['task']=='both' and x['config']['ablation_group']=='sensor_mode'
                  and x['config']['sensor_mode']=='random')
    assert sha(args.inputs/'weights.pth') == protocol['weights_sha256']
    assert sha(args.inputs/'truths.pt') == protocol['truth_sha256']
    env = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
               visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),batch_size=1,
               inputs=str(args.inputs),protocol_sha256=sha(args.inputs/'protocol.json'),
               weights_sha256=protocol['weights_sha256'],truth_sha256=protocol['truth_sha256'],
               tf32=False,seeds=list(range(5)),sample_id=0)
    (args.output/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    truths=torch.load(args.inputs/'truths.pt',map_location='cpu',weights_only=False)
    gt=combine_truths(truths,[0],'cuda:0')
    bundle=load_fm4pde_checkpoint_bundle(str(args.inputs/'weights.pth'),'helmholtz','cuda:0',model_profile='recommended')
    rows=[]
    for seed in range(5):
        for shared in [False,True]:
            folder=args.output/f'seed{seed}'/('shared' if shared else 'independent')
            receipt=folder/'receipt.json'
            if receipt.exists():
                rows.append(json.loads(receipt.read_text()))
                continue
            folder.mkdir(parents=True,exist_ok=True)
            c=dict(config,shared_mask=shared,sample_seed=seed,mask_seed=seed,
                   output_dir=str(folder),checkpoint_path=str(args.inputs/'weights.pth'),
                   device='cuda:0',batch_size=1,offset=0,save_plots=False,
                   save_intermediate=False,save_per_sample_curves=True,
                   ablation_name='shared_sensors_0912')
            cfg=AblationConfig(**c)
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            with (folder/'run.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
                result=run_single_ablation(cfg,checkpoint_bundle=bundle,ground_truth=gt)
            torch.cuda.synchronize()
            path=Path(result['run_dir'])/'result.pt'
            payload=torch.load(path,map_location='cpu',weights_only=False)
            masks=torch.load(path.parent/'masks.pt',map_location='cpu',weights_only=False)
            errors={}
            for f,k in [('a','coef'),('u','sol')]:
                t=payload[k+'_ground_truth'].double().cpu()
                v=payload[k+'_final'].double().cpu()
                m=masks[k].double().cpu()
                assert m.sum()==500 and torch.isfinite(v).all()
                errors[f]=dict(full=float((v-t).norm()/t.norm()),observed=float(((v-t)*m).norm()/(t*m).norm()))
            overlap=int((masks['coef']*masks['sol']).sum())
            assert (overlap==500)==shared
            row=dict(seed=seed,shared=shared,sample_ids=[0],config=dataclasses.asdict(cfg),
                     errors=errors,overlap=overlap,result_path=str(path),result_sha256=sha(path),
                     masks_sha256=sha(path.parent/'masks.pt'),seconds=time.perf_counter()-start,
                     peak_bytes=torch.cuda.max_memory_allocated(),status=result['status'])
            receipt.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print('DONE',seed,shared,errors,'peak GiB',row['peak_bytes']/2**30,flush=True)
            # The first full batch-one run is the memory pilot. Continue only
            # with comfortable device headroom; no batch-size extrapolation.
            assert row['peak_bytes'] < .7*torch.cuda.get_device_properties(0).total_memory
    (args.output/'complete.json').write_text(json.dumps(dict(environment=env,rows=rows),indent=2)+'\n')


if __name__=='__main__':
    main()

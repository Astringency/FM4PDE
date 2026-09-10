"""Paired physical-weight sweep with fixed development/confirmation cohorts.

Only zeta_pde varies. Four development inputs select a multiplier by lowest
physical MSE subject to no field's mean error increasing over Obs-only by
more than 2%. Eight separate confirmation inputs compare Obs-only, original,
and the selected multiplier. Every development result is retained.
"""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,json,math,os,socket,subprocess,sys,time
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sampling.config import AblationConfig
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.losses import _pde_params_with_residual_options
from sampling.pde_residuals import compute_pde_residual
from scripts.tuning.compare_pde_guidance_schedules import combine_truths
from run_paper_ablation_revision import digest,write,physical_errors
import sampling.runner as runner

MULTIPLIERS=[0,1,10,100,1000,10000,1000000]
DEV=[1100,1101,1102,1103]
CONFIRM=list(range(1500,1508))

def main(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    anchors=json.loads(args.anchors.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    for pde in args.pdes:
        target=args.output/pde;target.mkdir(exist_ok=True)
        with (target/'run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            source=args.inputs/pde;old=json.loads((source/'protocol.json').read_text())
            assert digest(source/'truths.pt')==old['truth_sha256']
            assert digest(source/'weights.pth')==old['weights_sha256']
            cfgbase=anchors[pde]['config']
            protocol=dict(pde=pde,anchor=anchors[pde],multipliers=MULTIPLIERS,
                development=DEV,confirmation=CONFIRM,batch_size=4,
                sample_seed=20260910,mask_seed=20260910,
                selection='min mean physical MSE; each field mean RelL2 <= 1.02 times Obs-only; multipliers > 1',
                weights_sha256=old['weights_sha256'],truths_sha256=old['truth_sha256'],
                source_sha256=digest(Path(__file__)),residual_sha256=digest(ROOT/'sampling/pde_residuals.py'))
            if (target/'protocol.json').exists():assert json.loads((target/'protocol.json').read_text())==protocol
            else:write(target/'protocol.json',protocol)
            write(target/'environment.json',dict(pid=os.getpid(),host=socket.gethostname(),
                python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
            free,_=torch.cuda.mem_get_info()
            assert free>60*2**30,(pde,'expected idle 80GB GPU',free)
            truths=torch.load(source/'truths.pt',map_location='cpu',weights_only=False)
            bundle=load_fm4pde_checkpoint_bundle(str(source/'weights.pth'),pde,'cuda:0',model_profile='recommended')
            captures={}
            original=runner._sample_initial_noise
            def capture(*a,**kw):
                x=original(*a,**kw)
                captures['initial_noise_sha256']=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
                captures['rng_after_initial_sha256']=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
                return x
            runner._sample_initial_noise=capture

            def one(stage,multiplier,ids):
                folder=target/stage/f'm{multiplier:g}_id{ids[0]}'
                receipt=folder/'receipt.json'
                if receipt.exists():
                    r=json.loads(receipt.read_text());assert r['sample_ids']==ids
                    assert digest(r['result_path'])==r['result_sha256']
                    return r
                folder.mkdir(parents=True,exist_ok=True)
                cfg=AblationConfig(**dict(cfgbase,checkpoint_path=str(source/'weights.pth'),
                    output_dir=str(folder),device='cuda:0',batch_size=len(ids),offset=ids[0],
                    zeta_pde=float(cfgbase['zeta_pde'])*multiplier,
                    guidance_components='obs_only' if multiplier==0 else 'obs_pde',
                    sample_seed=20260910+ids[0],mask_seed=20260910+ids[0],
                    save_plots=False,save_intermediate=False,save_per_sample_curves=True,
                    ablation_name='guidance_weight_0910',allow_synthetic_data=False))
                gt=combine_truths(truths,ids,'cuda:0')
                torch.cuda.reset_peak_memory_stats();captures.clear()
                begin=time.monotonic()
                with (folder/'run.log').open('w') as f,redirect_stdout(f),redirect_stderr(f):
                    result=runner.run_single_ablation(cfg,checkpoint_bundle=bundle,ground_truth=gt)
                path=Path(result['run_dir'])/'result.pt'
                payload=torch.load(path,map_location='cpu',weights_only=False)
                masks=torch.load(path.parent/'masks.pt',map_location='cpu',weights_only=False)
                errors={field:physical_errors(payload[prefix+'_final'],payload[prefix+'_ground_truth'])
                        for field,prefix in [('a','coef'),('u','sol')]}
                # Score the stored prediction tensors in float64 using exactly
                # the physical component convention, without any true trajectory.
                params=_pde_params_with_residual_options(combine_truths(truths,ids,'cpu').pde_params,cfg)
                with torch.no_grad():
                    out=compute_pde_residual(pde,payload['coef_final'].double(),payload['sol_final'].double(),
                        pde_params=params,residual_mode=cfg.residual_mode,k=cfg.k)
                    losses=np.zeros(len(ids))
                    for k,v in out.components.items():
                        if v is not None:
                            weight=cfg.bc_weight if k=='boundary' else cfg.endpoint_bc_weight if k=='endpoint' else 1.
                            losses+=weight*v.square().flatten(1).mean(1).numpy()
                r=dict(pde=pde,stage=stage,multiplier=multiplier,zeta_pde=cfg.zeta_pde,
                    sample_ids=ids,errors=errors,pde_mse=losses.tolist(),config=cfg.asdict(),
                    mask_sha256=hashlib.sha256(masks['coef'].numpy().tobytes()+masks['sol'].numpy().tobytes()).hexdigest(),
                    result_path=str(path),result_sha256=digest(path),seconds=time.monotonic()-begin,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,**captures)
                assert np.isfinite(losses).all() and all(math.isfinite(v) for e in errors.values() for v in e)
                write(receipt,r);print('DONE',pde,stage,multiplier,ids,'MSE',losses.mean(),flush=True)
                return r

            # Measured batch-one memory bounds the development batch before use.
            probe=one('probe',1,[0])
            free,_=torch.cuda.mem_get_info()
            assert 4*probe['peak_allocated_gib']<free/2**30,(pde,'batch-four conservative memory gate',probe['peak_allocated_gib'],free/2**30)
            dev=[one('development',m,DEV) for m in MULTIPLIERS]
            for r in dev:
                for k in ['initial_noise_sha256','rng_after_initial_sha256','mask_sha256']:
                    assert r[k]==dev[0][k],(pde,k)
            fields=['u'] if pde=='burger' else ['a','u']
            eligible=[r for r in dev if r['multiplier']>1 and
                all(np.mean(r['errors'][f])<=1.02*np.mean(dev[0]['errors'][f]) for f in fields)]
            selected=min(eligible,key=lambda r:np.mean(r['pde_mse'])) if eligible else None
            if selected is not None and np.mean(selected['pde_mse'])>=np.mean(dev[0]['pde_mse']):selected=None
            write(target/'selection.json',dict(selected_multiplier=selected['multiplier'] if selected else None,
                selected_on_development_only=True,eligible=[r['multiplier'] for r in eligible],
                development=[dict(multiplier=r['multiplier'],mean_pde=float(np.mean(r['pde_mse'])),
                    mean_errors={f:float(np.mean(r['errors'][f])) for f in fields}) for r in dev]))
            candidates=sorted(set([0,1]+([selected['multiplier']] if selected else [])))
            for i in range(0,len(CONFIRM),4):
                rs=[one('confirmation',m,CONFIRM[i:i+4]) for m in candidates]
                for r in rs:
                    for k in ['initial_noise_sha256','rng_after_initial_sha256','mask_sha256']:assert r[k]==rs[0][k]
            write(target/'complete.json',dict(complete=True,confirmation_candidates=candidates,
                protocol_sha256=digest(target/'protocol.json'),finished=time.time()))
            runner._sample_initial_noise=original
            del bundle,truths;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--anchors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pdes',nargs='+',required=True)
    main(p.parse_args())

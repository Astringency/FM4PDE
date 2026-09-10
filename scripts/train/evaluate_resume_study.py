"""Paired downstream sampling after all training jobs on a GPU have finished."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
IDS = list(range(1500, 1532))
SEEDS = (0, 1)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def field_scores(prediction, truth, pde):
    import numpy as np
    from scipy.fft import dctn
    p, t = prediction.cpu().double().numpy(), truth.cpu().double().numpy()
    axes = tuple(range(1, p.ndim))
    relative = np.sqrt(np.sum((p-t)**2, axis=axes) / np.maximum(np.sum(t*t, axis=axes), 1e-24))
    if pde == 'nsnonbounded':
        pc, tc = np.fft.fft2(p, norm='ortho'), np.fft.fft2(t, norm='ortho')
        fy, fx = np.fft.fftfreq(p.shape[-2])*p.shape[-2], np.fft.fftfreq(p.shape[-1])*p.shape[-1]
        radius = np.hypot(fy[:,None], fx[None,:])
        mask = (radius>0)&(radius<=8)
        pc, tc = pc[...,mask], tc[...,mask]
        basis = 'periodic spatial FFT2; 0<radial mode<=8'
    elif pde == 'burger':
        pc, tc = np.fft.rfft(p, axis=-1, norm='ortho')[...,1:9], np.fft.rfft(t, axis=-1, norm='ortho')[...,1:9]
        basis = 'periodic spatial FFT along x, modes 1..8; summed over time'
    else:
        pc, tc = dctn(p, axes=(-2,-1), norm='ortho'), dctn(t, axes=(-2,-1), norm='ortho')
        radius = np.hypot(np.arange(p.shape[-2])[:,None],np.arange(p.shape[-1])[None,:])
        mask = (radius>0)&(radius<=8)
        pc, tc = pc[...,mask], tc[...,mask]
        basis = 'nonperiodic orthonormal DCT2; 0<radial mode<=8'
    pc, tc = pc.reshape(len(p),-1), tc.reshape(len(t),-1)
    pp, tt = np.sum(abs(pc)**2,1), np.sum(abs(tc)**2,1)
    low = np.sqrt(np.sum(abs(pc-tc)**2,1)/np.maximum(tt,1e-24))
    alignment = np.real(np.sum(pc*np.conj(tc),1))/np.maximum(np.sqrt(pp*tt),1e-24)
    result = dict(relative_l2=relative.tolist(), low_relative_l2=low.tolist(),
                  low_power_ratio=(pp/np.maximum(tt,1e-24)).tolist(), low_alignment=alignment.tolist(), basis=basis)
    assert all(np.isfinite(result[k]).all() for k in ['relative_l2','low_relative_l2','low_power_ratio','low_alignment'])
    return result


def evaluate_pde(study, job):
    import numpy as np
    import torch
    from sampling.config import load_config
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from scripts.tuning.compare_pde_guidance_schedules import combine_truths
    import sampling.runner as runner
    pde = job['pde']
    out = Path(job['output'])/'evaluation'
    out.mkdir(parents=True,exist_ok=True)
    training = json.loads((Path(job['output'])/'complete.json').read_text())
    source = ROOT/'outputs/reproducibility/cleanup_20260910/retained_inputs/paper_revision_20260908/inputs'/pde
    old = json.loads((source/'protocol.json').read_text())
    assert sha(source/'truths.pt')==old['truth_sha256']
    truths = torch.load(source/'truths.pt',map_location='cpu',weights_only=False)
    assert all(i in truths and not truths[i].metadata['synthetic'] for i in IDS+[0])
    tasks = ['both'] if pde=='burger' else ['both','inverse']
    checkpoints = {'baseline':job['checkpoint'], 'resumed':training['best_checkpoint']}
    hashes = {k:sha(v) for k,v in checkpoints.items()}
    assert hashes['resumed']==training['best_sha256']
    protocol = dict(pde=pde, sample_ids=IDS, seeds=list(SEEDS), tasks=tasks, num_steps=100,
        checkpoint_sha256=hashes, truths_sha256=old['truth_sha256'],
        precision=dict(parameters='float32',outputs='float32',
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,torch=str(torch.__version__)),
        matched='same inputs, observation masks, initial noise, sampler random draws and guidance settings',
        selection='checkpoint selected on training-data validation only; sampling results are confirmation',
        configs={task:load_config(ROOT/f'configs/main/{task}/{pde}.yaml').asdict() for task in tasks})
    protocol=json.loads(json.dumps(protocol))
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text())==protocol
    else:
        write(out/'protocol.json',protocol)
    rows=[]
    captures={}
    original=runner._sample_initial_noise
    def capture(*a,**kw):
        x=original(*a,**kw)
        captures['initial_noise_sha256']=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
        captures['rng_after_initial_sha256']=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
        return x
    runner._sample_initial_noise=capture
    batch_size=None
    try:
        for label,path in checkpoints.items():
            free,_=torch.cuda.mem_get_info()
            assert free>65*2**30, 'Need an idle GPU for paired evaluation'
            bundle=load_fm4pde_checkpoint_bundle(path,pde,'cuda:0',model_profile='auto')
            def one(task, seed, ids, probe=False):
                folder=out/(f'probe_batch{len(ids)}' if probe else task)/label/f'seed{seed}_id{ids[0]}'
                receipt=folder/'receipt.json'
                if receipt.exists():
                    row=json.loads(receipt.read_text())
                    assert row['sample_ids']==ids and row['checkpoint_sha256']==hashes[label]
                    assert sha(row['result_path'])==row['result_sha256']
                    return row
                folder.mkdir(parents=True,exist_ok=True)
                cfg=load_config(ROOT/f'configs/main/{task}/{pde}.yaml',overrides=dict(
                    checkpoint_path=path,model_profile='auto',output_dir=str(folder),
                    device='cuda:0',dtype='float32',batch_size=len(ids),offset=ids[0],
                    sample_seed=20260911+10000*seed+ids[0],mask_seed=20260910+ids[0],
                    sensor_mode='per_sample_random',shared_mask=True,num_steps=100,
                    save_plots=False,save_intermediate=False,save_per_sample_curves=True,
                    ablation_name='paired_resume',allow_synthetic_data=False))
                gt=combine_truths(truths,ids,'cuda:0')
                torch.cuda.reset_peak_memory_stats(); captures.clear()
                start=time.monotonic()
                with (folder/'run.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
                    result=runner.run_single_ablation(cfg,checkpoint_bundle=bundle,ground_truth=gt)
                prediction_path=Path(result['run_dir'])/'result.pt'
                payload=torch.load(prediction_path,map_location='cpu',weights_only=False)
                masks=torch.load(prediction_path.parent/'masks.pt',map_location='cpu',weights_only=False)
                fields={'u':field_scores(payload['sol_final'],payload['sol_ground_truth'],pde)}
                if pde!='burger':fields['a']=field_scores(payload['coef_final'],payload['coef_ground_truth'],pde)
                row=dict(pde=pde,task=task,label=label,seed=seed,sample_ids=ids,fields=fields,
                    config=cfg.asdict(),checkpoint_sha256=hashes[label],
                    mask_sha256=hashlib.sha256(masks['coef'].numpy().tobytes()+masks['sol'].numpy().tobytes()).hexdigest(),
                    result_path=str(prediction_path),result_sha256=sha(prediction_path),
                    peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,**captures)
                write(receipt,row)
                print('PAIRED_RESULT',pde,task,label,seed,ids,'seconds',row['seconds'],flush=True)
                return row
            if batch_size is None:
                probe=one(tasks[0],0,[0],probe=True)
                batch_size=max(b for b in (1,2,4,8,16) if b*probe['peak_bytes']<48*2**30)
                measured=probe if batch_size==1 else one(tasks[0],0,[0]*batch_size,probe=True)
                assert measured['peak_bytes']<56*2**30, 'Selected batch exceeded the reserved memory margin'
                batches=2*len(tasks)*len(SEEDS)*math.ceil(len(IDS)/batch_size)
                estimate=batches*measured['seconds']
                write(out/'batch_selection.json',dict(batch_size=batch_size,probe_peak_bytes=probe['peak_bytes'],
                    selected_batch_peak_bytes=measured['peak_bytes'],selected_batch_seconds=measured['seconds'],
                    estimated_paired_sampling_seconds=estimate,probe_sample_id=0,probe_excluded_from_accuracy=True))
                print('SAMPLING_BUDGET',pde,'batch',batch_size,'estimated_seconds',estimate,flush=True)
            for task in tasks:
                for seed in SEEDS:
                    for start in range(0,len(IDS),batch_size):
                        rows.append(one(task,seed,IDS[start:start+batch_size]))
            del bundle
            torch.cuda.empty_cache()
    finally:
        runner._sample_initial_noise=original
    lookup={(r['task'],r['seed'],r['sample_ids'][0],r['label']):r for r in rows}
    comparisons=[]
    for task in tasks:
        by_field={f:{'baseline':{},'resumed':{}} for f in (['u'] if pde=='burger' else ['a','u'])}
        for seed in SEEDS:
            for start in range(0,len(IDS),batch_size):
                key=(task,seed,IDS[start])
                b,a=lookup[(*key,'baseline')],lookup[(*key,'resumed')]
                for name in ['mask_sha256','initial_noise_sha256','rng_after_initial_sha256']:
                    assert b[name]==a[name], (pde,task,name)
                for field in by_field:
                    for label,r in [('baseline',b),('resumed',a)]:
                        for j,sample in enumerate(r['sample_ids']):
                            by_field[field][label].setdefault(sample,[]).append(
                                {m:r['fields'][field][m][j] for m in ['relative_l2','low_relative_l2','low_power_ratio','low_alignment']})
        for field,values in by_field.items():
            for metric in ['relative_l2','low_relative_l2','low_power_ratio','low_alignment']:
                # Average seeds within each input before calculating a paired CI.
                b=np.array([np.mean([x[metric] for x in values['baseline'][i]]) for i in IDS])
                a=np.array([np.mean([x[metric] for x in values['resumed'][i]]) for i in IDS])
                d=a-b;half=1.96*d.std(ddof=1)/math.sqrt(len(d))
                comparisons.append(dict(task=task,field=field,metric=metric,baseline=float(b.mean()),
                    resumed=float(a.mean()),paired_change=float(d.mean()),paired_change_95ci=[float(d.mean()-half),float(d.mean()+half)],
                    per_input_baseline=b.tolist(),per_input_resumed=a.tolist()))
    write(out/'complete.json',dict(status='complete',pde=pde,samples=32,seeds=2,tasks=tasks,
        paired_masks_and_rng_verified=True,batch_size=batch_size,checkpoint_sha256=hashes,comparisons=comparisons))


def main(args):
    study=args.study.resolve()
    assert '/outputs/pretrained/' in str(study), 'Explicit pretrained output directory required'
    plan=json.loads((study/'study_plan.json').read_text())
    jobs=[j for j in plan['jobs'] if j['gpu']==args.gpu]
    status_path=study/f'evaluation_gpu{args.gpu}_state.json'
    write(status_path,dict(state='waiting_for_training',pid=os.getpid(),pdes=[j['pde'] for j in jobs]))
    while True:
        done=[]
        for job in jobs:
            out=Path(job['output'])
            if (out/'exit.json').exists():
                status=json.loads((out/'exit.json').read_text())
                assert status['exit_code']==0, 'Training failed: '+job['pde']
                assert (out/'complete.json').exists()
                done.append(job['pde'])
            else:
                state=json.loads((out/'queue_state.json').read_text())
                os.kill(state['pid'],0)  # Verify a live worker, not just a stale status file.
        if len(done)==len(jobs):break
        time.sleep(15)
    lock_directory = Path(plan.get('gpu_lock_directory', study)).resolve()
    assert '/outputs/pretrained/' in str(lock_directory)
    with (lock_directory/f'gpu{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=not args.strict_fp32
        torch.backends.cudnn.allow_tf32=not args.strict_fp32
        torch.backends.cudnn.benchmark=False
        write(status_path,dict(state='running',pid=os.getpid(),pdes=[j['pde'] for j in jobs]))
        for job in jobs:evaluate_pde(study,job)
    write(status_path,dict(state='complete',pid=os.getpid(),pdes=[j['pde'] for j in jobs]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
    parser.add_argument('--strict-fp32',action='store_true',help='Disable TF32 for both models; default matches training validation acceleration')
    args=parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        write(args.study/f'evaluation_gpu{args.gpu}_error.json',dict(error=repr(exc),pid=os.getpid(),time=time.time()))
        write(args.study/f'evaluation_gpu{args.gpu}_exit.json',dict(exit_code=1,pid=os.getpid(),time=time.time()))
        raise
    else:
        write(args.study/f'evaluation_gpu{args.gpu}_exit.json',dict(exit_code=0,pid=os.getpid(),time=time.time()))

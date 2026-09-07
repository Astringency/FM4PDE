"""Freeze and run a separate paired NS loss-exchange accuracy experiment."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ns_loss_exchange import build_native_diffusion, diffusion_pde_loss, fm_exchange, fm_pde_loss

TASKS = ['forward', 'inverse', 'both']
VARIANTS = [[method, steps, exchange] for method, steps in [('FM4PDE', 100), ('DiffusionPDE', 100), ('DiffusionPDE', 1000)] for exchange in [False, True]]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def numerical_failure(exc):
    return isinstance(exc, FloatingPointError) or any(k in str(exc).lower() for k in
             ['nan', 'nonfinite', 'non-finite', 'contains inf', ' or inf', 'infinite'])


def freeze(args):
    import h5py
    import numpy as np
    import yaml
    from sampling.config import load_config
    dest = args.inputs
    assert not (dest / 'source.json').exists(), 'Frozen source already exists'
    dest.mkdir(parents=True, exist_ok=True)
    excluded = [909, 383, 313, 466, 598, 100, 40, 602, 514, 260, 615, 604]
    ids = random.Random(20260908).sample([i for i in range(1000) if i not in excluded], 36)
    pilots, evaluation = ids[:4], ids[4:]
    arrays = {}
    with h5py.File(args.data_path, 'r') as f:
        params = {'nu': float(f.attrs['viscosity']), 'T': float(f.attrs['T']), 'solver_dt': float(f.attrs['dt'])}
        attrs = {k: np.asarray(v).tolist() for k, v in f.attrs.items()}
        for i in ids:
            arrays[f'a_{i}'] = f['w0'][i][None, None]
            arrays[f'u_{i}'] = f['w'][i, :, :, -1][None, None]
            rng = np.random.default_rng(20260908 + i)
            for field in ['a', 'u']:
                mask = np.zeros((1, 1, 128, 128), dtype=np.float32)
                mask.flat[rng.choice(128 * 128, 500, replace=False)] = 1
                arrays[f'mask_{field}_{i}'] = mask
    np.savez_compressed(dest / 'fields_masks.npz', **arrays)
    fm = {t: load_config(ROOT / f'configs/main/{t}/nsnonbounded.yaml').asdict() for t in TASKS}
    dm = {}
    for t in TASKS:
        for n in [100, 1000]:
            path = args.diffusion_root / f'outputs/MAIN1000_{n}/.sample_sweep/configs/nsnonbounded_{t}.yaml'
            c = yaml.safe_load(path.read_text())
            assert c['test']['iterations'] == n
            # generate_pde.py applies this after reading the stored YAML.
            if t == 'forward': c['generate']['zeta_obs_u'] = 0
            if t == 'inverse': c['generate']['zeta_obs_a'] = 0
            dm[f'{t}_{n}'] = dict(config=c, source=str(path), sha256=sha(path))
    source = dict(version=1, data_path=str(args.data_path), data_size=args.data_path.stat().st_size,
                  data_mtime_ns=args.data_path.stat().st_mtime_ns, data_attrs=attrs,
                  pde_params=params, pilot_ids=pilots, evaluation_ids=evaluation, excluded_ids=excluded,
                  selection_seed=20260908, inference_seeds=[0, 1, 2], examples=32,
                  tasks=TASKS, variants=VARIANTS, formal_calls=32 * 3 * 3 * 6,
                  fm_configs=fm, diffusion_configs=dm, fields_sha256=sha(dest/'fields_masks.npz'),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  diffusion_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.diffusion_root,text=True).strip(),
                  scope='Paired supplementary accuracy study. Native FM float32 and Diffusion float64 sampler arithmetic. Common physical inputs/masks; recipient weights, observation losses and solver retained. Full PDE operator and scalar reduction exchanged without retuning.')
    write(dest/'source.json', source)
    print('FROZEN', len(ids), 'NS fields,', source['formal_calls'], 'planned calls', flush=True)


def run(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    import pickle
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from run_matched_timing import fm_predict

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith('GPU-'): uuid = 'GPU-' + uuid
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
    foreign = [line for line in apps.splitlines() if line.startswith(uuid) and int(line.split(',')[1]) != os.getpid()]
    assert not foreign, f'Assigned GPU has another compute process: {foreign}'
    source = json.loads((args.inputs/'source.json').read_text())
    assert sha(args.inputs/'fields_masks.npz') == source['fields_sha256']
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/f'worker_{args.shard}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code_sha = {p: sha(ROOT/p) for p in ['plot/run_ns_loss_study.py','plot/ns_loss_exchange.py','plot/run_matched_timing.py','sampling/pde_residuals.py','sampling/losses.py','sampling/guidance.py']}
    protocol = dict(source_sha256=sha(args.inputs/'source.json'), code_sha256=code_sha,
                    fm_checkpoint_sha256=sha(args.fm_checkpoint), diffusion_checkpoint_sha256=sha(args.dm_checkpoint),
                    diffusion_source_sha256=sha(args.diffusion_root/'scripts/generate_ns_nonbounded.py'),
                    seeds=source['inference_seeds'], evaluation_ids=source['evaluation_ids'], tasks=TASKS,
                    variants=VARIANTS, formal_calls=source['formal_calls'])
    protocol_path = args.output/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        write(protocol_path, protocol)
    ph = sha(protocol_path)
    write(args.output/f'environment_{args.shard}.json', dict(protocol_sha256=ph, pid=os.getpid(),
          host=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda,
          gpu=torch.cuda.get_device_name(0), uuid=str(torch.cuda.get_device_properties(0).uuid),
          visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cpu_threads=2,
          native_arithmetic={'FM4PDE':'float32','DiffusionPDE':'float64 sampler, float32 network'}))
    data = np.load(args.inputs/'fields_masks.npz')
    fm = load_fm4pde_checkpoint_bundle(str(args.fm_checkpoint), 'nsnonbounded', 'cuda:0', model_profile='recommended')
    sys.path.append(str(args.diffusion_root))
    with args.dm_checkpoint.open('rb') as f:
        dm = pickle.load(f)['ema'].to('cuda:0').eval().requires_grad_(False)
    functions = {}
    for exchange in [False, True]:
        for diag in [False, True]:
            func, text = build_native_diffusion(args.diffusion_root, exchange=exchange, diagnostics=diag)
            functions[exchange, diag] = func
            path = args.output/f'diffusion_exchange{int(exchange)}_diagnostics{int(diag)}.py'
            if path.exists(): assert path.read_text() == text
            else: path.write_text(text)
    counts = {'FM4PDE':0, 'DiffusionPDE':0}
    def hook(method):
        def count(*unused): counts[method] += 1
        return count
    handles = [fm[0].model.register_forward_hook(hook('FM4PDE')), dm.register_forward_hook(hook('DiffusionPDE'))]

    def predict(task, method, n, exchange, sample_id, seed, *, diagnostics=False, hidden=False, obs_scale=1.):
        dtype = torch.float32 if method == 'FM4PDE' else torch.float64
        a = torch.as_tensor(data[f'a_{sample_id}'], device='cuda:0', dtype=dtype)
        u = torch.as_tensor(data[f'u_{sample_id}'], device='cuda:0', dtype=dtype)
        ma = torch.as_tensor(data[f'mask_a_{sample_id}'], device='cuda:0', dtype=dtype)
        mu = torch.as_tensor(data[f'mask_u_{sample_id}'], device='cuda:0', dtype=dtype)
        if task == 'both' and source.get('joint_shared_mask', False):
            # The task-specific Senseiver checkpoint consumes both measured
            # values at one coordinate. Match this information contract for
            # every method, rather than filling an unobserved channel with a
            # value that its encoder would interpret as an observation.
            mu = ma.clone()
        if task == 'forward': mu = torch.zeros_like(mu)
        if task == 'inverse': ma = torch.zeros_like(ma)
        masks = PairMasks(ma, mu, dict(source='frozen common 500-point mask'))
        obs_a, obs_u = a * ma * obs_scale, u * mu * obs_scale
        if hidden:
            obs_a = obs_a + 2.7 * (1-ma)
            obs_u = obs_u - 4.1 * (1-mu)
        c = copy.deepcopy(source['fm_configs'][task])
        c.update(device='cuda:0', sample_seed=seed, batch_size=1, offset=sample_id,
                 num_steps=100, save_plots=False, save_intermediate=False, save_per_sample_curves=False,
                 checkpoint_path=str(args.fm_checkpoint), output_dir=str(args.output/'pilot_reference'),
                 initial_noise_source_indices=[], initial_noise_source_batch_size=None)
        cfg = AblationConfig(**c)
        meta = dict(sample_ids=[str(sample_id)], sample_offsets=[sample_id], offset=sample_id,
                    batch_size=1, synthetic=False, endpoint_pair=True, data_path=source['data_path'])
        params = {k: torch.tensor([v], dtype=dtype, device='cuda:0') for k,v in source['pde_params'].items()}
        gt = PDEGroundTruth('nsnonbounded', obs_a, obs_u, torch.cat([obs_a,obs_u],1), params, ['w0'], ['wT'], meta)
        trace = []
        if method == 'FM4PDE':
            with fm_exchange(exchange, trace):
                prediction = fm_predict(cfg, fm, gt, masks)
        else:
            c = copy.deepcopy(source['diffusion_configs'][f'{task}_{n}']['config'])
            c['generate'].update(device='cuda:0', seed=seed, batch_size=1)
            prediction = functions[exchange, diagnostics](c, dm, obs_a, obs_u, ma[0,0], mu[0,0],
                           lambda aa,uu: fm_pde_loss(aa,uu,cfg,params), trace)
        return prediction, dict(truths=(a,u), masks=(ma,mu), config=cfg, params=params, trace=trace, gt=gt, pair_masks=masks)

    # Pilot precedes all formal calls; each worker has its own diagnostic receipt.
    pilot_path = args.output/f'pilot_{args.shard}.json'
    if not pilot_path.exists():
        i = source['pilot_ids'][args.shard]
        checks = []
        for method, n, exchange in VARIANTS:
            if n != 100: continue
            try:
                base, extra = predict('both',method,n,exchange,i,0)
            except (FloatingPointError, RuntimeError) as exc:
                if not exchange or not numerical_failure(exc): raise
                row = dict(method=method,exchange=exchange,sample_id=i,status='unstable',error=repr(exc))
                checks.append(row)
                print('PILOT_NUMERICAL_OUTCOME',row,flush=True)
                continue
            repeated, _ = predict('both',method,n,exchange,i,0)
            changed, _ = predict('both',method,n,exchange,i,0,obs_scale=.83)
            hidden, _ = predict('both',method,n,exchange,i,0,hidden=True)
            delta = lambda values: max(float((a-b).abs().max()) for a,b in zip(base,values))
            row = dict(method=method, exchange=exchange, sample_id=i, repeat=delta(repeated), hidden=delta(hidden), sensitivity=delta(changed))
            assert row['repeat'] == 0 and row['hidden'] == 0 and row['sensitivity'] > 0, row
            if method == 'DiffusionPDE':
                ref, _ = predict('both',method,n,exchange,i,0,diagnostics=True)
                row['diagnostic_reference'] = delta(ref)
                assert row['diagnostic_reference'] == 0, row
            elif not exchange:
                import sampling.runner as runner
                receipt = runner.run_single_ablation(extra['config'], fm, ground_truth=extra['gt'], observation_masks=extra['pair_masks'])
                ref = torch.load(Path(receipt['run_dir'])/'result.pt', weights_only=False, map_location='cuda:0')
                row['original_runner'] = delta((ref['coef_final'],ref['sol_final']))
                assert row['original_runner'] == 0, row
            assert all(bool(torch.isfinite(x).all()) for x in base), row
            checks.append(row)
            print('PILOT', method, exchange, row, flush=True)
        write(pilot_path, dict(status='pass', protocol_sha256=ph, checks=checks))
    if args.pilot_only:
        return
    jobs = [(t,m,n,e,i,s) for i in source['evaluation_ids'] for s in source['inference_seeds'] for t in TASKS for m,n,e in VARIANTS]
    random.Random(20260908).shuffle(jobs)
    jobs = jobs[args.shard::args.shards]
    for task, method, n, exchange, i, seed in jobs:
        stem = f'{task}/{method}_{n}/{"exchanged" if exchange else "original"}/sample{i}_seed{seed}'
        dest = args.output/'results'/stem
        receipt_path = dest.with_suffix('.json')
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            assert old['protocol_sha256'] == ph
            continue
        counts[method] = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        row = dict(task=task,method=method,steps=n,exchange=exchange,sample_id=i,seed=seed,protocol_sha256=ph)
        try:
            prediction, extra = predict(task,method,n,exchange,i,seed)
            torch.cuda.synchronize()
            finite = all(bool(torch.isfinite(x).all()) for x in prediction)
            row.update(status='complete' if finite else 'nonfinite', seconds=time.perf_counter()-start,
                       nfe=counts[method], relative_l2=[float((x-y).double().norm()/y.double().norm()) if finite else None for x,y in zip(prediction,extra['truths'])])
            assert row['nfe'] == (n if method=='FM4PDE' else 2*n-1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            output = dest.with_suffix('.pt')
            torch.save(dict(prediction=[x.cpu() for x in prediction], truth=[x.cpu() for x in extra['truths']],
                            masks=[x.cpu() for x in extra['masks']], trace=extra['trace'], receipt=row), output)
            row['prediction_sha256'] = sha(output)
        except (FloatingPointError, RuntimeError) as exc:
            # Instability is an outcome. Preserve it without replacing it by
            # a successful seed, clipping rule, observation or loss weight.
            if not numerical_failure(exc):
                raise
            row.update(status='unstable', error=repr(exc), nfe=counts[method], seconds=time.perf_counter()-start)
        write(receipt_path,row)
        print('RESULT',stem,row['status'],row.get('relative_l2'),flush=True)
    write(args.output/f'complete_{args.shard}.json',dict(status='complete',protocol_sha256=ph,jobs=len(jobs)))
    for handle in handles: handle.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','run'])
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--diffusion-root',type=Path,required=True)
    p.add_argument('--data-path',type=Path)
    p.add_argument('--fm-checkpoint',type=Path)
    p.add_argument('--dm-checkpoint',type=Path)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--pilot-only',action='store_true')
    args = p.parse_args()
    freeze(args) if args.mode == 'freeze' else run(args)


if __name__ == '__main__': main()

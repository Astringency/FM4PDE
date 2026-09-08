"""Matched NS checkpoint comparison; frozen inputs and separate resumable outputs.

The primary comparison changes only the checkpoint (including its saved network
and normalizer). A separately labelled backup configuration checks historical
guidance. No evaluation outcomes select checkpoints or hyperparameters.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_ns_loss_study import sha, write, numerical_failure

TASKS = ['forward', 'inverse', 'both']


def simple(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): simple(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [simple(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def freeze(args):
    from models.legacy_checkpoint import read_checkpoint
    from sampling.config import load_config
    assert not (args.output / 'protocol.json').exists()
    source = json.loads((args.inputs / 'source.json').read_text())
    assert len(source['evaluation_ids']) == 32 and source['inference_seeds'] == [0, 1, 2]
    assert set(source['evaluation_ids']).isdisjoint(source['pilot_ids'])
    specs = json.loads(args.checkpoints.read_text())
    inventory = {}
    for label, spec in specs.items():
        path = Path(spec['path'])
        payload = read_checkpoint(path, 'nsnonbounded')
        keys = ['epoch', 'checkpoint_schema_version', 'model_profile', 'model_config',
                'normalizer', 'normalization', 'legacy_compatibility', 'args',
                'data_metadata', 'inference_weight', 'has_ema']
        info = {k: simple(vars(payload[k]) if k == 'args' and not isinstance(payload[k], dict)
                          else payload[k]) for k in keys if k in payload}
        info.update(path=str(path), sha256=sha(path),
                    parameter_count=sum(v.numel() for v in payload['model'].values()),
                    model_tensor_dtypes=sorted({str(v.dtype) for v in payload['model'].values()}))
        evidence = args.output / 'evidence' / label
        evidence.mkdir(parents=True, exist_ok=True)
        info['evidence_sha256'] = {}
        for name in spec.get('evidence', []):
            src = Path(name)
            dest = evidence / src.name
            shutil.copy2(src, dest)
            info['evidence_sha256'][str(dest.relative_to(args.output))] = sha(dest)
        inventory[label] = info
        del payload
    variants = [dict(name=f'{label}_common100', checkpoint=label, configuration='common', steps=100)
                for label in ['current', 'v260904', 'bak']]
    variants.append(dict(name='bak_legacy100', checkpoint='bak', configuration='legacy', steps=100))
    code = [p for folder in ['sampling', 'models', 'flow_matching', 'data']
            for p in (ROOT / folder).rglob('*.py')]
    code += [ROOT / 'plot' / p for p in ['run_ns_checkpoint_comparison.py', 'run_matched_timing.py',
                                        'run_ns_loss_study.py', 'ns_loss_exchange.py']]
    protocol = dict(version=1, inputs=str(args.inputs), source_sha256=sha(args.inputs / 'source.json'),
                    fields_sha256=sha(args.inputs / 'fields_masks.npz'), checkpoints=inventory,
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    code_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(code))},
                    common_configs=source['fm_configs'],
                    legacy_configs={t: load_config(ROOT / f'configs/bak/{t}/nsnonbounded.yaml').asdict() for t in TASKS},
                    evaluation_ids=source['evaluation_ids'], pilot_ids=source['pilot_ids'],
                    seeds=[0, 1, 2], tasks=TASKS, variants=variants,
                    formal_calls=32 * 3 * 3 * len(variants),
                    selection='All specified final checkpoints, no score-based selection or retuning.',
                    pairing='Input sharding: every checkpoint, task and seed for one input stays on one GPU.',
                    primary='Forward u, inverse a, joint per-call max(a,u). Mean seeds within input, then mean and sample SD across 32 inputs.',
                    scope='Exploratory Smooth NS checkpoint comparison on the previous frozen 32-input cohort; not a replacement for 1000-input main experiments. Current and 260904 change architecture and potentially training protocol, not training duration alone. Backup normalization reproduces its archived sampling affine; training statistics were not saved. Native DiffusionPDE 100/1000 saved outcomes are external paired accuracy references, not same-environment timing benchmarks.')
    write(args.output / 'protocol.json', protocol)
    print('FROZEN', protocol['formal_calls'], 'calls', sha(args.output / 'protocol.json'), flush=True)
    for label, row in inventory.items():
        print(label, row['epoch'], row.get('model_profile'), row['parameter_count'], row['sha256'], flush=True)


def worker(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
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
    protocol = json.loads((args.output / 'protocol.json').read_text())
    ph = sha(args.output / 'protocol.json')
    assert sha(args.inputs / 'source.json') == protocol['source_sha256']
    assert sha(args.inputs / 'fields_masks.npz') == protocol['fields_sha256']
    for name, h in protocol['code_sha256'].items():
        assert sha(ROOT / name) == h, name
    lock = (args.output / f'worker{args.shard}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith('GPU-'):
        uuid = 'GPU-' + uuid
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not [r for r in apps.splitlines() if r.startswith(uuid) and int(r.split(',')[1]) != os.getpid()]
    write(args.output / f'environment_{args.shard}.json', dict(protocol_sha256=ph, pid=os.getpid(),
          host=socket.gethostname(), torch=torch.__version__, cuda=torch.version.cuda,
          gpu=torch.cuda.get_device_name(0), uuid=uuid, tf32=False, batch_size=1,
          arithmetic='float32 model and sampling states; float64 metric reductions'))
    source = json.loads((args.inputs / 'source.json').read_text())
    data = np.load(args.inputs / 'fields_masks.npz')
    models = {}
    for label, spec in protocol['checkpoints'].items():
        assert sha(spec['path']) == spec['sha256'], label
        models[label] = load_fm4pde_checkpoint_bundle(spec['path'], 'nsnonbounded', 'cuda:0', model_profile='auto')
    counts = {label: 0 for label in models}
    handles = []
    for label, model in models.items():
        def hook(*unused, label=label):
            counts[label] += 1
        handles.append(model[0].model.register_forward_hook(hook))

    def predict(task, variant, i, seed, hidden=False, scale=1.):
        label = variant['checkpoint']
        truth = [torch.from_numpy(data[f'{f}_{i}']).to('cuda:0', torch.float32) for f in ['a', 'u']]
        ma, mu = [torch.from_numpy(data[f'mask_{f}_{i}']).to('cuda:0', torch.float32) for f in ['a', 'u']]
        if task == 'both':
            mu = ma.clone()
        if task == 'forward':
            mu = torch.zeros_like(mu)
        if task == 'inverse':
            ma = torch.zeros_like(ma)
        aa, uu = truth[0] * ma * scale, truth[1] * mu * scale
        if hidden:
            aa, uu = aa + 2.7 * (1 - ma), uu - 4.1 * (1 - mu)
        masks = PairMasks(ma, mu, dict(source='frozen common 500-point masks'))
        conf = copy.deepcopy(protocol[f"{variant['configuration']}_configs"][task])
        conf.update(device='cuda:0', sample_seed=seed, batch_size=1, offset=i, num_steps=variant['steps'],
                    model_profile=protocol['checkpoints'][label]['model_profile'],
                    save_plots=False, save_intermediate=False, save_per_sample_curves=False,
                    checkpoint_path=protocol['checkpoints'][label]['path'],
                    output_dir=str(args.output / f'pilot_native_{args.shard}' / variant['name']),
                    initial_noise_source_indices=[], initial_noise_source_batch_size=None,
                    model_gradient_checkpointing=False)
        cfg = AblationConfig(**conf)
        params = {k: torch.tensor([v], device='cuda:0', dtype=torch.float32) for k, v in source['pde_params'].items()}
        gt = PDEGroundTruth('nsnonbounded', aa, uu, torch.cat([aa, uu], 1), params, ['w0'], ['wT'],
                            dict(sample_ids=[str(i)], sample_offsets=[i], offset=i, batch_size=1,
                                 synthetic=False, endpoint_pair=True, data_path=source['data_path']))
        counts[label] = 0
        pred = fm_predict(cfg, models[label], gt, masks)
        return pred, dict(truth=truth, masks=[ma, mu], config=cfg, gt=gt, pair_masks=masks, nfe=counts[label])

    pilot_path = args.output / f'implementation_check_{args.shard}.json'
    if not pilot_path.exists():
        checks = []
        i = protocol['pilot_ids'][args.shard]
        for variant in protocol['variants']:
            try:
                pred, d = predict('both', variant, i, 0)
                if not all(bool(torch.isfinite(x).all()) for x in pred):
                    raise FloatingPointError('Nonfinite pilot prediction')
            except (RuntimeError, FloatingPointError) as exc:
                if not numerical_failure(exc):
                    raise
                row = dict(variant=variant['name'], sample_id=i, status='unstable', error=repr(exc))
                checks.append(row)
                print('PILOT_NUMERICAL_OUTCOME', row, flush=True)
                continue
            repeat, _ = predict('both', variant, i, 0)
            hidden, _ = predict('both', variant, i, 0, hidden=True)
            changed, _ = predict('both', variant, i, 0, scale=.83)
            delta = lambda q: max(float((x - y).abs().max()) for x, y in zip(pred, q))
            row = dict(variant=variant['name'], sample_id=i, repeat=delta(repeat), hidden=delta(hidden),
                       observation_sensitivity=delta(changed), nfe=d['nfe'])
            assert row['repeat'] == row['hidden'] == 0 and row['observation_sensitivity'] > 0 and row['nfe'] == 100, row
            import sampling.runner as runner
            receipt = runner.run_single_ablation(d['config'], models[variant['checkpoint']],
                                                  ground_truth=d['gt'], observation_masks=d['pair_masks'])
            native = torch.load(Path(receipt['run_dir']) / 'result.pt', map_location='cuda:0', weights_only=False)
            row['native_runner'] = delta([native['coef_final'], native['sol_final']])
            assert row['native_runner'] == 0, row
            row['primary_error'] = max(float((x.double() - y.double()).norm() / y.double().norm()) for x, y in zip(pred, d['truth']))
            checks.append(row)
            print('PILOT', row, flush=True)
        write(pilot_path, dict(status='pass', protocol_sha256=ph, checks=checks))
    if args.pilot_only:
        return
    jobs = [(t, v, i, seed) for i in protocol['evaluation_ids'][args.shard::2]
            for seed in protocol['seeds'] for t in TASKS for v in protocol['variants']]
    random.Random(20260911 + args.shard).shuffle(jobs)
    for task, variant, i, seed in jobs:
        dest = args.output / 'evaluation' / task / variant['name'] / f'sample{i}_seed{seed}'
        rpath = dest.with_suffix('.json')
        if rpath.exists():
            old = json.loads(rpath.read_text())
            assert old['protocol_sha256'] == ph
            if 'prediction_sha256' in old:
                assert sha(dest.with_suffix('.pt')) == old['prediction_sha256']
            continue
        r = dict(task=task, variant=variant['name'], checkpoint=variant['checkpoint'], sample_id=i,
                 seed=seed, protocol_sha256=ph, worker=args.shard, steps=variant['steps'])
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            pred, d = predict(task, variant, i, seed)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            assert d['nfe'] == variant['steps']
            finite = all(bool(torch.isfinite(x).all()) for x in pred)
            errors = [float((x.double() - y.double()).norm() / y.double().norm()) for x, y in zip(pred, d['truth'])] if finite else [None, None]
            r.update(status='complete' if finite else 'nonfinite', seconds=seconds, nfe=d['nfe'],
                     rel_l2_a=errors[0], rel_l2_u=errors[1],
                     primary_error=(errors[1] if task == 'forward' else errors[0] if task == 'inverse' else max(errors)) if finite else None)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tensor = dest.with_suffix('.pt')
            tmp = tensor.with_suffix('.pt.tmp')
            torch.save(dict(prediction=[x.cpu() for x in pred], truth=[x.cpu() for x in d['truth']],
                            masks=[x.cpu() for x in d['masks']], config=d['config'].asdict(), receipt=r), tmp)
            tmp.replace(tensor)
            r['prediction_sha256'] = sha(tensor)
        except (RuntimeError, FloatingPointError) as exc:
            if not numerical_failure(exc):
                raise
            r.update(status='unstable', error=repr(exc), seconds=time.perf_counter() - started,
                     nfe=counts[variant['checkpoint']])
        write(rpath, r)
        print('RESULT', task, variant['name'], i, seed, r['status'], r.get('primary_error'), flush=True)
    write(args.output / f'complete_{args.shard}.json', dict(protocol_sha256=ph, expected_calls=len(jobs)))
    for h in handles:
        h.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'worker'])
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path)
    p.add_argument('--shard', type=int, choices=[0, 1], default=0)
    p.add_argument('--pilot-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args) if args.mode == 'freeze' else worker(args)


if __name__ == '__main__':
    main()

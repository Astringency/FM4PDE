"""Frozen Rough NS screening with native DiffusionPDE and archived paired bak."""
from __future__ import annotations

import argparse
import copy
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
TASKS = ['forward', 'inverse', 'both']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze(args):
    import numpy as np
    import torch
    assert not (args.inputs / 'protocol.json').exists()
    ids = random.Random(20260910).sample(range(1000), 39)
    evaluation, pilots = ids[:32], ids[32:]
    rows = {}
    archive = args.archive_root / 'outputs/bak_comparison_full'
    result_files = sorted(archive.glob('paired_results.worker*.jsonl'))
    for path in result_files:
        for line in path.open():
            row = json.loads(line)
            if row['pde'] == 'nsnonbounded' and row['test_type'] == 'rough' and row['sample_id'] in ids:
                key = (row['task'], row['sample_id'])
                assert key not in rows and row['status'] == 'ok' and row['steps'] == 100
                rows[key] = row
    assert len(rows) == 39 * 3
    arrays, evidence = {}, []
    for task in TASKS:
        for i in ids:
            row = rows[task, i]
            path = args.archive_root / row['run_dir'] / 'result.pt'
            payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            config, metadata = payload['config'], payload['ground_truth_metadata']
            assert config['task'] == task and config['test_type'] == 'rough'
            assert config['guidance_operator'] == 'legacy' and config['num_obs'] == 500
            index = [str(v) for v in metadata['sample_ids']].index(str(i))
            item = dict(task=task, sample_id=i, archive_path=str(path), archive_sha256=sha(path),
                        batch_index=index, metadata=metadata, config=config, recorded_errors=[row['bak_a'], row['bak_u']])
            item['pde_params'] = {k: float(v.reshape(-1)[index if v.numel() > 1 else 0])
                                  for k, v in payload['pde_params'].items()}
            assert abs(item['pde_params']['nu'] - .001) < 1e-9 and item['pde_params']['T'] == 1
            for f, name in [('a', 'coef'), ('u', 'sol')]:
                truth = payload[name + '_ground_truth'][index:index + 1].clone()
                pred = payload[name + '_final'][index:index + 1].clone()
                mask = payload['masks'][name][index:index + 1].clone()
                if (task == 'forward' and f == 'u') or (task == 'inverse' and f == 'a'):
                    mask.zero_()
                else:
                    assert int(mask.sum()) == 500
                assert all(bool(torch.isfinite(x).all()) for x in [truth, pred, mask])
                error = float((pred.double() - truth.double()).norm() / truth.double().norm())
                assert abs(error - row['bak_' + f]) < 2e-5
                item['recomputed_bak_' + f] = error
                arrays[f'{task}_{i}_{f}'] = truth.numpy()
                arrays[f'{task}_{i}_mask_{f}'] = mask.numpy()
                arrays[f'{task}_{i}_bak_{f}'] = pred.numpy()
                if task != 'forward':
                    assert np.array_equal(arrays[f'forward_{i}_{f}'], truth.numpy())
            evidence.append(item)
    args.inputs.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.inputs / 'fields_masks_bak.npz', **arrays)
    source = json.loads(args.template_source.read_text())
    dm_configs = copy.deepcopy(source['diffusion_configs'])
    for task in TASKS:
        for n in [100, 1000]:
            c = dm_configs[f'{task}_{n}']['config']
            assert c['test']['iterations'] == n and c['data']['obs_size'] == 500
            c['data']['datapath'] = 'frozen Rough archive tensors; no data file read by sampler'
            c['generate'].update(problem=task, seed=0, batch_size=1)
            if task == 'forward': c['generate']['zeta_obs_u'] = 0
            if task == 'inverse': c['generate']['zeta_obs_a'] = 0
    write(args.inputs / 'source.json', dict(evidence=evidence,
          source_result_hashes={str(p): sha(p) for p in result_files},
          template_source=str(args.template_source), template_source_sha256=sha(args.template_source)))
    assignment, start = {}, 0
    for shard, count in enumerate([6, 6, 4, 4, 4, 4, 4]):
        assignment[str(shard)] = evaluation[start:start + count]
        start += count
    protocol = dict(version=1, distribution='rough', pde='nsnonbounded', selection_seed=20260910,
        evaluation_ids=evaluation, pilot_ids=pilots, inference_seeds=[0], tasks=TASKS,
        steps=[100, 1000], formal_calls=192, workers=7, assignment=assignment,
        source_sha256=sha(args.inputs / 'source.json'), fields_sha256=sha(args.inputs / 'fields_masks_bak.npz'),
        diffusion_configs=dm_configs, checkpoint_sha256=sha(args.dm_checkpoint),
        native_source_sha256=sha(args.diffusion_root / 'scripts/generate_ns_nonbounded.py'),
        code_sha256={p: sha(ROOT / p) for p in ['plot/run_ns_rough_diffusion.py', 'plot/ns_loss_exchange.py']},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        selection='32 uniformly sampled IDs and seven disjoint pilots, fixed before new DiffusionPDE outcomes; no retuning or early stopping by score.',
        pairing='Exact archived bak Rough truth tensors and task-specific effective masks reused. Both Diffusion budgets for an input stay on one GPU. Archived bak retains its historical one reconstruction per input; no claim of matched initial model noise.',
        scope='Supplementary 32-input, one-seed Rough screening, three tasks, 100 and 1000 native diffusion steps. Not a 1000-input evaluation or repeated-seed robustness test. Archived Smooth hyperparameters and native NS spatial-derivative surrogate retained unchanged.')
    write(args.inputs / 'protocol.json', protocol)
    print('FROZEN', json.dumps(dict(ids=evaluation, calls=192, sha256=sha(args.inputs / 'protocol.json'))), flush=True)


def worker(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import pickle
    import torch
    from ns_loss_exchange import build_native_diffusion
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    p = json.loads((args.inputs / 'protocol.json').read_text())
    ph = sha(args.inputs / 'protocol.json')
    assert sha(args.inputs / 'source.json') == p['source_sha256']
    assert sha(args.inputs / 'fields_masks_bak.npz') == p['fields_sha256']
    assert sha(args.dm_checkpoint) == p['checkpoint_sha256']
    assert sha(args.diffusion_root / 'scripts/generate_ns_nonbounded.py') == p['native_source_sha256']
    for name, expected in p['code_sha256'].items(): assert sha(ROOT / name) == expected
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / f'worker{args.shard}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith('GPU-'): uuid = 'GPU-' + uuid
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not [r for r in apps.splitlines() if r.startswith(uuid) and int(r.split(',')[1]) != os.getpid()]
    write(args.output / f'environment_{args.shard}.json', dict(protocol_sha256=ph, host=socket.gethostname(),
          pid=os.getpid(), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
          uuid=uuid, visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], tf32=False,
          arithmetic='float64 diffusion states/time grid, float32 network, float64 metric reductions', batch_size=1))
    sys.path.append(str(args.diffusion_root))
    with args.dm_checkpoint.open('rb') as stream:
        model = pickle.load(stream)['ema'].to('cuda:0').eval().requires_grad_(False)
    functions = {}
    for diagnostics in [False, True]:
        functions[diagnostics], effective = build_native_diffusion(args.diffusion_root, diagnostics=diagnostics)
        dest = args.output / f'effective_sampler_{args.shard}_{int(diagnostics)}.py'
        if dest.exists(): assert dest.read_text() == effective
        else: dest.write_text(effective)
    data = np.load(args.inputs / 'fields_masks_bak.npz')
    nfe = [0]
    def hook(*unused): nfe[0] += 1
    model.register_forward_hook(hook)

    def predict(task, i, steps, hidden=False, scale=1., diagnostics=False):
        truth = [torch.from_numpy(data[f'{task}_{i}_{f}']).to('cuda:0', torch.float64) for f in ['a', 'u']]
        masks = [torch.from_numpy(data[f'{task}_{i}_mask_{f}']).to('cuda:0', torch.float64) for f in ['a', 'u']]
        observed = [t * m * scale for t, m in zip(truth, masks)]
        if hidden: observed = [t + 2.7 * (1 - m) for t, m in zip(observed, masks)]
        config = copy.deepcopy(p['diffusion_configs'][f'{task}_{steps}']['config'])
        config['generate'].update(device='cuda:0', seed=0, batch_size=1)
        config['data']['offset'] = i
        trace, nfe[0] = [], 0
        pred = functions[diagnostics](config, model, *observed, masks[0][0, 0], masks[1][0, 0], None, trace)
        assert nfe[0] == 2 * steps - 1
        return pred, truth, masks, config, trace

    check_path = args.output / f'pilot_{args.shard}.json'
    if not check_path.exists():
        i = p['pilot_ids'][args.shard]
        base = predict('both', i, 100)[0]
        check = dict(protocol_sha256=ph, sample_id=i)
        for label, options in [('repeat', {}), ('hidden', dict(hidden=True)),
                               ('sensitivity', dict(scale=.83)), ('diagnostic_reference', dict(diagnostics=True))]:
            pred = predict('both', i, 100, **options)[0]
            check[label] = max(float((x - y).abs().max()) for x, y in zip(base, pred))
        assert all(bool(torch.isfinite(x).all()) for x in base)
        assert check['repeat'] == check['hidden'] == check['diagnostic_reference'] == 0 and check['sensitivity'] > 0
        check.update(status='pass', nfe=199)
        write(check_path, check)
        print('PILOT', json.dumps(check), flush=True)
    else:
        assert json.loads(check_path.read_text())['protocol_sha256'] == ph
    jobs = [(task, i, n) for i in p['assignment'][str(args.shard)] for task in TASKS for n in [1000, 100]]
    for j, (task, i, steps) in enumerate(jobs):
        dest = args.output / 'results' / task / f'DiffusionPDE_{steps}' / f'sample{i}_seed0.pt'
        receipt = dest.with_suffix('.json')
        if receipt.exists():
            r = json.loads(receipt.read_text())
            assert r['protocol_sha256'] == ph and r['prediction_sha256'] == sha(dest)
            continue
        write(args.output / f'status_{args.shard}.json', dict(status='running', completed=j, total=len(jobs),
              task=task, sample_id=i, steps=steps, protocol_sha256=ph))
        print('START', task, i, steps, flush=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        pred, truth, masks, config, trace = predict(task, i, steps)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        finite = all(bool(torch.isfinite(x).all()) for x in pred)
        r = dict(task=task, sample_id=i, seed=0, steps=steps, method='DiffusionPDE', distribution='rough',
                 status='complete' if finite else 'nonfinite', seconds=elapsed, nfe=nfe[0], shard=args.shard,
                 protocol_sha256=ph, relative_l2=[float((x - y).norm() / y.norm()) if finite else None for x, y in zip(pred, truth)])
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.pt.tmp')
        torch.save(dict(prediction=[x.cpu() for x in pred], truth=[x.cpu() for x in truth],
                   masks=[x.cpu() for x in masks], trace=trace, effective_config=config, receipt=r), temp)
        temp.replace(dest)
        r['prediction_sha256'] = sha(dest)
        write(receipt, r)
        print('DONE', json.dumps(r), flush=True)
    write(args.output / f'status_{args.shard}.json', dict(status='complete', completed=len(jobs),
          total=len(jobs), protocol_sha256=ph))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'worker'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--template-source', type=Path)
    parser.add_argument('--diffusion-root', type=Path, required=True)
    parser.add_argument('--dm-checkpoint', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=range(7), default=0)
    args = parser.parse_args()
    freeze(args) if args.mode == 'freeze' else worker(args)


if __name__ == '__main__':
    main()

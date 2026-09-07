"""Six- or eight-worker continuation of the immutable four-worker NS plan.

The native sampler stays frozen. Completed calls and actual producer GPUs
are retained; only unfinished pairs are redistributed to newly idle GPUs.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import statistics
from types import SimpleNamespace

import run_ns_loss_study as frozen
from ns_acceleration import all_jobs, digest, stem, validate_plan as validate_parent

ROOT = Path(__file__).resolve().parents[1]
def dimensions(shards):
    assert shards in (6, 8)
    return shards, 1728 // shards


def producer(parent, parent_owner, key):
    return parent['initial_completed'][key]['legacy_worker'] if key in parent['initial_completed'] else parent_owner[key]


def validate_plan(plan, source, parent):
    SHARDS, SLOTS = dimensions(plan['shards'])
    old_owner = validate_parent(parent, source)
    assert plan['version'] == 2 and plan['shards'] == SHARDS
    assert plan['protocol_sha256'] == parent['protocol_sha256']
    assert plan['source_sha256'] == parent['source_sha256']
    jobs = [tuple(j) for j in plan['ordered_jobs']]
    assert len(jobs) == len(set(jobs)) == 1728 and set(jobs) == set(all_jobs(source))
    assert [w['shard'] for w in plan['workers']] == list(range(SHARDS))
    assert len({(w['environment']['host'], w['environment']['uuid']) for w in plan['workers']}) == SHARDS
    owner = {stem(j): i % SHARDS for i, j in enumerate(jobs)}
    initial = plan['initial_completed']
    assert set(parent['initial_completed']) <= set(initial) <= set(owner)
    for key, value in initial.items():
        assert value['producer_worker'] == producer(parent, old_owner, key)
        if key in parent['initial_completed']:
            assert all(value[k] == parent['initial_completed'][key][k] for k in ['receipt_sha256', 'prediction_sha256'])
    pairs = defaultdict(list)
    for job in jobs:
        pairs[job[:3]+job[4:]].append(job)
    for pair in pairs.values():
        pending = [j for j in pair if stem(j) not in initial]
        done = [j for j in pair if stem(j) in initial]
        if len(pending) == 2:
            assert owner[stem(pending[0])] == owner[stem(pending[1])]
        elif len(pending) == 1:
            assert owner[stem(pending[0])] == initial[stem(done[0])]['producer_worker']
    counts = [sum(stem(j) not in initial for j in jobs[i::SHARDS]) for i in range(SHARDS)]
    assert counts == plan['pending_counts'] and all(len(jobs[i::SHARDS]) == SLOTS for i in range(SHARDS))
    return owner


def make_plan(args):
    parent = json.loads(args.parent_plan.read_text())
    source = json.loads((args.inputs/'source.json').read_text())
    old_owner = validate_parent(parent, source)
    assert digest(args.results/'protocol.json') == parent['protocol_sha256']
    assert digest(args.inputs/'source.json') == parent['source_sha256']
    workers = json.loads(args.workers.read_text())
    SHARDS, SLOTS = dimensions(len(workers))
    for i in range(4):
        env = json.loads((args.results/f'acceleration_environment_{i}.json').read_text())
        assert workers[i]['environment'] == env
    initial, timings = {}, defaultdict(list)
    for path in sorted((args.results/'results').rglob('*.json')):
        receipt = json.loads(path.read_text())
        key = stem(tuple(receipt[k] for k in ['task', 'method', 'steps', 'exchange', 'sample_id', 'seed']))
        assert key in old_owner and receipt['protocol_sha256'] == parent['protocol_sha256']
        assert receipt['status'] == 'complete'
        assert digest(path.with_suffix('.pt')) == receipt['prediction_sha256']
        worker = producer(parent, old_owner, key)
        initial[key] = dict(receipt_sha256=digest(path), prediction_sha256=receipt['prediction_sha256'], producer_worker=worker)
        timings[worker, receipt['method'], receipt['steps']].append(receipt['seconds'])
    assert len(parent['initial_completed']) <= len(initial) < 1728
    costs = {}
    for i in range(SHARDS):
        for method, n in [('FM4PDE', 100), ('DiffusionPDE', 100), ('DiffusionPDE', 1000)]:
            measured = timings[i, method, n] if i < 4 else timings[2, method, n]+timings[3, method, n]
            fallback = (9.5 if method == 'FM4PDE' else 16. if n == 100 else 158.) * (1. if i < 2 else 1.6 if method == 'FM4PDE' else 2.2)
            costs[i, method, n] = statistics.median(measured) if len(measured) >= 3 else fallback
    cost = lambda j, i: costs[i, j[1], j[2]]
    pairs = defaultdict(list)
    jobs = all_jobs(source)
    for job in jobs:
        pairs[job[:3]+job[4:]].append(job)
    queues, load, free = [[] for _ in range(SHARDS)], [0.]*SHARDS, []
    for pair in pairs.values():
        pending = [j for j in pair if stem(j) not in initial]
        done = [j for j in pair if stem(j) in initial]
        if len(pending) == 1:
            worker = initial[stem(done[0])]['producer_worker']
            queues[worker].extend(pending)
            load[worker] += cost(pending[0], worker)
        elif len(pending) == 2:
            free.append(sorted(pending, key=lambda j: j[3]))
    assert all(len(q) <= SLOTS for q in queues)
    for pair in sorted(free, key=lambda pp: (-sum(cost(j, 0) for j in pp), stem(pp[0]))):
        available = [i for i in range(SHARDS) if len(queues[i])+2 <= SLOTS]
        worker = min(available, key=lambda i: load[i]+sum(cost(j, i) for j in pair))
        queues[worker].extend(pair)
        load[worker] += sum(cost(j, worker) for j in pair)
    pending_counts = [len(q) for q in queues]
    for job in jobs:
        if stem(job) in initial:
            worker = min((i for i in range(SHARDS) if len(queues[i]) < SLOTS), key=lambda i: len(queues[i]))
            queues[worker].append(job)
    plan = dict(version=2, shards=SHARDS, ordered_jobs=[queues[i][k] for k in range(SLOTS) for i in range(SHARDS)],
                workers=workers, initial_completed=initial, pending_counts=pending_counts, estimated_worker_seconds=load,
                source_sha256=parent['source_sha256'], protocol_sha256=parent['protocol_sha256'],
                parent_plan_sha256=digest(args.parent_plan), scheduler_sha256=digest(Path(__file__)),
                native_driver_sha256=parent['native_driver_sha256'],
                planning_costs=[dict(worker=i, method=m, steps=n, seconds=c) for (i,m,n),c in costs.items()],
                scope='Scheduling only. All committed results immutable; partly completed pairs keep their actual GPU, fully pending pairs share one GPU. No score-dependent allocation. Medians are scheduling estimates, not benchmark latency; A800 estimates use A100 observations.')
    validate_plan(plan, source, parent)
    assert not args.plan.exists()
    frozen.write(args.plan, plan)
    print(json.dumps(dict(plan_sha256=digest(args.plan), initial=len(initial), pending_counts=pending_counts,
                          estimated_hours=[v/3600 for v in load])), flush=True)


def run(args):
    plan = json.loads(args.plan.read_text())
    SHARDS, SLOTS = dimensions(plan['shards'])
    parent_path = args.output/'acceleration_plan.json'
    parent = json.loads(parent_path.read_text())
    assert digest(parent_path) == plan['parent_plan_sha256']
    source = json.loads((args.inputs/'source.json').read_text())
    assert digest(args.inputs/'source.json') == plan['source_sha256']
    assert digest(Path(frozen.__file__)) == plan['native_driver_sha256']
    assert digest(Path(__file__)) == plan['scheduler_sha256']
    assert digest(args.output/'protocol.json') == plan['protocol_sha256']
    assert digest(args.output/'acceleration_extension_plan.json') == digest(args.plan)
    validate_plan(plan, source, parent)
    assert args.shard in range(SHARDS)
    worker = plan['workers'][args.shard]
    assert socket.gethostname() == worker['environment']['host']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(worker['gpu'])
    import torch
    env = worker['environment']
    assert torch.__version__ == env['torch'] and torch.version.cuda == env['cuda']
    assert str(torch.cuda.get_device_properties(0).uuid) == env['uuid']
    assert torch.cuda.get_device_name(0) == env['gpu']
    # The native pilot uses four reserved inputs indexed by shard. Added-GPU
    # pilots run beforehand with indices 0..3 and are copied by verified hash.
    assert digest(args.output/'pilot_0.json') == worker['pilot_sha256']
    all_expected = [tuple(j) for j in plan['ordered_jobs']]
    expected = [j for j in all_expected[args.shard::SHARDS] if stem(j) not in plan['initial_completed']]
    assert len(expected) == plan['pending_counts'][args.shard]
    class PlannedOrder:
        def __init__(self, seed):
            assert seed == 20260908
        def shuffle(self, jobs):
            assert len(jobs) == len(all_expected) and set(jobs) == set(all_expected)
            jobs[:] = expected
    frozen.random = SimpleNamespace(Random=PlannedOrder)
    frozen.write(args.output/f'extension_worker_{args.shard}.json', dict(
        plan_sha256=digest(args.plan), scheduler_sha256=digest(Path(__file__)),
        native_driver_sha256=digest(Path(frozen.__file__)), shard=args.shard, shards=SHARDS,
        host=socket.gethostname(), pid=os.getpid(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        assigned_slots=SLOTS, initial_pending=plan['pending_counts'][args.shard],
        native_shard=0, native_shards=1))
    # Each continuation worker has its own output directory. Removing already
    # committed slots before the native loop is equivalent to its receipt skip
    # and avoids copying old prediction tensors onto every newly added host.
    native_args = SimpleNamespace(**vars(args))
    native_args.shard, native_args.shards, native_args.pilot_only = 0, 1, False
    frozen.run(native_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan','run'])
    for name in ['plan','inputs','results','output','parent-plan','workers','diffusion-root','fm-checkpoint','dm-checkpoint']:
        parser.add_argument('--'+name, type=Path, required=name in ['plan','inputs'])
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    make_plan(args) if args.mode == 'plan' else run(args)


if __name__ == '__main__':
    main()

"""Resume frozen NS sampling with four explicitly planned worker queues.

Only job ordering/ownership changes. The original sampling module, protocol,
seed, checkpoint, arithmetic, and native update loop remain unchanged.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import socket
import sys
from types import SimpleNamespace
import os

import run_ns_loss_study as frozen

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def stem(job):
    task,method,steps,exchange,i,seed=job
    return f'{task}/{method}_{steps}/{"exchanged" if exchange else "original"}/sample{i}_seed{seed}'


def all_jobs(source):
    return [(t,m,n,e,i,s) for i in source['evaluation_ids'] for s in source['inference_seeds']
            for t in frozen.TASKS for m,n,e in frozen.VARIANTS]


def validate_plan(plan, source):
    jobs=[tuple(j) for j in plan['ordered_jobs']]
    assert len(jobs)==1728 and len(set(jobs))==1728 and set(jobs)==set(all_jobs(source))
    assert plan['shards']==4 and all(len(jobs[i::4])==432 for i in range(4))
    initial=plan['initial_completed'];owner={stem(j):i%4 for i,j in enumerate(jobs)}
    pairs=defaultdict(list)
    for j in jobs:pairs[j[:3]+j[4:]].append(j)
    for pair in pairs.values():
        pending=[j for j in pair if stem(j) not in initial]
        completed=[j for j in pair if stem(j) in initial]
        if len(pending)==2:assert owner[stem(pending[0])]==owner[stem(pending[1])]
        if len(pending)==1:
            assert owner[stem(pending[0])]==initial[stem(completed[0])]['legacy_worker']
    return owner


def make_plan(args):
    source=json.loads((args.inputs/'source.json').read_text())
    protocol=json.loads((args.results/'protocol.json').read_text())
    ph=digest(args.results/'protocol.json')
    assert protocol['source_sha256']==digest(args.inputs/'source.json')
    jobs=all_jobs(source);random.Random(20260908).shuffle(jobs)
    legacy_owner={stem(j):i%2 for i,j in enumerate(jobs)}
    initial={}
    for path in sorted((args.results/'results').rglob('*.json')):
        receipt=json.loads(path.read_text())
        key=stem(tuple(receipt[k] for k in ['task','method','steps','exchange','sample_id','seed']))
        assert key in legacy_owner and receipt['protocol_sha256']==ph
        assert receipt['status']=='complete', 'Keep failures explicit before changing worker ownership.'
        assert digest(path.with_suffix('.pt'))==receipt['prediction_sha256']
        initial[key]=dict(receipt_sha256=digest(path),prediction_sha256=receipt['prediction_sha256'],
                          legacy_worker=legacy_owner[key])
    assert 0<len(initial)<1728
    pairs=defaultdict(list)
    for j in jobs:pairs[j[:3]+j[4:]].append(j)
    queues=[[] for _ in range(4)];load=[0.]*4
    def cost(job,worker):
        base=9.5 if job[1]=='FM4PDE' else 16. if job[2]==100 else 158.
        # Conservative scheduling estimates only, not reported latency results.
        multiplier=1. if worker<2 else 1.6 if job[1]=='FM4PDE' else 2.2
        return base*multiplier
    free=[]
    for pair in pairs.values():
        pending=[j for j in pair if stem(j) not in initial]
        completed=[j for j in pair if stem(j) in initial]
        if len(pending)==1:
            worker=initial[stem(completed[0])]['legacy_worker']
            queues[worker].extend(pending);load[worker]+=cost(pending[0],worker)
        elif len(pending)==2:free.append(sorted(pending,key=lambda j:j[3]))
    for pair in sorted(free,key=lambda pp:(-sum(cost(j,0) for j in pp),stem(pp[0]))):
        available=[i for i in range(4) if len(queues[i])+2<=432]
        worker=min(available,key=lambda i:load[i]+sum(cost(j,i) for j in pair))
        queues[worker].extend(pair);load[worker]+=sum(cost(j,worker) for j in pair)
    pending_counts=[len(q) for q in queues]
    completed_jobs=[j for j in jobs if stem(j) in initial]
    for j in completed_jobs:
        worker=min((i for i in range(4) if len(queues[i])<432),key=lambda i:len(queues[i]))
        queues[worker].append(j)
    ordered=[queues[i][k] for k in range(432) for i in range(4)]
    plan=dict(version=1,shards=4,source_sha256=digest(args.inputs/'source.json'),protocol_sha256=ph,
              native_driver_sha256=protocol['code_sha256']['plot/run_ns_loss_study.py'],
              scheduler_sha256=digest(Path(__file__)),initial_completed=initial,ordered_jobs=ordered,
              pending_counts=pending_counts,estimated_worker_seconds=load,
              scheduling_estimates='FM100=9.5s, Diff100=16s, Diff1000=158s on4090; A100 multipliers1.6/2.2. Queue estimates only; actual runtime is recorded per call.',
              legacy_environment={str(i):json.loads((args.results/f'environment_{i}.json').read_text()) for i in [0,1]},
              workers=[dict(shard=i,host='server193' if i<2 else 'server197',gpu=i%2,
                            hardware='RTX4090' if i<2 else 'A100') for i in range(4)],
              scope='Scheduling-only continuation after user requested acceleration. Completed calls preserved by hash. Every fully pending loss pair shares one GPU; partly completed pairs remain on their original GPU. No error-dependent selection.')
    validate_plan(plan,source)
    assert not args.plan.exists(), 'The acceleration plan is immutable once written.'
    frozen.write(args.plan,plan)
    print(json.dumps(dict(plan=str(args.plan),sha256=digest(args.plan),initial=len(initial),
                          pending_counts=pending_counts,estimated_hours=[v/3600 for v in load])),flush=True)


def run(args):
    plan=json.loads(args.plan.read_text());source=json.loads((args.inputs/'source.json').read_text())
    assert digest(args.inputs/'source.json')==plan['source_sha256']
    assert digest(Path(frozen.__file__))==plan['native_driver_sha256']
    assert digest(Path(__file__))==plan['scheduler_sha256']
    assert digest(args.output/'protocol.json')==plan['protocol_sha256']
    validate_plan(plan,source);args.shards=4;args.pilot_only=False
    assert args.shard in range(4)
    assert digest(args.output/'acceleration_plan.json')==digest(args.plan)
    expected=[tuple(j) for j in plan['ordered_jobs']]
    class PlannedOrder:
        def __init__(self,seed):assert seed==20260908
        def shuffle(self,jobs):
            assert len(jobs)==len(expected) and set(jobs)==set(expected)
            jobs[:]=expected
    # Module-local replacement: torch and numpy random generators are untouched.
    frozen.random=SimpleNamespace(Random=PlannedOrder)
    frozen.write(args.output/f'acceleration_worker_{args.shard}.json',dict(
        plan_sha256=digest(args.plan),scheduler_sha256=digest(Path(__file__)),
        native_driver_sha256=digest(Path(frozen.__file__)),shard=args.shard,shards=4,
        host=socket.gethostname(),pid=os.getpid(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        assigned_slots=432,initial_pending=plan['pending_counts'][args.shard]))
    frozen.run(args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','run'])
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--results',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--diffusion-root',type=Path)
    parser.add_argument('--fm-checkpoint',type=Path)
    parser.add_argument('--dm-checkpoint',type=Path)
    parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    make_plan(args) if args.mode=='plan' else run(args)


if __name__=='__main__':main()

"""Redistribute unfinished calls while retaining the frozen single-input worker.

The generated worker changes only its job-selection expression. The native
sampling function, seeds, guidance, precision, pilots and output tensors are
unchanged. Assignment and executor hashes are recorded with new receipts.
Stop the previous owners before making the final assignment or starting work.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import socket

import run_burgers_revision as native
from run_paper_ablation_revision import digest, write


def key(job):
    return job['cell'], job['method'], int(job['steps']), int(job['sample_id'])


def make_assignment(args):
    protocol = json.loads(args.sampling_protocol.read_text())
    ph = digest(args.sampling_protocol)
    jobs = {key(j):j for j in protocol['jobs']}
    assert len(jobs) == 7000
    complete = {}
    for root in args.results:
        for path in (root/'results').glob('*/*/*.json'):
            receipt = json.loads(path.read_text())
            ident = key(receipt)
            assert ident in jobs and ident not in complete, (ident,path)
            assert receipt['protocol_sha256'] == ph
            assert all(receipt[k] == v for k,v in jobs[ident].items())
            assert digest(path.with_suffix('.pt')) == receipt['tensor_sha256']
            complete[ident] = dict(job=jobs[ident], receipt=str(path),
                                   receipt_sha256=digest(path), tensor_sha256=receipt['tensor_sha256'])
    resources = json.loads(args.resources.read_text())
    workers = [dict(index=i, **row, jobs=[], work=0) for i,row in enumerate(resources)]
    assert len({(w['host'],w['gpu']) for w in workers}) == len(workers)
    for job in protocol['jobs']:
        if key(job) in complete: continue
        selected = min(workers, key=lambda w:(w['work']/w['relative_rate'],w['index']))
        selected['jobs'].append(job)
        selected['work'] += job['steps']
    pending = [key(j) for w in workers for j in w['jobs']]
    assert len(pending) == len(set(pending))
    assert set(pending).isdisjoint(complete) and set(pending)|set(complete) == set(jobs)
    assert not args.output.exists(), 'Assignments are immutable; use a new filename'
    write(args.output,dict(protocol_sha256=ph,executor_sha256=digest(Path(__file__)),
                          workers=workers,completed=list(complete.values()),
                          completed_calls=len(complete),assigned_calls=len(pending),
                          original_workers_stopped=args.original_workers_stopped))
    print('Assigned',len(pending),'remaining calls to',len(workers),'workers;',len(complete),'retained')


def selected_worker(assigned):
    path = Path(native.__file__)
    module = ast.parse(path.read_text())
    reference = next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    worker = copy.deepcopy(reference)
    found = [n for n in ast.walk(worker) if isinstance(n,ast.For) and
             isinstance(n.target,ast.Name) and n.target.id=='job']
    assert len(found) == 1
    loop = found[0]
    assert ast.unparse(loop.iter) == "[j for j in protocol['jobs'] if j['shard'] == args.shard]"
    old_iter = copy.deepcopy(loop.iter)
    loop.iter = ast.parse("[j for j in protocol['jobs'] if _job_key(j) in _assigned_jobs]",mode='eval').body
    generated = ast.unparse(ast.fix_missing_locations(worker))+'\n'
    loop.iter = old_iter
    assert ast.dump(worker,include_attributes=False) == ast.dump(reference,include_attributes=False)
    # The check above proves that the job selector is the only AST change.
    return generated, dict(vars(native),_job_key=key,_assigned_jobs=assigned)


def work(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    protocol = json.loads(args.sampling_protocol.read_text())
    assert digest(Path(native.__file__)) == protocol['runner_sha256']
    if args.pilot_only:
        assigned = set()
        assignment_hash = None
    else:
        ledger = json.loads(args.assignment.read_text())
        assert ledger['original_workers_stopped'], 'Previous owners must stop before reassignment'
        assert ledger['protocol_sha256'] == digest(args.sampling_protocol)
        assert ledger['executor_sha256'] == digest(Path(__file__))
        worker = ledger['workers'][args.worker_index]
        assert worker['index'] == args.worker_index and worker['host'] == socket.gethostname()
        assert str(worker['gpu']) == os.environ.get('CUDA_VISIBLE_DEVICES')
        assigned = {key(j) for j in worker['jobs']}
        assignment_hash = digest(args.assignment)
    args.shard = f'parallel_{args.worker_index:02d}'
    generated, scope = selected_worker(assigned)
    args.output.mkdir(parents=True,exist_ok=True)
    generated_path = args.output/f'worker_{args.shard}.py'
    generated_path.write_text(generated)
    evidence = dict(executor_sha256=digest(Path(__file__)),
                    generated_worker_sha256=digest(generated_path),assignment_sha256=assignment_hash,
                    reference_runner_sha256=protocol['runner_sha256'],
                    worker_index=args.worker_index,pilot_only=args.pilot_only,
                    only_ast_change='job-selection expression')
    write(args.output/f'executor_{args.shard}.json',evidence)

    def recorded_write(path,data):
        if isinstance(data,dict) and 'rel_l2_u' in data and 'sample_id' in data:
            data = dict(data,executor=evidence)
        write(path,data)

    scope['write'] = recorded_write
    exec(compile(generated,str(generated_path),'exec'),scope)
    import torch
    assert torch.cuda.mem_get_info()[0] >= 12*1024**3, 'At least 12 GiB free GPU memory required'
    scope['worker'](args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest='mode',required=True)
    plan = modes.add_parser('assign')
    plan.add_argument('--sampling-protocol',type=Path,required=True)
    plan.add_argument('--results',type=Path,nargs='+',required=True)
    plan.add_argument('--resources',type=Path,required=True)
    plan.add_argument('--output',type=Path,required=True)
    plan.add_argument('--original-workers-stopped',action='store_true')
    run = modes.add_parser('work')
    for name in ['inputs','sampling-protocol','output','fm-weights','dm-weights','diffusion-root']:
        run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--assignment',type=Path)
    run.add_argument('--worker-index',type=int,required=True)
    run.add_argument('--pilot-only',action='store_true')
    a = p.parse_args()
    make_assignment(a) if a.mode=='assign' else work(a)

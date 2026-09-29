"""Run one manuscript paragraph from a checked-in experiment manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def selected_jobs(spec, args):
    """Select existing cases without changing their seeds or batch membership."""
    selected = []
    for job in spec['jobs']:
        overrides = job.get('overrides', {})
        if getattr(args, 'job_ids', None) and job['id'] not in args.job_ids:
            continue
        if args.pdes and job['pde'] not in args.pdes:
            continue
        if getattr(args, 'tasks', None) and job['task'] not in args.tasks:
            continue
        if getattr(args, 'test_types', None) and overrides.get('test_type', 'id') not in args.test_types:
            continue
        if getattr(args, 'phases', None) and overrides.get('sampler_phase', 'stochastic') not in args.phases:
            continue
        if getattr(args, 'steps', None) and overrides.get('num_steps', 100) not in args.steps:
            continue
        selected.append(job)
    selected = selected[:args.limit]
    return selected[getattr(args, 'shard_index', 0)::getattr(args, 'num_shards', 1)]


def resolved_jobs(spec, args):
    from sampling.config import load_config, parse_cli_overrides
    extra = parse_cli_overrides(args.override)
    selected = selected_jobs(spec, args)
    if not selected:
        raise ValueError('No jobs match the requested PDEs')
    for job in selected:
        overrides = dict(job.get('overrides', {}))
        overrides.update(device=args.device, output_dir=str(args.output/job['id']),
                         save_plots=False, save_intermediate=False, save_per_sample_curves=True)
        overrides.update(extra)
        config = load_config(ROOT/f"configs/main/{job['task']}/{job['pde']}.yaml", overrides)
        yield job, config


def run_sampling(spec, args):
    jobs = list(resolved_jobs(spec, args))
    if args.plan_only:
        print(json.dumps(dict(paragraph=spec['paragraph'], jobs=len(jobs),
                              realizations=sum(c.batch_size for _, c in jobs),
                              example=jobs[0][1].asdict()), indent=2))
        return
    import torch
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from sampling.runner import run_single_ablation
    from sampling.data import load_ground_truth

    torch.set_num_threads(args.threads)
    from experiments.paper.provenance import runtime_identity
    runtimes = {}
    bundle, previous = None, None
    source_hashes = {str(p.relative_to(ROOT)): digest(p)
                     for directory in ['sampling', 'models', 'flow_matching', 'torchdiffeq', 'experiments/paper', 'data']
                     for p in (ROOT/directory).rglob('*') if p.suffix in {'.py', '.m'}}
    asset_hashes = {}
    records = []
    failures = []
    for job, config in jobs:
        if config.device not in runtimes:
            runtimes[config.device] = runtime_identity(config.device)
        # Detect stale outputs after either code, settings, data, or weights change.
        for path in [config.checkpoint_path, config.data_path]:
            if path not in asset_hashes:
                asset_hashes[path] = digest(path)
        identity = dict(config=config.asdict(), sources=source_hashes,
                        runtime=runtimes[config.device], postprocess=spec.get('postprocess'),
                        cohort=job.get('cohort'),
                        observation_protocol=job.get('observation_protocol'),
                        data_sha256=asset_hashes[config.data_path],
                        checkpoint_sha256=asset_hashes[config.checkpoint_path])
        folder = Path(config.output_dir)
        receipt = folder/'receipt.json'
        if receipt.exists() and args.resume:
            saved = json.loads(receipt.read_text())
            if saved['identity'] != identity:
                raise ValueError(f'{receipt}: settings or source changed; choose a new output directory')
            if digest(saved['result']) != saved['result_sha256']:
                raise ValueError(f'{receipt}: saved prediction checksum mismatch')
            records.append(saved)
            continue
        key = (config.pde, config.checkpoint_path, config.model_profile, config.device)
        if previous != key:
            bundle = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bundle = load_fm4pde_checkpoint_bundle(config.checkpoint_path, config.pde, config.device,
                                                  wrap=True, model_profile=config.model_profile)
            previous = key
        try:
            truth = load_ground_truth(config)
            from experiments.paper.observations import for_job
            masks = for_job(job, truth, config)
            result = run_single_ablation(config, checkpoint_bundle=bundle, ground_truth=truth,
                                         observation_masks=masks)
            if result.get('status') != 'ok' or result.get('pde_eval_error_count', 0):
                raise RuntimeError(f"{job['id']}: sampling or PDE evaluation failed")
        except Exception:
            failure = dict(id=job['id'], identity=identity, traceback=traceback.format_exc())
            write_json(folder/'failure.json', failure)
            failures.append(failure)
            write_json(args.output/'failures.json', failures)
            if not args.continue_on_error:
                raise
            print(f"{job['id']}: FAILED (see {folder/'failure.json'})", flush=True)
            continue
        path = Path(result['run_dir'])/'result.pt'
        saved = dict(id=job['id'], pde=config.pde, task=config.task,
                     identity=identity, result=str(path), result_sha256=digest(path))
        if spec.get('postprocess') == 'consistency':
            from experiments.paper.consistency import evaluate
            saved['consistency'] = evaluate(path, folder, args)
        write_json(receipt, saved)
        records.append(saved)
        write_json(args.output/'index.json', records)
        print(f"{job['id']}: complete", flush=True)
    write_json(args.output/'index.json', records)
    from experiments.paper.summarize import summarize
    if records:
        summarize(args.output)
    write_json(args.output/'completion.json', dict(expected=len(jobs), completed=len(records), failed=len(failures)))
    if spec.get('postprocess') == 'temporal_truth':
        from experiments.paper.temporal_truth import evaluate
        evaluate(spec, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--plan-only', action='store_true', default=os.environ.get('PLAN_ONLY', '').lower() in {'1','true'})
    parser.add_argument('--pdes', nargs='+', default=os.environ.get('PDE_LIST', '').split() or None)
    parser.add_argument('--job-ids', nargs='+', help='Select exact case IDs from the manifest')
    parser.add_argument('--truth-only', action='store_true', help='Recompute temporal losses on real endpoint pairs without sampling')
    parser.add_argument('--tasks', nargs='+', choices=['forward','inverse','both'])
    parser.add_argument('--test-types', nargs='+', choices=['id','smooth','rough','rough2','rough3'])
    parser.add_argument('--phases', nargs='+', choices=['stochastic','deterministic','hybrid_d2s','hybrid_s2d'])
    parser.add_argument('--steps', nargs='+', type=int, help='Select existing step budgets; does not override them')
    parser.add_argument('--methods', nargs='+', choices=['FM4PDE','FM4PDE-OFM','DiffusionPDE'])
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--continue-on-error', action='store_true', help='Record failed cases and continue; never replace non-finite updates')
    parser.add_argument('--device', default=os.environ.get('DEVICE', 'cuda:0'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--override', action='append', default=[])
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32, help='Conditional draws per batch')
    parser.add_argument('--matlab', default=os.environ.get('MATLAB_BIN','matlab'))
    parser.add_argument('--chebfun', default=os.environ.get('CHEBFUN_ROOT',''))
    args = parser.parse_args()
    os.chdir(ROOT)
    spec = yaml.safe_load(args.manifest.read_text())
    args.output = (args.output or Path(os.environ.get('OUTPUT_ROOT', 'outputs/paper'))/args.manifest.stem).resolve()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error('Require 0 <= --shard-index < --num-shards')
    if not args.plan_only:
        # Seeds alone do not fix CUDA kernel selection across repeated runs.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    if args.truth_only:
        if spec.get('postprocess') != 'temporal_truth':
            parser.error('--truth-only requires the temporal_residuals manifest')
        if args.plan_only:
            print(json.dumps(dict(jobs=len(selected_jobs(spec,args)), mode='truth_only')))
        else:
            from experiments.paper.temporal_truth import evaluate
            evaluate(spec,args)
    elif spec['engine'] == 'sampling':
        run_sampling(spec, args)
    else:
        from experiments.paper.special import run
        run(spec, args)


if __name__ == '__main__':
    main()

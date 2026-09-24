"""Run one manuscript paragraph from a checked-in experiment manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def resolved_jobs(spec, args):
    from sampling.config import load_config, parse_cli_overrides
    extra = parse_cli_overrides(args.override)
    selected = [j for j in spec['jobs'] if not args.pdes or j['pde'] in args.pdes]
    if not selected:
        raise ValueError('No jobs match the requested PDEs')
    for job in selected[:args.limit]:
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
    bundle, previous = None, None
    source_hashes = {str(p.relative_to(ROOT)): digest(p)
                     for directory in ['sampling', 'models', 'flow_matching', 'torchdiffeq']
                     for p in (ROOT/directory).rglob('*.py')}
    asset_hashes = {}
    records = []
    for job, config in jobs:
        # Detect stale outputs after either code, settings, data, or weights change.
        for path in [config.checkpoint_path, config.data_path]:
            if path not in asset_hashes:
                asset_hashes[path] = digest(path)
        identity = dict(config=config.asdict(), sources=source_hashes,
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
        truth = load_ground_truth(config)
        result = run_single_ablation(config, checkpoint_bundle=bundle, ground_truth=truth)
        if result.get('status') != 'ok' or result.get('pde_eval_error_count', 0):
            raise RuntimeError(f"{job['id']}: sampling or PDE evaluation failed")
        path = Path(result['run_dir'])/'result.pt'
        saved = dict(id=job['id'], pde=config.pde, task=config.task,
                     identity=identity, result=str(path), result_sha256=digest(path))
        if spec.get('postprocess') == 'consistency':
            from experiments.paper.consistency import evaluate
            saved['consistency'] = evaluate(path, folder, args)
        write_json(receipt, saved)
        records.append(saved)
        print(f"{job['id']}: complete", flush=True)
    write_json(args.output/'index.json', records)
    from experiments.paper.summarize import summarize
    summarize(args.output)
    if spec.get('postprocess') == 'temporal_truth':
        from experiments.paper.temporal_truth import evaluate
        evaluate(spec, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--plan-only', action='store_true', default=os.environ.get('PLAN_ONLY', '').lower() in {'1','true'})
    parser.add_argument('--pdes', nargs='+', default=os.environ.get('PDE_LIST', '').split() or None)
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
    if spec['engine'] == 'sampling':
        run_sampling(spec, args)
    else:
        from experiments.paper.special import run
        run(spec, args)


if __name__ == '__main__':
    main()

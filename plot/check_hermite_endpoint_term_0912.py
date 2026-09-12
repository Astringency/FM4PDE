"""Paired Hermite endpoint-term diagnostic; does not modify manuscript files."""
import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from sampling.config import AblationConfig
from sampling.masks import PairMasks
from sampling.model_io import load_fm4pde_checkpoint_bundle
from sampling.pde_residuals import compute_pde_residual
from sampling.runner import run_single_ablation
from scripts.tuning.compare_pde_guidance_schedules import combine_truths

PDES = ['nsnonbounded', 'reaction_diffusion', 'shallow_water', 'heat', 'wave', 'advection_diffusion']


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def cpu_double(v):
    if isinstance(v, torch.Tensor):
        return v.cpu().double() if v.is_floating_point() else v.cpu()
    if isinstance(v, dict):
        return {k: cpu_double(x) for k, x in v.items()}
    if isinstance(v, list):
        return [cpu_double(x) for x in v]
    return v


def diagnostics(pde, payload, config):
    params = cpu_double(copy.deepcopy(payload['pde_params']))
    for k in ['hermite_collocation_times', 'hermite_num_collocation', 'enforce_boundary_conditions',
              'boundary_condition_mode', 'boundary_residual_normalization', 'bc_weight',
              'endpoint_bc_weight', 'ns_operator_mode']:
        params[k] = config[k]
    params.update(hermite_include_integral_residual=True, hermite_integral_weight=1.0)
    out = compute_pde_residual(pde, payload['coef_final'].double(), payload['sol_final'].double(),
                               pde_params=params, residual_mode='hermite_bridge')
    return {k + '_mse': float(v.square().mean()) if v is not None else 0.0
            for k, v in out.components.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pdes', nargs='+', choices=PDES, default=PDES)
    args = p.parse_args()
    assert args.output.is_absolute() and args.reference.is_absolute()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = json.loads(args.reference.read_text())
    refs = {r['pde']: r for r in reference['rows'] if r['condition'] == 'hermite_bridge'}
    assert set(refs) == set(PDES)
    env = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               device=os.environ.get('CUDA_VISIBLE_DEVICES'), tf32=False, batch_size=1,
               reference=str(args.reference), reference_sha256=digest(args.reference))
    assert env['gpu'] == reference['environment']['gpu']
    assert env['torch'] == reference['environment']['torch']
    write(args.output / 'environment.json', env)
    write(args.output / 'reference.json', reference)
    rows = []
    for pde in args.pdes:
        old = refs[pde]
        old_path = Path(old['result_path'])
        assert digest(old_path) == old['result_sha256']
        assert digest(old_path.parent / 'masks.pt') == old['mask_sha256']
        old_payload = torch.load(old_path, map_location='cpu', weights_only=False)
        masks = torch.load(old_path.parent / 'masks.pt', map_location='cpu', weights_only=False)
        source = Path(old['config']['checkpoint_path']).parent
        assert digest(source / 'weights.pth') == old['weights_sha256']
        assert digest(source / 'truths.pt') == old['truth_sha256']
        truths = torch.load(source / 'truths.pt', map_location='cpu', weights_only=False)
        bundle = None
        for include in [True, False]:
            label = 'with_endpoint' if include else 'without_endpoint'
            folder = args.output / pde / label
            receipt = folder / 'receipt.json'
            if receipt.exists():
                row = json.loads(receipt.read_text())
                assert row['environment'] == env
                assert digest(row['result_path']) == row['result_sha256']
                rows.append(row)
                continue
            if bundle is None:
                bundle = load_fm4pde_checkpoint_bundle(str(source / 'weights.pth'), pde,
                                                       'cuda:0', model_profile=old['config']['model_profile'])
            config = copy.deepcopy(old['config'])
            assert config['hermite_include_integral_residual'] is True
            assert config['hermite_integral_weight'] == config['endpoint_bc_weight'] == 1.0
            config.update(output_dir=str(folder), ablation_name='hermite_endpoint_toggle_20260912',
                          hermite_include_integral_residual=include)
            cfg = AblationConfig(**config)
            cfg.validate()
            gt = combine_truths(truths, [0], 'cuda:0')
            assert torch.equal(gt.coef.cpu(), old_payload['coef_ground_truth'])
            assert torch.equal(gt.sol.cpu(), old_payload['sol_ground_truth'])
            pair_masks = PairMasks(masks['coef'].cuda(), masks['sol'].cuda(), copy.deepcopy(masks['metadata']))
            folder.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with (folder / 'run.log').open('w') as f, redirect_stdout(f), redirect_stderr(f):
                result = run_single_ablation(cfg, checkpoint_bundle=bundle, ground_truth=gt,
                                             observation_masks=pair_masks)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            assert result['status'] == 'ok', result['status']
            assert peak < .5 * torch.cuda.get_device_properties(0).total_memory
            path = Path(result['run_dir']) / 'result.pt'
            payload = torch.load(path, map_location='cpu', weights_only=False)
            saved_masks = torch.load(path.parent / 'masks.pt', map_location='cpu', weights_only=False)
            errors = {}
            replay = {}
            for f, prefix in [('a', 'coef'), ('u', 'sol')]:
                t = payload[prefix + '_ground_truth'].double()
                pred = payload[prefix + '_final'].double()
                mask = saved_masks[prefix].double()
                assert torch.equal(saved_masks[prefix], masks[prefix])
                assert torch.equal(payload[prefix + '_ground_truth'], old_payload[prefix + '_ground_truth'])
                assert torch.isfinite(pred).all()
                errors[f] = dict(full=float((pred-t).norm()/t.norm()),
                                 observed=float(((pred-t)*mask).norm()/(t*mask).norm()))
                replay[f] = float((pred-old_payload[prefix + '_final'].double()).norm()/t.norm())
            assert payload['config']['hermite_include_integral_residual'] is include
            metrics = payload['metrics']
            assert (metrics['pde_residual_channels_endpoint'] > 0) is include
            if not include:
                assert metrics['endpoint_residual_norm'] == metrics['pde_loss_endpoint'] == 0.0
            row = dict(pde=pde, include_endpoint=include, config=dataclasses.asdict(cfg), environment=env,
                       errors=errors, prediction_difference_from_archived=replay,
                       common_residual_diagnostics=diagnostics(pde, payload, config),
                       final_loss_components={k: metrics[k] for k in ['pde_loss_interior',
                                              'pde_loss_boundary', 'pde_loss_endpoint',
                                              'pde_residual_channels_endpoint']},
                       result_path=str(path), result_sha256=digest(path),
                       mask_sha256=digest(path.parent / 'masks.pt'),
                       source_weights_sha256=old['weights_sha256'], source_truth_sha256=old['truth_sha256'],
                       seconds=time.perf_counter()-start, peak_bytes=peak, status=result['status'])
            write(receipt, row)
            rows.append(row)
            print('DONE', pde, label, errors, 'peak GiB', peak / 2**30, flush=True)
            del gt, pair_masks, payload, saved_masks
        del bundle, truths, old_payload, masks
        gc.collect()
        torch.cuda.empty_cache()
    write(args.output / 'complete.json', dict(environment=env, rows=rows))


if __name__ == '__main__':
    main()

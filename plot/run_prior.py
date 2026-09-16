"""Generate an unconditional sample with the configured pretrained model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sampling.study_models import add_model_arguments, model_config, print_model_plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pde', default='poisson')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint-sha256', help='Optional expected checkpoint digest.')
    add_model_arguments(parser)
    args = parser.parse_args()
    args.pdes = [args.pde]
    if print_model_plan(args):
        return
    if args.steps <= 0:
        parser.error('--steps must be positive')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(args.output))
    cfg = model_config(args, args.pde)
    checkpoint_hash = sha(cfg.checkpoint_path)
    if args.checkpoint_sha256 and checkpoint_hash != args.checkpoint_sha256:
        raise ValueError('Checkpoint digest differs from --checkpoint-sha256')
    import torch
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    from sampling.sampler_wrappers import _call_velocity_model
    from sampling.state import standardized_to_physical_state
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, normalizer, payload = load_fm4pde_checkpoint_bundle(
        cfg.checkpoint_path, args.pde, args.device, model_profile=cfg.model_profile)
    architecture = payload['model_config']
    if architecture.get('scalar_conditioning', False) or architecture.get('num_classes') is not None:
        raise ValueError('This prior entry requires an unconditioned model; conditional models need their physical/class inputs.')
    shape = (1, cfg.img_channels, cfg.img_resolution, cfg.img_resolution)
    torch.manual_seed(args.seed)
    x = torch.randn(shape, device=args.device, dtype=torch.float32)
    noise_hash = hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    is_cuda = x.is_cuda
    if is_cuda:
        torch.cuda.synchronize(x.device)
    started = time.perf_counter()
    with torch.no_grad():
        for k in range(args.steps):
            t = torch.tensor(k / args.steps, device=x.device, dtype=x.dtype)
            x = x + (1 / args.steps) * _call_velocity_model(model, x, t, None)
        fields = standardized_to_physical_state(x, args.pde, normalizer=normalizer)
    if is_cuda:
        torch.cuda.synchronize(x.device)
    elapsed = time.perf_counter() - started
    if not (torch.isfinite(fields.coef).all() and torch.isfinite(fields.sol).all()):
        raise FloatingPointError('Nonfinite unconditional prediction')
    args.output.mkdir(parents=True)
    torch.save(dict(coef=fields.coef.cpu(), sol=fields.sol.cpu(), scalar_conditioning=None), args.output / 'sample.pt')
    report = dict(status='complete', pde=args.pde, seed=args.seed, steps=args.steps,
                  sampler='direct uniform Euler ODE', guidance='none', checkpoint=cfg.checkpoint_path,
                  model_profile=payload['selected_model_profile'], checkpoint_sha256=checkpoint_hash,
                  parameters=sum(p.numel() for p in model.model.parameters()),
                  initial_noise_sha256=noise_hash, sample_sha256=sha(args.output / 'sample.pt'),
                  elapsed_seconds=elapsed, torch=torch.__version__, cuda=torch.version.cuda,
                  device=args.device, gpu=torch.cuda.get_device_name(x.device) if is_cuda else None,
                  visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), script_sha256=sha(__file__))
    (args.output / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

"""Replay one recorded update before moving a continuation to another GPU.

Run this file directly so imports resolve from --scientific-checkout. It writes
only diagnostic output, never the training checkpoint or training logs.
"""
import argparse
import json
from pathlib import Path
import sys


def run(root, pde, arm, checkpoint, expected_sha, reference, checkout, output):
    sys.path.insert(0, str(checkout.resolve()))
    import torch
    from experiments.ema_timesteps import study
    from experiments.optimizer_diagnostics.study import sha, tensor_sha, write
    from training.load_and_save import _load_optimizer_state_preserving_runtime_options

    assert Path(study.__file__).resolve() == (checkout / 'experiments/ema_timesteps/study.py').resolve()
    assert sha(checkpoint) == expected_sha
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    meta = saved['ema_timesteps']
    profile = json.loads((root / 'profiles' / pde / 'complete.json').read_text())
    config = study.configuration(root, pde, arm)
    assert meta['config'] == config and meta['microbatch'] == profile['microbatch']
    environment = study.runtime()
    for key in ['gpu', 'torch', 'cuda', 'tf32', 'precision']:
        assert environment[key] == meta['environment'][key], key
    expected = json.loads(reference.read_text())
    epoch = int(meta['completed_epochs']) + 1
    step = int(meta['additional_updates']) + 1
    assert expected['epoch'] == epoch and expected['step'] == step
    assert expected['pde'] == pde and expected['arm'] == arm
    assert step == (epoch - 1) * 703 + 1

    _, data, record, model, opt = study.initialize(root, pde)
    assert meta['input_data_sha256'] == record['data_sha256']
    model.load_state_dict(saved['model_for_resume'], strict=True)
    _load_optimizer_state_preserving_runtime_options(opt, saved['optimizer'])
    torch.set_rng_state(meta['rng_cpu'])
    torch.cuda.set_rng_state(meta['rng_cuda'])
    assert int(model.num_updates) == step - 1
    for group in opt.param_groups:
        group['betas'] = tuple(config['betas'])
    del saved

    order = torch.randperm(len(data['train']), generator=torch.Generator().manual_seed(study.SEED + epoch + 10000))
    ids = order[:study.BATCH]
    model.train(True)
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    xt, t, target = study.draw(data['train'], ids, step, config)
    loss = study.backward(model, xt, t, target, profile['microbatch'])
    lr = config['lr'] * min(1.0, .1 + .9 * step / config['warmup_updates'])
    for group in opt.param_groups:
        group['lr'] = lr
        group['initial_lr'] = config['lr']
    grad = float(torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), float('inf'), error_if_nonfinite=True))
    before = next(model.model.parameters()).detach().clone()
    opt.step()
    model.update_ema()
    param = next(model.model.parameters()).detach()
    actual = dict(
        pde=pde, arm=arm, epoch=epoch, step=step, loss=loss, lr=lr, grad_norm=grad,
        first_parameter_relative_update=float((param - before).norm() / before.norm().clamp_min(1e-12)),
        input_ids_sha256=tensor_sha(data['train_ids'][ids]), target_sha256=tensor_sha(target),
        ema_updates=int(model.num_updates),
    )
    checks = {name: actual[name] == expected[name] for name in actual}
    result = dict(
        passed=all(checks.values()), checks=checks, actual=actual, reference=expected,
        checkpoint_sha256=expected_sha, reference_sha256=sha(reference),
        environment=environment, scientific_module=str(Path(study.__file__).resolve()),
        scientific_module_sha256=sha(Path(study.__file__)),
        gate_code_sha256=sha(Path(__file__)),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        optimizer_steps=sorted({int(state['step']) for state in opt.state.values()}),
        scope='One actual recorded update: exact loss, gradient norm, first-parameter relative update, input/noise hashes and EMA counter. This is not equality of all gradient/parameter tensors or all future updates.',
    )
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'result.json', result)
    assert result['passed'], result
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pde', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--scientific-checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.pde, args.arm, args.checkpoint, args.checkpoint_sha256,
        args.reference, args.scientific_checkout, args.output)

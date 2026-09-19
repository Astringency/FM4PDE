"""Export truthful native-resume metadata without changing experiment snapshots."""
import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess

import torch

from experiments.optimizer_diagnostics.study import sha, write
from models.ema import EMA
from models.model_configs import instantiate_model
from sampling.model_io import load_fm4pde_checkpoint_bundle
from training.load_and_save import load_model, OPTIMIZER_RUNTIME_OPTION_KEYS


def equal_tree(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


def verify(source, destination, pde):
    original = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    exported = torch.load(destination, map_location='cpu', weights_only=False, mmap=True)
    for key in ['model', 'model_ema', 'model_for_resume', 'optimizer', 'normalizer',
                'model_config', 'epoch', 'ema_timesteps']:
        equal_tree(original[key], exported[key])
    assert exported['scaler'] is None and exported['lr_schedule'] is None
    args = copy.deepcopy(exported['args'])
    args.resume = str(destination)
    model = EMA(instantiate_model(pde, use_ema=False, model_config=exported['model_config']),
                decay=.5, warmup=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=.1, betas=(.1,.2), fused=False, foreach=False)
    # None makes an accidental attempt to restore a stale scaler fail this check.
    restored = load_model(args, model, optimizer, None, None)
    equal_tree(model.state_dict(), original['model_for_resume'])
    equal_tree(restored['normalizer'], original['normalizer'])
    actual = optimizer.state_dict()
    equal_tree(actual['state'], original['optimizer']['state'])
    assert len(actual['param_groups']) == len(original['optimizer']['param_groups'])
    for group, expected in zip(actual['param_groups'], original['optimizer']['param_groups']):
        equal_tree({k:v for k,v in group.items() if k not in OPTIMIZER_RUNTIME_OPTION_KEYS},
                   {k:v for k,v in expected.items() if k not in OPTIMIZER_RUNTIME_OPTION_KEYS})
        assert group['fused'] is False and group['foreach'] is False
    assert model.decay == original['ema_decay'] and model.warmup == original['ema_warmup']
    assert args.start_epoch == original['epoch']+1
    updates = int(model.num_updates)
    del model, optimizer, restored, exported, actual
    gc.collect()
    for weight in ['raw', 'ema']:
        inference, normalizer, payload = load_fm4pde_checkpoint_bundle(
            str(destination), pde, 'cpu', wrap=False, prefer_ema=weight=='ema')
        assert payload['selected_inference_weight'] == weight
        equal_tree(inference.state_dict(), original['model' if weight=='raw' else 'model_ema'])
        equal_tree(payload['normalizer'], original['normalizer'])
        del inference, normalizer, payload
        gc.collect()
    return dict(native_load_model_passed=True, actual_raw_and_ema_loaders_passed=True,
                tensors_optimizer_normalizer_and_rng_unchanged=True,
                native_runtime_optimizer_options_retained=True, ema_updates=updates,
                next_native_epoch=args.start_epoch,
                scope='Verified native loading of real states; not a claim of identical future data/RNG trajectories')


def run(source, destination, pde):
    source, destination = source.resolve(), destination.resolve()
    assert source != destination and source.is_file()
    ready = json.loads(source.with_suffix('.ready.json').read_text())
    digest = sha(source)
    assert digest == ready['sha256'] and pde == ready['pde']
    receipt = destination.with_suffix('.export.json')
    if destination.exists():
        assert receipt.exists(), 'Export exists without successful verification; inspect it before retrying'
        record = json.loads(receipt.read_text())
        assert record['source_sha256'] == digest and record['export_sha256'] == sha(destination)
        return record
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    metadata = saved['ema_timesteps']; config = metadata['config']
    assert metadata['environment']['precision'] == 'float32'
    assert metadata['additional_updates'] >= config['warmup_updates']
    for group in saved['optimizer']['param_groups']:
        assert group['lr'] == config['lr'] and tuple(group['betas']) == tuple(config['betas'])
    microbatch = metadata['microbatch']
    assert config['effective_batch'] % microbatch == 0
    inherited = copy.deepcopy(saved.get('args', argparse.Namespace()))
    args = argparse.Namespace(**copy.deepcopy(inherited if isinstance(inherited, dict) else vars(inherited)))
    overrides = dict(dataset=pde, use_ema=True, resume_add_ema=False,
                     ema_decay=saved['ema_decay'], ema_warmup=saved['ema_warmup'],
                     sampling_dtype='float32', clip_grad=None, fused_adamw=True,
                     lr=config['lr'], min_lr=config['lr'], lr_scheduler='constant',
                     resolved_lr_scheduler='constant', warmup_epochs=0,
                     optimizer_betas=list(config['betas']), resume_optimizer_betas=None,
                     batch_size=microbatch, accum_iter=config['effective_batch']//microbatch,
                     timestep_sampling=config['time_sampling'], skewed_timesteps=False,
                     logit_time_mean=config['logit_mean'], logit_time_std=config['logit_std'])
    for key, value in overrides.items():
        setattr(args, key, value)
    saved.update(args=args, scaler=None, lr_schedule=None, lr_scheduler='constant',
                 resolved_lr_scheduler='constant', min_lr=config['lr'], warmup_epochs=0)
    saved['native_resume_export'] = dict(source=str(source), source_sha256=digest,
        inherited_args=inherited, normalized_args=overrides,
        explanation='Study used direct FP32 AdamW without GradScaler; inherited pretraining scaler is inapplicable. Both inference weights and complete training states are unchanged.',
        trajectory='Use study.run for exact study continuation. Native training has its own data order, RNG and validation stream; stratification is per native microbatch.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    assert not temporary.exists()
    torch.save(saved, temporary)
    del saved
    verification = verify(source, temporary, pde)
    temporary.replace(destination)
    record = dict(status='verified', source=str(source), source_sha256=digest,
                  path=str(destination), export_sha256=sha(destination), pde=pde,
                  normalized_args=overrides, verification=verification,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],
                             cwd=Path(__file__).resolve().parents[2],text=True).strip())
    write(receipt, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pde', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(run(args.source, args.output, args.pde)))

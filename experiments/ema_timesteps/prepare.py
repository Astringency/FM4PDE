"""Prepare complete original training splits, preserving checkpoint normalization."""
import argparse
import gc
import json
from pathlib import Path
import subprocess

import torch

from data.load import PDEloader
from data.specs import get_pde_spec
from data.training_manifest import load_training_file_manifest
from data.transform import PDEStandardizer
from experiments.optimizer_diagnostics.study import sha, tensor_sha, write, ROOT
from train import _train_val_split_indices


def prepare(root, previous, pde, data_root):
    out = root / 'inputs' / pde
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'prepared.json').exists():
        record = json.loads((out / 'prepared.json').read_text())
        assert sha(out / 'full_data.pt') == record['data_sha256']
        print('VERIFIED_EXISTING', pde, flush=True)
        return
    original = previous / 'inputs' / pde
    prior = json.loads((original / 'prepared.json').read_text())
    assert sha(original / 'source.pth') == prior['checkpoint_sha256']
    source = torch.load(original / 'source.pth', map_location='cpu', weights_only=False, mmap=True)
    assert not source.get('has_ema') and not source.get('use_ema')
    args = source['args'] if isinstance(source['args'], dict) else vars(source['args'])
    files = load_training_file_manifest(ROOT / 'configs/training_data.yaml',
        data_root=data_root, pde_names=[pde])[pde]
    print('LOADING_FULL_POOL', pde, flush=True)
    raw, _ = PDEloader(pde).load_data_files(files)
    assert len(raw) == 50000
    train_ids, val_ids = _train_val_split_indices(50000,
        seed=int(args.get('seed', 0)) + get_pde_spec(pde).label_id * 1009, val_ratio=.1)
    assert len(train_ids) == 45000 and len(val_ids) == 5000
    perm = torch.randperm(len(val_ids), generator=torch.Generator().manual_seed(20260921))
    dev_ids, confirm_ids = val_ids[perm[:512]], val_ids[perm[512:1024]]
    assert not set(train_ids.tolist()) & set(dev_ids.tolist() + confirm_ids.tolist())
    normalizer = PDEStandardizer.from_state_dict(source['normalizer'])
    tensors = dict(train=normalizer.transform(raw[train_ids]).contiguous(),
        development=normalizer.transform(raw[dev_ids]).contiguous(),
        confirmation=normalizer.transform(raw[confirm_ids]).contiguous(),
        train_ids=train_ids, development_ids=dev_ids, confirmation_ids=confirm_ids)
    stats = dict(normalized_mean=tensors['train'].mean((0, 2, 3)).tolist(),
        normalized_std=tensors['train'].std((0, 2, 3)).tolist(),
        shape=list(tensors['train'].shape))
    del raw
    gc.collect()
    temporary = out / 'full_data.tmp'
    torch.save(tensors, temporary)
    temporary.replace(out / 'full_data.pt')
    # Reuse immutable canonical inputs without copying full optimizer states.
    for name in ['source.pth', 'source_prepared.json']:
        destination = out / name
        target = original / ('prepared.json' if name == 'source_prepared.json' else name)
        if not destination.exists():
            destination.symlink_to(target)
    write(out / 'prepared.json', dict(pde=pde, source_checkpoint_sha256=prior['checkpoint_sha256'],
        source_original_checkpoint=prior['source_checkpoint'], source_optimizer_steps=prior['optimizer_steps'],
        data_sha256=sha(out / 'full_data.pt'), tensor_sha256={k:tensor_sha(v) for k,v in tensors.items()},
        counts={k:len(tensors[k]) for k in ['train', 'development', 'confirmation']},
        files=[dict(path=str(f), size=f.stat().st_size, sha256=sha(f)) for f in files],
        normalizer_retained=True, original_split_retained=True, stats=stats,
        historical_validation_caveat='Earlier June pretraining may have seen the validation pool; September NS has a documented disjoint split.',
        selection='512 development and 512 confirmation inputs chosen before new model evaluation from all 5000 original held-out IDs',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    print('FULL_POOL_READY', pde, stats, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--pdes', nargs='+', required=True)
    p.add_argument('--data-root', default='/large_storage/zhangxf/PDEdata')
    a = p.parse_args()
    torch.set_num_threads(4)
    for pde in a.pdes:
        prepare(a.root, a.previous, pde, a.data_root)

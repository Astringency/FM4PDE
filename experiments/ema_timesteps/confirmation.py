"""Evaluate a frozen list of checkpoints on the reserved confirmation inputs."""
import argparse
import gc
import json
from pathlib import Path

import torch

from experiments.ema_timesteps.evaluate import fixed_metrics
from experiments.optimizer_diagnostics.study import paired, sha, write
from sampling.model_io import load_fm4pde_checkpoint_bundle


def run(root, pde, selection):
    # This file is written only after inspecting development and physical sampling.
    decisions = json.loads(selection.read_text())
    assert decisions['choices_frozen_before_confirmation'] is True
    entries = decisions['pdes'][pde]['checkpoints']
    assert entries and len({e['name'] for e in entries}) == len(entries)
    folder = root/'confirmation'/pde
    folder.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha(selection)
    if (folder/'selection_identity.json').exists():
        assert json.loads((folder/'selection_identity.json').read_text())['sha256'] == manifest_sha
    else:
        write(folder/'selection_identity.json', dict(sha256=manifest_sha, selection=decisions))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    profile = json.loads((root/'profiles'/pde/'complete.json').read_text())
    assert torch.cuda.get_device_name() == profile['environment']['gpu']
    microbatch = profile['microbatch']
    data = torch.load(root/'inputs'/pde/'full_data.pt', map_location='cpu', mmap=True, weights_only=False)
    original = root/'inputs'/pde/'source.pth'
    jobs = [dict(name='original', path=str(original), weight='raw', sha256=sha(original))] + entries
    metrics = {}
    for entry in jobs:
        path = Path(entry['path'])
        if not path.is_absolute():
            path = root/path
        assert sha(path) == entry['sha256']
        destination = folder/(entry['name']+'.json')
        identity = dict(checkpoint_sha256=entry['sha256'], weight=entry['weight'], selection_sha256=manifest_sha)
        if destination.exists():
            record = json.loads(destination.read_text())
            assert record['identity'] == identity
        else:
            model, normalizer, payload = load_fm4pde_checkpoint_bundle(
                str(path), pde, 'cuda:0', wrap=False, prefer_ema=entry['weight']=='ema')
            assert payload['selected_inference_weight'] == entry['weight']
            result = fixed_metrics(model, data['confirmation'], microbatch, seed=20260925)
            record = dict(identity=identity, metrics=result,
                confirmation_ids=data['confirmation_ids'].tolist(), microbatch=microbatch)
            write(destination, record)
            del model, normalizer, payload
            gc.collect()
            torch.cuda.empty_cache()
        metrics[entry['name']] = record['metrics']
        print('CONFIRMATION', pde, entry['name'], record['metrics']['mean'], flush=True)
    baseline = metrics['original']
    result = dict(pde=pde, selection_sha256=manifest_sha, comparisons={})
    for name, values in metrics.items():
        if name == 'original':
            continue
        comparison = paired(values, baseline)
        comparison['time_bins'] = [paired(dict(per_input=c,mean=sum(c)/len(c)),
                                         dict(per_input=b,mean=sum(b)/len(b)))
                                  for c,b in zip(values['per_time_input'], baseline['per_time_input'])]
        comparison['time_bin_ci_scope'] = 'Exploratory marginal intervals; no multiplicity correction'
        result['comparisons'][name] = comparison
    write(folder/'summary.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pde', required=True)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.pde, args.selection)

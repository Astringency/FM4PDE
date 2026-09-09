#!/usr/bin/env python3
"""Read-only comparison of historical main checkpoints and ablation slim copies."""
import argparse
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bindings', type=Path, required=True)
    ap.add_argument('--ablation-inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from freeze_baseline_inputs import identical
    torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = json.loads(args.bindings.read_text())
    for row in rows:
        source = Path(row['path'])
        slim = args.ablation_inputs / row['pde'] / 'weights.pth'
        assert sha(source) == row['sha256']
        assert sha(slim) == row['ablation_archive_expected_sha256']
        a = torch.load(source, weights_only=False, map_location='cpu', mmap=True)
        b = torch.load(slim, weights_only=False, map_location='cpu', mmap=True)
        selected = a.get('model_ema')
        selection = 'ema'
        if not isinstance(selected, dict) or not selected:
            selected = a['model']
            selection = 'raw'
        state = b['model']
        assert set(selected) == set(state)
        hashes = {}
        for name in sorted(selected):
            x, y = selected[name], state[name]
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), (row['pde'], name)
            hashes[name] = hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()
        retained = {}
        for key in ['normalizer', 'normalization', 'data_metadata', 'model_config', 'model_config_metadata']:
            if key in a or key in b:
                retained[key] = identical(a.get(key), b.get(key))
                assert retained[key], (row['pde'], key)
        row.update(slim_path=str(slim), inference_selection=selection,
                   selected_state_dict_exact=True, selected_state_tensor_count=len(hashes),
                   selected_state_tensor_hashes=hashes, inference_metadata_equal=retained)
        print('PASS', row['pde'], selection, len(hashes), flush=True)
        del a, b, selected, state
    result = {'status': 'pass', 'source_files_differ_from_slim_files': True,
              'interpretation': 'Serialized files differ; the selected inference parameters and retained model/normalization metadata agree exactly.',
              'bindings_sha256': sha(args.bindings), 'rows': rows}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

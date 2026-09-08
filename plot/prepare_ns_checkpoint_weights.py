"""Copy inference-complete NS weights without optimizer buffers; verify equality."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_ns_checkpoint_comparison import inference_signature
from run_ns_loss_study import sha, write


def main():
    import torch
    from models.legacy_checkpoint import read_checkpoint
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {}
    for label, spec in json.loads(args.checkpoints.read_text()).items():
        if label == 'current':
            continue  # The identical prior inference copy is already on server216.
        path = args.output / f'{label}.pth'
        assert not path.exists()
        payload = read_checkpoint(spec['path'], 'nsnonbounded')
        signature = inference_signature(payload)
        slim = {k:v for k,v in payload.items() if k not in {'optimizer', 'model_for_resume', 'scaler'}}
        slim['slimmed_from'] = dict(path=spec['path'], sha256=sha(spec['path']))
        torch.save(slim, path)
        assert inference_signature(read_checkpoint(path, 'nsnonbounded')) == signature
        result[label] = dict(path=str(path), original=slim['slimmed_from'],
                             sha256=sha(path), bytes=path.stat().st_size, inference_signature=signature)
        print(label, result[label], flush=True)
    write(args.output / 'conversion_manifest.json', result)


if __name__ == '__main__':
    main()

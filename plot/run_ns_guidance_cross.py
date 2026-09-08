"""Complete the fixed NS checkpoint x guidance comparison without retuning."""
from pathlib import Path
import argparse
import copy
import json
import os
import socket
import subprocess

import run_ns_checkpoint_comparison as core
from run_ns_loss_study import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'worker'])
    p.add_argument('--base-results', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights-map', type=Path)
    p.add_argument('--shard', type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument('--pilot-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'freeze':
        assert not (args.output / 'protocol.json').exists()
        parent = args.base_results / 'protocol.json'
        base = json.loads(parent.read_text())
        assert base['workers'] == 4 and len(base['evaluation_ids']) == 32
        assert base['seeds'] == [0, 1, 2]
        for name in base['code_sha256']:
            assert sha(core.ROOT / name) == base['code_sha256'][name], name
        assert sha(args.inputs / 'source.json') == base['source_sha256']
        assert sha(args.inputs / 'fields_masks.npz') == base['fields_sha256']
        protocol = copy.deepcopy(base)
        protocol.pop('upstream_protocol_sha256', None)
        protocol.pop('upstream_reuse', None)
        protocol['checkpoints'] = {k: protocol['checkpoints'][k] for k in ['current', 'v260904']}
        protocol.update(
            version='guidance_cross_v1', parent_protocol_sha256=sha(parent),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=core.ROOT, text=True).strip(),
            variants=[dict(name=f'{label}_legacy100', checkpoint=label, configuration='legacy', steps=100)
                      for label in ['current', 'v260904']], formal_calls=576,
            selection='Both requested checkpoints, all 32 previously frozen inputs, all three seeds and tasks. No tuning, selection, changed normalization or additional training.',
            pairing='Input index modulo four retains exactly the same host and physical GPU UUID as the parent comparison.',
            scope='Complete the missing two cells of the 3-checkpoint x 2-guidance design at 100 steps. This compares the entire inherited guidance configuration, not individual guidance components. Legacy NS guidance is the historical spatial-derivative surrogate, not the full NS residual. Supplementary 32-input Smooth evaluation; not a replacement for 1000-input main tables.',
        )
        protocol['code_sha256']['plot/run_ns_guidance_cross.py'] = sha(Path(__file__))
        protocol['worker_reference_environments'] = {}
        for shard in range(4):
            path = args.base_results / f'environment_{shard}.json'
            env = json.loads(path.read_text())
            assert env['protocol_sha256'] == sha(parent)
            protocol['worker_reference_environments'][str(shard)] = dict(
                source_sha256=sha(path), host=env['host'], uuid=env['uuid'],
                torch=env['torch'], cuda=env['cuda'], gpu=env['gpu'])
        write(args.output / 'protocol.json', protocol)
        print('FROZEN', protocol['formal_calls'], sha(args.output / 'protocol.json'), flush=True)
    else:
        import torch
        protocol = json.loads((args.output / 'protocol.json').read_text())
        assert sha(args.base_results / 'protocol.json') == protocol['parent_protocol_sha256']
        reference = protocol['worker_reference_environments'][str(args.shard)]
        visible = os.environ['CUDA_VISIBLE_DEVICES']
        assert visible.isdigit(), 'Assign one physical GPU to each worker.'
        actual_uuid = subprocess.check_output([
            'nvidia-smi', f'--id={visible}', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        assert (socket.gethostname(), actual_uuid, torch.__version__, torch.version.cuda) == (
            reference['host'], reference['uuid'], reference['torch'], reference['cuda'])
        core.worker(args)


if __name__ == '__main__':
    main()

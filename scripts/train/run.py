"""Launch the eleven appendix training recipes (two GPUs, effective batch 64)."""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    recipe = yaml.safe_load((ROOT / 'configs/training.yaml').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdes', nargs='+', choices=list(recipe['pdes']),
                        default=os.environ.get('PDE_LIST', os.environ.get('PDE', '')).split() or list(recipe['pdes']))
    parser.add_argument('--nproc', type=int, default=int(os.environ.get('NPROC_PER_NODE', 2)))
    parser.add_argument('--plan-only', action='store_true', default=os.environ.get('DRY_RUN', '').lower() in {'1', 'true'})
    parser.add_argument('--output', default=os.environ.get('OUTPUT_DIR', 'outputs/pretrained/formal'))
    args, extra = parser.parse_known_args()
    if args.nproc < 1:
        parser.error('--nproc must be positive')
    for pde in args.pdes:
        config = dict(recipe['pdes'][pde])
        denominator = config['batch_size'] * args.nproc
        if 64 % denominator:
            parser.error(f'{pde}: batch_size × nproc must divide 64')
        config['accum_iter'] = 64 // denominator
        config.update(dataset=pde, data_path=os.environ.get('DATA_ROOT', str(ROOT/'datasets')).rstrip('/')+'/',
                      train_data_config='configs/training_data.yaml',
                      output_dir=str(Path(args.output)/pde)+'/', device='cuda',
                      timestep_sampling='uniform')
        command = [sys.executable]
        if args.nproc > 1:
            command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={args.nproc}']
        command += ['train.py']
        for key, value in config.items():
            command += ['--'+key] + [str(x) for x in (value if isinstance(value, list) else [value])]
        command += extra
        print(shlex.join(command), flush=True)
        if not args.plan_only:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

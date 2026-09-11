"""Check a transferred main checkpoint with the execution host's CPU environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import time

import torch

from models.legacy_checkpoint import read_checkpoint
from models.model_configs import instantiate_model
from scripts.train.resume_study import file_sha, optimizer_steps, restore, write


def main(args):
    assert torch.cuda.device_count() == 0, 'Run this read-only check with CUDA_VISIBLE_DEVICES empty'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    binding = next(row for row in json.loads(args.binding.read_text())['models'] if row['pde'] == args.pde)
    digest = file_sha(args.checkpoint)
    assert digest == binding['resume_sha256']
    source = read_checkpoint(args.checkpoint, args.pde)
    assert int(source['epoch']) == binding['epoch'] == 299
    model = instantiate_model(args.pde, use_ema=False, model_config=source['model_config']).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, fused=False)
    restore(model, optimizer, source, 3e-6)
    state = model.state_dict()
    assert state.keys() == source['model_for_resume'].keys()
    assert all(torch.equal(value, source['model_for_resume'][key]) for key, value in state.items())
    assert all(torch.isfinite(value).all() for value in state.values() if value.is_floating_point())
    steps = optimizer_steps(optimizer)
    assert len(steps) == binding['optimizer_state_count'] and sorted(set(steps)) == binding['optimizer_steps']
    saved_ids = [i for group in source['optimizer']['param_groups'] for i in group['params']]
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    assert len(saved_ids) == len(parameters)
    for key, parameter in zip(saved_ids, parameters):
        expected, actual = source['optimizer']['state'].get(key, {}), optimizer.state.get(parameter, {})
        assert expected.keys() == actual.keys()
        for name, value in expected.items():
            if torch.is_tensor(value):
                assert torch.equal(value, actual[name]) and torch.isfinite(actual[name]).all()
            else:
                assert value == actual[name]
    result = dict(status='verified', pde=args.pde, host=socket.gethostname(),
        checkpoint=str(args.checkpoint), source_sha256=digest, source_epoch=int(source['epoch']),
        model_tensors=len(state), parameter_elements=sum(p.numel() for p in model.parameters()),
        optimizer_states=len(steps), optimizer_steps=sorted(set(steps)),
        cpu_model_exact=True, cpu_adam_state_exact=True, finite=True, torch_version=torch.__version__,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        checked_unix=time.time(), seconds=time.monotonic()-started,
        scope='CPU loading and exact state restoration only; no optimizer update, GPU probe, or sampling accuracy claim')
    write(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pde', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

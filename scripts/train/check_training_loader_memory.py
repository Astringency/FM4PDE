"""Read one real training shard on CPU and record exact tensors and loading cost.

Run in separate processes with NUMPY_MADVISE_HUGEPAGE set before Python starts.
No model, optimizer, GPU workload, or training input file is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import time

import numpy as np
import torch

from data.load import PDEloader


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    assert value.device.type == 'cpu' and value.is_contiguous()
    view = memoryview(value.numpy()).cast('B')
    digest = hashlib.sha256()
    for start in range(0, len(view), 8 << 20):
        digest.update(view[start:start + (8 << 20)])
    return digest.hexdigest()


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and torch.cuda.device_count() == 0
    assert args.output.is_absolute() and '/outputs/pretrained/' in str(args.output)
    assert not args.output.exists(), 'Keep each independent observation'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    assert file_sha(args.file) == args.expected_sha256
    before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.monotonic()
    data, labels = PDEloader(args.pde).load_data_files([args.file])
    seconds = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    assert len(data) == len(labels) == args.expected_samples
    assert data.dtype == torch.float32 and torch.isfinite(data).all()
    row = dict(status='complete', pde=args.pde, host=socket.gethostname(),
               file=str(args.file), source_sha256=args.expected_sha256,
               shape=list(data.shape), dtype=str(data.dtype),
               tensor_sha256=tensor_sha(data), labels_sha256=tensor_sha(labels),
               load_seconds=seconds, load_user_seconds=after.ru_utime - before.ru_utime,
               load_system_seconds=after.ru_stime - before.ru_stime,
               peak_rss_kib=after.ru_maxrss, numpy_version=np.__version__,
               torch_version=torch.__version__, kernel=os.uname().release,
               numpy_madvise_hugepage=os.environ.get('NUMPY_MADVISE_HUGEPAGE'),
               checked_unix=time.time(),
               scope='Read-only full-shard CPU loading and exact tensor fingerprint; no training or sampling accuracy claim')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pde', required=True, choices=['poisson', 'darcy', 'burger'])
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-samples', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

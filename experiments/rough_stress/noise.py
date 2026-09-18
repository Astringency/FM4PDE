"""Replay the exact A100 canonical Gaussian stream on other GPU architectures.

CUDA Philox's launch geometry changes its full-pool layout and counter advance
with GPU architecture. A seed alone does not pair stochastic trajectories
between A100 and 4090. This file archives the original A100 draws, without
changing their law or any model, guidance, or sampler operation.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.rough_stress.run import sha256, write_json


def make_bank(root, device="cuda:0"):
    root = Path(root)
    if (root/"manifest.json").exists():
        raise ValueError("Noise bank already exists")
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    if "A100" not in torch.cuda.get_device_name():
        raise ValueError("Generate the reference stream on the original A100 architecture")
    root.mkdir(parents=True, exist_ok=True)
    files, chunks = [], []
    for lo in range(0, 1000, 16):
        hi = min(lo+16, 1000)
        filename = f"noise_{lo:04d}_{hi:04d}.npy"
        chunks.append(dict(start=lo, stop=hi, file=filename))
        files.append(np.lib.format.open_memmap(root/filename, mode="w+", dtype=np.float32,
            shape=(hi-lo, 101, 2, 128, 128)))
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    pools = []
    for step in range(101):
        if step == 0:
            noise = torch.randn((1000, 2, 128, 128), device=device, dtype=torch.float32)
        else:
            noise = torch.randn_like(torch.empty((1000, 2, 128, 128), device=device, dtype=torch.float32))
        array = noise.cpu().numpy()
        pools.append(hashlib.sha256(array.tobytes()).hexdigest())
        for chunk, dest in zip(chunks, files):
            dest[:, step] = array[chunk["start"]:chunk["stop"]]
        if step % 10 == 0:
            print(f"NOISE DRAW {step}/100", flush=True)
    for dest, chunk in zip(files, chunks):
        dest.flush()
        chunk["sha256"] = sha256(root/chunk["file"])
    write_json(root/"manifest.json", dict(seed=0, count=1000, steps=100, shape=[2,128,128],
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        multiprocessor_count=torch.cuda.get_device_properties(device).multi_processor_count,
        protocol="A100_native_canonical_pool_seed0_initial_plus_100_bridge_draws",
        pool_raw_sha256=pools, chunks=chunks))
    print("NOISE BANK COMPLETE", root, flush=True)


class NoiseBank:
    def __init__(self, root, cache_root=None):
        self.root = Path(root)
        self.cache_root = Path(cache_root) if cache_root else None
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        self.manifest_sha256 = sha256(self.root/"manifest.json")
        if (self.manifest["seed"],self.manifest["count"],self.manifest["steps"]) != (0,1000,100):
            raise ValueError("Unexpected canonical noise protocol")

    def load(self, indices):
        selected = {}
        for chunk in self.manifest["chunks"]:
            wanted = [i for i in indices if chunk["start"] <= i < chunk["stop"]]
            if not wanted:
                continue
            path = self.root/chunk["file"]
            # A cache may still be copying in parallel. Only use a complete,
            # digest-verified file; otherwise read the persistent reference.
            if self.cache_root is not None:
                cached = self.cache_root/chunk["file"]
                if cached.exists() and sha256(cached) == chunk["sha256"]:
                    path = cached
            if sha256(path) != chunk["sha256"]:
                raise ValueError(f"Noise chunk is corrupt: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            for i in wanted:
                selected[i] = np.array(array[i-chunk["start"]], copy=True)
        if set(selected) != set(indices):
            raise ValueError("Missing noise rows")
        return torch.from_numpy(np.stack([selected[i] for i in indices]))

    @contextmanager
    def replay(self, indices):
        # Explicit experiment-local hooks leave the stock sampler untouched.
        # Restored even on errors; each worker is a separate Python process.
        import sampling.runner as runner
        import sampling.sampler_wrappers as sampler
        data = self.load(indices)
        original_initial = runner._sample_initial_noise
        original_bridge = sampler._stochastic_bridge_noise_like
        used = [0]

        def initial(config, ground_truth, device):
            if config.sample_seed != 0 or config.initial_noise_source_batch_size != 1000:
                raise ValueError("Noise replay config differs from archived stream")
            return data[:, 0].to(device=device, dtype=ground_truth.pair.dtype)

        def bridge(x_cur, *, device, source_batch_size=None, source_indices=None):
            if source_batch_size != 1000 or list(source_indices) != list(indices):
                raise ValueError("Noise replay indices changed")
            used[0] += 1
            result = data[:, used[0]].to(device=device, dtype=x_cur.dtype)
            if result.shape != x_cur.shape:
                raise ValueError("Noise replay shape changed")
            return result

        runner._sample_initial_noise = initial
        sampler._stochastic_bridge_noise_like = bridge
        try:
            yield
            if used[0] != 100:
                raise ValueError(f"Expected 100 bridge draws, used {used[0]}")
        finally:
            runner._sample_initial_noise = original_initial
            sampler._stochastic_bridge_noise_like = original_bridge


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    make_bank(args.output_root, args.device)

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
DATAGEN_DIR = THIS_DIR.parent
if str(DATAGEN_DIR) not in sys.path:
    sys.path.insert(0, str(DATAGEN_DIR))

from generation_profiles import (
    DATASET_TYPE_ALIASES,
    SEED_OFFSETS,
    TEMPORAL_GRF_PROFILES,
    canonical_dataset_type,
)
from no_bound_ns.ns_2d import navier_stokes_2d
from no_bound_ns.random_fields import GaussianRF


DEFAULT_OUT_DIR = Path("/large_storage/zhangxf/PDEdata/nsnonbounded")
GENERATION_PROFILES = {
    name: {"alpha": values[0], "tau": values[1], "seed_offset": SEED_OFFSETS[name]}
    for name, values in TEMPORAL_GRF_PROFILES.items()
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FM4PDE non-bounded Navier-Stokes HDF5/MAT data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--dataset-type",
        choices=sorted(DATASET_TYPE_ALIASES),
        default=None,
        help="Distribution profile: train, id, smooth, or rough.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default=None,
        help="Legacy alias: train maps to train and test maps to id.",
    )
    parser.add_argument("--total-samples", type=int, default=1000)
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=None,
        help="Samples per train shard. Defaults to total-samples for test and 10000 for train.",
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--record-steps", type=int, default=10)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--viscosity", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-offset", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_dataset_type(args: argparse.Namespace) -> str:
    requested_type = getattr(args, "dataset_type", None)
    legacy_split = getattr(args, "split", None)
    if requested_type is None and legacy_split is None:
        return "id"
    if requested_type is None:
        return canonical_dataset_type(legacy_split)
    resolved = canonical_dataset_type(requested_type)
    if legacy_split is not None and canonical_dataset_type(legacy_split) != resolved:
        raise ValueError("--dataset-type and --split select different generation profiles")
    return resolved


def _sample_counts(total_samples: int, samples_per_file: int) -> list[int]:
    if total_samples < 1:
        raise ValueError("--total-samples must be positive")
    if samples_per_file < 1:
        raise ValueError("--samples-per-file must be positive")
    counts = []
    remaining = int(total_samples)
    while remaining > 0:
        count = min(int(samples_per_file), remaining)
        counts.append(count)
        remaining -= count
    return counts


def _output_path(
    out_dir: Path,
    *,
    split: str | None = None,
    dataset_type: str | None = None,
    sample_count: int,
    resolution: int,
    record_steps: int,
    file_index: int,
) -> Path:
    resolved_type = canonical_dataset_type(dataset_type or split or "id")
    if resolved_type == "train":
        stem = (
            f"nsnonbounded_{sample_count}-{resolution}-{resolution}-{record_steps}_"
            f"{file_index + 1}_new"
        )
    else:
        stem = f"nsnonbounded_test_{sample_count}-{resolution}-{resolution}-{record_steps}_{resolved_type}"
    return out_dir / f"{stem}.mat"


def generate_file(
    *,
    path: Path,
    dataset_type: str,
    sample_count: int,
    sample_start: int,
    resolution: int,
    device: torch.device,
    record_steps: int,
    T: float,
    dt: float,
    viscosity: float,
    seed_offset: int,
    overwrite: bool,
) -> bool:
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            print(f"Skipping existing nsnonbounded file: {path}", flush=True)
            return False

    dataset_type = canonical_dataset_type(dataset_type)
    profile = GENERATION_PROFILES[dataset_type]
    print(f"nsnonbounded {dataset_type} generating {sample_count} samples on {device}", flush=True)
    torch.manual_seed(int(seed_offset + sample_start))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed_offset + sample_start))

    s = int(resolution)
    alpha = float(profile["alpha"])
    tau = float(profile["tau"])
    grf = GaussianRF(2, s, alpha=alpha, tau=tau, device=device)
    grid = torch.linspace(0, 1, s + 1, device=device)[:-1]
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    forcing = 0.1 * (torch.sin(2 * math.pi * (x + y)) + torch.cos(2 * math.pi * (x + y)))

    w0 = grf.sample(int(sample_count))
    sol_vx0, sol_vy0, sol_w, sol_vx, sol_vy, sol_t = navier_stokes_2d(
        w0,
        forcing,
        float(viscosity),
        float(T),
        float(dt),
        int(record_steps),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_seed = seed_offset + np.arange(sample_start, sample_start + sample_count, dtype=np.int64)
    with h5py.File(str(path), "w") as h5:
        h5.attrs["pde_name"] = "nsnonbounded"
        h5.attrs["split"] = "train" if dataset_type == "train" else "test"
        h5.attrs["dataset_type"] = dataset_type
        h5.attrs["n_samples"] = int(sample_count)
        h5.attrs["sample_start"] = int(sample_start)
        h5.attrs["resolution"] = f"{s}x{s}"
        h5.attrs["record_steps"] = int(record_steps)
        h5.attrs["T"] = float(T)
        h5.attrs["dt"] = float(dt)
        h5.attrs["viscosity"] = float(viscosity)
        h5.attrs["grf_alpha"] = float(alpha)
        h5.attrs["grf_tau"] = float(tau)
        h5.create_dataset("sample_seed", data=sample_seed, dtype="int64")
        h5.create_dataset("w0", data=w0.real.detach().cpu().numpy().astype("float32"), dtype="float32")
        h5.create_dataset("w", data=sol_w.detach().cpu().numpy().astype("float32"), dtype="float32")
        h5.create_dataset("vx0", data=sol_vx0.detach().cpu().numpy().astype("float32"), dtype="float32")
        h5.create_dataset("vy0", data=sol_vy0.detach().cpu().numpy().astype("float32"), dtype="float32")
        h5.create_dataset("vx", data=sol_vx.detach().cpu().numpy().astype("float32"), dtype="float32")
        h5.create_dataset("vy", data=sol_vy.detach().cpu().numpy().astype("float32"), dtype="float32")
        h5.create_dataset("t", data=sol_t.detach().cpu().numpy().astype("float32"), dtype="float32")
    return True


def generate_dataset(args: argparse.Namespace) -> list[Path]:
    dataset_type = resolve_dataset_type(args)
    samples_per_file = args.samples_per_file
    if samples_per_file is None:
        samples_per_file = args.total_samples if dataset_type != "train" else 10000
    if dataset_type != "train" and int(samples_per_file) != int(args.total_samples):
        raise ValueError(
            "each id, smooth, or rough test split writes one file; "
            "omit --samples-per-file or set it equal to --total-samples"
        )
    counts = _sample_counts(args.total_samples, samples_per_file)
    device = torch.device(args.device)
    seed_offset = args.seed_offset
    if seed_offset is None:
        seed_offset = int(GENERATION_PROFILES[dataset_type]["seed_offset"])

    paths = []
    sample_start = 0
    for file_index, sample_count in enumerate(counts):
        path = _output_path(
            Path(args.out_dir),
            dataset_type=dataset_type,
            sample_count=sample_count,
            resolution=int(args.resolution),
            record_steps=int(args.record_steps),
            file_index=file_index,
        )
        generate_file(
            path=path,
            dataset_type=dataset_type,
            sample_count=sample_count,
            sample_start=sample_start,
            resolution=int(args.resolution),
            device=device,
            record_steps=int(args.record_steps),
            T=float(args.T),
            dt=float(args.dt),
            viscosity=float(args.viscosity),
            seed_offset=int(seed_offset),
            overwrite=bool(args.overwrite),
        )
        paths.append(path)
        sample_start += sample_count
    return paths


if __name__ == "__main__":
    generated = generate_dataset(parse_args())
    for item in generated:
        print(item)

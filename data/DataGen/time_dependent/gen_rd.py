from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from pdebench.data_gen.src.sim_diff_react import Simulator as DiffReactSimulator
from tqdm import tqdm

DATAGEN_DIR = Path(__file__).resolve().parent.parent
if str(DATAGEN_DIR) not in sys.path:
    sys.path.insert(0, str(DATAGEN_DIR))

from generation_profiles import (
    REACTION_DIFFUSION_GRF_PROFILES,
    canonical_dataset_type,
    seed_offset as profile_seed_offset,
)


DEFAULT_SAVE_PATH = Path("/large_storage/zhangxf/PDEdata/reaction_diffusion/")
DEFAULT_SPLIT_SEED_OFFSETS = {"train": 0, "test": 10_000_000}


def canonical_init_mode(init_mode: str) -> str:
    mode = str(init_mode).lower()
    aliases = {
        "iid": "iid",
        "standard_normal": "iid",
        "grf": "grf",
        "gaussian_random_field": "grf",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unknown init_mode={init_mode!r}; expected iid, standard_normal, grf, or gaussian_random_field"
        )
    return aliases[mode]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FM4PDE 2D reaction-diffusion HDF5 data.")
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--total-samples", type=int, default=1000)
    parser.add_argument("--samples-per-file", "--batch-size", dest="samples_per_file", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--n-save-steps", type=int, default=10)
    parser.add_argument(
        "--init-mode",
        choices=["iid", "standard_normal", "grf", "gaussian_random_field"],
        default="grf",
    )
    parser.add_argument("--init-mean", type=float, default=0.0)
    parser.add_argument("--init-std", type=float, default=1.0)
    parser.add_argument("--grf-length-scale", type=float, default=None)
    parser.add_argument("--grf-spectral-power", type=float, default=None)
    parser.add_argument("--no-grf-normalize", action="store_true")
    parser.add_argument("--Du", type=float, default=2e-3)
    parser.add_argument("--Dv", type=float, default=4e-3)
    parser.add_argument("--k", type=float, default=3e-3)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=None,
        help="First RNG seed. Defaults to 0 for train and 10,000,000 for test.",
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--dataset-type",
        choices=["train", "id", "smooth", "rough", "test", "easytest", "hardtest"],
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def output_file_name(
    *,
    split: str,
    init_mode: str,
    sample_count: int,
    resolution: int,
    T: float,
    n_save_steps: int,
    dataset_type: str,
    shard_id: int | None = None,
) -> str:
    if split == "train":
        stem = (
            f"reaction_diffusion_{init_mode}_{sample_count}-{resolution}-{resolution}-"
            f"T{T:g}-steps{n_save_steps}_{shard_id or 1}"
        )
    else:
        stem = (
            f"reaction_diffusion_test_{init_mode}_{sample_count}-{resolution}-{resolution}-"
            f"T{T:g}-steps{n_save_steps}_{canonical_dataset_type(dataset_type)}"
        )
    return f"{stem}.h5"


def resolve_seed_offset(split: str, seed_offset: int | None, dataset_type: str | None = None) -> int:
    if split not in DEFAULT_SPLIT_SEED_OFFSETS:
        raise ValueError(f"Unknown split={split!r}; expected one of {sorted(DEFAULT_SPLIT_SEED_OFFSETS)}")
    if seed_offset is not None:
        return int(seed_offset)
    if split == "train":
        return DEFAULT_SPLIT_SEED_OFFSETS[split]
    return profile_seed_offset(dataset_type or "id")


def apply_dataset_profile(args: argparse.Namespace) -> str:
    default_type = "train" if args.split == "train" else "id"
    dataset_type = canonical_dataset_type(getattr(args, "dataset_type", None) or default_type)
    if args.split == "train" and dataset_type != "train":
        raise ValueError("training files require dataset_type='train'")
    if args.split == "test" and dataset_type == "train":
        raise ValueError("test files cannot use dataset_type='train'")
    length_scale, spectral_power = REACTION_DIFFUSION_GRF_PROFILES[dataset_type]
    if getattr(args, "grf_length_scale", None) is None:
        args.grf_length_scale = length_scale
    if getattr(args, "grf_spectral_power", None) is None:
        args.grf_spectral_power = spectral_power
    args.dataset_type = dataset_type
    return dataset_type


def metadata_from_args(args: argparse.Namespace, *, tdim: int, init_mode: str) -> dict[str, object]:
    dataset_type = apply_dataset_profile(args)
    return {
        "pde_name": "reaction_diffusion",
        "split": args.split,
        "dataset_type": dataset_type,
        "total_samples": int(args.total_samples),
        "resolution": int(args.resolution),
        "xdim": int(args.resolution),
        "ydim": int(args.resolution),
        "tdim": int(tdim),
        "n_save_steps": int(args.n_save_steps),
        "T": float(args.T),
        "Du": float(args.Du),
        "Dv": float(args.Dv),
        "k": float(args.k),
        "init_mode": init_mode,
        "init_mean": float(args.init_mean),
        "init_std": float(args.init_std),
        "grf_length_scale": float(args.grf_length_scale),
        "grf_spectral_power": float(args.grf_spectral_power),
        "grf_normalize": bool(not args.no_grf_normalize),
        "seed_offset": resolve_seed_offset(args.split, args.seed_offset, dataset_type),
        "x_range": (-1.0, 1.0),
        "y_range": (-1.0, 1.0),
        "boundary_condition": "homogeneous_neumann",
    }


def write_attrs(obj: h5py.Group | h5py.File, attrs: dict[str, object]) -> None:
    for key, value in attrs.items():
        obj.attrs[key] = value


def generate_file(
    file_path: Path,
    *,
    args: argparse.Namespace,
    sample_start: int,
    sample_count: int,
    shard_id: int | None = None,
) -> bool:
    tdim = args.n_save_steps + 1
    init_mode = canonical_init_mode(args.init_mode)
    metadata = metadata_from_args(args, tdim=tdim, init_mode=init_mode)

    if file_path.exists():
        if args.overwrite:
            file_path.unlink()
        else:
            print(f"Skipping existing reaction_diffusion file: {file_path}")
            return False

    file_path.parent.mkdir(parents=True, exist_ok=True)
    seed_offset = resolve_seed_offset(args.split, args.seed_offset, args.dataset_type)
    seeds = seed_offset + np.arange(sample_start, sample_start + sample_count, dtype=np.int64)

    with h5py.File(file_path, "w") as h5:
        write_attrs(h5, metadata)
        h5.attrs["samples_in_file"] = int(sample_count)
        h5.attrs["sample_start"] = int(sample_start)
        if shard_id is not None:
            h5.attrs["shard_id"] = int(shard_id)
        h5.create_dataset("sample_seed", data=seeds, dtype="int64")

        for local_idx, seed in enumerate(tqdm(seeds, desc=file_path.name)):
            sample_id = sample_start + local_idx
            sim = DiffReactSimulator(
                xdim=args.resolution,
                ydim=args.resolution,
                Du=args.Du,
                Dv=args.Dv,
                k=args.k,
                t=args.T,
                tdim=tdim,
                x_left=-1.0,
                x_right=1.0,
                y_bottom=-1.0,
                y_top=1.0,
                seed=int(seed),
                init_mode=init_mode,
                init_mean=args.init_mean,
                init_std=args.init_std,
                grf_length_scale=args.grf_length_scale,
                grf_spectral_power=args.grf_spectral_power,
                grf_normalize=not args.no_grf_normalize,
            )
            data_sample = sim.generate_sample().astype("float32", copy=False)
            if data_sample.shape != (tdim, args.resolution, args.resolution, 2):
                raise RuntimeError(
                    f"Unexpected reaction-diffusion sample shape {data_sample.shape}; "
                    f"expected {(tdim, args.resolution, args.resolution, 2)}"
                )
            if len(sim.t) != tdim or not np.isclose(sim.t[0], 0.0) or not np.isclose(sim.t[-1], args.T):
                raise RuntimeError(
                    f"Unexpected time grid for seed={int(seed)}: len={len(sim.t)}, "
                    f"first={sim.t[0]}, last={sim.t[-1]}, expected len={tdim}, last={args.T}"
                )

            group = h5.create_group(f"{sample_id:06d}")
            group.create_dataset("data", data=data_sample, dtype="float32", compression="lzf")
            group.create_dataset("grid/x", data=sim.x.astype("float32"), dtype="float32")
            group.create_dataset("grid/y", data=sim.y.astype("float32"), dtype="float32")
            group.create_dataset("grid/t", data=sim.t.astype("float32"), dtype="float32")
            group_attrs = dict(metadata)
            group_attrs["seed"] = int(seed)
            group_attrs["sample_id"] = int(sample_id)
            write_attrs(group, group_attrs)
    return True


def generate_dataset(args: argparse.Namespace) -> list[Path]:
    dataset_type = apply_dataset_profile(args)
    if args.total_samples < 1:
        raise ValueError("--total-samples must be positive")
    if args.resolution < 2:
        raise ValueError("--resolution must be at least 2")
    if args.n_save_steps < 1:
        raise ValueError("--n-save-steps must be positive")

    init_mode = canonical_init_mode(args.init_mode)
    samples_per_file = args.samples_per_file or args.total_samples
    if samples_per_file < 1:
        raise ValueError("--samples-per-file/--batch-size must be positive")
    if args.split == "test" and int(samples_per_file) != int(args.total_samples):
        raise ValueError("test files are not sharded; samples-per-file must equal total-samples")

    paths = []
    sample_start = 0
    shard_id = 0
    while sample_start < args.total_samples:
        sample_count = min(samples_per_file, args.total_samples - sample_start)
        file_shard_id = shard_id + 1 if args.split == "train" else None
        file_path = args.save_path / output_file_name(
            split=args.split,
            init_mode=init_mode,
            sample_count=sample_count,
            resolution=args.resolution,
            T=args.T,
            n_save_steps=args.n_save_steps,
            dataset_type=dataset_type,
            shard_id=file_shard_id,
        )
        generate_file(
            file_path,
            args=args,
            sample_start=sample_start,
            sample_count=sample_count,
            shard_id=file_shard_id,
        )
        paths.append(file_path)
        sample_start += sample_count
        shard_id += 1
    return paths


def main() -> None:
    args = parse_args()
    paths = generate_dataset(args)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

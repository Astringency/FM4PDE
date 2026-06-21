from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from pdebench.data_gen.src.sim_diff_react import Simulator as DiffReactSimulator
from tqdm import tqdm


DEFAULT_SAVE_PATH = Path("/large_storage/zhangxf/PDEdata/reaction_diffusion/")


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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--grf-length-scale", type=float, default=0.15)
    parser.add_argument("--grf-spectral-power", type=float, default=2.0)
    parser.add_argument("--no-grf-normalize", action="store_true")
    parser.add_argument("--Du", type=float, default=2e-3)
    parser.add_argument("--Dv", type=float, default=4e-3)
    parser.add_argument("--k", type=float, default=3e-3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_file_name(
    *,
    split: str,
    init_mode: str,
    total_samples: int,
    resolution: int,
    T: float,
    n_save_steps: int,
    shard_id: int | None = None,
) -> str:
    prefix = "reaction_diffusion_test" if split == "test" else "reaction_diffusion"
    stem = f"{prefix}_{init_mode}_{total_samples}-{resolution}-{resolution}-T{T:g}-steps{n_save_steps}"
    if shard_id is not None:
        stem = f"{stem}_shard{shard_id:03d}"
    return f"{stem}.h5"


def metadata_from_args(args: argparse.Namespace, *, tdim: int, init_mode: str) -> dict[str, object]:
    return {
        "pde_name": "reaction_diffusion",
        "split": args.split,
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
        "seed_offset": int(args.seed_offset),
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
) -> None:
    tdim = args.n_save_steps + 1
    init_mode = canonical_init_mode(args.init_mode)
    metadata = metadata_from_args(args, tdim=tdim, init_mode=init_mode)

    if file_path.exists():
        if args.overwrite:
            file_path.unlink()
        else:
            raise FileExistsError(f"{file_path} already exists; pass --overwrite to replace it")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = args.seed_offset + np.arange(sample_start, sample_start + sample_count, dtype=np.int64)

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


def generate_dataset(args: argparse.Namespace) -> list[Path]:
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

    paths = []
    sample_start = 0
    shard_id = 0
    while sample_start < args.total_samples:
        sample_count = min(samples_per_file, args.total_samples - sample_start)
        file_shard_id = shard_id if samples_per_file < args.total_samples else None
        file_path = args.save_path / output_file_name(
            split=args.split,
            init_mode=init_mode,
            total_samples=args.total_samples,
            resolution=args.resolution,
            T=args.T,
            n_save_steps=args.n_save_steps,
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

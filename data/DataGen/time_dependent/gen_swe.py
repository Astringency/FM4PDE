from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


DEFAULT_OUT_DIR = Path("/large_storage/zhangxf/PDEdata/shallow_water")
DATAGEN_DIR = THIS_DIR.parent
if str(DATAGEN_DIR) not in sys.path:
    sys.path.insert(0, str(DATAGEN_DIR))

from generation_profiles import (
    SHALLOW_WATER_SPATIAL_PROFILES,
    canonical_dataset_type,
    seed_offset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FM4PDE shallow-water radial dam-break HDF5 data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--total-samples", type=int, default=1000)
    parser.add_argument("--samples-per-file", type=int, default=None)
    parser.add_argument(
        "--dataset-type",
        choices=["train", "id", "smooth", "rough", "test", "easytest", "hardtest"],
        default=None,
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--tsteps", type=int, default=10, help="Number of saved intervals; file n_time is tsteps + 1.")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def output_path(
    out_dir: Path,
    *,
    split: str,
    sample_count: int | None = None,
    total_samples: int | None = None,
    resolution: int,
    tsteps: int,
    dataset_type: str | None = None,
    file_index: int = 0,
) -> Path:
    count = int(sample_count if sample_count is not None else total_samples)
    resolved_type = canonical_dataset_type(dataset_type or ("train" if split == "train" else "id"))
    if split == "train":
        return out_dir / f"2d_swe_{resolution}_{resolution}_{tsteps}_{file_index}.h5"
    return out_dir / (
        f"shallow_water_test_{count}-{resolution}-{resolution}-{tsteps}_{resolved_type}.h5"
    )


def generate_dataset(args: argparse.Namespace) -> list[Path]:
    if args.total_samples < 1:
        raise ValueError("--total-samples must be positive")
    if args.resolution < 2:
        raise ValueError("--resolution must be at least 2")
    if args.tsteps < 1:
        raise ValueError("--tsteps must be positive")

    split = str(args.split)
    default_type = "train" if split == "train" else "id"
    dataset_type = canonical_dataset_type(getattr(args, "dataset_type", None) or default_type)
    if split == "train" and dataset_type != "train":
        raise ValueError("training files require dataset_type='train'")
    if split == "test" and dataset_type == "train":
        raise ValueError("test files cannot use dataset_type='train'")
    base_seed = args.base_seed
    if base_seed is None:
        base_seed = seed_offset(dataset_type)

    samples_per_file = getattr(args, "samples_per_file", None)
    if samples_per_file is None:
        samples_per_file = 10000 if split == "train" else args.total_samples
    if samples_per_file < 1:
        raise ValueError("--samples-per-file must be positive")
    if split == "test" and int(samples_per_file) != int(args.total_samples):
        raise ValueError("test files are not sharded; samples-per-file must equal total-samples")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    sample_start = 0
    file_index = 0
    while sample_start < int(args.total_samples):
        sample_count = min(int(samples_per_file), int(args.total_samples) - sample_start)
        path = output_path(
            out_dir,
            split=split,
            sample_count=sample_count,
            resolution=int(args.resolution),
            tsteps=int(args.tsteps),
            dataset_type=dataset_type,
            file_index=file_index,
        )
        paths.append(path)
        if path.exists() and not args.overwrite:
            print(f"Skipping existing shallow_water file: {path}")
            sample_start += sample_count
            file_index += 1
            continue
        if path.exists():
            path.unlink()
        generate_file(
            path,
            args=args,
            split=split,
            dataset_type=dataset_type,
            base_seed=int(base_seed),
            sample_start=sample_start,
            sample_count=sample_count,
        )
        sample_start += sample_count
        file_index += 1
    return paths


def generate_file(
    path: Path,
    *,
    args: argparse.Namespace,
    split: str,
    dataset_type: str,
    base_seed: int,
    sample_start: int,
    sample_count: int,
) -> None:
    try:
        from pdebench.data_gen.src.sim_radial_dam_break import RadialDamBreak2D
    except Exception as exc:
        raise RuntimeError(
            "shallow_water generation requires the Clawpack/PyClaw dependency used by "
            "pdebench.data_gen.src.sim_radial_dam_break"
        ) from exc

    spatial_profile = SHALLOW_WATER_SPATIAL_PROFILES[dataset_type]
    with h5py.File(path, "w") as h5:
        h5.attrs["pde_name"] = "shallow_water_radial_dam_break"
        h5.attrs["split"] = split
        h5.attrs["dataset_type"] = dataset_type
        h5.attrs["n_samples"] = int(sample_count)
        h5.attrs["sample_start"] = int(sample_start)
        h5.attrs["resolution"] = f"{args.resolution}x{args.resolution}"
        h5.attrs["n_time"] = int(args.tsteps) + 1
        h5.attrs["tsteps"] = int(args.tsteps)
        h5.attrs["T"] = float(args.T)
        h5.attrs["base_seed"] = int(base_seed)
        h5.attrs["transition_width"] = float(spatial_profile["transition_width"])
        h5.attrs["boundary_roughness"] = float(spatial_profile["boundary_roughness"])
        h5.attrs["boundary_mode"] = int(spatial_profile["boundary_mode"])
        h5.create_dataset(
            "sample_seed",
            data=int(base_seed) + np.arange(sample_start, sample_start + sample_count, dtype=np.int64),
            dtype="int64",
        )
        for local_idx in tqdm(range(int(sample_count)), desc=f"shallow_water {dataset_type}"):
            sample_id = sample_start + local_idx
            seed = int(base_seed) + sample_id
            rng = np.random.default_rng(seed)
            dam_radius = float(rng.uniform(0.4, 0.8))
            inner_height = float(rng.uniform(2.0, 3.0))
            boundary_phase = float(rng.uniform(0.0, 2.0 * np.pi))

            sim = RadialDamBreak2D(
                xdim=int(args.resolution),
                ydim=int(args.resolution),
                grav=1.0,
                dam_radius=dam_radius,
                inner_height=inner_height,
                transition_width=float(spatial_profile["transition_width"]),
                boundary_roughness=float(spatial_profile["boundary_roughness"]),
                boundary_mode=int(spatial_profile["boundary_mode"]),
                boundary_phase=boundary_phase,
            )
            sim.run(T=float(args.T), tsteps=int(args.tsteps))

            seed_group = f"{sample_id:06d}"
            sim.save_state_to_disk(h5, seed_group)
            group = h5[seed_group]
            group.attrs["seed"] = seed
            group.attrs["xdim"] = sim.xdim
            group.attrs["ydim"] = sim.ydim
            group.attrs["grav"] = sim.grav
            group.attrs["dam_radius"] = dam_radius
            group.attrs["inner_height"] = inner_height
            group.attrs["dataset_type"] = dataset_type
            group.attrs["transition_width"] = float(spatial_profile["transition_width"])
            group.attrs["boundary_roughness"] = float(spatial_profile["boundary_roughness"])
            group.attrs["boundary_mode"] = int(spatial_profile["boundary_mode"])
            group.attrs["boundary_phase"] = boundary_phase
            group.attrs["x_range"] = (sim.xlower, sim.xupper)
            group.attrs["y_range"] = (sim.ylower, sim.yupper)
            group.attrs["T"] = float(args.T)


if __name__ == "__main__":
    for generated_path in generate_dataset(parse_args()):
        print(generated_path)

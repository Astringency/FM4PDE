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
TRAIN_BASE_SEED = 0
TEST_BASE_SEED = 10_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FM4PDE shallow-water radial dam-break HDF5 data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--total-samples", type=int, default=1000)
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
    total_samples: int,
    resolution: int,
    tsteps: int,
) -> Path:
    prefix = "shallow_water_test" if split == "test" else "shallow_water"
    return out_dir / f"{prefix}_{total_samples}-{resolution}-{resolution}-{tsteps}.h5"


def generate_dataset(args: argparse.Namespace) -> Path:
    if args.total_samples < 1:
        raise ValueError("--total-samples must be positive")
    if args.resolution < 2:
        raise ValueError("--resolution must be at least 2")
    if args.tsteps < 1:
        raise ValueError("--tsteps must be positive")

    try:
        from pdebench.data_gen.src.sim_radial_dam_break import RadialDamBreak2D
    except Exception as exc:
        raise RuntimeError(
            "shallow_water generation requires the Clawpack/PyClaw dependency used by "
            "pdebench.data_gen.src.sim_radial_dam_break"
        ) from exc

    split = str(args.split)
    base_seed = args.base_seed
    if base_seed is None:
        base_seed = TEST_BASE_SEED if split == "test" else TRAIN_BASE_SEED

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = output_path(
        out_dir,
        split=split,
        total_samples=int(args.total_samples),
        resolution=int(args.resolution),
        tsteps=int(args.tsteps),
    )
    if path.exists():
        if args.overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    with h5py.File(path, "w") as h5:
        h5.attrs["pde_name"] = "shallow_water_radial_dam_break"
        h5.attrs["split"] = split
        h5.attrs["n_samples"] = int(args.total_samples)
        h5.attrs["resolution"] = f"{args.resolution}x{args.resolution}"
        h5.attrs["n_time"] = int(args.tsteps) + 1
        h5.attrs["tsteps"] = int(args.tsteps)
        h5.attrs["T"] = float(args.T)
        h5.attrs["base_seed"] = int(base_seed)
        h5.create_dataset(
            "sample_seed",
            data=int(base_seed) + np.arange(int(args.total_samples), dtype=np.int64),
            dtype="int64",
        )
        for idx in tqdm(range(int(args.total_samples)), desc=f"shallow_water {split}"):
            seed = int(base_seed) + idx
            rng = np.random.default_rng(seed)
            dam_radius = float(rng.uniform(0.4, 0.8))
            inner_height = float(rng.uniform(2.0, 3.0))

            sim = RadialDamBreak2D(
                xdim=int(args.resolution),
                ydim=int(args.resolution),
                grav=1.0,
                dam_radius=dam_radius,
                inner_height=inner_height,
            )
            sim.run(T=float(args.T), tsteps=int(args.tsteps))

            seed_group = f"{idx:06d}"
            sim.save_state_to_disk(h5, seed_group)
            group = h5[seed_group]
            group.attrs["seed"] = seed
            group.attrs["xdim"] = sim.xdim
            group.attrs["ydim"] = sim.ydim
            group.attrs["grav"] = sim.grav
            group.attrs["dam_radius"] = dam_radius
            group.attrs["inner_height"] = inner_height
            group.attrs["x_range"] = (sim.xlower, sim.xupper)
            group.attrs["y_range"] = (sim.ylower, sim.yupper)
            group.attrs["T"] = float(args.T)

    return path


def process_batch(batch_start, batch_size):
    """Backward-compatible helper used by older ad-hoc calls."""
    args = argparse.Namespace(
        out_dir=DEFAULT_OUT_DIR,
        split="test",
        total_samples=int(batch_size),
        resolution=128,
        tsteps=10,
        T=1.0,
        base_seed=int(batch_start),
        overwrite=True,
    )
    return generate_dataset(args)


if __name__ == "__main__":
    print(generate_dataset(parse_args()))

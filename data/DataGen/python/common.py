from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from tqdm import tqdm


TRAIN_BASE_SEED = 0
TEST_BASE_SEED = 10_000_000
NO_LEAKAGE_NOTE = (
    "Train and test samples are generated independently. Each sample uses "
    "seed=base_seed+global_sample_id, and train/test base_seed ranges do not overlap."
)


@dataclass(frozen=True)
class FuturePDEConfig:
    pde: str
    out_root: Path
    resolution: int = 128
    n_train: int = 50_000
    n_val: int = 0
    n_test: int = 1_000
    train_shards: int = 5
    samples_per_shard: int | None = None
    n_time: int = 11
    T: float = 1.0
    base_seed_train: int = TRAIN_BASE_SEED
    base_seed_val: int = 5_000_000
    base_seed_test: int = TEST_BASE_SEED
    chunk_size: int = 128
    compression: str | None = "lzf"
    compression_level: int = 4
    dtype: str = "float32"
    save_trajectory: bool = True
    materialize_constant_fields: bool = False
    recfno_split: bool = False
    bc: str = "periodic"
    overwrite: bool = False
    dry_run: bool = False
    quick_test: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def shard_size(self) -> int:
        if self.samples_per_shard is not None:
            return int(self.samples_per_shard)
        if self.n_train % self.train_shards != 0:
            raise ValueError("n_train must be divisible by train_shards when samples_per_shard is omitted")
        return self.n_train // self.train_shards


@dataclass
class ChunkResult:
    input_data: np.ndarray
    output_data: np.ndarray
    data: np.ndarray | None
    trajectory: np.ndarray | None
    params: dict[str, np.ndarray]


SolverFn = Callable[[np.ndarray, FuturePDEConfig], ChunkResult]


def apply_quick_test_defaults(args: Any) -> None:
    if not getattr(args, "quick_test", False):
        return
    if args.n_train == 50_000:
        args.n_train = 8
    if args.n_test == 1_000:
        args.n_test = 4
    if getattr(args, "n_val", 0):
        args.n_val = min(args.n_val, 4)
    if args.train_shards == 5:
        args.train_shards = 2
    if args.samples_per_shard is None:
        args.samples_per_shard = args.n_train // args.train_shards
    if args.resolution == 128:
        args.resolution = 16
    if args.n_time == 11:
        args.n_time = 5
    args.chunk_size = min(args.chunk_size, 4)


def add_common_arguments(parser: Any) -> None:
    parser.add_argument(
        "--pde",
        choices=["heat", "wave", "advection_diffusion", "steady_heat_conduction", "all"],
        default="all",
    )
    parser.add_argument("--out-root", required=True, help="Output root containing one subdirectory per PDE.")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-val", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=1_000)
    parser.add_argument("--train-shards", type=int, default=5)
    parser.add_argument("--samples-per-shard", type=int, default=None)
    parser.add_argument("--n-time", type=int, default=11)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--base-seed-train", type=int, default=TRAIN_BASE_SEED)
    parser.add_argument("--base-seed-val", type=int, default=5_000_000)
    parser.add_argument("--base-seed-test", type=int, default=TEST_BASE_SEED)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--save-trajectory", dest="save_trajectory", action="store_true", default=True)
    parser.add_argument("--no-trajectory", dest="save_trajectory", action="store_false")
    parser.add_argument("--materialize-constant-fields", action="store_true")
    parser.add_argument("--recfno-split", action="store_true")
    parser.add_argument("--bc", choices=["periodic", "neumann"], default="periodic")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick-test", action="store_true")


def namespace_to_config(args: Any, pde: str, extra: dict[str, Any] | None = None) -> FuturePDEConfig:
    if args.recfno_split and args.n_train == 50_000 and args.n_val == 0 and args.n_test == 1_000:
        args.n_train = 4_000
        args.n_val = 1_000
        args.n_test = 1_000
        if args.samples_per_shard is None:
            args.samples_per_shard = args.n_train // args.train_shards
    samples_per_shard = args.samples_per_shard
    if samples_per_shard is None:
        if args.n_train % args.train_shards != 0:
            raise ValueError("n_train must be divisible by train_shards")
        samples_per_shard = args.n_train // args.train_shards
    if args.n_train != samples_per_shard * args.train_shards:
        raise ValueError(
            "n_train must equal samples_per_shard * train_shards "
            f"({args.n_train} != {samples_per_shard} * {args.train_shards})"
        )
    return FuturePDEConfig(
        pde=pde,
        out_root=Path(args.out_root),
        resolution=args.resolution,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        train_shards=args.train_shards,
        samples_per_shard=samples_per_shard,
        n_time=args.n_time,
        T=args.T,
        base_seed_train=args.base_seed_train,
        base_seed_val=args.base_seed_val,
        base_seed_test=args.base_seed_test,
        chunk_size=args.chunk_size,
        compression=None if args.compression == "none" else args.compression,
        compression_level=args.compression_level,
        dtype=args.dtype,
        save_trajectory=args.save_trajectory,
        materialize_constant_fields=args.materialize_constant_fields,
        recfno_split=args.recfno_split,
        bc=args.bc,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        quick_test=args.quick_test,
        extra=extra or {},
    )


def generate_dataset(config: FuturePDEConfig, solver: SolverFn, metadata: dict[str, Any]) -> list[Path]:
    validate_config(config)
    pde_dir = config.out_root / config.pde
    files = plan_files(config)
    if config.dry_run:
        print_plan(config, files)
        return files

    pde_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for shard_id, n_samples, sample_start, split, path in files_with_splits(config):
        write_h5_shard(
            path=path,
            config=config,
            split=split,
            shard_id=shard_id,
            n_samples=n_samples,
            sample_id_start=sample_start,
            base_seed=base_seed_for_split(config, split),
            solver=solver,
            metadata=metadata,
        )
        written.append(path)

    check = run_no_leakage_check(config, written)
    check_path = pde_dir / "no_leakage_check.json"
    check_path.write_text(json.dumps(check, indent=2), encoding="utf-8")
    return written


def validate_config(config: FuturePDEConfig) -> None:
    if config.resolution <= 1:
        raise ValueError("resolution must be > 1")
    if config.n_time < 2:
        raise ValueError("n_time must be at least 2")
    if config.n_train < 0 or config.n_val < 0 or config.n_test < 0:
        raise ValueError("sample counts must be non-negative")
    if config.train_shards < 1:
        raise ValueError("train_shards must be positive")
    if config.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if config.base_seed_train == config.base_seed_test:
        raise ValueError("train and test base seeds must differ")
    train_seed_end = config.base_seed_train + max(config.n_train - 1, 0)
    val_seed_end = config.base_seed_val + max(config.n_val - 1, 0)
    test_seed_end = config.base_seed_test + max(config.n_test - 1, 0)
    if ranges_overlap(config.base_seed_train, train_seed_end, config.base_seed_test, test_seed_end):
        raise ValueError("train/test seed ranges overlap")
    if config.n_val:
        if ranges_overlap(config.base_seed_train, train_seed_end, config.base_seed_val, val_seed_end):
            raise ValueError("train/val seed ranges overlap")
        if ranges_overlap(config.base_seed_val, val_seed_end, config.base_seed_test, test_seed_end):
            raise ValueError("val/test seed ranges overlap")


def plan_files(config: FuturePDEConfig) -> list[Path]:
    return [item[-1] for item in files_with_splits(config)]


def files_with_splits(config: FuturePDEConfig) -> list[tuple[int, int, int, str, Path]]:
    pde_dir = config.out_root / config.pde
    files: list[tuple[int, int, int, str, Path]] = []
    for shard_idx in range(config.train_shards):
        shard_id = shard_idx + 1
        sample_start = shard_idx * config.shard_size
        path = pde_dir / f"{config.pde}_{config.shard_size}-{config.resolution}-{config.resolution}_{shard_id}.h5"
        files.append((shard_id, config.shard_size, sample_start, "train", path))
    if config.n_val:
        val_path = pde_dir / f"{config.pde}_val_{config.n_val}-{config.resolution}-{config.resolution}.h5"
        files.append((0, config.n_val, 0, "val", val_path))
    test_path = pde_dir / f"{config.pde}_test_{config.n_test}-{config.resolution}-{config.resolution}.h5"
    files.append((0, config.n_test, 0, "test", test_path))
    return files


def print_plan(config: FuturePDEConfig, files: list[Path]) -> None:
    print(
        json.dumps(
            {
                "pde": config.pde,
                "resolution": config.resolution,
                "n_train": config.n_train,
                "n_test": config.n_test,
                "train_shards": config.train_shards,
                "samples_per_shard": config.shard_size,
                "n_time": config.n_time,
                "T": config.T,
                "files": [str(path) for path in files],
            },
            indent=2,
        )
    )


def write_h5_shard(
    path: Path,
    config: FuturePDEConfig,
    split: str,
    shard_id: int,
    n_samples: int,
    sample_id_start: int,
    base_seed: int,
    solver: SolverFn,
    metadata: dict[str, Any],
) -> None:
    if path.exists() and not config.overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    if path.exists():
        path.unlink()

    sample_ids = np.arange(sample_id_start, sample_id_start + n_samples, dtype=np.int64)
    seeds = base_seed + sample_ids
    with h5py.File(path, "w") as h5:
        h5.create_dataset("sample_id", data=sample_ids, dtype="int64")
        h5.create_dataset("sample_seed", data=seeds, dtype="int64")
        h5.create_dataset("x", data=np.linspace(0.0, 1.0, config.resolution, endpoint=False, dtype=np.float32))
        h5.create_dataset("y", data=np.linspace(0.0, 1.0, config.resolution, endpoint=False, dtype=np.float32))
        h5.create_dataset("t", data=np.linspace(0.0, config.T, config.n_time, dtype=np.float32))

        datasets_created = False
        dsets: dict[str, h5py.Dataset] = {}
        param_dsets: dict[str, h5py.Dataset] = {}
        iterator = chunk_slices(n_samples, config.chunk_size)
        desc = f"{config.pde} {split} shard {shard_id}" if split == "train" else f"{config.pde} test"
        for sl in tqdm(list(iterator), desc=desc):
            global_ids = sample_ids[sl]
            solver_config = replace(config, base_seed_train=base_seed)
            result = solver(global_ids, solver_config)
            result = cast_chunk_result(result, config.dtype)
            n_chunk = sl.stop - sl.start
            if result.input_data.shape[0] != n_chunk:
                raise ValueError("solver returned a chunk with the wrong sample count")
            if not datasets_created:
                dsets = create_main_datasets(h5, n_samples, result, config)
                for name, values in result.params.items():
                    param_dsets[name] = h5.create_dataset(
                        name,
                        shape=(n_samples,) + values.shape[1:],
                        dtype=values.dtype,
                        chunks=dataset_chunks((n_samples,) + values.shape[1:], values.dtype, config.chunk_size),
                        **compression_kwargs(config),
                    )
                datasets_created = True

            dsets["input_data"][sl] = result.input_data
            dsets["output_data"][sl] = result.output_data
            if result.data is not None and "materialized_data" in dsets:
                dsets["materialized_data"][sl] = result.data
            if result.trajectory is not None and "full_trajectory" in dsets:
                dsets["full_trajectory"][sl] = result.trajectory
            for name, values in result.params.items():
                param_dsets[name][sl] = values

        write_attrs(
            h5,
            config=config,
            split=split,
            shard_id=shard_id,
            n_samples=n_samples,
            sample_id_start=sample_id_start,
            sample_id_end=sample_id_start + n_samples - 1,
            seed_start=int(seeds[0]) if len(seeds) else int(base_seed),
            seed_end=int(seeds[-1]) if len(seeds) else int(base_seed),
            metadata=metadata,
        )


def create_main_datasets(h5: h5py.File, n_samples: int, result: ChunkResult, config: FuturePDEConfig) -> dict[str, h5py.Dataset]:
    kwargs = compression_kwargs(config)
    dsets = {
        "input_data": h5.create_dataset(
            "input_data",
            shape=(n_samples,) + result.input_data.shape[1:],
            dtype=result.input_data.dtype,
            chunks=dataset_chunks((n_samples,) + result.input_data.shape[1:], result.input_data.dtype, config.chunk_size),
            **kwargs,
        ),
        "output_data": h5.create_dataset(
            "output_data",
            shape=(n_samples,) + result.output_data.shape[1:],
            dtype=result.output_data.dtype,
            chunks=dataset_chunks((n_samples,) + result.output_data.shape[1:], result.output_data.dtype, config.chunk_size),
            **kwargs,
        ),
    }
    if result.data is not None:
        dsets["materialized_data"] = h5.create_dataset(
            "materialized_data",
            shape=(n_samples,) + result.data.shape[1:],
            dtype=result.data.dtype,
            chunks=dataset_chunks((n_samples,) + result.data.shape[1:], result.data.dtype, config.chunk_size),
            **kwargs,
        )
    if result.trajectory is not None:
        dsets["full_trajectory"] = h5.create_dataset(
            "full_trajectory",
            shape=(n_samples,) + result.trajectory.shape[1:],
            dtype=result.trajectory.dtype,
            chunks=dataset_chunks(
                (n_samples,) + result.trajectory.shape[1:],
                result.trajectory.dtype,
                config.chunk_size,
            ),
            **kwargs,
        )
    return dsets


def cast_chunk_result(result: ChunkResult, dtype: str) -> ChunkResult:
    np_dtype = np.dtype(dtype)
    trajectory = None if result.trajectory is None else result.trajectory.astype(np_dtype, copy=False)
    params = {
        name: values.astype(np_dtype, copy=False) if np.issubdtype(values.dtype, np.floating) else values
        for name, values in result.params.items()
    }
    return ChunkResult(
        input_data=result.input_data.astype(np_dtype, copy=False),
        output_data=result.output_data.astype(np_dtype, copy=False),
        data=None if result.data is None else result.data.astype(np_dtype, copy=False),
        trajectory=trajectory,
        params=params,
    )


def write_attrs(
    h5: h5py.File,
    config: FuturePDEConfig,
    split: str,
    shard_id: int,
    n_samples: int,
    sample_id_start: int,
    sample_id_end: int,
    seed_start: int,
    seed_end: int,
    metadata: dict[str, Any],
) -> None:
    attrs = {
        "pde_name": config.pde,
        "resolution": f"{config.resolution}x{config.resolution}",
        "n_samples": n_samples,
        "split": split,
        "shard_id": shard_id,
        "sample_id_start": sample_id_start,
        "sample_id_end": sample_id_end,
        "base_seed": base_seed_for_split(config, split),
        "seed_start": seed_start,
        "seed_end": seed_end,
        "T": config.T,
        "n_time": config.n_time,
        "boundary_condition": config.bc,
        "initial_condition_type": "periodic Gaussian random field",
        "save_trajectory": bool(config.save_trajectory),
        "materialized_constant_fields": bool(config.materialize_constant_fields),
        "dtype": config.dtype,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_git_commit": generation_git_commit(),
        "no_data_leakage_note": NO_LEAKAGE_NOTE,
    }
    attrs.update(metadata)
    for key, value in attrs.items():
        h5.attrs[key] = encode_attr(value)


def encode_attr(value: Any) -> Any:
    if isinstance(value, (str, int, float, np.integer, np.floating, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def compression_kwargs(config: FuturePDEConfig) -> dict[str, Any]:
    if config.compression is None:
        return {}
    if config.compression == "gzip":
        return {"compression": "gzip", "compression_opts": config.compression_level, "shuffle": True}
    if config.compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    raise ValueError(f"Unsupported compression: {config.compression}")


def dataset_chunks(shape: tuple[int, ...], dtype: np.dtype, requested: int) -> tuple[int, ...]:
    if not shape:
        return shape
    itemsize = np.dtype(dtype).itemsize
    sample_items = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    max_chunk_bytes = 8 * 1024 * 1024
    max_n = max(1, max_chunk_bytes // max(1, sample_items * itemsize))
    return (min(shape[0], requested, max_n),) + shape[1:]


def chunk_slices(n_samples: int, chunk_size: int) -> list[slice]:
    return [slice(start, min(start + chunk_size, n_samples)) for start in range(0, n_samples, chunk_size)]


def periodic_wavenumbers(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return angular Fourier wavenumbers on the periodic unit square.

    np.fft.fftfreq returns cycles per unit; PDE spectral derivatives require
    angular wavenumbers, so this function multiplies by 2*pi.
    """
    k = 2.0 * np.pi * np.fft.fftfreq(resolution, d=1.0 / resolution)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    ksq = kx**2 + ky**2
    return kx.astype(np.float64), ky.astype(np.float64), ksq.astype(np.float64)


def neumann_eigenvalues(resolution: int) -> np.ndarray:
    k = np.arange(resolution, dtype=np.float64) * np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx**2 + ky**2


def sample_periodic_grf(
    rng: np.random.Generator,
    resolution: int,
    smoothness: float = 3.0,
    tau: float = 5.0,
    scale: float = 1.0,
) -> np.ndarray:
    white = rng.standard_normal((resolution, resolution))
    _, _, ksq = periodic_wavenumbers(resolution)
    spectral_filter = (tau**2 + ksq) ** (-smoothness / 4.0)
    spectral_filter[0, 0] = 0.0
    field = np.fft.ifft2(np.fft.fft2(white) * spectral_filter).real
    field = normalize_field(field)
    return (scale * field).astype(np.float64)


def normalize_field(field: np.ndarray) -> np.ndarray:
    field = field - np.mean(field)
    std = np.std(field)
    if std < 1e-12:
        return field
    return field / std


def sample_rng(global_sample_id: int, config: FuturePDEConfig, split_base_seed: int | None = None) -> np.random.Generator:
    base = config.base_seed_train if split_base_seed is None else split_base_seed
    return np.random.default_rng(base + int(global_sample_id))


def sample_indices(n: int, max_count: int = 16, seed: int = 20260617) -> list[int]:
    if n <= 0:
        return []
    if n <= max_count:
        return list(range(n))
    rng = np.random.default_rng(seed)
    raw = rng.choice(n, size=max_count, replace=False)
    return sorted(int(i) for i in raw)


def hash_array(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def run_no_leakage_check(config: FuturePDEConfig, files: list[Path]) -> dict[str, Any]:
    train_files = [path for path in files if "_test_" not in path.name]
    val_files = [path for path in train_files if "_val_" in path.name]
    train_files = [path for path in train_files if "_val_" not in path.name]
    test_files = [path for path in files if "_test_" in path.name]
    train_seeds = collect_seeds(train_files)
    val_seeds = collect_seeds(val_files)
    test_seeds = collect_seeds(test_files)
    train_hashes = collect_input_hashes(train_files)
    test_hashes = collect_input_hashes(test_files)
    overlap_files = sorted(set(path.name for path in train_files).intersection(path.name for path in test_files))
    seed_overlap = sorted(train_seeds.intersection(test_seeds))
    val_seed_overlap = sorted(train_seeds.intersection(val_seeds).union(val_seeds.intersection(test_seeds)))
    hash_overlap = sorted(train_hashes.intersection(test_hashes))
    ok = not overlap_files and not seed_overlap and not val_seed_overlap and not hash_overlap
    return {
        "pde": config.pde,
        "ok": ok,
        "train_files": [str(path) for path in train_files],
        "val_files": [str(path) for path in val_files],
        "test_files": [str(path) for path in test_files],
        "file_name_overlap": overlap_files,
        "seed_overlap_count": len(seed_overlap),
        "seed_overlap_examples": seed_overlap[:10],
        "val_seed_overlap_count": len(val_seed_overlap),
        "val_seed_overlap_examples": val_seed_overlap[:10],
        "input_hash_overlap_count": len(hash_overlap),
        "input_hash_overlap_examples": hash_overlap[:10],
        "train_seed_min": min(train_seeds) if train_seeds else None,
        "train_seed_max": max(train_seeds) if train_seeds else None,
        "test_seed_min": min(test_seeds) if test_seeds else None,
        "test_seed_max": max(test_seeds) if test_seeds else None,
        "val_seed_min": min(val_seeds) if val_seeds else None,
        "val_seed_max": max(val_seeds) if val_seeds else None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "note": NO_LEAKAGE_NOTE,
    }


def collect_seeds(files: list[Path]) -> set[int]:
    seeds: set[int] = set()
    for path in files:
        with h5py.File(path, "r") as h5:
            seeds.update(int(seed) for seed in h5["sample_seed"][:])
    return seeds


def collect_input_hashes(files: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in files:
        with h5py.File(path, "r") as h5:
            n = int(h5["input_data"].shape[0])
            for idx in sample_indices(n):
                hashes.add(hash_array(h5["input_data"][idx]))
    return hashes


def base_seed_for_split(config: FuturePDEConfig, split: str) -> int:
    if split == "train":
        return config.base_seed_train
    if split == "val":
        return config.base_seed_val
    if split == "test":
        return config.base_seed_test
    raise ValueError(f"Unknown split={split!r}")


def ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def generation_git_commit() -> str:
    env_commit = os.environ.get("FM4PDE_GIT_COMMIT")
    if env_commit:
        return env_commit
    root = Path(__file__).resolve().parents[3]
    head = root / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            ref = text.split(" ", 1)[1]
            ref_path = root / ".git" / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed = root / ".git" / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#") and line.endswith(f" {ref}"):
                        return line.split(" ", 1)[0]
            return "unknown"
        return text
    except OSError:
        return "unknown"


def ensure_finite(*arrays: np.ndarray) -> None:
    for array in arrays:
        if not np.isfinite(array).all():
            raise FloatingPointError("generated array contains NaN or Inf")


def scalar_field(value: float, n: int, resolution: int) -> np.ndarray:
    return np.full((n, resolution, resolution), value, dtype=np.float64)


def format_float_range(bounds: tuple[float, float]) -> dict[str, float]:
    return {"min": float(bounds[0]), "max": float(bounds[1])}


def finite_difference_periodic_laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(u, 1, axis=-2)
        + np.roll(u, -1, axis=-2)
        + np.roll(u, 1, axis=-1)
        + np.roll(u, -1, axis=-1)
        - 4.0 * u
    ) / (dx * dx)


def minmax_scale_for_storage(field: np.ndarray, limit: float = 3.0) -> np.ndarray:
    return np.clip(field, -limit, limit) / limit


def safe_int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def estimate_size_bytes(n_samples: int, channels: int, resolution: int, dtype: str) -> int:
    return n_samples * channels * resolution * resolution * np.dtype(dtype).itemsize


def human_bytes(num: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num} B"


def n_internal_steps_for_cfl(T: float, resolution: int, c_max: float, cfl: float = 0.25) -> int:
    dx = 1.0 / resolution
    dt_max = cfl * dx / max(c_max, 1e-12) / math.sqrt(2.0)
    return max(1, int(math.ceil(T / dt_max)))

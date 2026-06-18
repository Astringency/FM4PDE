from __future__ import annotations

import argparse
import binascii
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback for minimal environments
    plt = None

THIS_DIR = Path(__file__).resolve().parent
FM4PDE_ROOT = THIS_DIR.parents[2]
TIME_DEPENDENT_DIR = FM4PDE_ROOT / "data" / "DataGen" / "time_dependent"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(TIME_DEPENDENT_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_DEPENDENT_DIR))

from common import FuturePDEConfig, generate_dataset  # noqa: E402
from generate_advection_diffusion import advection_diffusion_metadata, solve_advection_diffusion_chunk  # noqa: E402
from generate_heat import heat_metadata, solve_heat_chunk  # noqa: E402
from generate_steady_heat_conduction import (  # noqa: E402
    steady_heat_conduction_metadata,
    solve_steady_heat_conduction_chunk,
)
from generate_wave import solve_wave_chunk, wave_metadata  # noqa: E402


TRAIN_BASE_SEED = 0
TEST_BASE_SEED = 10_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small DataGen smoke check with HDF5 outputs and PNG previews.")
    parser.add_argument("--out-root", type=Path, default=Path("outputs/datacheck"))
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument(
        "--n-time",
        type=int,
        default=11,
        help="Number of saved frames including the initial state for non-reaction-diffusion generators.",
    )
    parser.add_argument(
        "--n-save-steps",
        type=int,
        default=10,
        help="Reaction-diffusion saved time intervals; tdim is n_save_steps + 1.",
    )
    parser.add_argument(
        "--reaction-diffusion-init-mode",
        choices=["iid", "standard_normal", "grf", "gaussian_random_field"],
        default="grf",
    )
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--device", default=None, help="Device for torch-based generators; defaults to cuda when available.")
    parser.add_argument("--ns-dt", type=float, default=1e-4, help="Internal time step for nsnonbounded datacheck generation.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    out_root = args.out_root.resolve()
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "out_root": str(out_root),
        "resolution": args.resolution,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "n_time": args.n_time,
        "reaction_diffusion_n_save_steps": args.n_save_steps,
        "reaction_diffusion_tdim": args.n_save_steps + 1,
        "reaction_diffusion_init_mode": args.reaction_diffusion_init_mode,
        "entries": {},
    }

    future_entries = [
        ("heat", solve_heat_chunk, heat_metadata, {"alpha_mode": "random", "alpha": 1e-3}, True),
        ("wave", solve_wave_chunk, wave_metadata, {"random_v0": False, "variable_c": False, "c": 1.0, "c_mode": "fixed"}, True),
        ("advection_diffusion", solve_advection_diffusion_chunk, advection_diffusion_metadata, {}, True),
        (
            "steady_heat_conduction",
            solve_steady_heat_conduction_chunk,
            steady_heat_conduction_metadata,
            {"picard_max_iter": 30, "picard_tol": 1e-5},
            False,
        ),
    ]

    for pde, solver, metadata_fn, extra, time_dependent in future_entries:
        config = FuturePDEConfig(
            pde=pde,
            out_root=out_root,
            resolution=args.resolution,
            n_train=args.n_train,
            n_test=args.n_test,
            train_shards=1,
            samples_per_shard=args.n_train,
            n_time=args.n_time,
            T=args.T,
            chunk_size=args.chunk_size,
            overwrite=True,
            save_trajectory=time_dependent,
            extra=extra,
        )
        written = generate_dataset(config, solver, metadata_fn(config))
        entry = {
            "status": "ok",
            "files": [str(path) for path in written],
            "no_leakage_check": str(out_root / pde / "no_leakage_check.json"),
        }
        try:
            if time_dependent:
                entry["previews"] = preview_future_trajectory(out_root, pde, args)
            else:
                entry["previews"] = preview_future_static(out_root, pde, args)
        except Exception as exc:  # pragma: no cover - diagnostic path
            entry["preview_error"] = repr(exc)
        summary["entries"][pde] = entry

    summary["entries"]["reaction_diffusion"] = generate_reaction_diffusion(out_root, args)
    summary["entries"]["shallow_water"] = generate_shallow_water(out_root, args)
    summary["entries"]["nsnonbounded"] = generate_nsnonbounded(out_root, args)
    summary["no_leakage_check"] = write_combined_no_leakage_check(out_root, summary["entries"])

    summary_path = out_root / "datacheck_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def future_train_path(out_root: Path, pde: str, args: argparse.Namespace) -> Path:
    return out_root / pde / f"{pde}_{args.n_train}-{args.resolution}-{args.resolution}_1.h5"


def future_test_path(out_root: Path, pde: str, args: argparse.Namespace) -> Path:
    return out_root / pde / f"{pde}_test_{args.n_test}-{args.resolution}-{args.resolution}.h5"


def canonical_reaction_diffusion_init_mode(init_mode: str) -> str:
    aliases = {
        "iid": "iid",
        "standard_normal": "iid",
        "grf": "grf",
        "gaussian_random_field": "grf",
    }
    return aliases[str(init_mode).lower()]


def preview_future_trajectory(out_root: Path, pde: str, args: argparse.Namespace) -> dict[str, str]:
    train_path = future_train_path(out_root, pde, args)
    test_path = future_test_path(out_root, pde, args)
    previews = {}
    label_map = {
        "heat": ["u"],
        "wave": ["u"],
        "advection_diffusion": ["u"],
    }
    title_map = {
        "heat": "time-dependent heat equation: u(t)",
        "wave": "wave equation: u(t); v is saved at endpoints",
        "advection_diffusion": "advection-diffusion equation: u(t)",
    }
    for split, path in (("train", train_path), ("test", test_path)):
        with h5py.File(path, "r") as h5:
            frames = h5["full_trajectory"][0, 0]
            times = h5["t"][:] if "t" in h5 else None
        preview_path = path.parent / f"preview_{split}.png"
        write_trajectory_preview(
            frames,
            preview_path,
            max_frames=10,
            labels=label_map.get(pde, ["field"]),
            title=f"{title_map.get(pde, pde)} ({split}, sample 0)",
            times=times,
        )
        previews[split] = str(preview_path)
    return previews


def preview_future_static(out_root: Path, pde: str, args: argparse.Namespace) -> dict[str, str]:
    train_path = future_train_path(out_root, pde, args)
    test_path = future_test_path(out_root, pde, args)
    previews = {}
    for split, path in (("train", train_path), ("test", test_path)):
        with h5py.File(path, "r") as h5:
            source = h5["input_data"][0, 0]
            solution = h5["output_data"][0, 0]
        preview_path = path.parent / f"preview_{split}.png"
        write_static_preview(
            [source, solution],
            preview_path,
            labels=["f(x,y)", "u(x,y)"],
            title=f"steady heat conduction ({split}, sample 0)",
        )
        previews[split] = str(preview_path)
    return previews


def generate_reaction_diffusion(out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    pde_dir = out_root / "reaction_diffusion"
    pde_dir.mkdir(parents=True, exist_ok=True)
    try:
        from pdebench.data_gen.src.sim_diff_react import Simulator as DiffReactSimulator
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        return write_error(pde_dir, "reaction_diffusion", "import_failed", exc)

    init_mode = canonical_reaction_diffusion_init_mode(args.reaction_diffusion_init_mode)
    rd_tdim = args.n_save_steps + 1
    init_mean = 0.0
    init_std = 1.0
    grf_length_scale = 0.15
    grf_spectral_power = 2.0
    grf_normalize = True
    du = 2e-3
    dv = 4e-3
    k_react = 3e-3
    train_path = (
        pde_dir
        / f"reaction_diffusion_{init_mode}_{args.n_train}-{args.resolution}-{args.resolution}-T{args.T:g}-steps{args.n_save_steps}.h5"
    )
    test_path = (
        pde_dir
        / f"reaction_diffusion_test_{init_mode}_{args.n_test}-{args.resolution}-{args.resolution}-T{args.T:g}-steps{args.n_save_steps}.h5"
    )
    for path, split, count, base_seed in (
        (train_path, "train", args.n_train, TRAIN_BASE_SEED),
        (test_path, "test", args.n_test, TEST_BASE_SEED),
    ):
        if path.exists():
            path.unlink()
        with h5py.File(path, "w") as h5:
            h5.attrs["pde_name"] = "reaction_diffusion"
            h5.attrs["split"] = split
            h5.attrs["n_samples"] = count
            h5.attrs["resolution"] = f"{args.resolution}x{args.resolution}"
            h5.attrs["n_time"] = rd_tdim
            h5.attrs["tdim"] = rd_tdim
            h5.attrs["n_save_steps"] = args.n_save_steps
            h5.attrs["T"] = args.T
            h5.attrs["Du"] = du
            h5.attrs["Dv"] = dv
            h5.attrs["k"] = k_react
            h5.attrs["init_mode"] = init_mode
            h5.attrs["init_mean"] = init_mean
            h5.attrs["init_std"] = init_std
            h5.attrs["grf_length_scale"] = grf_length_scale
            h5.attrs["grf_spectral_power"] = grf_spectral_power
            h5.attrs["grf_normalize"] = grf_normalize
            h5.attrs["x_range"] = (-1.0, 1.0)
            h5.attrs["y_range"] = (-1.0, 1.0)
            h5.attrs["base_seed"] = base_seed
            h5.create_dataset("sample_seed", data=base_seed + np.arange(count, dtype=np.int64), dtype="int64")
            for idx in range(count):
                print(f"reaction_diffusion {split} sample {idx + 1}/{count}", flush=True)
                seed = base_seed + idx
                sim = DiffReactSimulator(
                    xdim=args.resolution,
                    ydim=args.resolution,
                    Du=du,
                    Dv=dv,
                    k=k_react,
                    t=args.T,
                    tdim=rd_tdim,
                    x_left=-1.0,
                    x_right=1.0,
                    y_bottom=-1.0,
                    y_top=1.0,
                    seed=seed,
                    init_mode=init_mode,
                    init_mean=init_mean,
                    init_std=init_std,
                    grf_length_scale=grf_length_scale,
                    grf_spectral_power=grf_spectral_power,
                    grf_normalize=grf_normalize,
                )
                data_sample = sim.generate_sample().astype("float32", copy=False)
                group = h5.create_group(f"{idx:06d}")
                group.create_dataset("data", data=data_sample, dtype="float32", compression="lzf")
                group.create_dataset("grid/x", data=sim.x.astype("float32"), dtype="float32")
                group.create_dataset("grid/y", data=sim.y.astype("float32"), dtype="float32")
                group.create_dataset("grid/t", data=sim.t.astype("float32"), dtype="float32")
                group.attrs["seed"] = seed
                group.attrs["xdim"] = args.resolution
                group.attrs["ydim"] = args.resolution
                group.attrs["tdim"] = rd_tdim
                group.attrs["n_save_steps"] = args.n_save_steps
                group.attrs["Du"] = du
                group.attrs["Dv"] = dv
                group.attrs["k"] = k_react
                group.attrs["T"] = args.T
                group.attrs["init_mode"] = init_mode
                group.attrs["init_mean"] = init_mean
                group.attrs["init_std"] = init_std
                group.attrs["grf_length_scale"] = grf_length_scale
                group.attrs["grf_spectral_power"] = grf_spectral_power
                group.attrs["grf_normalize"] = grf_normalize
                group.attrs["x_range"] = (-1.0, 1.0)
                group.attrs["y_range"] = (-1.0, 1.0)

    previews = {
        "train": str(
            write_group_multichannel_trajectory_preview(
                train_path,
                pde_dir / "preview_train.png",
                "data",
                labels=["u", "v"],
                title=f"reaction-diffusion equation, init={init_mode} (train, sample 0)",
            )
        ),
        "test": str(
            write_group_multichannel_trajectory_preview(
                test_path,
                pde_dir / "preview_test.png",
                "data",
                labels=["u", "v"],
                title=f"reaction-diffusion equation, init={init_mode} (test, sample 0)",
            )
        ),
    }
    check_path = write_split_no_leakage_check(pde_dir, "reaction_diffusion", train_path, test_path)
    return {"status": "ok", "files": [str(train_path), str(test_path)], "previews": previews, "no_leakage_check": str(check_path)}


def generate_shallow_water(out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    pde_dir = out_root / "shallow_water"
    pde_dir.mkdir(parents=True, exist_ok=True)
    try:
        from pdebench.data_gen.src.sim_radial_dam_break import RadialDamBreak2D
    except Exception as exc:
        return write_error(pde_dir, "shallow_water", "missing_clawpack_dependency", exc)

    train_path = pde_dir / f"shallow_water_{args.n_train}-{args.resolution}-{args.resolution}-{args.n_time}.h5"
    test_path = pde_dir / f"shallow_water_test_{args.n_test}-{args.resolution}-{args.resolution}-{args.n_time}.h5"
    for path, split, count, base_seed in (
        (train_path, "train", args.n_train, TRAIN_BASE_SEED),
        (test_path, "test", args.n_test, TEST_BASE_SEED),
    ):
        if path.exists():
            path.unlink()
        with h5py.File(path, "w") as h5:
            h5.attrs["pde_name"] = "shallow_water_radial_dam_break"
            h5.attrs["split"] = split
            h5.attrs["n_samples"] = count
            h5.attrs["resolution"] = f"{args.resolution}x{args.resolution}"
            h5.attrs["n_time"] = args.n_time
            h5.attrs["base_seed"] = base_seed
            h5.create_dataset("sample_seed", data=base_seed + np.arange(count, dtype=np.int64), dtype="int64")
            for idx in range(count):
                print(f"shallow_water {split} sample {idx + 1}/{count}", flush=True)
                seed = base_seed + idx
                rng = np.random.default_rng(seed)
                dam_radius = float(rng.uniform(0.4, 0.8))
                inner_height = float(rng.uniform(2.0, 3.0))
                sim = RadialDamBreak2D(
                    xdim=args.resolution,
                    ydim=args.resolution,
                    grav=1.0,
                    dam_radius=dam_radius,
                    inner_height=inner_height,
                )
                sim.run(T=args.T, tsteps=args.n_time - 1)
                seed_group = f"{idx:06d}"
                sim.save_state_to_disk(h5, seed_group)
                group = h5[seed_group]
                group.attrs["seed"] = seed
                group.attrs["dam_radius"] = dam_radius
                group.attrs["inner_height"] = inner_height
                group.attrs["T"] = args.T

    previews = {
        "train": str(
            write_shallow_water_preview(
                train_path,
                pde_dir / "preview_train.png",
                title="shallow-water radial dam break (train, sample 0)",
            )
        ),
        "test": str(
            write_shallow_water_preview(
                test_path,
                pde_dir / "preview_test.png",
                title="shallow-water radial dam break (test, sample 0)",
            )
        ),
    }
    check_path = write_split_no_leakage_check(pde_dir, "shallow_water", train_path, test_path)
    return {"status": "ok", "files": [str(train_path), str(test_path)], "previews": previews, "no_leakage_check": str(check_path)}


def generate_nsnonbounded(out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    pde_dir = out_root / "nsnonbounded"
    pde_dir.mkdir(parents=True, exist_ok=True)
    try:
        import math

        import torch
        from no_bound_ns.ns_2d import navier_stokes_2d
        from no_bound_ns.random_fields import GaussianRF
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        return write_error(pde_dir, "nsnonbounded", "import_failed", exc)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record_steps = args.n_time - 1
    if record_steps < 1:
        raise ValueError("--n-time must be at least 2 for nsnonbounded")

    train_path = pde_dir / f"nsnonbounded_{args.n_train}-{args.resolution}-{args.resolution}-{record_steps}_1_new.mat"
    test_path = pde_dir / f"nsnonbounded_test_{args.n_test}-{args.resolution}-{args.resolution}-{record_steps}.mat"

    for path, split, count, base_seed, alpha, tau in (
        (train_path, "train", args.n_train, TRAIN_BASE_SEED, 2.5, 7.0),
        (test_path, "test", args.n_test, TEST_BASE_SEED, 3.0, 6.5),
    ):
        if path.exists():
            path.unlink()
        print(f"nsnonbounded {split} generating {count} samples on {device}", flush=True)
        torch.manual_seed(int(base_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(base_seed))
        s = int(args.resolution)
        grf = GaussianRF(2, s, alpha=alpha, tau=tau, device=device)
        grid = torch.linspace(0, 1, s + 1, device=device)[:-1]
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        forcing = 0.1 * (torch.sin(2 * math.pi * (x + y)) + torch.cos(2 * math.pi * (x + y)))
        w0 = grf.sample(int(count))
        sol_vx0, sol_vy0, sol_w, sol_vx, sol_vy, sol_t = navier_stokes_2d(
            w0,
            forcing,
            1e-3,
            float(args.T),
            float(args.ns_dt),
            record_steps,
        )
        with h5py.File(path, "w") as h5:
            h5.attrs["pde_name"] = "nsnonbounded"
            h5.attrs["split"] = split
            h5.attrs["n_samples"] = int(count)
            h5.attrs["resolution"] = f"{s}x{s}"
            h5.attrs["record_steps"] = int(record_steps)
            h5.attrs["T"] = float(args.T)
            h5.attrs["dt"] = float(args.ns_dt)
            h5.attrs["viscosity"] = 1e-3
            h5.attrs["forcing"] = "0.1*(sin(2*pi*(x+y))+cos(2*pi*(x+y)))"
            h5.attrs["grf_alpha"] = float(alpha)
            h5.attrs["grf_tau"] = float(tau)
            h5.create_dataset("sample_seed", data=base_seed + np.arange(count, dtype=np.int64), dtype="int64")
            h5.create_dataset("w0", data=w0.detach().cpu().numpy().astype("float32"), dtype="float32")
            h5.create_dataset("w", data=sol_w.detach().cpu().numpy().astype("float32"), dtype="float32")
            h5.create_dataset("vx0", data=sol_vx0.detach().cpu().numpy().astype("float32"), dtype="float32")
            h5.create_dataset("vy0", data=sol_vy0.detach().cpu().numpy().astype("float32"), dtype="float32")
            h5.create_dataset("vx", data=sol_vx.detach().cpu().numpy().astype("float32"), dtype="float32")
            h5.create_dataset("vy", data=sol_vy.detach().cpu().numpy().astype("float32"), dtype="float32")
            h5.create_dataset("t", data=sol_t.detach().cpu().numpy().astype("float32"), dtype="float32")

    previews = {
        "train": str(write_nsnonbounded_preview(train_path, pde_dir / "preview_train.png", "nsnonbounded (train, sample 0)")),
        "test": str(write_nsnonbounded_preview(test_path, pde_dir / "preview_test.png", "nsnonbounded (test, sample 0)")),
    }
    check_path = write_split_no_leakage_check(pde_dir, "nsnonbounded", train_path, test_path)
    return {"status": "ok", "files": [str(train_path), str(test_path)], "previews": previews, "no_leakage_check": str(check_path)}


def write_error(pde_dir: Path, pde: str, status: str, exc: Exception) -> dict[str, Any]:
    error = {
        "status": status,
        "pde": pde,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    (pde_dir / "ERROR.txt").write_text(
        f"{pde} datacheck failed before data generation.\n{type(exc).__name__}: {exc}\n",
        encoding="utf-8",
    )
    (pde_dir / "generation_status.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
    return error


def write_group_trajectory_preview(h5_path: Path, preview_path: Path, dataset_suffix: str, channel: int = 0) -> Path:
    with h5py.File(h5_path, "r") as h5:
        sample_key = sorted(key for key in h5.keys() if key.isdigit())[0]
        arr = h5[f"{sample_key}/{dataset_suffix}"][:]
        times = h5[f"{sample_key}/grid/t"][:] if f"{sample_key}/grid/t" in h5 else None
    if arr.ndim == 4:
        frames = arr[..., channel]
    elif arr.ndim == 3:
        frames = arr
    else:
        raise ValueError(f"Unsupported trajectory shape {arr.shape} in {h5_path}")
    write_trajectory_preview(frames, preview_path, max_frames=10, labels=["field"], title=h5_path.stem, times=times)
    return preview_path


def write_group_multichannel_trajectory_preview(
    h5_path: Path,
    preview_path: Path,
    dataset_suffix: str,
    labels: list[str],
    title: str,
) -> Path:
    with h5py.File(h5_path, "r") as h5:
        sample_key = sorted(key for key in h5.keys() if key.isdigit())[0]
        arr = h5[f"{sample_key}/{dataset_suffix}"][:]
        times = h5[f"{sample_key}/grid/t"][:] if f"{sample_key}/grid/t" in h5 else None
    if arr.ndim != 4:
        raise ValueError(f"Expected [T,H,W,C] data in {h5_path}, got {arr.shape}")
    series = np.moveaxis(arr, -1, 0)
    write_multichannel_trajectory_preview(series, preview_path, labels=labels, title=title, times=times)
    return preview_path


def write_nsnonbounded_preview(h5_path: Path, preview_path: Path, title: str) -> Path:
    with h5py.File(h5_path, "r") as h5:
        w = np.concatenate([h5["w0"][0][None, ...], np.moveaxis(h5["w"][0], -1, 0)], axis=0)
        vx = np.concatenate([h5["vx0"][0][None, ...], np.moveaxis(h5["vx"][0], -1, 0)], axis=0)
        vy = np.concatenate([h5["vy0"][0][None, ...], np.moveaxis(h5["vy"][0], -1, 0)], axis=0)
        times = np.concatenate([[0.0], h5["t"][:].astype(np.float64)])
    series = np.stack([w, vx, vy], axis=0)
    write_multichannel_trajectory_preview(series, preview_path, labels=["w", "vx", "vy"], title=title, times=times)
    return preview_path


def write_shallow_water_preview(h5_path: Path, preview_path: Path, title: str) -> Path:
    labels = ["h", "u", "v", "hu", "hv"]
    with h5py.File(h5_path, "r") as h5:
        sample_key = sorted(key for key in h5.keys() if key.isdigit())[0]
        times = h5[f"{sample_key}/grid/t"][:] if f"{sample_key}/grid/t" in h5 else None
        fields = []
        for label in labels:
            arr = h5[f"{sample_key}/data/{label}"][:]
            fields.append(arr[..., 0] if arr.ndim == 4 else arr)
    series = np.stack(fields, axis=0)
    write_multichannel_trajectory_preview(series, preview_path, labels=labels, title=title, times=times)
    return preview_path


def write_trajectory_preview(
    frames: np.ndarray,
    path: Path,
    max_frames: int = 10,
    labels: list[str] | None = None,
    title: str | None = None,
    times: np.ndarray | None = None,
) -> None:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3:
        raise ValueError(f"trajectory preview expects [T,H,W], got {frames.shape}")
    write_multichannel_trajectory_preview(frames[None, ...], path, labels=labels or ["field"], title=title, times=times, max_frames=max_frames)


def write_multichannel_trajectory_preview(
    series: np.ndarray,
    path: Path,
    labels: list[str],
    title: str | None = None,
    times: np.ndarray | None = None,
    max_frames: int = 10,
) -> None:
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 4:
        raise ValueError(f"multichannel trajectory preview expects [Q,T,H,W], got {series.shape}")
    count = min(max_frames, series.shape[1])
    indices = np.linspace(0, series.shape[1] - 1, count, dtype=int)
    chosen = series[:, indices]
    if plt is not None:
        write_matplotlib_trajectory_preview(chosen, indices, path, labels, title=title, times=times)
        return
    first_channel = chosen[0]
    vmin, vmax = finite_min_max(first_channel)
    tiles = [colorize(frame, vmin, vmax) for frame in first_channel]
    grid = tile_images(tiles, cols=5)
    write_png(path, grid)


def write_static_preview(fields: list[np.ndarray], path: Path, labels: list[str] | None = None, title: str | None = None) -> None:
    if plt is not None:
        write_matplotlib_static_preview(fields, path, labels=labels, title=title)
        return
    tiles = []
    for field in fields:
        vmin, vmax = finite_min_max(field)
        tiles.append(colorize(np.asarray(field, dtype=np.float64), vmin, vmax))
    grid = tile_images(tiles, cols=len(tiles))
    write_png(path, grid)


def write_matplotlib_trajectory_preview(
    chosen: np.ndarray,
    indices: np.ndarray,
    path: Path,
    labels: list[str],
    title: str | None = None,
    times: np.ndarray | None = None,
) -> None:
    n_fields, n_times = chosen.shape[:2]
    labels = labels[:n_fields] + [f"field {idx}" for idx in range(len(labels), n_fields)]
    fig_w = max(12.0, 1.7 * n_times)
    fig_h = max(2.4, 1.8 * n_fields + 0.8)
    fig, axes = plt.subplots(n_fields, n_times, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    for row in range(n_fields):
        vmin, vmax = finite_min_max(chosen[row])
        for col in range(n_times):
            ax = axes[row, col]
            ax.imshow(chosen[row, col], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                if times is not None:
                    ax.set_title(f"t={float(times[indices[col]]):.3g}", fontsize=8)
                else:
                    ax.set_title(f"step {int(indices[col])}", fontsize=8)
            if col == 0:
                ax.text(
                    -0.15,
                    0.5,
                    labels[row],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10,
                    fontweight="bold",
                )
    if title:
        fig.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_matplotlib_static_preview(
    fields: list[np.ndarray],
    path: Path,
    labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    labels = labels or [f"field {idx}" for idx in range(len(fields))]
    fig, axes = plt.subplots(1, len(fields), figsize=(4.0 * len(fields), 3.8), squeeze=False, constrained_layout=True)
    for idx, field in enumerate(fields):
        ax = axes[0, idx]
        arr = np.asarray(field, dtype=np.float64)
        vmin, vmax = finite_min_max(arr)
        image = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(labels[idx] if idx < len(labels) else f"field {idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def finite_min_max(arr: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(arr)[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    return vmin, vmax


def colorize(field: np.ndarray, vmin: float, vmax: float, tile_size: int = 96) -> np.ndarray:
    normalized = np.clip((field - vmin) / (vmax - vmin), 0.0, 1.0)
    stops = np.asarray(
        [
            [31, 31, 84],
            [59, 130, 246],
            [34, 197, 94],
            [250, 204, 21],
            [239, 68, 68],
        ],
        dtype=np.float64,
    )
    scaled = normalized * (len(stops) - 1)
    low = np.floor(scaled).astype(np.int64)
    high = np.clip(low + 1, 0, len(stops) - 1)
    frac = (scaled - low)[..., None]
    rgb = (1.0 - frac) * stops[low] + frac * stops[high]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    scale = max(1, int(np.ceil(tile_size / max(field.shape))))
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def tile_images(images: list[np.ndarray], cols: int, gap: int = 4, background: int = 245) -> np.ndarray:
    if not images:
        raise ValueError("No images to tile")
    cols = max(1, cols)
    rows = int(np.ceil(len(images) / cols))
    tile_h = max(image.shape[0] for image in images)
    tile_w = max(image.shape[1] for image in images)
    canvas_h = rows * tile_h + (rows + 1) * gap
    canvas_w = cols * tile_w + (cols + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), background, dtype=np.uint8)
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        y0 = gap + row * (tile_h + gap)
        x0 = gap + col * (tile_w + gap)
        canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
    return canvas


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"PNG writer expects RGB uint8, got {arr.shape}")
    height, width, _ = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", binascii.crc32(kind + data) & 0xFFFFFFFF)


def write_split_no_leakage_check(pde_dir: Path, pde: str, train_path: Path, test_path: Path) -> Path:
    with h5py.File(train_path, "r") as h5:
        train_seeds = set(int(x) for x in h5["sample_seed"][:])
    with h5py.File(test_path, "r") as h5:
        test_seeds = set(int(x) for x in h5["sample_seed"][:])
    check = {
        "pde": pde,
        "ok": not train_seeds.intersection(test_seeds),
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_seed_min": min(train_seeds),
        "train_seed_max": max(train_seeds),
        "test_seed_min": min(test_seeds),
        "test_seed_max": max(test_seeds),
        "seed_overlap_count": len(train_seeds.intersection(test_seeds)),
    }
    check_path = pde_dir / "no_leakage_check.json"
    check_path.write_text(json.dumps(check, indent=2), encoding="utf-8")
    return check_path


def write_combined_no_leakage_check(out_root: Path, entries: dict[str, Any]) -> str:
    combined = {}
    for pde, entry in entries.items():
        check_path = entry.get("no_leakage_check") if isinstance(entry, dict) else None
        if check_path and Path(check_path).exists():
            combined[pde] = json.loads(Path(check_path).read_text(encoding="utf-8"))
        elif isinstance(entry, dict):
            combined[pde] = {"ok": False, "status": entry.get("status"), "error": entry.get("error")}
    combined["ok"] = all(bool(item.get("ok")) for key, item in combined.items() if key != "ok")
    combined_path = out_root / "no_leakage_check.json"
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    return str(combined_path)


if __name__ == "__main__":
    main()

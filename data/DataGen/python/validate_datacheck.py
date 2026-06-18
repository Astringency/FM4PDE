from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
FM4PDE_ROOT = THIS_DIR.parents[2]
TIME_DEPENDENT_DIR = FM4PDE_ROOT / "data" / "DataGen" / "time_dependent"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(TIME_DEPENDENT_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_DEPENDENT_DIR))

from common import periodic_wavenumbers  # noqa: E402
from generate_steady_heat_conduction import conductivity_lambda, nonlinear_residual  # noqa: E402


EXACT_REL_TOL = 5e-5
BOUNDARY_TOL = 5e-5
STEADY_RESIDUAL_TOL = 1e-2
MOMENTUM_TOL = 5e-5
MASS_REL_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 128x128 datacheck PDE outputs numerically.")
    parser.add_argument("--root", type=Path, required=True, help="Root produced by datacheck_generate.py")
    parser.add_argument("--output", type=Path, default=None, help="Validation JSON path")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    root = args.root.resolve()
    report = validate_root(root)
    output = args.output or root / "numeric_validation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)
    return report


def validate_root(root: Path) -> dict[str, Any]:
    summary_path = root / "datacheck_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = summary["entries"]
    report: dict[str, Any] = {
        "root": str(root),
        "resolution": summary.get("resolution"),
        "n_train": summary.get("n_train"),
        "n_test": summary.get("n_test"),
        "n_time": summary.get("n_time"),
        "checks": {},
    }
    report["checks"]["no_leakage"] = validate_no_leakage(root)
    report["checks"]["heat"] = validate_heat(entries["heat"]["files"])
    report["checks"]["wave"] = validate_wave(entries["wave"]["files"])
    report["checks"]["advection_diffusion"] = validate_advection_diffusion(entries["advection_diffusion"]["files"])
    report["checks"]["steady_heat_conduction"] = validate_steady_heat_conduction(entries["steady_heat_conduction"]["files"])
    report["checks"]["reaction_diffusion"] = validate_reaction_diffusion(entries["reaction_diffusion"]["files"])
    report["checks"]["shallow_water"] = validate_shallow_water(entries["shallow_water"]["files"])
    if "nsnonbounded" in entries and entries["nsnonbounded"].get("status") == "ok":
        report["checks"]["nsnonbounded"] = validate_nsnonbounded(entries["nsnonbounded"]["files"])
    report["ok"] = all(check["ok"] for check in report["checks"].values())
    return report


def validate_no_leakage(root: Path) -> dict[str, Any]:
    path = root / "no_leakage_check.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"ok": bool(payload.get("ok")), "path": str(path)}


def validate_heat(files: list[str]) -> dict[str, Any]:
    metrics = {
        "max_exact_rel_l2": 0.0,
        "max_output_terminal_rel_l2": 0.0,
        "max_positive_variance_jump": 0.0,
        "files_checked": len(files),
        "samples_checked": 0,
    }
    for file_name in files:
        with h5py.File(file_name, "r") as h5:
            require_keys(h5, ["input_data", "output_data", "full_trajectory", "t"])
            u0_all = h5["input_data"][:, 0].astype(np.float64)
            uT_all = h5["output_data"][:, 0].astype(np.float64)
            traj_all = h5["full_trajectory"][:, 0].astype(np.float64)
            times = h5["t"][:].astype(np.float64)
            alpha_all = read_param_or_attr(h5, "alpha", "fixed_alpha", len(u0_all))
            _, _, ksq = periodic_wavenumbers(u0_all.shape[-1])
            for idx, (u0, uT, traj, alpha) in enumerate(zip(u0_all, uT_all, traj_all, alpha_all)):
                coeff = np.fft.fft2(u0)
                expected = np.stack([np.fft.ifft2(coeff * np.exp(-float(alpha) * ksq * t)).real for t in times], axis=0)
                metrics["max_exact_rel_l2"] = max(metrics["max_exact_rel_l2"], rel_l2(traj, expected))
                metrics["max_output_terminal_rel_l2"] = max(metrics["max_output_terminal_rel_l2"], rel_l2(uT, traj[-1]))
                variances = np.var(traj.reshape(traj.shape[0], -1), axis=1)
                jumps = np.diff(variances)
                metrics["max_positive_variance_jump"] = max(metrics["max_positive_variance_jump"], float(np.max(jumps)) if jumps.size else 0.0)
                metrics["samples_checked"] += 1
    ok = (
        metrics["max_exact_rel_l2"] <= EXACT_REL_TOL
        and metrics["max_output_terminal_rel_l2"] <= EXACT_REL_TOL
        and metrics["max_positive_variance_jump"] <= 1e-6
    )
    return {"ok": ok, "thresholds": {"exact_rel_l2": EXACT_REL_TOL, "variance_jump": 1e-6}, "metrics": metrics}


def validate_wave(files: list[str]) -> dict[str, Any]:
    metrics = {
        "max_u_exact_rel_l2": 0.0,
        "max_uT_terminal_rel_l2": 0.0,
        "max_vT_exact_rel_l2": 0.0,
        "min_vT_l2": float("inf"),
        "files_checked": len(files),
        "samples_checked": 0,
    }
    for file_name in files:
        with h5py.File(file_name, "r") as h5:
            require_keys(h5, ["input_data", "output_data", "full_trajectory", "t"])
            input_data = h5["input_data"][:].astype(np.float64)
            output_data = h5["output_data"][:].astype(np.float64)
            traj_all = h5["full_trajectory"][:, 0].astype(np.float64)
            times = h5["t"][:].astype(np.float64)
            c_all = read_param_or_attr(h5, "c", "fixed_c", input_data.shape[0], default=1.0)
            _, _, ksq = periodic_wavenumbers(input_data.shape[-1])
            kabs = np.sqrt(ksq)
            for inp, out, traj, c in zip(input_data, output_data, traj_all, c_all):
                u0, v0 = inp[0], inp[1]
                u0_hat = np.fft.fft2(u0)
                v0_hat = np.fft.fft2(v0)
                u_frames = []
                v_last = None
                for t in times:
                    ck_t = float(c) * kabs * t
                    cos_term = np.cos(ck_t)
                    sin_over_ck = np.zeros_like(kabs)
                    nonzero = kabs > 0.0
                    sin_over_ck[nonzero] = np.sin(ck_t[nonzero]) / (float(c) * kabs[nonzero])
                    u_hat = u0_hat * cos_term + v0_hat * sin_over_ck
                    u_hat[0, 0] = u0_hat[0, 0] + t * v0_hat[0, 0]
                    v_hat = -u0_hat * float(c) * kabs * np.sin(ck_t) + v0_hat * cos_term
                    v_hat[0, 0] = v0_hat[0, 0]
                    u_frames.append(np.fft.ifft2(u_hat).real)
                    v_last = np.fft.ifft2(v_hat).real
                expected_u = np.stack(u_frames, axis=0)
                metrics["max_u_exact_rel_l2"] = max(metrics["max_u_exact_rel_l2"], rel_l2(traj, expected_u))
                metrics["max_uT_terminal_rel_l2"] = max(metrics["max_uT_terminal_rel_l2"], rel_l2(out[0], traj[-1]))
                metrics["max_vT_exact_rel_l2"] = max(metrics["max_vT_exact_rel_l2"], rel_l2(out[1], v_last))
                metrics["min_vT_l2"] = min(metrics["min_vT_l2"], float(np.linalg.norm(out[1])))
                metrics["samples_checked"] += 1
    ok = (
        metrics["max_u_exact_rel_l2"] <= EXACT_REL_TOL
        and metrics["max_uT_terminal_rel_l2"] <= EXACT_REL_TOL
        and metrics["max_vT_exact_rel_l2"] <= EXACT_REL_TOL
        and metrics["min_vT_l2"] > 1e-5
    )
    return {"ok": ok, "thresholds": {"exact_rel_l2": EXACT_REL_TOL, "min_vT_l2": 1e-5}, "metrics": metrics}


def validate_advection_diffusion(files: list[str]) -> dict[str, Any]:
    metrics = {
        "max_exact_rel_l2": 0.0,
        "max_output_terminal_rel_l2": 0.0,
        "min_kappa": float("inf"),
        "sign_test_max_abs": advection_sign_test(),
        "files_checked": len(files),
        "samples_checked": 0,
    }
    for file_name in files:
        with h5py.File(file_name, "r") as h5:
            require_keys(h5, ["input_data", "output_data", "full_trajectory", "t", "b_x", "b_y", "kappa"])
            u0_all = h5["input_data"][:, 0].astype(np.float64)
            uT_all = h5["output_data"][:, 0].astype(np.float64)
            traj_all = h5["full_trajectory"][:, 0].astype(np.float64)
            bx_all = h5["b_x"][:].astype(np.float64)
            by_all = h5["b_y"][:].astype(np.float64)
            kappa_all = h5["kappa"][:].astype(np.float64)
            times = h5["t"][:].astype(np.float64)
            kx, ky, ksq = periodic_wavenumbers(u0_all.shape[-1])
            for u0, uT, traj, bx, by, kappa in zip(u0_all, uT_all, traj_all, bx_all, by_all, kappa_all):
                coeff = np.fft.fft2(u0)
                phase = bx * kx + by * ky
                expected = np.stack(
                    [np.fft.ifft2(coeff * np.exp(-(kappa * ksq + 1j * phase) * t)).real for t in times],
                    axis=0,
                )
                metrics["max_exact_rel_l2"] = max(metrics["max_exact_rel_l2"], rel_l2(traj, expected))
                metrics["max_output_terminal_rel_l2"] = max(metrics["max_output_terminal_rel_l2"], rel_l2(uT, traj[-1]))
                metrics["min_kappa"] = min(metrics["min_kappa"], float(kappa))
                metrics["samples_checked"] += 1
    ok = (
        metrics["max_exact_rel_l2"] <= EXACT_REL_TOL
        and metrics["max_output_terminal_rel_l2"] <= EXACT_REL_TOL
        and metrics["min_kappa"] > 0.0
        and metrics["sign_test_max_abs"] <= 1e-10
    )
    return {"ok": ok, "thresholds": {"exact_rel_l2": EXACT_REL_TOL, "sign_test_max_abs": 1e-10}, "metrics": metrics}


def validate_steady_heat_conduction(files: list[str]) -> dict[str, Any]:
    metrics = {
        "max_bottom_dirichlet_abs": 0.0,
        "max_neumann_abs": 0.0,
        "min_lambda": float("inf"),
        "max_residual_norm": 0.0,
        "max_recomputed_float32_residual_norm": 0.0,
        "min_converged": 1,
        "files_checked": len(files),
        "samples_checked": 0,
    }
    for file_name in files:
        with h5py.File(file_name, "r") as h5:
            require_keys(h5, ["input_data", "output_data", "u_D", "converged", "residual_norm"])
            f_all = h5["input_data"][:, 0].astype(np.float64)
            u_all = h5["output_data"][:, 0].astype(np.float64)
            u_d_all = h5["u_D"][:].astype(np.float64)
            converged = h5["converged"][:]
            residual_norm = h5["residual_norm"][:].astype(np.float64)
            for f, u, u_d, conv, res in zip(f_all, u_all, u_d_all, converged, residual_norm):
                conductivity = conductivity_lambda(u)
                residual = nonlinear_residual(u, conductivity, f)
                metrics["max_bottom_dirichlet_abs"] = max(metrics["max_bottom_dirichlet_abs"], float(np.max(np.abs(u[0, :] - u_d))))
                metrics["max_neumann_abs"] = max(
                    metrics["max_neumann_abs"],
                    float(
                        max(
                            np.max(np.abs(u[-1, :] - u[-2, :])),
                            np.max(np.abs(u[:, 0] - u[:, 1])),
                            np.max(np.abs(u[:, -1] - u[:, -2])),
                        )
                    ),
                )
                metrics["min_lambda"] = min(metrics["min_lambda"], float(np.min(conductivity)))
                metrics["max_residual_norm"] = max(metrics["max_residual_norm"], float(res))
                metrics["max_recomputed_float32_residual_norm"] = max(
                    metrics["max_recomputed_float32_residual_norm"],
                    float(np.linalg.norm(residual) / np.sqrt(residual.size)),
                )
                metrics["min_converged"] = min(metrics["min_converged"], int(conv))
                metrics["samples_checked"] += 1
    ok = (
        metrics["max_bottom_dirichlet_abs"] <= BOUNDARY_TOL
        and metrics["max_neumann_abs"] <= BOUNDARY_TOL
        and metrics["min_lambda"] > 0.0
        and metrics["max_residual_norm"] <= STEADY_RESIDUAL_TOL
        and metrics["min_converged"] == 1
    )
    return {
        "ok": ok,
        "thresholds": {
            "boundary_abs": BOUNDARY_TOL,
            "steady_residual_norm": STEADY_RESIDUAL_TOL,
        },
        "metrics": metrics,
    }


def validate_reaction_diffusion(files: list[str]) -> dict[str, Any]:
    from pdebench.data_gen.src.sim_diff_react import Simulator as DiffReactSimulator

    metrics = {
        "max_reproduction_abs": 0.0,
        "max_abs_value": 0.0,
        "min_channel_std": float("inf"),
        "files_checked": len(files),
        "samples_checked": 0,
    }
    for file_name in files:
        with h5py.File(file_name, "r") as h5:
            for key in sorted(k for k in h5.keys() if k.isdigit()):
                group = h5[key]
                data = group["data"][:].astype(np.float64)
                require_finite(data, f"{file_name}:{key}/data")
                seed = int(group.attrs["seed"])
                du = float(group.attrs["Du"])
                dv = float(group.attrs["Dv"])
                k = float(group.attrs["k"])
                T = float(group.attrs["T"])
                init_mode = group.attrs.get("init_mode", "iid")
                if isinstance(init_mode, bytes):
                    init_mode = init_mode.decode("utf-8")
                init_mean = float(group.attrs.get("init_mean", 0.0))
                init_std = float(group.attrs.get("init_std", 1.0))
                grf_length_scale = float(group.attrs.get("grf_length_scale", 0.15))
                grf_spectral_power = float(group.attrs.get("grf_spectral_power", 2.0))
                grf_normalize = bool(group.attrs.get("grf_normalize", True))
                sim = DiffReactSimulator(
                    xdim=data.shape[2],
                    ydim=data.shape[1],
                    Du=du,
                    Dv=dv,
                    k=k,
                    t=T,
                    tdim=data.shape[0],
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
                expected = sim.generate_sample().astype(np.float32).astype(np.float64)
                metrics["max_reproduction_abs"] = max(metrics["max_reproduction_abs"], float(np.max(np.abs(data - expected))))
                metrics["max_abs_value"] = max(metrics["max_abs_value"], float(np.max(np.abs(data))))
                metrics["min_channel_std"] = min(metrics["min_channel_std"], float(np.min(np.std(data.reshape(data.shape[0], -1, data.shape[-1]), axis=(0, 1)))))
                metrics["samples_checked"] += 1
    ok = metrics["max_reproduction_abs"] <= 1e-6 and metrics["max_abs_value"] < 1e3 and metrics["min_channel_std"] > 1e-6
    return {
        "ok": ok,
        "thresholds": {"reproduction_abs": 1e-6, "max_abs_value": 1e3, "min_channel_std": 1e-6},
        "metrics": metrics,
    }


def validate_shallow_water(files: list[str]) -> dict[str, Any]:
    metrics = {
        "min_h": float("inf"),
        "max_mass_rel_change": 0.0,
        "max_hu_consistency_abs": 0.0,
        "max_hv_consistency_abs": 0.0,
        "max_initial_outer_height_abs": 0.0,
        "max_initial_inner_height_abs": 0.0,
        "files_checked": len(files),
        "samples_checked": 0,
    }
    for file_name in files:
        with h5py.File(file_name, "r") as h5:
            for key in sorted(k for k in h5.keys() if k.isdigit()):
                group = h5[key]
                h = group["data/h"][:, :, :, 0].astype(np.float64)
                u = group["data/u"][:, :, :, 0].astype(np.float64)
                v = group["data/v"][:, :, :, 0].astype(np.float64)
                hu = group["data/hu"][:, :, :, 0].astype(np.float64)
                hv = group["data/hv"][:, :, :, 0].astype(np.float64)
                require_finite(h, f"{file_name}:{key}/h")
                require_finite(u, f"{file_name}:{key}/u")
                require_finite(v, f"{file_name}:{key}/v")
                mass = np.sum(h, axis=(1, 2))
                metrics["min_h"] = min(metrics["min_h"], float(np.min(h)))
                metrics["max_mass_rel_change"] = max(metrics["max_mass_rel_change"], float(np.max(np.abs(mass - mass[0])) / max(abs(mass[0]), 1e-12)))
                metrics["max_hu_consistency_abs"] = max(metrics["max_hu_consistency_abs"], float(np.max(np.abs(hu - h * u))))
                metrics["max_hv_consistency_abs"] = max(metrics["max_hv_consistency_abs"], float(np.max(np.abs(hv - h * v))))
                inner_height = float(group.attrs["inner_height"])
                metrics["max_initial_outer_height_abs"] = max(metrics["max_initial_outer_height_abs"], float(abs(np.min(h[0]) - 1.0)))
                metrics["max_initial_inner_height_abs"] = max(metrics["max_initial_inner_height_abs"], float(abs(np.max(h[0]) - inner_height)))
                metrics["samples_checked"] += 1
    ok = (
        metrics["min_h"] > 0.0
        and metrics["max_mass_rel_change"] <= MASS_REL_TOL
        and metrics["max_hu_consistency_abs"] <= MOMENTUM_TOL
        and metrics["max_hv_consistency_abs"] <= MOMENTUM_TOL
        and metrics["max_initial_outer_height_abs"] <= MOMENTUM_TOL
        and metrics["max_initial_inner_height_abs"] <= MOMENTUM_TOL
    )
    return {
        "ok": ok,
        "thresholds": {"mass_rel_change": MASS_REL_TOL, "momentum_abs": MOMENTUM_TOL},
        "metrics": metrics,
    }


def validate_nsnonbounded(files: list[str]) -> dict[str, Any]:
    metrics = {
        "max_abs_w": 0.0,
        "max_abs_velocity": 0.0,
        "min_w0_std": float("inf"),
        "max_time_monotonic_violation": 0.0,
        "files_checked": len(files),
        "samples_checked": 0,
    }
    for file_name in files:
        with h5py.File(file_name, "r") as h5:
            require_keys(h5, ["sample_seed", "w0", "w", "vx0", "vy0", "vx", "vy", "t"])
            w0 = h5["w0"][:].astype(np.float64)
            w = h5["w"][:].astype(np.float64)
            vx0 = h5["vx0"][:].astype(np.float64)
            vy0 = h5["vy0"][:].astype(np.float64)
            vx = h5["vx"][:].astype(np.float64)
            vy = h5["vy"][:].astype(np.float64)
            times = h5["t"][:].astype(np.float64)
            for name, arr in (("w0", w0), ("w", w), ("vx0", vx0), ("vy0", vy0), ("vx", vx), ("vy", vy), ("t", times)):
                require_finite(arr, f"{file_name}:{name}")
            if w0.ndim != 3 or vx0.shape != w0.shape or vy0.shape != w0.shape:
                raise ValueError(f"Unexpected nsnonbounded initial shapes in {file_name}: w0={w0.shape}, vx0={vx0.shape}, vy0={vy0.shape}")
            if w.ndim != 4 or vx.shape != w.shape or vy.shape != w.shape:
                raise ValueError(f"Unexpected nsnonbounded trajectory shapes in {file_name}: w={w.shape}, vx={vx.shape}, vy={vy.shape}")
            if w.shape[:3] != w0.shape or times.shape != (w.shape[-1],):
                raise ValueError(f"Inconsistent nsnonbounded shapes in {file_name}: w0={w0.shape}, w={w.shape}, t={times.shape}")
            time_diffs = np.diff(times)
            if time_diffs.size:
                metrics["max_time_monotonic_violation"] = max(
                    metrics["max_time_monotonic_violation"],
                    float(np.max(np.maximum(-time_diffs, 0.0))),
                )
            metrics["max_abs_w"] = max(metrics["max_abs_w"], float(max(np.max(np.abs(w0)), np.max(np.abs(w)))))
            metrics["max_abs_velocity"] = max(
                metrics["max_abs_velocity"],
                float(max(np.max(np.abs(vx0)), np.max(np.abs(vy0)), np.max(np.abs(vx)), np.max(np.abs(vy)))),
            )
            metrics["min_w0_std"] = min(metrics["min_w0_std"], float(np.min(np.std(w0.reshape(w0.shape[0], -1), axis=1))))
            metrics["samples_checked"] += int(w0.shape[0])
    ok = (
        metrics["samples_checked"] > 0
        and metrics["min_w0_std"] > 1e-6
        and metrics["max_abs_w"] < 1e4
        and metrics["max_abs_velocity"] < 1e4
        and metrics["max_time_monotonic_violation"] <= 0.0
    )
    return {
        "ok": ok,
        "thresholds": {"min_w0_std": 1e-6, "max_abs_value": 1e4, "time_monotonic_violation": 0.0},
        "metrics": metrics,
    }


def advection_sign_test() -> float:
    n = 128
    t = 0.37
    bx = 0.25
    x = np.arange(n, dtype=np.float64) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u0 = np.sin(2.0 * np.pi * xx)
    kx, ky, ksq = periodic_wavenumbers(n)
    coeff = np.fft.fft2(u0)
    evolved = np.fft.ifft2(coeff * np.exp(-1j * bx * kx * t)).real
    expected = np.sin(2.0 * np.pi * (xx - bx * t))
    return float(np.max(np.abs(evolved - expected)))


def read_param_or_attr(h5: h5py.File, dataset_name: str, attr_name: str, n: int, default: float | None = None) -> np.ndarray:
    if dataset_name in h5:
        return h5[dataset_name][:].astype(np.float64)
    if attr_name in h5.attrs:
        return np.full((n,), float(h5.attrs[attr_name]), dtype=np.float64)
    if default is not None:
        return np.full((n,), default, dtype=np.float64)
    raise KeyError(f"Missing {dataset_name!r} dataset or {attr_name!r} attr in {h5.filename}")


def rel_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    require_finite(actual, "actual")
    require_finite(expected, "expected")
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))


def require_keys(h5: h5py.File, keys: list[str]) -> None:
    missing = [key for key in keys if key not in h5]
    if missing:
        raise KeyError(f"Missing keys in {h5.filename}: {missing}")


def require_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")


if __name__ == "__main__":
    main()

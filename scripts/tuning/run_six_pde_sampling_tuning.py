#!/usr/bin/env python3
"""Resumable sampling-parameter tuning for the six endpoint/static PDEs.

The tuner deliberately works from the original ID/smooth/rough datasets rather
than requiring a previous 1,000-sample benchmark.  It screens paired candidate
bundles on tune offsets, selects a task-aware winner, and validates that winner
against the current main config on disjoint holdout offsets.

The main configs are never edited.  Validated recommendations are written below
``<root>/recommended_configs/<profile>/<task>/<pde>.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.specs import TEMPORAL_ENDPOINT_PDES
from sampling.config import dump_yaml, load_config, load_yaml_file


PDES = (
    "advection_diffusion",
    "reaction_diffusion",
    "steady_heat_conduction",
    "heat",
    "shallow_water",
    "wave",
)
TASKS = ("both", "forward", "inverse")
TEST_TYPES = ("id", "smooth", "rough")
PROFILES = ("quick", "standard", "thorough")
CANDIDATE_SETS = ("broad", "refined")

DATA_TEMPLATES = {
    "advection_diffusion": "advection_diffusion/advection_diffusion_test_10000-128-128_{test_type}.h5",
    "reaction_diffusion": (
        "reaction_diffusion/"
        "reaction_diffusion_test_grf_10000-128-128-T1-steps10_{test_type}.h5"
    ),
    "steady_heat_conduction": (
        "steady_heat_conduction/steady_heat_conduction_test_10000-128-128_{test_type}.h5"
    ),
    "heat": "heat/heat_test_10000-128-128_{test_type}.h5",
    "shallow_water": "shallow_water/shallow_water_test_10000-128-128-10_{test_type}.h5",
    "wave": "wave/wave_test_10000-128-128_{test_type}.h5",
}

# These ranges bracket the stable values already used by the inverse-tuning
# scripts.  Forward/both tune the active coefficient and solution observation
# sides at the same PDE-specific scale, instead of retaining placeholder 1.0.
OBS_LEVELS = {
    "advection_diffusion": (1.0e4, 1.0e5, 1.0e6),
    "reaction_diffusion": (1.0e6, 4.0e6, 1.6e7),
    "steady_heat_conduction": (1.0e4, 1.0e5, 1.0e6),
    "heat": (1.0e3, 1.0e4, 1.0e5),
    "shallow_water": (5.0e4, 2.0e5, 8.0e5),
    "wave": (1.0e4, 1.0e5, 1.0e6),
}
PDE_LEVELS = {
    "advection_diffusion": (0.01, 0.1, 1.0),
    "reaction_diffusion": (0.1, 1.0, 10.0),
    "steady_heat_conduction": (0.01, 0.1, 1.0),
    "heat": (0.01, 0.1, 1.0),
    "shallow_water": (0.1, 1.0, 10.0),
    "wave": (0.01, 0.1, 1.0),
}

CANDIDATE_FIELDS = (
    "zeta_obs_a",
    "zeta_obs_u",
    "zeta_pde",
    "clip_mode",
    "clip_threshold",
    "sampler_phase",
    "switch_ratio",
    "time_grid",
    "time_grid_eta",
    "step_method",
    "residual_mode",
    "pde_guidance_start_ratio",
    "pde_guidance_ramp_ratio",
)

PROFILE_DEFAULTS = {
    "quick": {
        "test_types": ("id",),
        "tune_offsets": (0,),
        "holdout_offsets": (1000,),
        "batch_size": 1,
    },
    "standard": {
        "test_types": TEST_TYPES,
        "tune_offsets": (0, 1000),
        "holdout_offsets": (3000, 5000),
        "batch_size": 2,
    },
    "thorough": {
        "test_types": TEST_TYPES,
        "tune_offsets": (0, 1000, 2000),
        "holdout_offsets": (4000, 6000, 8000),
        "batch_size": 2,
    },
}

TUNE_MAX_TEST_MEAN_RATIO = 1.10
TUNE_MAX_AUXILIARY_RATIO = 1.10
TUNE_MAX_PDE_RATIO = 1.50
HOLDOUT_MAX_TEST_MEAN_RATIO = 1.05
HOLDOUT_MAX_AUXILIARY_RATIO = 1.10
HOLDOUT_MAX_PDE_RATIO = 1.50


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def robust_score(values: list[float]) -> float:
    if not values:
        return math.nan
    return 0.5 * statistics.fmean(values) + 0.3 * percentile(values, 0.9) + 0.2 * max(values)


def safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 1.0 if value == 0.0 else math.inf
    return value / baseline


def _parse_distribution_weights(value: str, test_types: list[str]) -> dict[str, float]:
    """Parse ``id=1,smooth=1,rough=2`` and require every selected split."""
    weights: dict[str, float] = {}
    for item in value.replace(" ", ",").split(","):
        if not item:
            continue
        try:
            name, raw_weight = item.split("=", 1)
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid distribution weight: {item!r}") from exc
        name = name.strip()
        if name not in TEST_TYPES or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Invalid distribution weight: {item!r}")
        if name in weights:
            raise ValueError(f"Duplicate distribution weight: {name!r}")
        weights[name] = weight
    missing = [name for name in test_types if name not in weights]
    extra = [name for name in weights if name not in test_types]
    if missing or extra:
        raise ValueError(
            "Distribution weights must match selected test types exactly; "
            f"missing={missing}, extra={extra}"
        )
    return weights


def _parse_names(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    selected = [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]
    invalid = [item for item in selected if item not in allowed]
    if invalid or not selected or len(set(selected)) != len(selected):
        raise ValueError(f"Invalid {label} list: {value!r}")
    return selected


def _parse_offsets(value: str) -> list[int]:
    offsets = [int(item.strip()) for item in value.replace(" ", ",").split(",") if item.strip()]
    if not offsets or any(item < 0 for item in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError(f"Invalid offset list: {value!r}")
    return offsets


def data_path_for(data_root: Path, pde: str, test_type: str) -> Path:
    return data_root / DATA_TEMPLATES[pde].format(test_type=test_type)


def config_path_for(pde: str, task: str) -> Path:
    return REPO_ROOT / "configs" / "main" / task / f"{pde}.yaml"


@lru_cache(maxsize=None)
def baseline_for(pde: str, task: str) -> dict[str, Any]:
    config = load_config(config_path_for(pde, task))
    candidate: dict[str, Any] = {"candidate": "baseline"}
    for field in CANDIDATE_FIELDS:
        candidate[field] = getattr(config, field)
    return candidate


def _active_observation_zetas(task: str, level: float) -> tuple[float, float]:
    if task == "forward":
        return float(level), 0.0
    if task == "inverse":
        return 0.0, float(level)
    return float(level), float(level)


def _stable_candidate(
    pde: str,
    task: str,
    name: str,
    *,
    obs_level: float,
    pde_level: float,
    **updates: Any,
) -> dict[str, Any]:
    zeta_obs_a, zeta_obs_u = _active_observation_zetas(task, obs_level)
    candidate: dict[str, Any] = {
        "candidate": name,
        "zeta_obs_a": zeta_obs_a,
        "zeta_obs_u": zeta_obs_u,
        "zeta_pde": float(pde_level),
        "clip_mode": "global_norm",
        "clip_threshold": 50.0,
        "sampler_phase": "stochastic",
        "switch_ratio": 0.5,
        "time_grid": "uniform",
        "time_grid_eta": 0.4,
        "step_method": "euler",
        "residual_mode": "auto" if pde == "steady_heat_conduction" else "hermite_bridge",
        "pde_guidance_start_ratio": 0.8,
        "pde_guidance_ramp_ratio": 0.1,
    }
    candidate.update(updates)
    return candidate


def _candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(candidate[field] for field in CANDIDATE_FIELDS)


def _scaled_active_observations(
    candidate: dict[str, Any], task: str, factor: float
) -> dict[str, float]:
    updates: dict[str, float] = {}
    if task in {"forward", "both"}:
        updates["zeta_obs_a"] = float(candidate["zeta_obs_a"]) * factor
    if task in {"inverse", "both"}:
        updates["zeta_obs_u"] = float(candidate["zeta_obs_u"]) * factor
    return updates


def refined_candidates_for(pde: str, task: str) -> list[dict[str, Any]]:
    """Generate a compact, numerically conservative grid around the main config.

    The first sweep already identified the useful sampler family for each
    equation/task.  This grid therefore changes one local factor at a time and
    adds one conservative combined candidate instead of repeating the original
    coarse cross-family screen.
    """
    center = dict(baseline_for(pde, task))

    def variant(name: str, **updates: Any) -> dict[str, Any]:
        return {**center, "candidate": f"refine_{name}", **updates}

    candidates: list[dict[str, Any]] = [center]
    for label, factor in (("obs_x0p5", 0.5), ("obs_x2", 2.0)):
        candidates.append(variant(label, **_scaled_active_observations(center, task, factor)))

    if task == "both":
        candidates.extend(
            [
                variant("obs_a_x2", zeta_obs_a=float(center["zeta_obs_a"]) * 2.0),
                variant("obs_u_x2", zeta_obs_u=float(center["zeta_obs_u"]) * 2.0),
            ]
        )

    center_pde = float(center["zeta_pde"])
    pde_anchor = float(PDE_LEVELS[pde][1])
    candidates.append(variant("pde_off", zeta_pde=0.0))
    candidates.append(
        variant("pde_low", zeta_pde=0.25 * (center_pde if center_pde > 0.0 else pde_anchor))
    )
    candidates.extend(
        [
            variant("clip20", clip_mode="global_norm", clip_threshold=20.0),
            variant(
                "late_pde",
                pde_guidance_start_ratio=0.9,
                pde_guidance_ramp_ratio=0.05,
            ),
        ]
    )

    phase = str(center["sampler_phase"])
    if phase == "hybrid_s2d":
        candidates.extend(
            [
                variant("switch0p25", switch_ratio=0.25),
                variant("stochastic", sampler_phase="stochastic"),
            ]
        )
    else:
        candidates.extend(
            [
                variant("hybrid0p25", sampler_phase="hybrid_s2d", switch_ratio=0.25),
                variant("hybrid0p5", sampler_phase="hybrid_s2d", switch_ratio=0.5),
            ]
        )

    if str(center["time_grid"]) == "geometric":
        candidates.append(variant("grid_eta0p2", time_grid_eta=0.2))
    else:
        candidates.append(
            variant("geometric_eta0p2", time_grid="geometric", time_grid_eta=0.2)
        )

    alternate_step = "euler" if str(center["step_method"]) == "midpoint" else "midpoint"
    candidates.append(variant(alternate_step, step_method=alternate_step))
    if pde in TEMPORAL_ENDPOINT_PDES:
        alternate_residual = (
            "hermite_bridge"
            if str(center["residual_mode"]) == "near_endpoint_temporal"
            else "near_endpoint_temporal"
        )
        candidates.append(variant("alternate_residual", residual_mode=alternate_residual))

    conservative_pde = 0.25 * (center_pde if center_pde > 0.0 else pde_anchor)
    candidates.append(
        variant(
            "conservative",
            **_scaled_active_observations(center, task, 0.5),
            zeta_pde=conservative_pde,
            clip_mode="global_norm",
            clip_threshold=20.0,
            pde_guidance_start_ratio=0.9,
            pde_guidance_ramp_ratio=0.05,
        )
    )

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        signature = _candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        deduplicated.append(candidate)
    return deduplicated


def candidates_for(
    pde: str,
    task: str,
    profile: str = "standard",
    candidate_set: str = "broad",
) -> list[dict[str, Any]]:
    """Return a compact, PDE-scaled bundle grid with the exact baseline first."""
    if candidate_set == "refined":
        return refined_candidates_for(pde, task)
    if candidate_set != "broad":
        raise ValueError(f"Unknown candidate_set={candidate_set!r}")
    low_obs, mid_obs, high_obs = OBS_LEVELS[pde]
    low_pde, mid_pde, high_pde = PDE_LEVELS[pde]
    candidates = [
        dict(baseline_for(pde, task)),
        _stable_candidate(pde, task, "obs_low", obs_level=low_obs, pde_level=0.0),
        _stable_candidate(pde, task, "obs_mid", obs_level=mid_obs, pde_level=0.0),
        _stable_candidate(pde, task, "obs_high", obs_level=high_obs, pde_level=0.0),
        _stable_candidate(
            pde, task, "obs_mid_pde_low", obs_level=mid_obs, pde_level=low_pde
        ),
        _stable_candidate(
            pde, task, "obs_mid_pde_mid", obs_level=mid_obs, pde_level=mid_pde
        ),
        _stable_candidate(
            pde, task, "obs_mid_pde_high", obs_level=mid_obs, pde_level=high_pde
        ),
    ]
    if profile in {"standard", "thorough"}:
        candidates.extend(
            [
                _stable_candidate(
                    pde,
                    task,
                    "deterministic",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    sampler_phase="deterministic",
                ),
                _stable_candidate(
                    pde,
                    task,
                    "hybrid_s2d",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    sampler_phase="hybrid_s2d",
                    switch_ratio=0.5,
                ),
                _stable_candidate(
                    pde,
                    task,
                    "geometric_grid",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    time_grid="geometric",
                    time_grid_eta=0.4,
                ),
                _stable_candidate(
                    pde,
                    task,
                    "midpoint",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    step_method="midpoint",
                ),
            ]
        )
        if pde in TEMPORAL_ENDPOINT_PDES:
            candidates.append(
                _stable_candidate(
                    pde,
                    task,
                    "near_endpoint",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    residual_mode="near_endpoint_temporal",
                )
            )
    if profile == "thorough":
        candidates.extend(
            [
                _stable_candidate(
                    pde,
                    task,
                    "obs_high_pde_mid",
                    obs_level=high_obs,
                    pde_level=mid_pde,
                ),
                _stable_candidate(
                    pde,
                    task,
                    "clip_20",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    clip_threshold=20.0,
                ),
                _stable_candidate(
                    pde,
                    task,
                    "clip_100",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    clip_threshold=100.0,
                ),
                _stable_candidate(
                    pde,
                    task,
                    "gate_0p6",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    pde_guidance_start_ratio=0.6,
                    pde_guidance_ramp_ratio=0.2,
                ),
                _stable_candidate(
                    pde,
                    task,
                    "hybrid_d2s",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    sampler_phase="hybrid_d2s",
                    switch_ratio=0.5,
                ),
                _stable_candidate(
                    pde,
                    task,
                    "cosine_grid",
                    obs_level=mid_obs,
                    pde_level=mid_pde,
                    time_grid="cosine",
                ),
            ]
        )

    # A provisional inverse winner can be identical to an explicit grid point.
    # Keep the named baseline and avoid spending GPU time on duplicate bundles.
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        signature = _candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        deduplicated.append(candidate)
    return deduplicated


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def inference_checkpoint_path(root: Path, pde: str) -> Path:
    return root / "checkpoints" / f"{pde}_inference.pth"


def prepare_inputs(
    root: Path,
    data_root: Path,
    pdes: list[str],
    tasks: list[str],
    test_types: list[str],
) -> None:
    missing: list[Path] = []
    for pde in pdes:
        for task in tasks:
            config_path = config_path_for(pde, task)
            if not config_path.is_file():
                missing.append(config_path)
        for test_type in test_types:
            path = data_path_for(data_root, pde, test_type)
            if not path.is_file():
                missing.append(path)
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing tuning inputs:\n"
            f"{joined}\nSet PDE_DATA_ROOT to the directory containing the six PDE subdirectories."
        )

    root.mkdir(parents=True, exist_ok=True)
    for pde in pdes:
        destination = inference_checkpoint_path(root, pde)
        if destination.is_file():
            print(f"SKIP checkpoint {destination}", flush=True)
            continue
        source_config = load_config(config_path_for(pde, tasks[0]))
        source = Path(source_config.checkpoint_path)
        if not source.is_absolute():
            source = REPO_ROOT / source
        if not source.is_file():
            raise FileNotFoundError(source)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "tuning" / "make_inference_checkpoint.py"),
            str(source.resolve()),
            str(destination.resolve()),
        ]
        if pde in {"shallow_water", "wave"}:
            command.append("--gradient-checkpointing")
        print(f"PREPARE {pde} inference checkpoint", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _find_artifact(job_root: Path, filename: str) -> Path | None:
    paths = sorted(job_root.rglob(filename), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return paths[0] if paths else None


def run_job(
    root: Path,
    *,
    phase: str,
    profile: str,
    pde: str,
    task: str,
    candidate: dict[str, Any],
    test_type: str,
    offset: int,
    batch_size: int,
    num_steps: int,
    sample_seed: int,
    mask_seed: int,
    data_root: Path,
    checkpoint: Path,
    device: str,
    resume: bool,
    retry_non_finite: bool,
    checkpoint_bundle: tuple[Any, Any, dict[str, Any]],
) -> dict[str, Any]:
    name = str(candidate["candidate"])
    job_root = (
        root
        / "runs"
        / phase
        / profile
        / pde
        / task
        / name
        / test_type
        / f"offset-{offset}-batch-{batch_size}"
    )
    status_path = job_root / "job.json"
    metrics_path = job_root / "missing"
    config_path = config_path_for(pde, task).resolve()
    data_path = data_path_for(data_root, pde, test_type).resolve()
    effective_sample_seed = sample_seed + offset
    effective_mask_seed = mask_seed + offset
    signature = {
        "phase": phase,
        "profile": profile,
        "pde": pde,
        "task": task,
        "candidate": name,
        "test_type": test_type,
        "offset": offset,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "sample_seed": effective_sample_seed,
        "mask_seed": effective_mask_seed,
        "parameters": {field: candidate[field] for field in CANDIDATE_FIELDS},
        "config": _file_fingerprint(config_path),
        "checkpoint": _file_fingerprint(checkpoint),
        "data": _file_fingerprint(data_path),
    }
    if resume and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        metrics_path = Path(str(status.get("metrics_path", "")))
        prior_final_path = Path(str(status.get("final_path", "")))
        artifacts_valid = False
        if metrics_path.is_file() and prior_final_path.is_file():
            try:
                prior_rows = read_csv(metrics_path)
                prior_final = json.loads(prior_final_path.read_text(encoding="utf-8"))
                artifacts_valid = (
                    len(prior_rows) == batch_size
                    and all(_row_metrics_are_finite(row) for row in prior_rows)
                    and _final_losses_are_finite(prior_final)
                    and prior_final.get("status") == "ok"
                )
            except (json.JSONDecodeError, OSError):
                artifacts_valid = False
        if (
            status.get("status") == "ok"
            and status.get("job_signature") == signature
            and artifacts_valid
        ):
            print(
                f"SKIP {phase} {pde}/{task} {name} {test_type} "
                f"offset={offset} batch={batch_size}",
                flush=True,
            )
            return status
        if (
            not retry_non_finite
            and status.get("status") == "failed"
            and status.get("failure_kind") == "non_finite_metrics"
            and status.get("job_signature") == signature
        ):
            print(
                f"SKIP known-nonfinite {phase} {pde}/{task} {name} "
                f"{test_type} offset={offset}",
                flush=True,
            )
            return status
        print(f"STALE {phase} {pde}/{task} {name} {test_type} offset={offset}", flush=True)

    job_root.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, Any] = {
        "data_path": str(data_path),
        "checkpoint_path": str(checkpoint.resolve()),
        "test_type": test_type,
        "output_dir": str(job_root.resolve()),
        "ablation_group": f"six_pde_{profile}_{phase}_{name}_{test_type}",
        "task": task,
        "batch_size": batch_size,
        "offset": offset,
        "num_steps": num_steps,
        "sample_seed": effective_sample_seed,
        "mask_seed": effective_mask_seed,
        "initial_noise_source_batch_size": batch_size,
        "initial_noise_source_indices": list(range(batch_size)),
        "device": device,
        "save_plots": False,
        "save_intermediate": False,
        "save_per_sample_curves": False,
    }
    overrides.update({field: candidate[field] for field in CANDIDATE_FIELDS})
    log_path = job_root / "runner.log"
    print(
        f"RUN  {phase} {pde}/{task} {name} {test_type} offset={offset} batch={batch_size}",
        flush=True,
    )
    started = time.time()
    error_message = ""
    returncode = 0
    try:
        from sampling.runner import run_single_ablation

        config = load_config(config_path, overrides=overrides)
        with log_path.open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                run_single_ablation(config, checkpoint_bundle=checkpoint_bundle)
    except Exception as exc:
        returncode = 1
        error_message = f"{type(exc).__name__}: {exc}"
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
    elapsed = time.time() - started
    metrics_path = _find_artifact(job_root, "metrics_per_sample.csv") or job_root / "missing"
    final_path = _find_artifact(job_root, "metrics_final.json") or job_root / "missing"
    valid_rows = []
    if metrics_path.is_file():
        valid_rows = read_csv(metrics_path)
    final_payload: dict[str, Any] = {}
    if final_path.is_file():
        try:
            final_payload = json.loads(final_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            final_payload = {}
    finite = len(valid_rows) == batch_size and all(
        _row_metrics_are_finite(row) for row in valid_rows
    )
    finite_losses = _final_losses_are_finite(final_payload)
    runner_ok = final_payload.get("status") == "ok"
    normalized_error = error_message.lower()
    numerical_error = any(
        marker in normalized_error
        for marker in ("non-finite", "nonfinite", "nan", "infinity", "overflow")
    )
    if returncode != 0:
        failure_kind = "non_finite_metrics" if numerical_error else "runner_exception"
    elif not metrics_path.is_file() or not final_path.is_file():
        failure_kind = "missing_artifacts"
    elif not finite or not finite_losses:
        failure_kind = "non_finite_metrics"
    elif not runner_ok:
        failure_kind = f"runner_status_{final_payload.get('status', 'missing')}"
    else:
        failure_kind = ""
    status = {
        "status": (
            "ok" if returncode == 0 and runner_ok and finite and finite_losses else "failed"
        ),
        "runner_status": final_payload.get("status", "missing"),
        "returncode": returncode,
        "error_message": error_message,
        "failure_kind": failure_kind,
        "phase": phase,
        "profile": profile,
        "pde": pde,
        "task": task,
        "candidate": name,
        "test_type": test_type,
        "offset": offset,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "metrics_path": str(metrics_path.resolve()),
        "final_path": str(final_path.resolve()),
        "log_path": str(log_path.resolve()),
        "elapsed_seconds": elapsed,
        "job_signature": signature,
    }
    _atomic_write_json(status_path, status)
    if status["status"] == "ok":
        print(f"DONE {pde}/{task} {name} {test_type} elapsed={elapsed:.1f}s", flush=True)
    else:
        tail = ""
        if log_path.is_file():
            tail = " | ".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-3:])
        print(
            f"FAIL {phase} {pde}/{task} {name} {test_type} offset={offset}: "
            f"{error_message or 'missing/non-finite artifacts'} {tail}",
            file=sys.stderr,
            flush=True,
        )
    return status


def _row_metrics_are_finite(row: dict[str, str]) -> bool:
    try:
        return all(
            math.isfinite(float(row[field]))
            for field in ("rel_l2_a", "rel_l2_u", "pde_residual_norm")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _final_losses_are_finite(payload: dict[str, Any]) -> bool:
    try:
        return all(
            math.isfinite(float(payload[field]))
            for field in ("L_obs_a", "L_obs_u", "L_pde")
        )
    except (KeyError, TypeError, ValueError):
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Some candidates cannot be paired with a complete baseline and therefore
    # omit the derived ``overall_*`` and per-distribution ratio fields.  Build
    # the schema from every row so their position in the list cannot make CSV
    # serialization fail after a long tuning run.
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _analysis_path(
    root: Path,
    stem: str,
    pdes: list[str],
    tasks: list[str],
    profile: str,
    suffix: str = ".csv",
) -> Path:
    pde_suffix = "" if pdes == list(PDES) else "_" + "_".join(pdes)
    task_suffix = "" if tasks == list(TASKS) else "_" + "_".join(tasks)
    return root / f"{stem}{pde_suffix}{task_suffix}_{profile}{suffix}"


def collect_rows(
    root: Path,
    *,
    phase: str,
    profile: str,
    pdes: list[str],
    tasks: list[str],
    allowed_candidates: dict[tuple[str, str], set[str]],
    test_types: list[str],
    offsets: list[int],
    batch_size: int,
    num_steps: int,
) -> list[dict[str, Any]]:
    collected: dict[tuple[str, str, str, str, int], tuple[int, dict[str, Any]]] = {}
    phase_root = root / "runs" / phase / profile
    if not phase_root.is_dir():
        return []
    for status_path in sorted(phase_root.rglob("job.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        signature = status.get("job_signature", {})
        pde = str(status.get("pde", ""))
        task = str(status.get("task", ""))
        candidate = str(status.get("candidate", ""))
        test_type = str(status.get("test_type", ""))
        offset = int(status.get("offset", -1))
        if (
            status.get("status") != "ok"
            or pde not in pdes
            or task not in tasks
            or candidate not in allowed_candidates.get((pde, task), set())
            or test_type not in test_types
            or offset not in offsets
            or int(status.get("batch_size", -1)) != batch_size
            or int(status.get("num_steps", -1)) != num_steps
            or signature.get("profile") != profile
        ):
            continue
        metrics_path = Path(str(status.get("metrics_path", "")))
        final_path = Path(str(status.get("final_path", "")))
        if not metrics_path.is_file():
            fallback = _find_artifact(status_path.parent, "metrics_per_sample.csv")
            if fallback is None:
                continue
            metrics_path = fallback
        if not final_path.is_file():
            fallback = _find_artifact(status_path.parent, "metrics_final.json")
            if fallback is None:
                continue
            final_path = fallback
        try:
            final_metrics = json.loads(final_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not _final_losses_are_finite(final_metrics):
            continue
        parameters = signature.get("parameters", {})
        for row in read_csv(metrics_path):
            if not _row_metrics_are_finite(row):
                continue
            sample_offset = offset + int(float(row["sample_index"]))
            logical_key = (pde, task, candidate, test_type, sample_offset)
            collected[logical_key] = (
                status_path.stat().st_mtime_ns,
                {
                    "phase": phase,
                    "profile": profile,
                    "pde": pde,
                    "task": task,
                    "candidate": candidate,
                    "test_type": test_type,
                    "sample_offset": sample_offset,
                    "sample_id": row.get("sample_id", str(sample_offset)),
                    "rel_l2_a": float(row["rel_l2_a"]),
                    "rel_l2_u": float(row["rel_l2_u"]),
                    "pde_residual_norm": float(row["pde_residual_norm"]),
                    # The runner evaluates losses as a batch mean.  Repeating
                    # that value on each member keeps weighting correct because
                    # every tuning job uses the same batch size.
                    "obs_loss_a_batch_mean": float(final_metrics["L_obs_a"]),
                    "obs_loss_u_batch_mean": float(final_metrics["L_obs_u"]),
                    "pde_loss_batch_mean": float(final_metrics["L_pde"]),
                    **{field: parameters[field] for field in CANDIDATE_FIELDS},
                    "metrics_path": str(metrics_path.resolve()),
                },
            )
    return [collected[key][1] for key in sorted(collected)]


def collect_failures(
    root: Path,
    *,
    phase: str,
    profile: str,
    pdes: list[str],
    tasks: list[str],
    allowed_candidates: dict[tuple[str, str], set[str]],
    test_types: list[str],
    offsets: list[int],
    batch_size: int,
    num_steps: int,
) -> list[dict[str, Any]]:
    """Return an inspectable inventory of failed jobs in the active plan."""
    failures: list[dict[str, Any]] = []
    phase_root = root / "runs" / phase / profile
    if not phase_root.is_dir():
        return failures
    for status_path in sorted(phase_root.rglob("job.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        pde = str(status.get("pde", ""))
        task = str(status.get("task", ""))
        candidate = str(status.get("candidate", ""))
        test_type = str(status.get("test_type", ""))
        offset = int(status.get("offset", -1))
        if (
            status.get("status") != "failed"
            or pde not in pdes
            or task not in tasks
            or candidate not in allowed_candidates.get((pde, task), set())
            or test_type not in test_types
            or offset not in offsets
            or int(status.get("batch_size", -1)) != batch_size
            or int(status.get("num_steps", -1)) != num_steps
        ):
            continue
        failures.append(
            {
                "phase": phase,
                "profile": profile,
                "pde": pde,
                "task": task,
                "candidate": candidate,
                "test_type": test_type,
                "offset": offset,
                "batch_size": batch_size,
                "failure_kind": status.get("failure_kind", "legacy_unclassified"),
                "runner_status": status.get("runner_status", ""),
                "error_message": status.get("error_message", ""),
                "log_path": status.get("log_path", ""),
                "status_path": str(status_path.resolve()),
            }
        )
    return failures


def _primary_values(rows: list[dict[str, Any]], task: str) -> list[float]:
    if task == "forward":
        return [float(row["rel_l2_u"]) for row in rows]
    if task == "inverse":
        return [float(row["rel_l2_a"]) for row in rows]
    return [max(float(row["rel_l2_a"]), float(row["rel_l2_u"])) for row in rows]


def _scope_ratios(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    task: str,
) -> dict[str, float]:
    key = lambda row: (str(row["test_type"]), int(row["sample_offset"]))
    candidate_map = {key(row): row for row in candidate_rows}
    baseline_map = {key(row): row for row in baseline_rows}
    if candidate_map.keys() != baseline_map.keys() or not candidate_map:
        raise RuntimeError("Candidate and baseline scopes are not paired")
    candidates = [candidate_map[item] for item in sorted(candidate_map)]
    baselines = [baseline_map[item] for item in sorted(baseline_map)]
    candidate_primary = _primary_values(candidates, task)
    baseline_primary = _primary_values(baselines, task)
    ca = [float(row["rel_l2_a"]) for row in candidates]
    ba = [float(row["rel_l2_a"]) for row in baselines]
    cu = [float(row["rel_l2_u"]) for row in candidates]
    bu = [float(row["rel_l2_u"]) for row in baselines]
    cp = [float(row["pde_residual_norm"]) for row in candidates]
    bp = [float(row["pde_residual_norm"]) for row in baselines]
    coa = [float(row.get("obs_loss_a_batch_mean", row["rel_l2_a"])) for row in candidates]
    boa = [float(row.get("obs_loss_a_batch_mean", row["rel_l2_a"])) for row in baselines]
    cou = [float(row.get("obs_loss_u_batch_mean", row["rel_l2_u"])) for row in candidates]
    bou = [float(row.get("obs_loss_u_batch_mean", row["rel_l2_u"])) for row in baselines]
    cpl = [float(row.get("pde_loss_batch_mean", row["pde_residual_norm"])) for row in candidates]
    bpl = [float(row.get("pde_loss_batch_mean", row["pde_residual_norm"])) for row in baselines]
    if task == "forward":
        auxiliary_ratio = safe_ratio(statistics.fmean(ca), statistics.fmean(ba))
    elif task == "inverse":
        auxiliary_ratio = safe_ratio(statistics.fmean(cu), statistics.fmean(bu))
    else:
        auxiliary_ratio = max(
            safe_ratio(statistics.fmean(ca), statistics.fmean(ba)),
            safe_ratio(statistics.fmean(cu), statistics.fmean(bu)),
        )
    return {
        "n": float(len(candidates)),
        "primary_mean_ratio": safe_ratio(
            statistics.fmean(candidate_primary), statistics.fmean(baseline_primary)
        ),
        "primary_p90_ratio": safe_ratio(
            percentile(candidate_primary, 0.9), percentile(baseline_primary, 0.9)
        ),
        "primary_robust_ratio": safe_ratio(
            robust_score(candidate_primary), robust_score(baseline_primary)
        ),
        "rel_l2_a_mean_ratio": safe_ratio(statistics.fmean(ca), statistics.fmean(ba)),
        "rel_l2_u_mean_ratio": safe_ratio(statistics.fmean(cu), statistics.fmean(bu)),
        "pde_residual_ratio": safe_ratio(statistics.fmean(cp), statistics.fmean(bp)),
        "obs_loss_a_ratio": safe_ratio(statistics.fmean(coa), statistics.fmean(boa)),
        "obs_loss_u_ratio": safe_ratio(statistics.fmean(cou), statistics.fmean(bou)),
        "pde_loss_ratio": safe_ratio(statistics.fmean(cpl), statistics.fmean(bpl)),
        "auxiliary_error_ratio": auxiliary_ratio,
        "candidate_primary_mean": statistics.fmean(candidate_primary),
        "baseline_primary_mean": statistics.fmean(baseline_primary),
    }


def _distribution_weighted_score(
    rows: list[dict[str, Any]],
    task: str,
    test_types: list[str],
    distribution_weights: dict[str, float],
) -> float:
    weighted_total = 0.0
    total_weight = 0.0
    for test_type in test_types:
        values = _primary_values(_filter_test(rows, test_type), task)
        if not values:
            return math.inf
        weight = distribution_weights[test_type]
        weighted_total += weight * robust_score(values)
        total_weight += weight
    return weighted_total / total_weight


def _filter_test(rows: list[dict[str, Any]], test_type: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["test_type"] == test_type]


def summarize_tune(
    rows: list[dict[str, Any]],
    *,
    pdes: list[str],
    tasks: list[str],
    candidate_map: dict[tuple[str, str], list[dict[str, Any]]],
    test_types: list[str],
    expected_n: int,
    distribution_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    distribution_weights = distribution_weights or {name: 1.0 for name in test_types}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["pde"]), str(row["task"]), str(row["candidate"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for pde in pdes:
        for task in tasks:
            baseline_rows = grouped.get((pde, task, "baseline"), [])
            baseline_complete = len(baseline_rows) == expected_n
            for candidate in candidate_map[(pde, task)]:
                name = str(candidate["candidate"])
                candidate_rows = grouped.get((pde, task, name), [])
                if len(candidate_rows) != expected_n:
                    continue
                primary = _primary_values(candidate_rows, task)
                rel_a = [float(row["rel_l2_a"]) for row in candidate_rows]
                rel_u = [float(row["rel_l2_u"]) for row in candidate_rows]
                pde_values = [float(row["pde_residual_norm"]) for row in candidate_rows]
                obs_loss_a = [
                    float(row.get("obs_loss_a_batch_mean", row["rel_l2_a"]))
                    for row in candidate_rows
                ]
                obs_loss_u = [
                    float(row.get("obs_loss_u_batch_mean", row["rel_l2_u"]))
                    for row in candidate_rows
                ]
                pde_loss = [
                    float(row.get("pde_loss_batch_mean", row["pde_residual_norm"]))
                    for row in candidate_rows
                ]
                summary: dict[str, Any] = {
                    "pde": pde,
                    "task": task,
                    "candidate": name,
                    **{field: candidate[field] for field in CANDIDATE_FIELDS},
                    "n": len(candidate_rows),
                    "primary_mean": statistics.fmean(primary),
                    "primary_p90": percentile(primary, 0.9),
                    "primary_max": max(primary),
                    "selection_score": _distribution_weighted_score(
                        candidate_rows, task, test_types, distribution_weights
                    ),
                    "distribution_weights": ",".join(
                        f"{name}={distribution_weights[name]:g}" for name in test_types
                    ),
                    "rel_l2_a_mean": statistics.fmean(rel_a),
                    "rel_l2_u_mean": statistics.fmean(rel_u),
                    "pde_residual_mean": statistics.fmean(pde_values),
                    "obs_loss_a_mean": statistics.fmean(obs_loss_a),
                    "obs_loss_u_mean": statistics.fmean(obs_loss_u),
                    "pde_loss_mean": statistics.fmean(pde_loss),
                    "baseline_complete": baseline_complete,
                    "passes_guardrails": True,
                }
                if baseline_complete:
                    overall = _scope_ratios(candidate_rows, baseline_rows, task)
                    summary.update({f"overall_{key}": value for key, value in overall.items()})
                    passes = overall["auxiliary_error_ratio"] <= TUNE_MAX_AUXILIARY_RATIO
                    passes = passes and overall["pde_residual_ratio"] <= TUNE_MAX_PDE_RATIO
                    if task == "both":
                        passes = passes and overall["rel_l2_a_mean_ratio"] <= TUNE_MAX_TEST_MEAN_RATIO
                        passes = passes and overall["rel_l2_u_mean_ratio"] <= TUNE_MAX_TEST_MEAN_RATIO
                    for test_type in test_types:
                        scoped = _scope_ratios(
                            _filter_test(candidate_rows, test_type),
                            _filter_test(baseline_rows, test_type),
                            task,
                        )
                        summary[f"primary_mean_ratio_{test_type}"] = scoped["primary_mean_ratio"]
                        passes = passes and scoped["primary_mean_ratio"] <= TUNE_MAX_TEST_MEAN_RATIO
                    summary["passes_guardrails"] = passes or name == "baseline"
                summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            pdes.index(str(row["pde"])),
            tasks.index(str(row["task"])),
            float(row["selection_score"]),
        )
    )
    return summaries


def select_tune_winners(
    summaries: list[dict[str, Any]], pdes: list[str], tasks: list[str]
) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for pde in pdes:
        for task in tasks:
            candidates = [
                row
                for row in summaries
                if row["pde"] == pde
                and row["task"] == task
                and bool(row["passes_guardrails"])
            ]
            if not candidates:
                raise RuntimeError(f"No complete finite tuning candidate for {pde}/{task}")
            winner = dict(min(candidates, key=lambda row: float(row["selection_score"])))
            winners.append(winner)
            print(
                f"SELECT {pde}/{task} {winner['candidate']} "
                f"score={float(winner['selection_score']):.6g} "
                f"mean={float(winner['primary_mean']):.6g}",
                flush=True,
            )
    return winners


def _candidate_map(
    pdes: list[str], tasks: list[str], profile: str, candidate_set: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    return {
        (pde, task): candidates_for(pde, task, profile, candidate_set)
        for pde in pdes
        for task in tasks
    }


def analyze_tune(
    root: Path,
    *,
    profile: str,
    pdes: list[str],
    tasks: list[str],
    test_types: list[str],
    offsets: list[int],
    batch_size: int,
    num_steps: int,
    candidate_set: str = "broad",
    distribution_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    candidate_map = _candidate_map(pdes, tasks, profile, candidate_set)
    allowed = {
        key: {str(candidate["candidate"]) for candidate in candidates}
        for key, candidates in candidate_map.items()
    }
    rows = collect_rows(
        root,
        phase="tune",
        profile=profile,
        pdes=pdes,
        tasks=tasks,
        allowed_candidates=allowed,
        test_types=test_types,
        offsets=offsets,
        batch_size=batch_size,
        num_steps=num_steps,
    )
    failures = collect_failures(
        root,
        phase="tune",
        profile=profile,
        pdes=pdes,
        tasks=tasks,
        allowed_candidates=allowed,
        test_types=test_types,
        offsets=offsets,
        batch_size=batch_size,
        num_steps=num_steps,
    )
    expected_n = len(test_types) * len(offsets) * batch_size
    summaries = summarize_tune(
        rows,
        pdes=pdes,
        tasks=tasks,
        candidate_map=candidate_map,
        test_types=test_types,
        expected_n=expected_n,
        distribution_weights=distribution_weights,
    )
    winners = select_tune_winners(summaries, pdes, tasks)
    write_csv(_analysis_path(root, "tune_per_sample", pdes, tasks, profile), rows)
    write_csv(_analysis_path(root, "stability_failures", pdes, tasks, profile), failures)
    write_csv(_analysis_path(root, "tune_summary", pdes, tasks, profile), summaries)
    selected_csv = _analysis_path(root, "selected_params", pdes, tasks, profile)
    write_csv(selected_csv, winners)
    _analysis_path(root, "selected_params", pdes, tasks, profile, ".json").write_text(
        json.dumps(winners, indent=2), encoding="utf-8"
    )
    return winners


def _selected_candidates(winners: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["pde"]), str(row["task"])): {
            "candidate": str(row["candidate"]),
            **{field: row[field] for field in CANDIDATE_FIELDS},
        }
        for row in winners
    }


def _write_recommended_config(
    root: Path,
    data_root: Path,
    profile: str,
    pde: str,
    task: str,
    candidate: dict[str, Any],
) -> Path:
    payload = load_yaml_file(config_path_for(pde, task))
    for field in CANDIDATE_FIELDS:
        payload[field] = candidate[field]
    payload["checkpoint_path"] = str(inference_checkpoint_path(root, pde).resolve())
    payload["data_paths"] = {
        test_type: str(data_path_for(data_root, pde, test_type).resolve())
        for test_type in TEST_TYPES
    }
    payload["data_path"] = payload["data_paths"].get(
        str(payload.get("test_type", "id")), payload["data_paths"]["id"]
    )
    destination = root / "recommended_configs" / profile / task / f"{pde}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dump_yaml(payload), encoding="utf-8")
    return destination


def analyze_holdout(
    root: Path,
    *,
    data_root: Path,
    profile: str,
    pdes: list[str],
    tasks: list[str],
    test_types: list[str],
    offsets: list[int],
    batch_size: int,
    num_steps: int,
    winners: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = _selected_candidates(winners)
    candidate_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pde in pdes:
        for task in tasks:
            candidates = [dict(baseline_for(pde, task))]
            winner = selected[(pde, task)]
            if winner["candidate"] != "baseline":
                candidates.append(winner)
            candidate_map[(pde, task)] = candidates
    allowed = {
        key: {str(candidate["candidate"]) for candidate in candidates}
        for key, candidates in candidate_map.items()
    }
    rows = collect_rows(
        root,
        phase="holdout",
        profile=profile,
        pdes=pdes,
        tasks=tasks,
        allowed_candidates=allowed,
        test_types=test_types,
        offsets=offsets,
        batch_size=batch_size,
        num_steps=num_steps,
    )
    write_csv(_analysis_path(root, "holdout_per_sample", pdes, tasks, profile), rows)
    expected_n = len(test_types) * len(offsets) * batch_size
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["pde"]), str(row["task"]), str(row["candidate"]))].append(row)
    comparisons: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    for pde in pdes:
        for task in tasks:
            winner = selected[(pde, task)]
            name = str(winner["candidate"])
            baseline_rows = grouped.get((pde, task, "baseline"), [])
            winner_rows = grouped.get((pde, task, name), [])
            baseline_complete = len(baseline_rows) == expected_n
            winner_complete = len(winner_rows) == expected_n
            validated = False
            reason = ""
            overall: dict[str, float] = {}
            if name == "baseline" and baseline_complete:
                validated = True
                reason = "tuning_selected_baseline"
                winner_rows = baseline_rows
            elif winner_complete and not baseline_complete:
                validated = True
                reason = "baseline_failed_but_winner_completed_finite_holdout"
            elif winner_complete and baseline_complete:
                overall = _scope_ratios(winner_rows, baseline_rows, task)
                validated = overall["primary_robust_ratio"] <= 1.0
                validated = validated and overall["auxiliary_error_ratio"] <= HOLDOUT_MAX_AUXILIARY_RATIO
                validated = validated and overall["pde_residual_ratio"] <= HOLDOUT_MAX_PDE_RATIO
                if task == "both":
                    validated = validated and overall["rel_l2_a_mean_ratio"] <= HOLDOUT_MAX_TEST_MEAN_RATIO
                    validated = validated and overall["rel_l2_u_mean_ratio"] <= HOLDOUT_MAX_TEST_MEAN_RATIO
                for test_type in test_types:
                    scoped = _scope_ratios(
                        _filter_test(winner_rows, test_type),
                        _filter_test(baseline_rows, test_type),
                        task,
                    )
                    validated = validated and (
                        scoped["primary_mean_ratio"] <= HOLDOUT_MAX_TEST_MEAN_RATIO
                    )
                reason = "paired_holdout_guardrails_passed" if validated else "paired_holdout_guardrails_failed"
            else:
                reason = "winner_holdout_incomplete"

            recommended = winner if validated else dict(baseline_for(pde, task))
            config_path = _write_recommended_config(
                root, data_root, profile, pde, task, recommended
            )
            recommendation = {
                "pde": pde,
                "task": task,
                "selected_candidate": name,
                "holdout_validated": validated,
                "validation_reason": reason,
                "recommended_candidate": recommended["candidate"],
                **{field: recommended[field] for field in CANDIDATE_FIELDS},
                "recommended_config": str(config_path.resolve()),
            }
            if overall:
                recommendation.update(
                    {
                        "holdout_primary_robust_ratio": overall["primary_robust_ratio"],
                        "holdout_primary_mean_ratio": overall["primary_mean_ratio"],
                        "holdout_pde_residual_ratio": overall["pde_residual_ratio"],
                        "holdout_obs_loss_a_ratio": overall["obs_loss_a_ratio"],
                        "holdout_obs_loss_u_ratio": overall["obs_loss_u_ratio"],
                        "holdout_pde_loss_ratio": overall["pde_loss_ratio"],
                    }
                )
            recommendations.append(recommendation)

            if baseline_complete and winner_complete:
                for test_type in ("all", *test_types):
                    baseline_scope = baseline_rows if test_type == "all" else _filter_test(baseline_rows, test_type)
                    winner_scope = winner_rows if test_type == "all" else _filter_test(winner_rows, test_type)
                    stats = _scope_ratios(winner_scope, baseline_scope, task)
                    comparisons.append(
                        {
                            "pde": pde,
                            "task": task,
                            "winner": name,
                            "test_type": test_type,
                            "n": int(stats["n"]),
                            **stats,
                            "holdout_validated": validated,
                            "validation_reason": reason,
                        }
                    )
            print(
                f"RECOMMEND {pde}/{task} {recommended['candidate']} "
                f"selected={name} validated={validated} reason={reason}",
                flush=True,
            )
    write_csv(_analysis_path(root, "holdout_comparison", pdes, tasks, profile), comparisons)
    recommended_csv = _analysis_path(root, "recommended_params", pdes, tasks, profile)
    write_csv(recommended_csv, recommendations)
    _analysis_path(root, "recommended_params", pdes, tasks, profile, ".json").write_text(
        json.dumps(recommendations, indent=2), encoding="utf-8"
    )
    return recommendations


def run_phase(
    root: Path,
    *,
    phase: str,
    profile: str,
    pdes: list[str],
    tasks: list[str],
    test_types: list[str],
    offsets: list[int],
    batch_size: int,
    num_steps: int,
    sample_seed: int,
    mask_seed: int,
    data_root: Path,
    device: str,
    resume: bool,
    winners: list[dict[str, Any]] | None = None,
    candidate_set: str = "broad",
    retry_non_finite: bool = True,
) -> int:
    import torch
    from sampling.model_io import load_fm4pde_checkpoint_bundle

    selected = {} if winners is None else _selected_candidates(winners)
    failures = 0
    for pde in pdes:
        checkpoint = inference_checkpoint_path(root, pde)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing {checkpoint}; run phase=prepare or the one-click shell script first."
            )
        print(f"LOAD {pde} checkpoint once on {device}", flush=True)
        bundle = load_fm4pde_checkpoint_bundle(
            str(checkpoint), pde, device=torch.device(device), wrap=True
        )
        for task in tasks:
            if phase == "tune":
                candidates = candidates_for(pde, task, profile, candidate_set)
            else:
                winner = selected[(pde, task)]
                candidates = [dict(baseline_for(pde, task))]
                if winner["candidate"] != "baseline":
                    candidates.append(winner)
            for candidate in candidates:
                for test_type in test_types:
                    for offset in offsets:
                        status = run_job(
                            root,
                            phase=phase,
                            profile=profile,
                            pde=pde,
                            task=task,
                            candidate=candidate,
                            test_type=test_type,
                            offset=offset,
                            batch_size=batch_size,
                            num_steps=num_steps,
                            sample_seed=sample_seed,
                            mask_seed=mask_seed,
                            data_root=data_root,
                            checkpoint=checkpoint,
                            device=device,
                            resume=resume,
                            retry_non_finite=retry_non_finite,
                            checkpoint_bundle=bundle,
                        )
                        failures += status["status"] != "ok"
                        gc.collect()
                        torch.cuda.empty_cache()
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    return failures


def print_plan(
    *,
    phase: str,
    profile: str,
    pdes: list[str],
    tasks: list[str],
    test_types: list[str],
    tune_offsets: list[int],
    holdout_offsets: list[int],
    batch_size: int,
    num_steps: int,
    device: str,
    candidate_set: str = "broad",
    distribution_weights: dict[str, float] | None = None,
) -> None:
    tune_jobs = 0
    for pde in pdes:
        for task in tasks:
            count = len(candidates_for(pde, task, profile, candidate_set))
            jobs = count * len(test_types) * len(tune_offsets)
            tune_jobs += jobs
            print(
                f"PLAN pde={pde} task={task} candidates={count} "
                f"tune_jobs={jobs} device={device}",
                flush=True,
            )
    holdout_jobs = len(pdes) * len(tasks) * 2 * len(test_types) * len(holdout_offsets)
    print(
        f"PLAN SUMMARY phase={phase} profile={profile} candidate_set={candidate_set} "
        f"distribution_weights={distribution_weights or {name: 1.0 for name in test_types}} "
        f"steps={num_steps} batch={batch_size} "
        f"samples_per_job={batch_size} tune_jobs={tune_jobs} "
        f"holdout_jobs_at_most={holdout_jobs}",
        flush=True,
    )


def _resolved_profile_values(args: argparse.Namespace) -> tuple[list[str], list[int], list[int], int]:
    defaults = PROFILE_DEFAULTS[args.profile]
    test_types = (
        _parse_names(args.test_types, TEST_TYPES, "test type")
        if args.test_types
        else list(defaults["test_types"])
    )
    tune_offsets = _parse_offsets(args.tune_offsets) if args.tune_offsets else list(defaults["tune_offsets"])
    holdout_offsets = (
        _parse_offsets(args.holdout_offsets)
        if args.holdout_offsets
        else list(defaults["holdout_offsets"])
    )
    if set(tune_offsets).intersection(holdout_offsets):
        raise ValueError("Tune and holdout offsets must be disjoint")
    batch_size = args.batch_size or int(defaults["batch_size"])
    return test_types, tune_offsets, holdout_offsets, batch_size


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/tuning/six_pde_sampling")
    )
    parser.add_argument(
        "--phase",
        choices=("prepare", "tune", "holdout", "all", "analyze", "analyze-tune"),
        default="all",
    )
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--candidate-set", choices=CANDIDATE_SETS, default="broad")
    parser.add_argument("--pdes", default=",".join(PDES))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--test-types", default="")
    parser.add_argument(
        "--distribution-weights",
        default="",
        help="Per-split selection weights, for example id=1,smooth=1,rough=2",
    )
    parser.add_argument("--tune-offsets", default="")
    parser.add_argument("--holdout-offsets", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--data-root", type=Path, default=Path.home() / "share" / "PDEdata"
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-known-nonfinite",
        action="store_true",
        help="Do not rerun a deterministic job already classified as NaN/Inf",
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        pdes = _parse_names(args.pdes, PDES, "PDE")
        tasks = _parse_names(args.tasks, TASKS, "task")
        test_types, tune_offsets, holdout_offsets, batch_size = _resolved_profile_values(args)
        distribution_weights = (
            _parse_distribution_weights(args.distribution_weights, test_types)
            if args.distribution_weights
            else {name: 1.0 for name in test_types}
        )
    except ValueError as exc:
        parser.error(str(exc))
    if batch_size < 1 or args.num_steps < 1:
        parser.error("--batch-size and --num-steps must be positive")
    root = args.root.resolve()
    data_root = args.data_root.resolve()
    if args.plan_only:
        print_plan(
            phase=args.phase,
            profile=args.profile,
            pdes=pdes,
            tasks=tasks,
            test_types=test_types,
            tune_offsets=tune_offsets,
            holdout_offsets=holdout_offsets,
            batch_size=batch_size,
            num_steps=args.num_steps,
            device=args.device,
            candidate_set=args.candidate_set,
            distribution_weights=distribution_weights,
        )
        return 0
    if args.phase == "prepare":
        prepare_inputs(root, data_root, pdes, tasks, test_types)
        return 0
    if args.phase in {"tune", "holdout", "all"}:
        prepare_inputs(root, data_root, pdes, tasks, test_types)

    failures = 0
    winners: list[dict[str, Any]] | None = None
    if args.phase in {"tune", "all"}:
        failures += run_phase(
            root,
            phase="tune",
            profile=args.profile,
            pdes=pdes,
            tasks=tasks,
            test_types=test_types,
            offsets=tune_offsets,
            batch_size=batch_size,
            num_steps=args.num_steps,
            sample_seed=args.sample_seed,
            mask_seed=args.mask_seed,
            data_root=data_root,
            device=args.device,
            resume=not args.no_resume,
            candidate_set=args.candidate_set,
            retry_non_finite=not args.skip_known_nonfinite,
        )
        winners = analyze_tune(
            root,
            profile=args.profile,
            pdes=pdes,
            tasks=tasks,
            test_types=test_types,
            offsets=tune_offsets,
            batch_size=batch_size,
            num_steps=args.num_steps,
            candidate_set=args.candidate_set,
            distribution_weights=distribution_weights,
        )
    if args.phase in {"holdout", "all"}:
        if winners is None:
            winners = analyze_tune(
                root,
                profile=args.profile,
                pdes=pdes,
                tasks=tasks,
                test_types=test_types,
                offsets=tune_offsets,
                batch_size=batch_size,
                num_steps=args.num_steps,
                candidate_set=args.candidate_set,
                distribution_weights=distribution_weights,
            )
        failures += run_phase(
            root,
            phase="holdout",
            profile=args.profile,
            pdes=pdes,
            tasks=tasks,
            test_types=test_types,
            offsets=holdout_offsets,
            batch_size=batch_size,
            num_steps=args.num_steps,
            sample_seed=args.sample_seed,
            mask_seed=args.mask_seed,
            data_root=data_root,
            device=args.device,
            resume=not args.no_resume,
            winners=winners,
            candidate_set=args.candidate_set,
            retry_non_finite=not args.skip_known_nonfinite,
        )
        analyze_holdout(
            root,
            data_root=data_root,
            profile=args.profile,
            pdes=pdes,
            tasks=tasks,
            test_types=test_types,
            offsets=holdout_offsets,
            batch_size=batch_size,
            num_steps=args.num_steps,
            winners=winners,
        )
    if args.phase in {"analyze", "analyze-tune"}:
        winners = analyze_tune(
            root,
            profile=args.profile,
            pdes=pdes,
            tasks=tasks,
            test_types=test_types,
            offsets=tune_offsets,
            batch_size=batch_size,
            num_steps=args.num_steps,
            candidate_set=args.candidate_set,
            distribution_weights=distribution_weights,
        )
        holdout_root = root / "runs" / "holdout" / args.profile
        if args.phase == "analyze" and holdout_root.is_dir():
            analyze_holdout(
                root,
                data_root=data_root,
                profile=args.profile,
                pdes=pdes,
                tasks=tasks,
                test_types=test_types,
                offsets=holdout_offsets,
                batch_size=batch_size,
                num_steps=args.num_steps,
                winners=winners,
            )
        elif args.phase == "analyze":
            print(
                f"SKIP holdout analysis: no jobs below {holdout_root}",
                flush=True,
            )
    if failures:
        print(
            f"Completed with {failures} failed jobs. Rerun the same command to retry them; "
            "finite complete candidates were still analyzed.",
            file=sys.stderr,
        )
    # A numerically unstable candidate is an expected outcome of parameter
    # screening, not a launcher failure. Fatal/preflight errors still raise and
    # return non-zero; successful jobs remain resumable on the next invocation.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

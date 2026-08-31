#!/usr/bin/env python3
"""Tune all main sampling tasks with hard-tail and regression protection.

Candidates are selected on hard, good, and deterministic random samples from
ID, smooth, and rough distributions.  The untouched holdout split is then run
with both the current baseline and the tuning winner using paired initial
noise.  One winner is produced for every (PDE, task), not one numerically
shared zeta value across tasks whose active observations have different units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sampling.config import load_config, load_yaml_file
from scripts.tuning.make_inference_checkpoint import make_inference_checkpoint


TEST_TYPES = ("id", "smooth", "rough")
PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded")
TASKS = ("both", "forward", "inverse")
STRATA = ("hard", "good", "random")
DEFAULT_MICROBATCH = {"poisson": 10, "helmholtz": 10, "darcy": 10, "nsnonbounded": 4}

# Predeclared selection/validation guardrails.
MAX_HARD_MEAN_RATIO = 1.00
MAX_HARD_P90_RATIO = 1.02
MAX_PROTECTION_MEAN_RATIO = 1.02
MAX_PROTECTION_P90_RATIO = 1.05
MAX_PDE_RESIDUAL_RATIO = 1.25
MAX_AUXILIARY_ERROR_RATIO = 1.05


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


def robust(values: list[float]) -> float:
    if not values:
        return math.nan
    return 0.5 * statistics.fmean(values) + 0.3 * percentile(values, 0.9) + 0.2 * max(values)


def safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 1.0 if value == 0.0 else math.inf
    return value / baseline


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _parse_list(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in selected if item not in allowed]
    if invalid or not selected or len(set(selected)) != len(selected):
        raise ValueError(f"Invalid {label} list: {value!r}")
    return selected


def parse_microbatches(values: list[str]) -> dict[str, int]:
    resolved = dict(DEFAULT_MICROBATCH)
    for value in values:
        pde, separator, raw_size = value.partition("=")
        if not separator or pde not in PDES or int(raw_size) < 1:
            raise ValueError(f"Invalid --microbatch value: {value!r}")
        resolved[pde] = int(raw_size)
    return resolved


@lru_cache(maxsize=None)
def baseline_for(pde: str, task: str) -> dict[str, Any]:
    config = load_config(REPO_ROOT / "configs" / "main" / task / f"{pde}.yaml")
    return {
        "candidate": "baseline",
        "zeta_obs_a": float(config.zeta_obs_a),
        "zeta_obs_u": float(config.zeta_obs_u),
        "zeta_pde": float(config.zeta_pde),
        "clip_threshold": float(config.clip_threshold),
    }


def _variant(base: dict[str, Any], name: str, **updates: float) -> dict[str, Any]:
    result = dict(base)
    result.update(updates)
    result["candidate"] = name
    return result


def _clip_candidates(base: dict[str, Any]) -> tuple[tuple[str, float], tuple[str, float]]:
    threshold = float(base["clip_threshold"])
    if threshold >= 1e9:
        return (("clip_100", 100.0), ("clip_200", 200.0))
    return (("clip_75", 75.0), ("clip_125", 125.0))


def candidates_for(pde: str, task: str) -> list[dict[str, Any]]:
    """Return the round-three grid, including the exact current baseline."""
    base = baseline_for(pde, task)
    if task == "inverse":
        if pde == "poisson":
            return [
                dict(base),
                *[
                    _variant(base, f"obs_{factor:g}x", zeta_obs_u=base["zeta_obs_u"] * factor)
                    for factor in (8.0, 10.0, 12.0, 14.0, 16.0)
                ],
            ]
        if pde == "helmholtz":
            return [
                dict(base),
                *[
                    _variant(base, f"obs_{factor:g}x", zeta_obs_u=base["zeta_obs_u"] * factor)
                    for factor in (12.0, 16.0, 20.0, 24.0)
                ],
            ]
        if pde == "darcy":
            return [
                dict(base),
                _variant(
                    base,
                    "obs_48x_clip90",
                    zeta_obs_u=base["zeta_obs_u"] * 48.0,
                    clip_threshold=90.0,
                ),
                _variant(
                    base,
                    "obs_64x_clip90",
                    zeta_obs_u=base["zeta_obs_u"] * 64.0,
                    clip_threshold=90.0,
                ),
                _variant(
                    base,
                    "obs_64x_clip100",
                    zeta_obs_u=base["zeta_obs_u"] * 64.0,
                    clip_threshold=100.0,
                ),
                _variant(
                    base,
                    "obs_80x_clip100",
                    zeta_obs_u=base["zeta_obs_u"] * 80.0,
                    clip_threshold=100.0,
                ),
            ]
        if pde == "nsnonbounded":
            return [
                dict(base),
                *[
                    _variant(base, f"clip_{int(threshold)}", clip_threshold=threshold)
                    for threshold in (100.0, 125.0, 150.0)
                ],
            ]
        raise ValueError(f"Unsupported inverse PDE: {pde}")

    low_clip, high_clip = _clip_candidates(base)
    if task == "forward":
        return [
            dict(base),
            _variant(base, "obs_a_0p75x", zeta_obs_a=base["zeta_obs_a"] * 0.75),
            _variant(base, "obs_a_1p25x", zeta_obs_a=base["zeta_obs_a"] * 1.25),
            _variant(base, "pde_0p5x", zeta_pde=base["zeta_pde"] * 0.5),
            _variant(base, "pde_2x", zeta_pde=base["zeta_pde"] * 2.0),
            _variant(
                base,
                "obs_a_1p25x_pde_2x",
                zeta_obs_a=base["zeta_obs_a"] * 1.25,
                zeta_pde=base["zeta_pde"] * 2.0,
            ),
            _variant(base, low_clip[0], clip_threshold=low_clip[1]),
            _variant(base, high_clip[0], clip_threshold=high_clip[1]),
        ]
    if task == "both":
        return [
            dict(base),
            _variant(base, "obs_a_0p75x", zeta_obs_a=base["zeta_obs_a"] * 0.75),
            _variant(base, "obs_a_1p25x", zeta_obs_a=base["zeta_obs_a"] * 1.25),
            _variant(base, "obs_u_0p75x", zeta_obs_u=base["zeta_obs_u"] * 0.75),
            _variant(base, "obs_u_1p25x", zeta_obs_u=base["zeta_obs_u"] * 1.25),
            _variant(
                base,
                "both_obs_1p25x",
                zeta_obs_a=base["zeta_obs_a"] * 1.25,
                zeta_obs_u=base["zeta_obs_u"] * 1.25,
            ),
            _variant(base, "pde_2x", zeta_pde=base["zeta_pde"] * 2.0),
            _variant(
                base,
                "both_obs_1p25x_pde_2x",
                zeta_obs_a=base["zeta_obs_a"] * 1.25,
                zeta_obs_u=base["zeta_obs_u"] * 1.25,
                zeta_pde=base["zeta_pde"] * 2.0,
            ),
            _variant(base, low_clip[0], clip_threshold=low_clip[1]),
            _variant(base, high_clip[0], clip_threshold=high_clip[1]),
        ]
    raise ValueError(f"Unsupported task: {task}")


def chunk_plan(sample_count: int, microbatch: int) -> list[tuple[int, int]]:
    return [
        (offset, min(microbatch, sample_count - offset))
        for offset in range(0, sample_count, microbatch)
    ]


def analysis_path(
    root: Path,
    stem: str,
    pdes: list[str],
    tasks: list[str],
    label: str = "",
) -> Path:
    selected_pdes = [pde for pde in PDES if pde in set(pdes)]
    selected_tasks = [task for task in TASKS if task in set(tasks)]
    suffix = "" if selected_pdes == list(PDES) else "_" + "_".join(selected_pdes)
    if selected_tasks != list(TASKS):
        suffix += "_" + "_".join(selected_tasks)
    if label:
        if re.fullmatch(r"[A-Za-z0-9_-]+", label) is None:
            raise ValueError(f"Invalid analysis label: {label!r}")
        suffix += "_" + label
    return root / f"{stem}{suffix}.csv"


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def ensure_checkpoint(root: Path, pde: str, task: str) -> Path:
    destination = root / "checkpoints" / f"{pde}_inference.pth"
    if destination.is_file():
        return destination.resolve()
    config = load_config(REPO_ROOT / "configs" / "main" / task / f"{pde}.yaml")
    make_inference_checkpoint(
        Path(config.checkpoint_path),
        destination,
        gradient_checkpointing=pde == "nsnonbounded",
    )
    return destination.resolve()


def find_run_artifact(job_root: Path, filename: str) -> Path | None:
    candidates = sorted(job_root.rglob(filename), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_job(
    root: Path,
    *,
    pde: str,
    task: str,
    phase: str,
    split: str,
    candidate: dict[str, Any],
    test_type: str,
    offset: int,
    batch_size: int,
    source_batch_size: int,
    checkpoint: Path,
    num_steps: int,
    device: str,
    resume: bool,
    checkpoint_bundle: tuple[Any, Any, dict[str, Any]] | None,
) -> dict[str, Any]:
    name = str(candidate["candidate"])
    job_root = (
        root / "runs" / phase / pde / task / name / test_type / f"offset-{offset}-batch-{batch_size}"
    )
    status_path = job_root / "job.json"
    data_path = (root / "subsets" / split / f"{pde}_{task}_{test_type}.mat").resolve()
    config_path = (REPO_ROOT / "configs" / "main" / task / f"{pde}.yaml").resolve()
    source_indices = list(range(offset, offset + batch_size))
    signature = {
        "phase": phase,
        "split": split,
        "pde": pde,
        "task": task,
        "candidate": name,
        "test_type": test_type,
        "offset": offset,
        "batch_size": batch_size,
        "source_batch_size": source_batch_size,
        "source_indices": source_indices,
        "zeta_obs_a": candidate["zeta_obs_a"],
        "zeta_obs_u": candidate["zeta_obs_u"],
        "zeta_pde": candidate["zeta_pde"],
        "clip_threshold": candidate["clip_threshold"],
        "num_steps": num_steps,
        "config": _file_fingerprint(config_path),
        "checkpoint": _file_fingerprint(checkpoint),
        "data": _file_fingerprint(data_path),
    }
    if resume and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        metrics_path = Path(str(status.get("metrics_path", "")))
        if (
            status.get("status") == "ok"
            and status.get("job_signature") == signature
            and (metrics_path.is_file() or find_run_artifact(status_path.parent, "metrics_per_sample.csv"))
        ):
            print(
                f"SKIP {phase} {pde}/{task} {name} {test_type} offset={offset} batch={batch_size}",
                flush=True,
            )
            return status
        print(f"STALE {phase} {pde}/{task} {name} {test_type} offset={offset}", flush=True)

    job_root.mkdir(parents=True, exist_ok=True)
    overrides = {
        "data_path": str(data_path),
        "checkpoint_path": str(checkpoint),
        "test_type": test_type,
        "output_dir": str(job_root.resolve()),
        "ablation_group": f"balanced_{phase}_{name}_{test_type}",
        "task": task,
        "batch_size": batch_size,
        "offset": offset,
        "num_steps": num_steps,
        "sample_seed": 0,
        "mask_seed": 0,
        "initial_noise_source_batch_size": source_batch_size,
        "initial_noise_source_indices": source_indices,
        "zeta_obs_a": candidate["zeta_obs_a"],
        "zeta_obs_u": candidate["zeta_obs_u"],
        "zeta_pde": candidate["zeta_pde"],
        "clip_mode": "global_norm",
        "clip_threshold": candidate["clip_threshold"],
        "device": device,
        "save_plots": False,
    }
    command = [sys.executable, "-u", "-m", "sampling.runner", "--config", str(config_path)]
    for key, value in overrides.items():
        command.extend(("--override", f"{key}={value}"))

    log_path = job_root / "runner.log"
    print(
        f"RUN  {phase} {pde}/{task} {name} {test_type} offset={offset} batch={batch_size}",
        flush=True,
    )
    started = time.time()
    error_message = ""
    with log_path.open("w", encoding="utf-8") as log:
        if checkpoint_bundle is None:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            returncode = completed.returncode
        else:
            from sampling.runner import run_single_ablation

            try:
                config = load_config(config_path, overrides=overrides)
                with redirect_stdout(log), redirect_stderr(log):
                    run_single_ablation(config, checkpoint_bundle=checkpoint_bundle)
                returncode = 0
            except Exception as exc:
                import traceback

                traceback.print_exc(file=log)
                error_message = f"{type(exc).__name__}: {exc}"
                returncode = 1
    elapsed = time.time() - started
    metrics_path = find_run_artifact(job_root, "metrics_per_sample.csv")
    final_path = find_run_artifact(job_root, "metrics_final.json")
    status = {
        "status": "ok" if returncode == 0 and metrics_path and final_path else "failed",
        "returncode": returncode,
        "error_message": error_message,
        "phase": phase,
        "split": split,
        "pde": pde,
        "task": task,
        "candidate": name,
        "test_type": test_type,
        "offset": offset,
        "batch_size": batch_size,
        "zeta_obs_a": candidate["zeta_obs_a"],
        "zeta_obs_u": candidate["zeta_obs_u"],
        "zeta_pde": candidate["zeta_pde"],
        "clip_threshold": candidate["clip_threshold"],
        "num_steps": num_steps,
        "metrics_path": "" if metrics_path is None else str(metrics_path.resolve()),
        "final_path": "" if final_path is None else str(final_path.resolve()),
        "log_path": str(log_path.resolve()),
        "elapsed_seconds": elapsed,
        "job_signature": signature,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if status["status"] != "ok":
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(f"Job failed: {pde}/{task}/{name}/{test_type}/{offset}\n{tail}")
    print(
        f"DONE {pde}/{task} {name} {test_type} offset={offset} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return status


def _local_artifact(status_path: Path, status: dict[str, Any], key: str, filename: str) -> Path | None:
    recorded = Path(str(status.get(key, "")))
    if recorded.is_file():
        return recorded
    return find_run_artifact(status_path.parent, filename)


def collect_phase_rows(
    root: Path,
    phase: str,
    manifest: list[dict[str, str]],
    *,
    source_batch_size: int,
    num_steps: int,
) -> list[dict[str, Any]]:
    lookup = {
        (
            row["pde"],
            row["task"],
            row["test_type"],
            row["split"],
            int(row["subset_index"]),
        ): row
        for row in manifest
    }
    collected: dict[tuple[str, str, str, str, int], tuple[int, dict[str, Any]]] = {}
    for status_path in sorted(
        (root / "runs" / phase).rglob("job.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    ):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        signature = status.get("job_signature", {})
        if (
            status.get("status") != "ok"
            or int(signature.get("source_batch_size", -1)) != source_batch_size
            or int(signature.get("num_steps", -1)) != num_steps
        ):
            continue
        metrics_path = _local_artifact(status_path, status, "metrics_path", "metrics_per_sample.csv")
        if metrics_path is None:
            continue
        offset = int(status["offset"])
        for row in read_csv(metrics_path):
            subset_index = offset + int(float(row["sample_index"]))
            source = lookup[
                (
                    status["pde"],
                    status["task"],
                    status["test_type"],
                    status["split"],
                    subset_index,
                )
            ]
            logical_key = (
                str(status["pde"]),
                str(status["task"]),
                str(status["candidate"]),
                str(status["test_type"]),
                subset_index,
            )
            collected[logical_key] = (
                status_path.stat().st_mtime_ns,
                {
                    "phase": phase,
                    "split": status["split"],
                    "pde": status["pde"],
                    "task": status["task"],
                    "candidate": status["candidate"],
                    "test_type": status["test_type"],
                    "stratum": source["stratum"],
                    "subset_index": subset_index,
                    "source_sample_id": int(source["source_sample_id"]),
                    "zeta_obs_a": float(status["zeta_obs_a"]),
                    "zeta_obs_u": float(status["zeta_obs_u"]),
                    "zeta_pde": float(status["zeta_pde"]),
                    "clip_threshold": float(status["clip_threshold"]),
                    "rel_l2_a": float(row["rel_l2_a"]),
                    "rel_l2_u": float(row["rel_l2_u"]),
                    "pde_residual_norm": float(row["pde_residual_norm"]),
                    "metrics_path": str(metrics_path.resolve()),
                },
            )
    return [collected[key][1] for key in sorted(collected)]


def _scope_stats(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    task: str,
) -> dict[str, float]:
    key = lambda row: (str(row["test_type"]), int(row["subset_index"]))
    candidate_map = {key(row): row for row in candidate_rows}
    baseline_map = {key(row): row for row in baseline_rows}
    if candidate_map.keys() != baseline_map.keys() or not candidate_map:
        raise RuntimeError("Candidate and baseline scopes are not paired")
    pairs = [(baseline_map[item], candidate_map[item]) for item in sorted(candidate_map)]
    ba = [float(pair[0]["rel_l2_a"]) for pair in pairs]
    ca = [float(pair[1]["rel_l2_a"]) for pair in pairs]
    bu = [float(pair[0]["rel_l2_u"]) for pair in pairs]
    cu = [float(pair[1]["rel_l2_u"]) for pair in pairs]
    bp = [float(pair[0]["pde_residual_norm"]) for pair in pairs]
    cp = [float(pair[1]["pde_residual_norm"]) for pair in pairs]

    mean_ratio_a = safe_ratio(statistics.fmean(ca), statistics.fmean(ba))
    mean_ratio_u = safe_ratio(statistics.fmean(cu), statistics.fmean(bu))
    p90_ratio_a = safe_ratio(percentile(ca, 0.9), percentile(ba, 0.9))
    p90_ratio_u = safe_ratio(percentile(cu, 0.9), percentile(bu, 0.9))
    robust_ratio_a = safe_ratio(robust(ca), robust(ba))
    robust_ratio_u = safe_ratio(robust(cu), robust(bu))
    if task == "forward":
        primary_mean_ratio = mean_ratio_u
        primary_p90_ratio = p90_ratio_u
        primary_robust_ratio = robust_ratio_u
        wins = [candidate < baseline for baseline, candidate in zip(bu, cu)]
        auxiliary_ratio = mean_ratio_a
        baseline_primary = statistics.fmean(bu)
        candidate_primary = statistics.fmean(cu)
    elif task == "inverse":
        primary_mean_ratio = mean_ratio_a
        primary_p90_ratio = p90_ratio_a
        primary_robust_ratio = robust_ratio_a
        wins = [candidate < baseline for baseline, candidate in zip(ba, ca)]
        auxiliary_ratio = mean_ratio_u
        baseline_primary = statistics.fmean(ba)
        candidate_primary = statistics.fmean(ca)
    else:
        primary_mean_ratio = max(mean_ratio_a, mean_ratio_u)
        primary_p90_ratio = max(p90_ratio_a, p90_ratio_u)
        primary_robust_ratio = max(robust_ratio_a, robust_ratio_u)
        wins = [
            candidate_a < baseline_a and candidate_u < baseline_u
            for baseline_a, candidate_a, baseline_u, candidate_u in zip(ba, ca, bu, cu)
        ]
        auxiliary_ratio = max(mean_ratio_a, mean_ratio_u)
        baseline_primary = max(statistics.fmean(ba), statistics.fmean(bu))
        candidate_primary = max(statistics.fmean(ca), statistics.fmean(cu))
    return {
        "n": float(len(pairs)),
        "baseline_rel_l2_a_mean": statistics.fmean(ba),
        "candidate_rel_l2_a_mean": statistics.fmean(ca),
        "rel_l2_a_mean_ratio": mean_ratio_a,
        "baseline_rel_l2_u_mean": statistics.fmean(bu),
        "candidate_rel_l2_u_mean": statistics.fmean(cu),
        "rel_l2_u_mean_ratio": mean_ratio_u,
        "baseline_primary_error_mean": baseline_primary,
        "candidate_primary_error_mean": candidate_primary,
        "primary_mean_ratio": primary_mean_ratio,
        "primary_p90_ratio": primary_p90_ratio,
        "primary_robust_ratio": primary_robust_ratio,
        "primary_win_rate": sum(wins) / len(wins),
        "baseline_pde_residual_mean": statistics.fmean(bp),
        "candidate_pde_residual_mean": statistics.fmean(cp),
        "pde_residual_ratio": safe_ratio(statistics.fmean(cp), statistics.fmean(bp)),
        "auxiliary_error_ratio": auxiliary_ratio,
    }


def _filter_scope(
    rows: list[dict[str, Any]],
    *,
    test_type: str = "all",
    stratum: str = "all",
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (test_type == "all" or row["test_type"] == test_type)
        and (stratum == "all" or row["stratum"] == stratum)
    ]


def summarize_tune(
    rows: list[dict[str, Any]],
    pdes: list[str],
    tasks: list[str],
    sample_count: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["pde"] in pdes
            and row["task"] in tasks
            and int(row["subset_index"]) < sample_count
        ):
            grouped[(row["pde"], row["task"], row["candidate"])].append(row)
    expected = sample_count * len(TEST_TYPES)
    summaries: list[dict[str, Any]] = []
    for pde in pdes:
        for task in tasks:
            baseline_rows = grouped.get((pde, task, "baseline"), [])
            if len(baseline_rows) != expected:
                continue
            for (gpde, gtask, candidate), candidate_rows in sorted(grouped.items()):
                if (gpde, gtask) != (pde, task) or len(candidate_rows) != expected:
                    continue
                summary: dict[str, Any] = {
                    "pde": pde,
                    "task": task,
                    "candidate": candidate,
                    "zeta_obs_a": candidate_rows[0]["zeta_obs_a"],
                    "zeta_obs_u": candidate_rows[0]["zeta_obs_u"],
                    "zeta_pde": candidate_rows[0]["zeta_pde"],
                    "clip_threshold": candidate_rows[0]["clip_threshold"],
                    "n": len(candidate_rows),
                }
                hard = _scope_stats(
                    _filter_scope(candidate_rows, stratum="hard"),
                    _filter_scope(baseline_rows, stratum="hard"),
                    task,
                )
                good = _scope_stats(
                    _filter_scope(candidate_rows, stratum="good"),
                    _filter_scope(baseline_rows, stratum="good"),
                    task,
                )
                random_scope = _scope_stats(
                    _filter_scope(candidate_rows, stratum="random"),
                    _filter_scope(baseline_rows, stratum="random"),
                    task,
                )
                summary.update(
                    {
                        "hard_primary_mean_ratio": hard["primary_mean_ratio"],
                        "hard_primary_p90_ratio": hard["primary_p90_ratio"],
                        "hard_primary_robust_ratio": hard["primary_robust_ratio"],
                        "hard_primary_win_rate": hard["primary_win_rate"],
                        "good_primary_mean_ratio": good["primary_mean_ratio"],
                        "good_primary_p90_ratio": good["primary_p90_ratio"],
                        "random_primary_mean_ratio": random_scope["primary_mean_ratio"],
                        "random_primary_p90_ratio": random_scope["primary_p90_ratio"],
                    }
                )
                hard_test_ratios = []
                passes = True
                for test_type in TEST_TYPES:
                    for stratum in STRATA:
                        scoped = _scope_stats(
                            _filter_scope(candidate_rows, test_type=test_type, stratum=stratum),
                            _filter_scope(baseline_rows, test_type=test_type, stratum=stratum),
                            task,
                        )
                        summary[f"{stratum}_primary_mean_ratio_{test_type}"] = scoped[
                            "primary_mean_ratio"
                        ]
                        summary[f"{stratum}_primary_p90_ratio_{test_type}"] = scoped[
                            "primary_p90_ratio"
                        ]
                        if stratum == "hard":
                            hard_test_ratios.append(scoped["primary_mean_ratio"])
                            passes = passes and scoped["primary_mean_ratio"] <= MAX_HARD_MEAN_RATIO
                            passes = passes and scoped["primary_p90_ratio"] <= MAX_HARD_P90_RATIO
                        else:
                            passes = passes and scoped["primary_mean_ratio"] <= MAX_PROTECTION_MEAN_RATIO
                            passes = passes and scoped["primary_p90_ratio"] <= MAX_PROTECTION_P90_RATIO
                    all_strata = _scope_stats(
                        _filter_scope(candidate_rows, test_type=test_type),
                        _filter_scope(baseline_rows, test_type=test_type),
                        task,
                    )
                    summary[f"pde_residual_ratio_{test_type}"] = all_strata["pde_residual_ratio"]
                    summary[f"auxiliary_error_ratio_{test_type}"] = all_strata[
                        "auxiliary_error_ratio"
                    ]
                    passes = passes and all_strata["pde_residual_ratio"] <= MAX_PDE_RESIDUAL_RATIO
                    passes = passes and all_strata["auxiliary_error_ratio"] <= MAX_AUXILIARY_ERROR_RATIO
                summary["hard_worst_test_mean_ratio"] = max(hard_test_ratios)
                summary["selection_score"] = (
                    0.55 * hard["primary_robust_ratio"]
                    + 0.20 * summary["hard_worst_test_mean_ratio"]
                    + 0.15 * random_scope["primary_mean_ratio"]
                    + 0.10 * good["primary_mean_ratio"]
                )
                summary["passes_guardrails"] = passes
                summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            pdes.index(str(row["pde"])),
            tasks.index(str(row["task"])),
            float(row["selection_score"]),
        )
    )
    return summaries


def select_winners(
    summaries: list[dict[str, Any]], pdes: list[str], tasks: list[str]
) -> list[dict[str, Any]]:
    winners = []
    for pde in pdes:
        for task in tasks:
            candidates = [row for row in summaries if row["pde"] == pde and row["task"] == task]
            if not candidates:
                raise RuntimeError(f"No complete tuning candidate for {pde}/{task}")
            baseline = next(row for row in candidates if row["candidate"] == "baseline")
            eligible = [row for row in candidates if bool(row["passes_guardrails"])]
            if not eligible:
                raise RuntimeError(f"No tuning candidate passed guardrails for {pde}/{task}")
            winner = dict(min(eligible, key=lambda row: float(row["selection_score"])))
            winner["hard_robust_improvement_vs_baseline"] = 1.0 - float(
                winner["hard_primary_robust_ratio"]
            )
            winner["selection_score_improvement_vs_baseline"] = safe_ratio(
                float(baseline["selection_score"]) - float(winner["selection_score"]),
                float(baseline["selection_score"]),
            )
            winners.append(winner)
    return winners


def _allowed_candidates(pdes: list[str], tasks: list[str]) -> set[tuple[str, str, str]]:
    return {
        (pde, task, str(candidate["candidate"]))
        for pde in pdes
        for task in tasks
        for candidate in candidates_for(pde, task)
    }


def analyze_tune(
    root: Path,
    manifest: list[dict[str, str]],
    pdes: list[str],
    tasks: list[str],
    sample_count: int,
    num_steps: int,
    analysis_label: str,
) -> list[dict[str, Any]]:
    allowed = _allowed_candidates(pdes, tasks)
    rows = [
        row
        for row in collect_phase_rows(
            root, "tune", manifest, source_batch_size=sample_count, num_steps=num_steps
        )
        if (row["pde"], row["task"], row["candidate"]) in allowed
        and int(row["subset_index"]) < sample_count
    ]
    write_csv(analysis_path(root, "tune_per_sample", pdes, tasks, analysis_label), rows)
    summaries = summarize_tune(rows, pdes, tasks, sample_count)
    winners = select_winners(summaries, pdes, tasks)
    write_csv(analysis_path(root, "tune_summary", pdes, tasks, analysis_label), summaries)
    selected_path = analysis_path(root, "selected_params", pdes, tasks, analysis_label)
    write_csv(selected_path, winners)
    selected_path.with_suffix(".json").write_text(json.dumps(winners, indent=2), encoding="utf-8")
    for row in winners:
        print(
            f"SELECT {row['pde']}/{row['task']} {row['candidate']} "
            f"hard_improvement={100 * float(row['hard_robust_improvement_vs_baseline']):.2f}% "
            f"score={float(row['selection_score']):.4f}",
            flush=True,
        )
    return winners


def candidate_from_winner(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": str(row["candidate"]),
        "zeta_obs_a": float(row["zeta_obs_a"]),
        "zeta_obs_u": float(row["zeta_obs_u"]),
        "zeta_pde": float(row["zeta_pde"]),
        "clip_threshold": float(row["clip_threshold"]),
    }


def _validated_holdout(
    candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], task: str
) -> bool:
    for test_type in TEST_TYPES:
        for stratum in STRATA:
            stats = _scope_stats(
                _filter_scope(candidate_rows, test_type=test_type, stratum=stratum),
                _filter_scope(baseline_rows, test_type=test_type, stratum=stratum),
                task,
            )
            if stratum == "hard":
                if (
                    stats["primary_mean_ratio"] > MAX_HARD_MEAN_RATIO
                    or stats["primary_p90_ratio"] > MAX_HARD_P90_RATIO
                ):
                    return False
            elif (
                stats["primary_mean_ratio"] > MAX_PROTECTION_MEAN_RATIO
                or stats["primary_p90_ratio"] > MAX_PROTECTION_P90_RATIO
            ):
                return False
        all_stats = _scope_stats(
            _filter_scope(candidate_rows, test_type=test_type),
            _filter_scope(baseline_rows, test_type=test_type),
            task,
        )
        if (
            all_stats["pde_residual_ratio"] > MAX_PDE_RESIDUAL_RATIO
            or all_stats["auxiliary_error_ratio"] > MAX_AUXILIARY_ERROR_RATIO
        ):
            return False
    return True


def _write_recommended_configs(
    root: Path, recommendations: list[dict[str, Any]]
) -> None:
    import yaml

    for row in recommendations:
        pde = str(row["pde"])
        task = str(row["task"])
        source_path = REPO_ROOT / "configs" / "main" / task / f"{pde}.yaml"
        payload = load_yaml_file(source_path)
        for field in ("zeta_obs_a", "zeta_obs_u", "zeta_pde", "clip_threshold"):
            payload[field] = float(row[field])
        destination = root / "recommended_configs" / task / f"{pde}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def analyze_holdout(
    root: Path,
    manifest: list[dict[str, str]],
    pdes: list[str],
    tasks: list[str],
    sample_count: int,
    num_steps: int,
    winners: list[dict[str, Any]],
    analysis_label: str,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in collect_phase_rows(
            root, "holdout", manifest, source_batch_size=sample_count, num_steps=num_steps
        )
        if row["pde"] in pdes
        and row["task"] in tasks
        and int(row["subset_index"]) < sample_count
    ]
    write_csv(analysis_path(root, "holdout_per_sample", pdes, tasks, analysis_label), rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["pde"], row["task"], row["candidate"])].append(row)
    comparisons: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    for winner in winners:
        pde = str(winner["pde"])
        task = str(winner["task"])
        name = str(winner["candidate"])
        baseline_rows = grouped[(pde, task, "baseline")]
        candidate_rows = grouped[(pde, task, name)]
        validated = _validated_holdout(candidate_rows, baseline_rows, task)
        for test_type in ("all", *TEST_TYPES):
            for stratum in ("all", *STRATA):
                candidate_scope = _filter_scope(
                    candidate_rows, test_type=test_type, stratum=stratum
                )
                baseline_scope = _filter_scope(
                    baseline_rows, test_type=test_type, stratum=stratum
                )
                stats = _scope_stats(candidate_scope, baseline_scope, task)
                comparisons.append(
                    {
                        "pde": pde,
                        "task": task,
                        "winner": name,
                        "test_type": test_type,
                        "stratum": stratum,
                        "n": int(stats["n"]),
                        "baseline_primary_error_mean": stats["baseline_primary_error_mean"],
                        "winner_primary_error_mean": stats["candidate_primary_error_mean"],
                        "primary_error_ratio_vs_baseline": stats["primary_mean_ratio"],
                        "relative_primary_improvement": 1.0 - stats["primary_mean_ratio"],
                        "primary_p90_ratio_vs_baseline": stats["primary_p90_ratio"],
                        "primary_win_rate": stats["primary_win_rate"],
                        "baseline_rel_l2_a_mean": stats["baseline_rel_l2_a_mean"],
                        "winner_rel_l2_a_mean": stats["candidate_rel_l2_a_mean"],
                        "rel_l2_a_ratio_vs_baseline": stats["rel_l2_a_mean_ratio"],
                        "baseline_rel_l2_u_mean": stats["baseline_rel_l2_u_mean"],
                        "winner_rel_l2_u_mean": stats["candidate_rel_l2_u_mean"],
                        "rel_l2_u_ratio_vs_baseline": stats["rel_l2_u_mean_ratio"],
                        "pde_residual_ratio_vs_baseline": stats["pde_residual_ratio"],
                        "auxiliary_error_ratio_vs_baseline": stats["auxiliary_error_ratio"],
                        "validated_across_test_types_and_strata": validated,
                    }
                )
        recommended = candidate_from_winner(winner) if validated else baseline_for(pde, task)
        recommendations.append(
            {
                "pde": pde,
                "task": task,
                "selected_candidate": name,
                "holdout_validated": validated,
                "recommended_candidate": recommended["candidate"],
                "zeta_obs_a": recommended["zeta_obs_a"],
                "zeta_obs_u": recommended["zeta_obs_u"],
                "zeta_pde": recommended["zeta_pde"],
                "clip_threshold": recommended["clip_threshold"],
            }
        )
        overall = _scope_stats(candidate_rows, baseline_rows, task)
        hard = _scope_stats(
            _filter_scope(candidate_rows, stratum="hard"),
            _filter_scope(baseline_rows, stratum="hard"),
            task,
        )
        print(
            f"HOLDOUT {pde}/{task} {name} hard_improvement="
            f"{100 * (1.0 - hard['primary_robust_ratio']):.2f}% "
            f"all_improvement={100 * (1.0 - overall['primary_mean_ratio']):.2f}% "
            f"validated={validated}",
            flush=True,
        )
    write_csv(analysis_path(root, "holdout_comparison", pdes, tasks, analysis_label), comparisons)
    recommended_path = analysis_path(root, "recommended_params", pdes, tasks, analysis_label)
    write_csv(recommended_path, recommendations)
    recommended_path.with_suffix(".json").write_text(
        json.dumps(recommendations, indent=2), encoding="utf-8"
    )
    _write_recommended_configs(root, recommendations)
    return recommendations


def run_phase(
    root: Path,
    *,
    phase: str,
    pdes: list[str],
    tasks: list[str],
    sample_count: int,
    microbatches: dict[str, int],
    num_steps: int,
    device: str,
    resume: bool,
    winners: list[dict[str, Any]] | None = None,
    in_process: bool = True,
) -> None:
    winner_map = (
        {}
        if winners is None
        else {(row["pde"], row["task"]): candidate_from_winner(row) for row in winners}
    )
    for pde in pdes:
        checkpoint = ensure_checkpoint(root, pde, tasks[0])
        checkpoint_bundle = None
        if in_process:
            import torch
            from sampling.model_io import load_fm4pde_checkpoint_bundle

            print(f"LOAD {pde} checkpoint once for all task/candidate jobs", flush=True)
            checkpoint_bundle = load_fm4pde_checkpoint_bundle(
                str(checkpoint), pde, device=torch.device(device), wrap=True
            )
        for task in tasks:
            if phase == "tune":
                candidates = candidates_for(pde, task)
                split = "tune"
            else:
                winner = winner_map[(pde, task)]
                candidates = [baseline_for(pde, task)]
                if winner["candidate"] != "baseline":
                    candidates.append(winner)
                split = "holdout"
            for candidate in candidates:
                for test_type in TEST_TYPES:
                    for offset, batch_size in chunk_plan(sample_count, microbatches[pde]):
                        run_job(
                            root,
                            pde=pde,
                            task=task,
                            phase=phase,
                            split=split,
                            candidate=candidate,
                            test_type=test_type,
                            offset=offset,
                            batch_size=batch_size,
                            source_batch_size=sample_count,
                            checkpoint=checkpoint,
                            num_steps=num_steps,
                            device=device,
                            resume=resume,
                            checkpoint_bundle=checkpoint_bundle,
                        )
        if checkpoint_bundle is not None:
            del checkpoint_bundle
            import torch

            torch.cuda.empty_cache()


def _validate_sample_count(parser: argparse.ArgumentParser, value: int, flag: str) -> None:
    if value not in {20, 40}:
        parser.error(f"{flag} must be 20 (screening prefix) or 40 (complete split)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/artifacts/balanced_sampling_tuning")
    )
    parser.add_argument("--phase", choices=("tune", "holdout", "all", "analyze"), default="all")
    parser.add_argument("--pdes", default=",".join(PDES))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--tune-samples-per-test-type", type=int, default=20)
    parser.add_argument("--holdout-samples-per-test-type", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", action="append", default=[], help="Override as pde=N")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--subprocess", action="store_true")
    parser.add_argument("--analysis-label", default="round3")
    args = parser.parse_args()
    _validate_sample_count(parser, args.tune_samples_per_test_type, "--tune-samples-per-test-type")
    _validate_sample_count(
        parser, args.holdout_samples_per_test_type, "--holdout-samples-per-test-type"
    )
    pdes = _parse_list(args.pdes, PDES, "PDE")
    tasks = _parse_list(args.tasks, TASKS, "task")
    microbatches = parse_microbatches(args.microbatch)
    root = args.root.resolve()
    manifest_path = root / "balanced_samples.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Run prepare_balanced_sampling_samples.py first: {manifest_path}"
        )
    manifest = read_csv(manifest_path)

    if args.phase in {"tune", "all"}:
        run_phase(
            root,
            phase="tune",
            pdes=pdes,
            tasks=tasks,
            sample_count=args.tune_samples_per_test_type,
            microbatches=microbatches,
            num_steps=args.num_steps,
            device=args.device,
            resume=not args.no_resume,
            in_process=not args.subprocess,
        )
    winners = analyze_tune(
        root,
        manifest,
        pdes,
        tasks,
        args.tune_samples_per_test_type,
        args.num_steps,
        args.analysis_label,
    )
    if args.phase in {"holdout", "all"}:
        run_phase(
            root,
            phase="holdout",
            pdes=pdes,
            tasks=tasks,
            sample_count=args.holdout_samples_per_test_type,
            microbatches=microbatches,
            num_steps=args.num_steps,
            device=args.device,
            resume=not args.no_resume,
            winners=winners,
            in_process=not args.subprocess,
        )
        analyze_holdout(
            root,
            manifest,
            pdes,
            tasks,
            args.holdout_samples_per_test_type,
            args.num_steps,
            winners,
            args.analysis_label,
        )
    elif args.phase == "analyze" and (root / "runs" / "holdout").is_dir():
        analyze_holdout(
            root,
            manifest,
            pdes,
            tasks,
            args.holdout_samples_per_test_type,
            args.num_steps,
            winners,
            args.analysis_label,
        )


if __name__ == "__main__":
    main()

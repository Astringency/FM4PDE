#!/usr/bin/env python3
"""Tune inverse zeta values on hard id/smooth/rough samples and validate them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sampling.config import load_config
from scripts.tuning.make_inference_checkpoint import make_inference_checkpoint


TEST_TYPES = ("id", "smooth", "rough")
PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded")
BASELINES = {
    "poisson": {"zeta_obs_u": 360_000_000.0, "zeta_pde": 0.3, "clip_threshold": 50.0},
    "helmholtz": {"zeta_obs_u": 320_000_000.0, "zeta_pde": 0.03, "clip_threshold": 50.0},
    "darcy": {"zeta_obs_u": 2_000_000_000.0, "zeta_pde": 0.03, "clip_threshold": 50.0},
    "nsnonbounded": {"zeta_obs_u": 240_000.0, "zeta_pde": 0.3, "clip_threshold": 50.0},
}
DEFAULT_MICROBATCH = {"poisson": 5, "helmholtz": 3, "darcy": 2, "nsnonbounded": 1}
MAX_PDE_RESIDUAL_RATIO = 1.25
MAX_SOLUTION_ERROR_RATIO = 1.05


def candidates_for(pde: str) -> list[dict[str, Any]]:
    base = BASELINES[pde]
    if pde == "nsnonbounded":
        return [
            {"candidate": "obs_half", **base, "zeta_obs_u": base["zeta_obs_u"] / 2.0},
            {"candidate": "baseline", **base},
            {"candidate": "obs_double", **base, "zeta_obs_u": base["zeta_obs_u"] * 2.0},
            {"candidate": "obs_quadruple", **base, "zeta_obs_u": base["zeta_obs_u"] * 4.0},
            {"candidate": "clip_double", **base, "clip_threshold": base["clip_threshold"] * 2.0},
            {"candidate": "obs_only", **base, "zeta_pde": 0.0},
        ]
    # The earlier cross-offset sweep was monotone up to the current value, so
    # spend the hard-sample budget on the unexplored upward range.
    return [
        {"candidate": "baseline", **base},
        {"candidate": "obs_double", **base, "zeta_obs_u": base["zeta_obs_u"] * 2.0},
        {"candidate": "obs_quadruple", **base, "zeta_obs_u": base["zeta_obs_u"] * 4.0},
        {"candidate": "obs_octuple", **base, "zeta_obs_u": base["zeta_obs_u"] * 8.0},
        {"candidate": "obs_16x", **base, "zeta_obs_u": base["zeta_obs_u"] * 16.0},
        {"candidate": "obs_32x", **base, "zeta_obs_u": base["zeta_obs_u"] * 32.0},
        {"candidate": "obs_64x", **base, "zeta_obs_u": base["zeta_obs_u"] * 64.0},
        {"candidate": "obs_only", **base, "zeta_pde": 0.0},
    ]


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


def safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 1.0 if value == 0.0 else math.inf
    return value / baseline


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def parse_pdes(value: str) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in selected if item not in PDES]
    if invalid or not selected:
        raise ValueError(f"Invalid PDE list: {value!r}")
    return selected


def parse_microbatches(values: list[str]) -> dict[str, int]:
    resolved = dict(DEFAULT_MICROBATCH)
    for value in values:
        pde, separator, raw_size = value.partition("=")
        if not separator or pde not in PDES or int(raw_size) < 1:
            raise ValueError(f"Invalid --microbatch value: {value!r}")
        resolved[pde] = int(raw_size)
    return resolved


def ensure_checkpoint(root: Path, pde: str) -> Path:
    destination = root / "checkpoints" / f"{pde}_inference.pth"
    if destination.is_file():
        return destination.resolve()
    config = load_config(Path("configs/main/inverse") / f"{pde}.yaml")
    make_inference_checkpoint(
        Path(config.checkpoint_path),
        destination,
        gradient_checkpointing=pde == "nsnonbounded",
    )
    return destination.resolve()


def chunk_plan(sample_count: int, microbatch: int) -> list[tuple[int, int]]:
    return [(offset, min(microbatch, sample_count - offset)) for offset in range(0, sample_count, microbatch)]


def analysis_path(root: Path, stem: str, pdes: list[str]) -> Path:
    """Keep partial/distributed analyses from overwriting one another."""
    selected = [pde for pde in PDES if pde in set(pdes)]
    suffix = "" if selected == list(PDES) else "_" + "_".join(selected)
    return root / f"{stem}{suffix}.csv"


def find_run_artifact(job_root: Path, filename: str) -> Path | None:
    candidates = sorted(job_root.rglob(filename), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _resume_status_matches(
    status: dict[str, Any],
    signature: dict[str, Any],
    *,
    metrics_path: Path,
) -> bool:
    if status.get("status") != "ok" or not metrics_path.is_file():
        return False
    if status.get("job_signature") == signature:
        return True

    # Results written before job signatures were introduced are safe only for
    # a single full batch. Chunked jobs cannot prove that their stochastic
    # noise came from the requested source-batch shape and must be rerun.
    if int(signature["source_batch_size"]) != int(signature["batch_size"]):
        return False
    legacy_fields = (
        "phase",
        "split",
        "pde",
        "candidate",
        "test_type",
        "offset",
        "batch_size",
        "zeta_obs_u",
        "zeta_pde",
        "clip_threshold",
        "num_steps",
    )
    return all(status.get(field) == signature[field] for field in legacy_fields)


def run_job(
    root: Path,
    *,
    pde: str,
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
    job_root = root / "runs" / phase / pde / name / test_type / f"offset-{offset}-batch-{batch_size}"
    status_path = job_root / "job.json"
    data_path = (root / "subsets" / split / f"{pde}_{test_type}.mat").resolve()
    source_indices = list(range(offset, offset + batch_size))
    signature = {
        "phase": phase,
        "split": split,
        "pde": pde,
        "candidate": name,
        "test_type": test_type,
        "offset": offset,
        "batch_size": batch_size,
        "source_batch_size": source_batch_size,
        "source_indices": source_indices,
        "zeta_obs_u": candidate["zeta_obs_u"],
        "zeta_pde": candidate["zeta_pde"],
        "clip_threshold": candidate["clip_threshold"],
        "num_steps": num_steps,
        "checkpoint": _file_fingerprint(checkpoint),
        "data": _file_fingerprint(data_path),
    }
    if resume and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        metrics_path = Path(status.get("metrics_path", ""))
        if _resume_status_matches(status, signature, metrics_path=metrics_path):
            print(f"SKIP {phase} {pde} {name} {test_type} offset={offset} batch={batch_size}", flush=True)
            return status
        print(f"STALE {phase} {pde} {name} {test_type} offset={offset}; rerunning", flush=True)

    job_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "sampling.runner",
        "--config",
        f"configs/main/inverse/{pde}.yaml",
    ]
    overrides = {
        "data_path": str(data_path),
        "checkpoint_path": str(checkpoint),
        "test_type": test_type,
        "output_dir": str(job_root.resolve()),
        "ablation_group": f"hard_inverse_{phase}_{name}_{test_type}",
        "task": "inverse",
        "batch_size": batch_size,
        "offset": offset,
        "num_steps": num_steps,
        "sample_seed": 0,
        "mask_seed": 0,
        "initial_noise_source_batch_size": source_batch_size,
        "initial_noise_source_indices": source_indices,
        "zeta_obs_a": 0,
        "zeta_obs_u": candidate["zeta_obs_u"],
        "zeta_pde": candidate["zeta_pde"],
        "clip_mode": "global_norm",
        "clip_threshold": candidate["clip_threshold"],
        "device": device,
        "save_plots": False,
    }
    for key, value in overrides.items():
        command.extend(("--override", f"{key}={value}"))

    log_path = job_root / "runner.log"
    print(
        f"RUN  {phase} {pde} {name} {test_type} offset={offset} batch={batch_size} "
        f"zu={candidate['zeta_obs_u']:g}",
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
                config = load_config(f"configs/main/inverse/{pde}.yaml", overrides=overrides)
                with redirect_stdout(log), redirect_stderr(log):
                    run_single_ablation(config, checkpoint_bundle=checkpoint_bundle)
                returncode = 0
            except Exception as exc:  # Keep a resumable job record before surfacing the failure.
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
        "candidate": name,
        "test_type": test_type,
        "offset": offset,
        "batch_size": batch_size,
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
        raise RuntimeError(f"Job failed: {pde}/{name}/{test_type}/{offset}\n{tail}")
    print(f"DONE {pde} {name} {test_type} offset={offset} elapsed={elapsed:.1f}s", flush=True)
    return status


def _status_matches_analysis_plan(
    status: dict[str, Any],
    *,
    source_batch_size: int,
    num_steps: int,
) -> bool:
    """Reject jobs from an earlier pilot or a different sampling budget."""
    signature = status.get("job_signature")
    if isinstance(signature, dict) and "source_batch_size" in signature:
        return (
            int(signature["source_batch_size"]) == source_batch_size
            and int(signature["num_steps"]) == num_steps
        )
    # Legacy full-batch jobs predate source-shape signatures. They are only
    # unambiguous when a single job covers the complete requested batch.
    return (
        int(status.get("offset", -1)) == 0
        and int(status.get("batch_size", -1)) == source_batch_size
        and int(status.get("num_steps", -1)) == num_steps
    )


def _local_status_artifact(
    status_path: Path,
    status: dict[str, Any],
    key: str,
    filename: str,
) -> Path | None:
    """Resolve an artifact after a run tree has been copied between servers."""
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
    source_lookup = {
        (row["pde"], row["test_type"], row["split"], int(row["subset_index"])): row
        for row in manifest
    }
    collected: dict[tuple[str, str, str, int], tuple[int, dict[str, Any]]] = {}
    status_paths = sorted(
        (root / "runs" / phase).rglob("job.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    for status_path in status_paths:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "ok" or not _status_matches_analysis_plan(
            status,
            source_batch_size=source_batch_size,
            num_steps=num_steps,
        ):
            continue
        metrics_path = _local_status_artifact(
            status_path, status, "metrics_path", "metrics_per_sample.csv"
        )
        if metrics_path is None:
            continue
        offset = int(status["offset"])
        for row in read_csv(metrics_path):
            subset_index = offset + int(float(row["sample_index"]))
            source = source_lookup[(status["pde"], status["test_type"], status["split"], subset_index)]
            logical_key = (
                str(status["pde"]),
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
                    "candidate": status["candidate"],
                    "test_type": status["test_type"],
                    "subset_index": subset_index,
                    "source_sample_id": int(source["source_sample_id"]),
                    "hard_rank": int(source["hard_rank"]),
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


def summarize_tune(rows: list[dict[str, Any]], pdes: list[str], sample_count: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["pde"] in pdes and int(row["subset_index"]) < sample_count:
            grouped[(row["pde"], row["candidate"])].append(row)
    summaries: list[dict[str, Any]] = []
    expected = sample_count * len(TEST_TYPES)
    for (pde, candidate), group in sorted(grouped.items()):
        if len(group) != expected:
            continue
        errors = [float(row["rel_l2_a"]) for row in group]
        summary = {
            "pde": pde,
            "candidate": candidate,
            "zeta_obs_u": group[0]["zeta_obs_u"],
            "zeta_pde": group[0]["zeta_pde"],
            "clip_threshold": group[0]["clip_threshold"],
            "n": len(group),
            "rel_l2_a_mean": statistics.fmean(errors),
            "rel_l2_a_median": statistics.median(errors),
            "rel_l2_a_p90": percentile(errors, 0.9),
            "rel_l2_a_max": max(errors),
            "rel_l2_u_mean": statistics.fmean(float(row["rel_l2_u"]) for row in group),
            "pde_residual_norm_mean": statistics.fmean(float(row["pde_residual_norm"]) for row in group),
        }
        summary["robust_score"] = (
            0.5 * summary["rel_l2_a_mean"]
            + 0.3 * summary["rel_l2_a_p90"]
            + 0.2 * summary["rel_l2_a_max"]
        )
        for test_type in TEST_TYPES:
            test_rows = [row for row in group if row["test_type"] == test_type]
            summary[f"rel_l2_a_mean_{test_type}"] = statistics.fmean(
                float(row["rel_l2_a"]) for row in test_rows
            )
            summary[f"rel_l2_u_mean_{test_type}"] = statistics.fmean(
                float(row["rel_l2_u"]) for row in test_rows
            )
            summary[f"pde_residual_norm_mean_{test_type}"] = statistics.fmean(
                float(row["pde_residual_norm"]) for row in test_rows
            )
        summaries.append(summary)
    summaries.sort(key=lambda row: (pdes.index(str(row["pde"])), float(row["robust_score"])))
    return summaries


def select_winners(summaries: list[dict[str, Any]], pdes: list[str]) -> list[dict[str, Any]]:
    winners = []
    for pde in pdes:
        candidates = [row for row in summaries if row["pde"] == pde]
        if not candidates:
            raise RuntimeError(f"No complete tuning candidate for {pde}")
        baseline = next(row for row in candidates if row["candidate"] == "baseline")
        for candidate in candidates:
            scopes = ("all", *TEST_TYPES)
            for scope in scopes:
                suffix = "" if scope == "all" else f"_{scope}"
                residual_key = f"pde_residual_norm_mean{suffix}"
                solution_key = f"rel_l2_u_mean{suffix}"
                candidate[f"pde_residual_ratio_vs_baseline{suffix}"] = safe_ratio(
                    float(candidate[residual_key]), float(baseline[residual_key])
                )
                candidate[f"solution_error_ratio_vs_baseline{suffix}"] = safe_ratio(
                    float(candidate[solution_key]), float(baseline[solution_key])
                )
            candidate["passes_guardrails"] = all(
                float(candidate[f"pde_residual_ratio_vs_baseline_{test_type}"])
                <= MAX_PDE_RESIDUAL_RATIO
                and float(candidate[f"solution_error_ratio_vs_baseline_{test_type}"])
                <= MAX_SOLUTION_ERROR_RATIO
                for test_type in TEST_TYPES
            )
        eligible = [candidate for candidate in candidates if candidate["passes_guardrails"]]
        if not eligible:
            raise RuntimeError(f"No tuning candidate passed guardrails for {pde}")
        winner = dict(min(eligible, key=lambda row: float(row["robust_score"])))
        winner["relative_robust_improvement_vs_baseline"] = (
            float(baseline["robust_score"]) - float(winner["robust_score"])
        ) / float(baseline["robust_score"])
        winners.append(winner)
    return winners


def analyze_tune(
    root: Path,
    manifest: list[dict[str, str]],
    pdes: list[str],
    sample_count: int,
    num_steps: int,
) -> list[dict[str, Any]]:
    rows = [
        row for row in collect_phase_rows(
            root,
            "tune",
            manifest,
            source_batch_size=sample_count,
            num_steps=num_steps,
        )
        if row["pde"] in pdes and int(row["subset_index"]) < sample_count
    ]
    write_csv(analysis_path(root, "tune_per_sample", pdes), rows)
    summaries = summarize_tune(rows, pdes, sample_count)
    winners = select_winners(summaries, pdes)
    write_csv(analysis_path(root, "tune_summary", pdes), summaries)
    selected_path = analysis_path(root, "selected_params", pdes)
    write_csv(selected_path, winners)
    selected_path.with_suffix(".json").write_text(json.dumps(winners, indent=2), encoding="utf-8")
    for row in winners:
        print(
            f"SELECT {row['pde']} {row['candidate']} zu={float(row['zeta_obs_u']):g} "
            f"score={100 * float(row['robust_score']):.2f}% "
            f"improvement={100 * float(row['relative_robust_improvement_vs_baseline']):.2f}%",
            flush=True,
        )
    return winners


def candidate_from_winner(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": str(row["candidate"]),
        "zeta_obs_u": float(row["zeta_obs_u"]),
        "zeta_pde": float(row["zeta_pde"]),
        "clip_threshold": float(row["clip_threshold"]),
    }


def analyze_holdout(
    root: Path,
    manifest: list[dict[str, str]],
    pdes: list[str],
    sample_count: int,
    num_steps: int,
    winners: list[dict[str, Any]],
) -> None:
    rows = [
        row for row in collect_phase_rows(
            root,
            "holdout",
            manifest,
            source_batch_size=sample_count,
            num_steps=num_steps,
        )
        if row["pde"] in pdes and int(row["subset_index"]) < sample_count
    ]
    write_csv(analysis_path(root, "holdout_per_sample", pdes), rows)
    lookup = {
        (row["pde"], row["candidate"], row["test_type"], row["subset_index"]): row
        for row in rows
    }
    winner_map = {str(row["pde"]): row for row in winners if row["pde"] in pdes}
    comparisons: list[dict[str, Any]] = []
    for pde in pdes:
        pde_start = len(comparisons)
        winner_name = str(winner_map[pde]["candidate"])
        paired = []
        for test_type in TEST_TYPES:
            for subset_index in range(sample_count):
                baseline = lookup[(pde, "baseline", test_type, subset_index)]
                winner = lookup[(pde, winner_name, test_type, subset_index)]
                paired.append((test_type, baseline, winner))
        for test_type in ("all", *TEST_TYPES):
            selected = [item for item in paired if test_type == "all" or item[0] == test_type]
            baseline_values = [float(item[1]["rel_l2_a"]) for item in selected]
            winner_values = [float(item[2]["rel_l2_a"]) for item in selected]
            deltas = [winner - baseline for baseline, winner in zip(baseline_values, winner_values)]
            baseline_solution = statistics.fmean(float(item[1]["rel_l2_u"]) for item in selected)
            winner_solution = statistics.fmean(float(item[2]["rel_l2_u"]) for item in selected)
            baseline_residual = statistics.fmean(
                float(item[1]["pde_residual_norm"]) for item in selected
            )
            winner_residual = statistics.fmean(
                float(item[2]["pde_residual_norm"]) for item in selected
            )
            solution_ratio = safe_ratio(winner_solution, baseline_solution)
            residual_ratio = safe_ratio(winner_residual, baseline_residual)
            comparisons.append(
                {
                    "pde": pde,
                    "winner": winner_name,
                    "test_type": test_type,
                    "n": len(selected),
                    "baseline_rel_l2_a_mean": statistics.fmean(baseline_values),
                    "winner_rel_l2_a_mean": statistics.fmean(winner_values),
                    "absolute_delta": statistics.fmean(deltas),
                    "relative_improvement": -statistics.fmean(deltas) / statistics.fmean(baseline_values),
                    "win_rate": sum(delta < 0 for delta in deltas) / len(deltas),
                    "baseline_rel_l2_a_p90": percentile(baseline_values, 0.9),
                    "winner_rel_l2_a_p90": percentile(winner_values, 0.9),
                    "baseline_rel_l2_u_mean": baseline_solution,
                    "winner_rel_l2_u_mean": winner_solution,
                    "solution_error_ratio_vs_baseline": solution_ratio,
                    "baseline_pde_residual_norm_mean": baseline_residual,
                    "winner_pde_residual_norm_mean": winner_residual,
                    "pde_residual_ratio_vs_baseline": residual_ratio,
                    "passes_guardrails": (
                        residual_ratio <= MAX_PDE_RESIDUAL_RATIO
                        and solution_ratio <= MAX_SOLUTION_ERROR_RATIO
                    ),
                }
            )
        validated = all(
            bool(row["passes_guardrails"])
            for row in comparisons[pde_start:]
            if row["test_type"] in TEST_TYPES
        )
        for row in comparisons[pde_start:]:
            row["validated_across_test_types"] = validated
    write_csv(analysis_path(root, "holdout_comparison", pdes), comparisons)
    for row in comparisons:
        if row["test_type"] == "all":
            print(
                f"HOLDOUT {row['pde']} {row['winner']} n={row['n']} "
                f"mean {100 * row['baseline_rel_l2_a_mean']:.2f}% -> "
                f"{100 * row['winner_rel_l2_a_mean']:.2f}% "
                f"improvement={100 * row['relative_improvement']:.2f}% "
                f"win_rate={100 * row['win_rate']:.1f}% "
                f"validated={row['validated_across_test_types']}",
                flush=True,
            )


def run_phase(
    root: Path,
    *,
    phase: str,
    pdes: list[str],
    sample_count: int,
    microbatches: dict[str, int],
    num_steps: int,
    device: str,
    resume: bool,
    winners: list[dict[str, Any]] | None = None,
    in_process: bool = True,
) -> None:
    winner_map = {} if winners is None else {row["pde"]: candidate_from_winner(row) for row in winners}
    for pde in pdes:
        checkpoint = ensure_checkpoint(root, pde)
        checkpoint_bundle = None
        if in_process:
            import torch
            from sampling.model_io import load_fm4pde_checkpoint_bundle

            print(f"LOAD {pde} checkpoint once for persistent tuning jobs", flush=True)
            checkpoint_bundle = load_fm4pde_checkpoint_bundle(
                str(checkpoint), pde, device=torch.device(device), wrap=True
            )
        if phase == "tune":
            candidates = candidates_for(pde)
            split = "tune"
        else:
            winner = winner_map[pde]
            candidates = [{"candidate": "baseline", **BASELINES[pde]}]
            if winner["candidate"] != "baseline":
                candidates.append(winner)
            split = "holdout"
        for candidate in candidates:
            for test_type in TEST_TYPES:
                for offset, batch_size in chunk_plan(sample_count, microbatches[pde]):
                    run_job(
                        root,
                        pde=pde,
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
        if phase == "holdout" and winner_map[pde]["candidate"] == "baseline":
            # Materialize an alias so paired analysis has both logical candidates.
            source = root / "runs" / phase / pde / "baseline"
            alias = root / "runs" / phase / pde / winner_map[pde]["candidate"]
            if source != alias:
                raise AssertionError("Unexpected baseline alias path")
        if checkpoint_bundle is not None:
            del checkpoint_bundle
            import torch

            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/inverse_hard_tuning"))
    parser.add_argument("--phase", choices=("tune", "holdout", "all", "analyze"), default="all")
    parser.add_argument("--pdes", default=",".join(PDES))
    parser.add_argument("--samples-per-test-type", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", action="append", default=[], help="Override as pde=N")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--subprocess", action="store_true", help="Reload the checkpoint in an isolated process per job")
    args = parser.parse_args()
    if args.samples_per_test_type < 1 or args.samples_per_test_type > 10:
        parser.error("--samples-per-test-type must be in [1, 10]")
    pdes = parse_pdes(args.pdes)
    microbatches = parse_microbatches(args.microbatch)
    root = args.root.resolve()
    manifest_path = root / "hard_samples.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run prepare_hard_inverse_samples.py first: {manifest_path}")
    manifest = read_csv(manifest_path)

    if args.phase in {"tune", "all"}:
        run_phase(
            root,
            phase="tune",
            pdes=pdes,
            sample_count=args.samples_per_test_type,
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
        args.samples_per_test_type,
        args.num_steps,
    )
    if args.phase in {"holdout", "all"}:
        run_phase(
            root,
            phase="holdout",
            pdes=pdes,
            sample_count=args.samples_per_test_type,
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
            args.samples_per_test_type,
            args.num_steps,
            winners,
        )
    elif args.phase == "analyze" and any((root / "runs" / "holdout").rglob("job.json")):
        analyze_holdout(
            root,
            manifest,
            pdes,
            args.samples_per_test_type,
            args.num_steps,
            winners,
        )


if __name__ == "__main__":
    main()

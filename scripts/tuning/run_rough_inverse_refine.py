#!/usr/bin/env python3
"""Rough-priority refinement for the four sparse inverse configurations.

The current ``configs/main/inverse`` values are treated as the baseline (that
is, the promoted tuned1 settings).  Candidate selection gives Rough twice the
weight of ID and Smooth by default, while retaining per-distribution primary,
auxiliary-error, and PDE-residual guardrails.  The runner reuses the balanced
hard/good/random tune and holdout subsets prepared for round3.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tuning import run_balanced_sampling_tuning as balanced


PDES = ("poisson", "helmholtz", "darcy", "nsnonbounded")
TEST_TYPES = ("id", "smooth", "rough")
PROFILES = ("quick", "standard")


def _parse_names(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    selected = [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]
    invalid = [item for item in selected if item not in allowed]
    if invalid or not selected or len(set(selected)) != len(selected):
        raise ValueError(f"Invalid {label} list: {value!r}")
    return selected


def parse_distribution_weights(value: str) -> dict[str, float]:
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
    missing = [name for name in TEST_TYPES if name not in weights]
    if missing:
        raise ValueError(f"Missing distribution weights: {missing}")
    return weights


def _number_token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _candidate(
    base: dict[str, Any],
    *,
    obs_scale: float = 1.0,
    pde_scale: float = 1.0,
    clip_threshold: float | None = None,
) -> dict[str, Any]:
    clip = float(base["clip_threshold"] if clip_threshold is None else clip_threshold)
    return {
        **base,
        "candidate": (
            f"rough_obs{_number_token(obs_scale)}_pde{_number_token(pde_scale)}"
            f"_clip{_number_token(clip)}"
        ),
        "zeta_obs_u": float(base["zeta_obs_u"]) * obs_scale,
        "zeta_pde": float(base["zeta_pde"]) * pde_scale,
        "clip_threshold": clip,
    }


def _deduplicate(base: dict[str, Any], variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("zeta_obs_a", "zeta_obs_u", "zeta_pde", "clip_threshold")
    baseline = {**base, "candidate": "baseline"}
    result = [baseline]
    seen = {tuple(float(baseline[field]) for field in fields)}
    for candidate in variants:
        signature = tuple(float(candidate[field]) for field in fields)
        if signature not in seen:
            result.append(candidate)
            seen.add(signature)
    return result


def candidates_for(pde: str, task: str = "inverse", profile: str = "quick") -> list[dict[str, Any]]:
    """Return a local grid centered on the currently promoted tuned1 config."""
    if pde not in PDES or task != "inverse" or profile not in PROFILES:
        raise ValueError(f"Unsupported refinement request: {pde}/{task}/{profile}")
    base = dict(balanced.baseline_for(pde, task))

    if profile == "standard":
        if pde == "poisson":
            obs_scales, pde_scales, clips = (0.875, 1.0, 1.125), (1.0, 1.5, 2.0), (50.0,)
        elif pde == "helmholtz":
            obs_scales, pde_scales, clips = (5.0 / 6.0, 1.0, 7.0 / 6.0), (1.0, 1.5, 2.0), (50.0,)
        elif pde == "darcy":
            obs_scales, pde_scales, clips = (0.875, 1.0, 1.125), (1.0, 1.5, 2.0), (80.0, 90.0, 100.0)
        else:
            obs_scales, pde_scales, clips = (0.75, 1.0, 1.25), (1.0, 1.5, 2.0), (100.0, 110.0, 120.0)
        variants = [
            _candidate(base, obs_scale=obs, pde_scale=pde_weight, clip_threshold=clip)
            for obs in obs_scales
            for pde_weight in pde_scales
            for clip in clips
        ]
        return _deduplicate(base, variants)

    if pde == "poisson":
        specs = [
            (0.875, 1.0, 50.0),
            (1.125, 1.0, 50.0),
            (1.0, 1.5, 50.0),
            (1.125, 1.5, 50.0),
            (1.0, 2.0, 50.0),
        ]
    elif pde == "helmholtz":
        specs = [
            (5.0 / 6.0, 1.0, 50.0),
            (7.0 / 6.0, 1.0, 50.0),
            (1.0, 1.5, 50.0),
            (7.0 / 6.0, 1.5, 50.0),
            (1.0, 2.0, 50.0),
        ]
    elif pde == "darcy":
        specs = [
            (0.875, 1.0, 90.0),
            (1.125, 1.0, 90.0),
            (1.0, 1.0, 80.0),
            (1.0, 1.0, 100.0),
            (1.0, 1.5, 90.0),
            (1.0, 2.0, 90.0),
            (0.875, 1.5, 90.0),
            (1.125, 1.5, 90.0),
            (1.0, 1.5, 80.0),
            (1.0, 1.5, 100.0),
        ]
    else:
        specs = [
            (1.0, 1.0, 105.0),
            (1.0, 1.0, 110.0),
            (1.0, 1.0, 115.0),
            (1.0, 1.0, 120.0),
            (1.0, 1.5, 100.0),
            (1.0, 2.0, 100.0),
            (1.0, 1.5, 110.0),
            (1.0, 1.5, 120.0),
            (0.75, 1.5, 110.0),
            (1.25, 1.5, 110.0),
        ]
    return _deduplicate(
        base,
        [
            _candidate(base, obs_scale=obs, pde_scale=pde_weight, clip_threshold=clip)
            for obs, pde_weight, clip in specs
        ],
    )


def apply_weighted_scores(
    summaries: list[dict[str, Any]],
    distribution_weights: dict[str, float],
    residual_penalty: float,
) -> list[dict[str, Any]]:
    """Replace the round3 score with a Rough-weighted, residual-aware score."""
    total_weight = sum(distribution_weights.values())
    for row in summaries:
        row["round3_selection_score"] = float(row["selection_score"])
        weighted_score = 0.0
        for test_type in TEST_TYPES:
            hard_mean = float(row[f"hard_primary_mean_ratio_{test_type}"])
            hard_p90 = float(row[f"hard_primary_p90_ratio_{test_type}"])
            good_mean = float(row[f"good_primary_mean_ratio_{test_type}"])
            random_mean = float(row[f"random_primary_mean_ratio_{test_type}"])
            pde_ratio = float(row[f"pde_residual_ratio_{test_type}"])
            hard_center = 0.5 * hard_mean + 0.5 * hard_p90
            score = (
                0.55 * hard_center
                + 0.20 * hard_mean
                + 0.15 * random_mean
                + 0.10 * good_mean
                + residual_penalty * max(0.0, pde_ratio - 1.0)
            )
            row[f"weighted_selection_score_{test_type}"] = score
            weighted_score += distribution_weights[test_type] * score
        row["selection_score"] = weighted_score / total_weight
        row["distribution_weights"] = ",".join(
            f"{name}={distribution_weights[name]:g}" for name in TEST_TYPES
        )
        row["residual_penalty"] = residual_penalty
    return summaries


def _ensure_link(destination: Path, source: Path, *, directory: bool = False) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Refusing to replace existing path: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=directory)


def prepare_artifact_root(root: Path, source_root: Path, pdes: list[str]) -> None:
    if root.resolve() == source_root.resolve():
        return
    root.mkdir(parents=True, exist_ok=True)
    _ensure_link(root / "balanced_samples.csv", source_root / "balanced_samples.csv")
    _ensure_link(root / "subsets", source_root / "subsets", directory=True)
    source_checkpoints = source_root / "checkpoints"
    if source_checkpoints.is_dir():
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        for pde in pdes:
            source = source_checkpoints / f"{pde}_inference.pth"
            if source.is_file():
                _ensure_link(root / "checkpoints" / source.name, source)


def print_plan(
    *,
    phase: str,
    pdes: list[str],
    profile: str,
    tune_count: int,
    holdout_count: int,
    microbatches: dict[str, int],
    weights: dict[str, float],
    num_steps: int,
    device: str,
    baseline_only: bool = False,
) -> None:
    tune_jobs = 0
    holdout_jobs = 0
    for pde in pdes:
        candidate_count = 1 if baseline_only else len(candidates_for(pde, profile=profile))
        tune_chunks = len(balanced.chunk_plan(tune_count, microbatches[pde]))
        holdout_chunks = len(balanced.chunk_plan(holdout_count, microbatches[pde]))
        pde_tune_jobs = candidate_count * len(TEST_TYPES) * tune_chunks
        pde_holdout_jobs = 2 * len(TEST_TYPES) * holdout_chunks
        tune_jobs += pde_tune_jobs
        holdout_jobs += pde_holdout_jobs
        print(
            f"PLAN pde={pde} task=inverse candidates={candidate_count} "
            f"tune_jobs={pde_tune_jobs} holdout_jobs_at_most={pde_holdout_jobs} device={device}",
            flush=True,
        )
    print(
        f"PLAN SUMMARY phase={phase} profile={profile} distribution_weights={weights} "
        f"steps={num_steps} tune_jobs={tune_jobs} holdout_jobs_at_most={holdout_jobs}",
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/artifacts/rough_inverse_refine")
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/artifacts/balanced_sampling_tuning"),
        help="Round3 artifact root containing balanced_samples.csv and subsets/",
    )
    parser.add_argument("--phase", choices=("tune", "holdout", "all", "analyze"), default="all")
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--pdes", default=",".join(PDES))
    parser.add_argument("--distribution-weights", default="id=1,smooth=1,rough=2")
    parser.add_argument("--max-pde-residual-ratio", type=float, default=1.05)
    parser.add_argument("--residual-penalty", type=float, default=0.10)
    parser.add_argument("--tune-samples-per-test-type", type=int, default=20)
    parser.add_argument("--holdout-samples-per-test-type", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", action="append", default=[], help="Override as pde=N")
    parser.add_argument("--analysis-label", default="rough_refine")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the current tuned1 config; useful for a paired num_steps screen",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--subprocess", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        pdes = _parse_names(args.pdes, PDES, "PDE")
        weights = parse_distribution_weights(args.distribution_weights)
    except ValueError as exc:
        parser.error(str(exc))
    balanced._validate_sample_count(
        parser, args.tune_samples_per_test_type, "--tune-samples-per-test-type"
    )
    balanced._validate_sample_count(
        parser, args.holdout_samples_per_test_type, "--holdout-samples-per-test-type"
    )
    if (
        not math.isfinite(args.max_pde_residual_ratio)
        or args.max_pde_residual_ratio < 1.0
        or not math.isfinite(args.residual_penalty)
        or args.residual_penalty < 0.0
    ):
        parser.error("residual ratio must be >= 1 and residual penalty must be >= 0")
    microbatches = balanced.parse_microbatches(args.microbatch)

    if args.plan_only:
        print_plan(
            phase=args.phase,
            pdes=pdes,
            profile=args.profile,
            tune_count=args.tune_samples_per_test_type,
            holdout_count=args.holdout_samples_per_test_type,
            microbatches=microbatches,
            weights=weights,
            num_steps=args.num_steps,
            device=args.device,
            baseline_only=args.baseline_only,
        )
        return 0

    root = args.root.resolve()
    source_root = args.source_root.resolve()
    prepare_artifact_root(root, source_root, pdes)
    manifest = balanced.read_csv(root / "balanced_samples.csv")

    original_summarize = balanced.summarize_tune

    def weighted_summarize(
        rows: list[dict[str, Any]],
        selected_pdes: list[str],
        tasks: list[str],
        sample_count: int,
    ) -> list[dict[str, Any]]:
        return apply_weighted_scores(
            original_summarize(rows, selected_pdes, tasks, sample_count),
            weights,
            args.residual_penalty,
        )

    # Reuse the tested round3 runner/collector with a local grid and score.
    def active_candidates(pde: str, task: str) -> list[dict[str, Any]]:
        candidates = candidates_for(pde, task, args.profile)
        return candidates[:1] if args.baseline_only else candidates

    balanced.candidates_for = active_candidates
    balanced.summarize_tune = weighted_summarize
    balanced.MAX_PDE_RESIDUAL_RATIO = args.max_pde_residual_ratio
    tasks = ["inverse"]

    if args.phase in {"tune", "all"}:
        balanced.run_phase(
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
    winners = balanced.analyze_tune(
        root,
        manifest,
        pdes,
        tasks,
        args.tune_samples_per_test_type,
        args.num_steps,
        args.analysis_label,
    )
    if args.phase in {"holdout", "all"}:
        balanced.run_phase(
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
        balanced.analyze_holdout(
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
        balanced.analyze_holdout(
            root,
            manifest,
            pdes,
            tasks,
            args.holdout_samples_per_test_type,
            args.num_steps,
            winners,
            args.analysis_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

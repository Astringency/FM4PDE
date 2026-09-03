import csv
import os
import subprocess
from pathlib import Path

from scripts.tuning.run_hard_inverse_tuning import (
    _resume_status_matches,
    _status_matches_analysis_plan,
    analysis_path,
    candidates_for,
    chunk_plan,
    select_winners,
)
from scripts.tuning.prepare_balanced_sampling_samples import _select_strata
from scripts.tuning.run_balanced_sampling_tuning import (
    _scope_stats as balanced_scope_stats,
    analysis_path as balanced_analysis_path,
    candidates_for as balanced_candidates_for,
)
from scripts.tuning.run_six_pde_sampling_tuning import (
    CANDIDATE_FIELDS as SIX_PDE_CANDIDATE_FIELDS,
    PDES as SIX_PDES,
    TASKS as SIX_PDE_TASKS,
    baseline_for as six_pde_baseline_for,
    candidates_for as six_pde_candidates_for,
    _final_losses_are_finite as six_pde_final_losses_are_finite,
    _parse_distribution_weights as parse_six_pde_distribution_weights,
    refined_candidates_for as six_pde_refined_candidates_for,
    select_tune_winners as select_six_pde_tune_winners,
    summarize_tune as summarize_six_pde_tune,
    write_csv as write_six_pde_csv,
)
from scripts.tuning.select_inverse_params import select_rows


def _run_plan(script: str, updates: dict[str, str]) -> list[str]:
    env = os.environ.copy()
    env.update({"PLAN_ONLY": "true", "AGGREGATE": "false", **updates})
    result = subprocess.run(
        ["bash", script],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        env=env,
    )
    return [line for line in result.stdout.splitlines() if line.startswith("PLAN ")]


def test_burger_tuning_plan_expands_the_requested_cross_product():
    lines = _run_plan(
        "scripts/tuning/run_burger_tuning.sh",
        {
            "STAGE": "screen",
            "DEVICE_LIST": "cuda:0 cuda:1",
            "SENSOR_MODE_LIST": "random sensor_column",
            "SAMPLER_LIST": "stochastic deterministic",
            "ZETA_OBS_U_LIST": "3200 6400",
            "ZETA_PDE_LIST": "1 10",
            "OFFSET_LIST": "0 1000",
            "MASK_SEED_LIST": "0 11",
            "CLIP_MODE_LIST": "global_norm per_component_norm",
            "CLIP_THRESHOLD_LIST": "50 100",
            "BATCH_SIZE": "2",
        },
    )

    assert len(lines) == 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    assert {line.rsplit("device=", 1)[1] for line in lines} == {"cuda:0", "cuda:1"}


def test_inverse_debug_plan_covers_ten_non_burger_equations():
    lines = _run_plan(
        "scripts/tuning/run_inverse_debug.sh",
        {"DEVICE_LIST": "cuda:0 cuda:1"},
    )

    assert len(lines) == 10
    assert all("pde=burger" not in line for line in lines)


def test_inverse_debug_preflight_reports_missing_data_without_aggregation(tmp_path):
    output_dir = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "PDE_LIST": "darcy",
            "PDE_DATA_ROOT": str(tmp_path / "missing-data"),
            "OUTPUT_DIR": str(output_dir),
            "DEVICE_LIST": "cuda:0",
            "MAX_PARALLEL_TASKS": "1",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/tuning/run_inverse_debug.sh"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2
    assert "MISSING DATA   darcy:" in result.stderr
    assert "Set PDE_DATA_ROOT" in result.stderr
    assert "Summary:" not in result.stdout
    assert not (output_dir / "summary").exists()


def test_inverse_tuning_stabilize_plan_uses_two_offsets_and_nine_configs():
    lines = _run_plan(
        "scripts/tuning/run_inverse_tuning.sh",
        {
            "STAGE": "stabilize",
            "PDE_LIST": "heat",
            "DEVICE_LIST": "cuda:0 cuda:1",
            "OFFSET_LIST": "0 1000",
        },
    )

    assert len(lines) == 18
    assert {line.split(" offset=", 1)[1].split()[0] for line in lines} == {
        "0",
        "1000",
    }
    assert all("batch=2" in line and "pde=heat" in line for line in lines)


def test_inverse_tuning_refine_default_plan_has_96_jobs():
    lines = _run_plan(
        "scripts/tuning/run_inverse_tuning.sh",
        {
            "STAGE": "refine",
            "DEVICE_LIST": "cuda:0 cuda:1",
            "OFFSET_LIST": "0 1000",
        },
    )

    assert len(lines) == 96
    assert {line.split("pde=", 1)[1].split()[0] for line in lines} == {
        "darcy",
        "poisson",
        "helmholtz",
        "nsnonbounded",
    }


def test_six_pde_tuning_plan_is_available_without_data_or_checkpoints(tmp_path):
    lines = _run_plan(
        "scripts/tuning/run_six_pde_sampling_tuning.sh",
        {
            "PROFILE": "quick",
            "PDE_LIST": "heat",
            "TASK_LIST": "forward",
            "DEVICE_LIST": "cuda:0",
            "MAX_PARALLEL_TASKS": "1",
            "OUTPUT_DIR": str(tmp_path / "unused"),
        },
    )

    assert len(lines) == 2
    assert "pde=heat task=forward candidates=7 tune_jobs=7" in lines[0]
    assert "holdout_jobs_at_most=2" in lines[1]
    assert not (tmp_path / "unused").exists()


def test_six_pde_candidate_grids_cover_all_equations_tasks_and_sampler_families():
    for pde in SIX_PDES:
        for task in SIX_PDE_TASKS:
            candidates = six_pde_candidates_for(pde, task, "standard")
            by_name = {candidate["candidate"]: candidate for candidate in candidates}
            assert candidates[0]["candidate"] == "baseline"
            assert len(by_name) == len(candidates)
            # A newly promoted main baseline can be identical to one of these
            # named candidates, in which case candidates_for intentionally
            # deduplicates the redundant job.  Check family coverage by the
            # effective parameters instead of requiring every historical name.
            assert any(
                candidate["sampler_phase"] == "deterministic" for candidate in candidates
            )
            assert any(
                candidate["sampler_phase"] == "hybrid_s2d" for candidate in candidates
            )
            assert any(candidate["time_grid"] == "geometric" for candidate in candidates)
            assert any(candidate["step_method"] == "midpoint" for candidate in candidates)
            if task == "forward":
                assert all(
                    candidate["zeta_obs_u"] == 0.0
                    for candidate in candidates
                    if candidate["candidate"] != "baseline"
                )
            elif task == "inverse":
                assert all(
                    candidate["zeta_obs_a"] == 0.0
                    for candidate in candidates
                    if candidate["candidate"] != "baseline"
                )
            if pde == "steady_heat_conduction":
                assert "near_endpoint" not in by_name
            else:
                assert by_name["near_endpoint"]["residual_mode"] == "near_endpoint_temporal"


def test_six_pde_refined_grid_is_local_task_aware_and_deduplicated():
    for pde in SIX_PDES:
        for task in SIX_PDE_TASKS:
            baseline = six_pde_baseline_for(pde, task)
            candidates = six_pde_refined_candidates_for(pde, task)
            signatures = {
                tuple(candidate[field] for field in SIX_PDE_CANDIDATE_FIELDS)
                for candidate in candidates
            }

            assert candidates[0]["candidate"] == "baseline"
            assert len(signatures) == len(candidates)
            assert any(candidate["candidate"] == "refine_conservative" for candidate in candidates)
            if task == "forward":
                assert all(candidate["zeta_obs_u"] == baseline["zeta_obs_u"] for candidate in candidates)
            elif task == "inverse":
                assert all(candidate["zeta_obs_a"] == baseline["zeta_obs_a"] for candidate in candidates)


def test_six_pde_distribution_weights_and_final_loss_validation():
    assert parse_six_pde_distribution_weights(
        "id=1 smooth=1 rough=2", ["id", "smooth", "rough"]
    ) == {"id": 1.0, "smooth": 1.0, "rough": 2.0}
    assert six_pde_final_losses_are_finite(
        {"L_obs_a": 0.1, "L_obs_u": 0.2, "L_pde": 3.0}
    )
    assert not six_pde_final_losses_are_finite(
        {"L_obs_a": 0.1, "L_obs_u": float("nan"), "L_pde": 3.0}
    )


def test_six_pde_refine_plan_uses_rough_weighted_local_search(tmp_path):
    lines = _run_plan(
        "scripts/tuning/run_six_pde_sampling_refine.sh",
        {
            "PDE_LIST": "heat",
            "TASK_LIST": "forward",
            "DEVICE_LIST": "cuda:0",
            "MAX_PARALLEL_TASKS": "1",
            "OUTPUT_DIR": str(tmp_path / "unused"),
        },
    )

    assert len(lines) == 2
    assert "candidate_set=refined" in lines[1]
    assert "'rough': 2.0" in lines[1]
    assert not (tmp_path / "unused").exists()


def test_six_pde_winner_rejects_incomplete_and_auxiliary_regression():
    pde = "heat"
    task = "forward"
    base = dict(six_pde_baseline_for(pde, task))
    candidates = [
        base,
        {**base, "candidate": "good"},
        {**base, "candidate": "unsafe_auxiliary"},
        {**base, "candidate": "incomplete"},
    ]
    rows = []
    for candidate in candidates:
        name = candidate["candidate"]
        for test_index, test_type in enumerate(("id", "rough")):
            for index in range(2):
                if name == "incomplete" and test_type == "rough" and index == 1:
                    continue
                rel_u = {
                    "baseline": 1.0,
                    "good": 0.6,
                    "unsafe_auxiliary": 0.4,
                    "incomplete": 0.1,
                }[name]
                rows.append(
                    {
                        "pde": pde,
                        "task": task,
                        "candidate": name,
                        "test_type": test_type,
                        "sample_offset": test_index * 2 + index,
                        "rel_l2_a": 1.2 if name == "unsafe_auxiliary" else 1.0,
                        "rel_l2_u": rel_u,
                        "pde_residual_norm": 1.0,
                    }
                )

    summaries = summarize_six_pde_tune(
        rows,
        pdes=[pde],
        tasks=[task],
        candidate_map={(pde, task): candidates},
        test_types=["id", "rough"],
        expected_n=4,
    )
    by_name = {row["candidate"]: row for row in summaries}
    winner = select_six_pde_tune_winners(summaries, [pde], [task])[0]

    assert set(SIX_PDE_CANDIDATE_FIELDS) <= set(by_name["good"])
    assert "incomplete" not in by_name
    assert by_name["unsafe_auxiliary"]["passes_guardrails"] is False
    assert winner["candidate"] == "good"


def test_six_pde_csv_writer_unions_fields_from_heterogeneous_rows(tmp_path):
    destination = tmp_path / "summary.csv"

    write_six_pde_csv(
        destination,
        [
            {"candidate": "unpaired", "score": 0.5},
            {"candidate": "paired", "score": 0.7, "overall_primary_mean_ratio": 0.9},
        ],
    )

    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"candidate": "unpaired", "score": "0.5", "overall_primary_mean_ratio": ""},
        {"candidate": "paired", "score": "0.7", "overall_primary_mean_ratio": "0.9"},
    ]


def test_inverse_selector_rejects_incomplete_configs_and_ranks_robust_error():
    base = {
        "pde": "darcy",
        "clip_mode": "global_norm",
        "clip_threshold": "50",
        "rel_l2_a_median": "0.1",
        "rel_l2_u_mean": "0.02",
        "pde_residual_norm_mean": "0.5",
    }
    rows = [
        {
            **base,
            "zeta_obs_u": "1",
            "zeta_pde": "0.1",
            "rel_l2_a_n": "4",
            "rel_l2_a_mean": "0.10",
            "rel_l2_a_p90": "0.15",
            "rel_l2_a_max": "0.20",
        },
        {
            **base,
            "zeta_obs_u": "2",
            "zeta_pde": "0.1",
            "rel_l2_a_n": "4",
            "rel_l2_a_mean": "0.11",
            "rel_l2_a_p90": "0.12",
            "rel_l2_a_max": "0.13",
        },
        {
            **base,
            "zeta_obs_u": "3",
            "zeta_pde": "0.1",
            "rel_l2_a_n": "2",
            "rel_l2_a_mean": "0.01",
            "rel_l2_a_p90": "0.01",
            "rel_l2_a_max": "0.01",
        },
    ]

    selected = select_rows(rows, expected_n=4, top_k=2)

    assert [row["zeta_obs_u"] for row in selected] == ["2", "1"]


def test_hard_inverse_grid_brackets_the_previous_upper_boundary():
    poisson = {row["candidate"]: row for row in candidates_for("poisson")}
    ns = {row["candidate"]: row for row in candidates_for("nsnonbounded")}

    assert poisson["obs_octuple"]["zeta_obs_u"] == 8 * poisson["baseline"]["zeta_obs_u"]
    assert poisson["obs_16x"]["zeta_obs_u"] == 16 * poisson["baseline"]["zeta_obs_u"]
    assert poisson["obs_64x"]["zeta_obs_u"] == 64 * poisson["baseline"]["zeta_obs_u"]
    assert poisson["obs_only"]["zeta_pde"] == 0.0
    assert ns["obs_quadruple"]["zeta_obs_u"] == 4 * ns["baseline"]["zeta_obs_u"]
    assert ns["clip_double"]["clip_threshold"] == 2 * ns["baseline"]["clip_threshold"]


def test_hard_inverse_refined_grid_targets_observed_frontiers():
    poisson = {row["candidate"]: row for row in candidates_for("poisson", "refined")}
    helmholtz = {row["candidate"]: row for row in candidates_for("helmholtz", "refined")}
    darcy = {row["candidate"]: row for row in candidates_for("darcy", "refined")}
    ns = {row["candidate"]: row for row in candidates_for("nsnonbounded", "refined")}

    assert poisson["obs_12x"]["zeta_obs_u"] == 12 * poisson["baseline"]["zeta_obs_u"]
    assert helmholtz["obs_24x"]["zeta_obs_u"] == 24 * helmholtz["baseline"]["zeta_obs_u"]
    assert darcy["obs_64x_clip100"]["clip_threshold"] == 100.0
    assert darcy["obs_64x_clip100_pde2x"]["zeta_pde"] == 2 * darcy["baseline"]["zeta_pde"]
    assert ns["clip_150"]["clip_threshold"] == 150.0
    assert ns["obs_half_clip100"]["zeta_obs_u"] == ns["baseline"]["zeta_obs_u"] / 2


def test_balanced_grid_covers_all_tasks_and_prior_inverse_frontiers():
    for pde in ("poisson", "helmholtz", "darcy", "nsnonbounded"):
        for task in ("both", "forward", "inverse"):
            candidates = balanced_candidates_for(pde, task)
            assert candidates[0]["candidate"] == "baseline"
            assert len({candidate["candidate"] for candidate in candidates}) == len(candidates)

    poisson_inverse = {
        row["candidate"]: row for row in balanced_candidates_for("poisson", "inverse")
    }
    darcy_inverse = {
        row["candidate"]: row for row in balanced_candidates_for("darcy", "inverse")
    }
    ns_inverse = {
        row["candidate"]: row for row in balanced_candidates_for("nsnonbounded", "inverse")
    }
    assert poisson_inverse["obs_14x"]["zeta_obs_u"] == 14 * poisson_inverse["baseline"]["zeta_obs_u"]
    assert darcy_inverse["obs_64x_clip100"]["clip_threshold"] == 100.0
    assert ns_inverse["clip_125"]["clip_threshold"] == 125.0


def test_balanced_sample_selection_is_disjoint_and_deterministic():
    rows = [
        {
            "sample_id": str(index),
            "rel_l2_a": str(index + 1),
            "rel_l2_u": str(200 - index),
        }
        for index in range(100)
    ]
    first = _select_strata(
        rows,
        pde="poisson",
        task="forward",
        test_type="id",
        tail_count=4,
        random_count=8,
    )
    second = _select_strata(
        rows,
        pde="poisson",
        task="forward",
        test_type="id",
        tail_count=4,
        random_count=8,
    )
    assert [(item[0], item[2]["sample_id"]) for item in first] == [
        (item[0], item[2]["sample_id"]) for item in second
    ]
    assert len(first) == len({item[2]["sample_id"] for item in first}) == 16
    assert [item[2]["sample_id"] for item in first if item[0] == "hard"] == [
        "0",
        "1",
        "2",
        "3",
    ]


def test_balanced_both_scope_requires_both_sides_to_improve():
    baseline = [
        {
            "test_type": "id",
            "subset_index": index,
            "rel_l2_a": 1.0,
            "rel_l2_u": 1.0,
            "pde_residual_norm": 1.0,
        }
        for index in range(2)
    ]
    candidate = [
        {
            **row,
            "rel_l2_a": 0.8,
            "rel_l2_u": 1.1 if index == 0 else 0.9,
        }
        for index, row in enumerate(baseline)
    ]
    stats = balanced_scope_stats(candidate, baseline, "both")
    assert stats["primary_mean_ratio"] == 1.0
    assert stats["primary_win_rate"] == 0.5


def test_balanced_analysis_paths_scope_distributed_servers_and_tasks(tmp_path):
    assert balanced_analysis_path(
        tmp_path,
        "selected_params",
        ["poisson", "helmholtz"],
        ["both", "forward", "inverse"],
        "round3",
    ) == tmp_path / "selected_params_poisson_helmholtz_round3.csv"
    assert balanced_analysis_path(
        tmp_path,
        "selected_params",
        ["poisson"],
        ["inverse"],
        "round3",
    ) == tmp_path / "selected_params_poisson_inverse_round3.csv"


def test_hard_inverse_chunking_and_scoped_analysis_paths(tmp_path):
    assert chunk_plan(10, 4) == [(0, 4), (4, 4), (8, 2)]
    assert analysis_path(tmp_path, "selected_params", ["poisson"]) == (
        tmp_path / "selected_params_poisson.csv"
    )
    assert analysis_path(
        tmp_path,
        "selected_params",
        ["nsnonbounded", "poisson", "darcy", "helmholtz"],
    ) == (tmp_path / "selected_params.csv")
    assert analysis_path(tmp_path, "selected_params", ["poisson"], "round2") == (
        tmp_path / "selected_params_poisson_round2.csv"
    )


def test_hard_inverse_resume_rejects_changed_or_unprovable_jobs(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text("rel_l2_a\n0.5\n", encoding="utf-8")
    signature = {
        "phase": "tune",
        "split": "tune",
        "pde": "poisson",
        "candidate": "baseline",
        "test_type": "id",
        "offset": 0,
        "batch_size": 4,
        "source_batch_size": 4,
        "source_indices": [0, 1, 2, 3],
        "zeta_obs_u": 360_000_000.0,
        "zeta_pde": 0.3,
        "clip_threshold": 50.0,
        "num_steps": 100,
        "checkpoint": {"path": "checkpoint", "size": 1, "mtime_ns": 1},
        "data": {"path": "data", "size": 1, "mtime_ns": 1},
    }
    exact = {"status": "ok", "job_signature": signature}
    assert _resume_status_matches(exact, signature, metrics_path=metrics_path)

    changed = {**signature, "num_steps": 200}
    assert not _resume_status_matches(exact, changed, metrics_path=metrics_path)

    legacy = {"status": "ok", **{key: value for key, value in signature.items() if key not in {
        "source_batch_size", "source_indices", "checkpoint", "data"
    }}}
    assert _resume_status_matches(legacy, signature, metrics_path=metrics_path)
    chunked = {**signature, "source_batch_size": 10}
    assert not _resume_status_matches(legacy, chunked, metrics_path=metrics_path)


def test_hard_inverse_analysis_excludes_prior_pilots_and_ambiguous_chunks():
    signed = {
        "job_signature": {"source_batch_size": 10, "num_steps": 100},
        "offset": 0,
        "batch_size": 10,
        "num_steps": 100,
    }
    assert _status_matches_analysis_plan(signed, source_batch_size=10, num_steps=100)
    assert not _status_matches_analysis_plan(signed, source_batch_size=4, num_steps=100)
    assert not _status_matches_analysis_plan(signed, source_batch_size=10, num_steps=200)

    legacy_full_batch = {"offset": 0, "batch_size": 4, "num_steps": 100}
    legacy_chunk = {"offset": 0, "batch_size": 2, "num_steps": 100}
    assert _status_matches_analysis_plan(
        legacy_full_batch, source_batch_size=4, num_steps=100
    )
    assert not _status_matches_analysis_plan(
        legacy_chunk, source_batch_size=4, num_steps=100
    )


def test_hard_inverse_winner_respects_per_distribution_guardrails():
    def row(candidate, robust, residual, solution, *, rough_residual=None):
        result = {
            "pde": "poisson",
            "candidate": candidate,
            "robust_score": robust,
            "pde_residual_norm_mean": residual,
            "rel_l2_u_mean": solution,
        }
        for test_type in ("id", "smooth", "rough"):
            result[f"pde_residual_norm_mean_{test_type}"] = residual
            result[f"rel_l2_u_mean_{test_type}"] = solution
        if rough_residual is not None:
            result["pde_residual_norm_mean_rough"] = rough_residual
        return result

    summaries = [
        row("baseline", 1.0, 1.0, 1.0),
        row("hidden_rough_regression", 0.3, 1.1, 0.8, rough_residual=1.3),
        row("unsafe_residual", 0.4, 1.3, 0.8),
        row("unsafe_solution", 0.5, 0.9, 1.1),
        row("eligible", 0.6, 1.2, 1.0),
    ]
    winner = select_winners(summaries, ["poisson"])[0]

    assert winner["candidate"] == "eligible"
    assert winner["passes_guardrails"] is True


def test_hard_inverse_a100_plan_splits_pdes_between_servers(tmp_path: Path):
    artifact_root = tmp_path / "inverse_hard_tuning"
    (artifact_root / "hard_samples.csv").parent.mkdir(parents=True)
    (artifact_root / "hard_samples.csv").touch()
    for split in ("tune", "holdout"):
        for pde in ("poisson", "helmholtz", "darcy", "nsnonbounded"):
            for test_type in ("id", "smooth", "rough"):
                subset = artifact_root / "subsets" / split / f"{pde}_{test_type}.mat"
                subset.parent.mkdir(parents=True, exist_ok=True)
                subset.touch()

    def run_plan(**updates: str) -> str:
        return subprocess.run(
            ["bash", "scripts/tuning/run_hard_inverse_a100.sh"],
            check=True,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "ARTIFACT_ROOT": str(artifact_root),
                "PLAN_ONLY": "true",
                **updates,
            },
        ).stdout

    output_zero = run_plan(SERVER_RANK="0")
    output_one = run_plan(SERVER_RANK="1")
    assert "--pdes poisson\\,helmholtz" in output_zero
    assert "--pdes darcy\\,nsnonbounded" in output_one

    refined = run_plan(
        SERVER_RANK="0", CANDIDATE_SET="refined", ANALYSIS_LABEL="round2"
    )
    assert "--candidate-set refined" in refined
    assert "--analysis-label round2" in refined
